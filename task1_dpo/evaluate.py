from __future__ import annotations

import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from common.data import (
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.generation import (
    batch_generate,
    response_sequence_logprobs,
    score_reward_pairs,
)
from common.metrics import preference_accuracy, sampled_kl
from common.models import (
    load_policy,
    load_reward_model,
    load_tokenizer,
    reference_mode,
)
from task1_dpo.dpo import dpo_loss
from task1_dpo.train import make_collate


def load_evaluation_bundle(
    config_path: str,
    adapter: str,
    eval_dataset_path: str | None = None,
    load_reward: bool = True,
):
    cfg = load_yaml(config_path)
    eval_path = eval_dataset_path or cfg["paths"]["dpo_standard_eval"]
    rows = read_jsonl(eval_path)
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=adapter, trainable=False)
    reward_bundle = load_reward_model(cfg) if load_reward else (None, None)

    return {
        "cfg": cfg,
        "rows": rows,
        "tokenizer": tokenizer,
        "policy": policy,
        "reward": reward_bundle,
        "eval_path": eval_path,
    }


def evaluate_dpo_policy(
    config_path: str,
    adapter: str,
    name: str = "standard",
    eval_dataset_path: str | None = None,
    max_eval_examples: int | None = None,
    generate_eval: bool = True,
    max_gen_examples: int | None = 64,
    load_reward: bool = True,
    bundle: dict | None = None,
):
    if bundle is None:
        bundle = load_evaluation_bundle(
            config_path, adapter, eval_dataset_path, load_reward=load_reward
        )
    cfg = bundle["cfg"]
    policy = bundle["policy"]
    tokenizer = bundle["tokenizer"]
    rm_model, rm_tokenizer = bundle["reward"]
    rows = bundle["rows"]
    if max_eval_examples is not None:
        rows = rows[: int(max_eval_examples)]

    device = next(policy.parameters()).device
    beta = float(cfg.get("beta", 0.10))
    batch_size = int(cfg.get("batch_size", 2))
    max_seq_len = int(cfg.get("max_sequence_length", 768))

    loader = DataLoader(
        rows,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_collate(tokenizer, max_seq_len),
    )

    print(f"\n--- Evaluating DPO Policy: {name} ---")
    print(f"Total evaluation pairs: {len(rows)} | Adapter: {adapter}")

    all_losses = []
    all_margins = []
    all_accuracies = []
    all_chosen_kls = []
    all_rejected_kls = []

    # 1. Held-out Preference Margin, Accuracy, and KL
    with torch.no_grad():
        for chosen_batch, rejected_batch in loader:
            chosen_batch = {k: v.to(device) for k, v in chosen_batch.items()}
            rejected_batch = {k: v.to(device) for k, v in rejected_batch.items()}

            # Reference logps
            with reference_mode(policy):
                ref_chosen_seq_logp, ref_chosen_tok_logp, chosen_mask = response_sequence_logprobs(
                    policy, chosen_batch
                )
                ref_rej_seq_logp, ref_rej_tok_logp, rej_mask = response_sequence_logprobs(
                    policy, rejected_batch
                )

            # Policy logps
            pol_chosen_seq_logp, pol_chosen_tok_logp, _ = response_sequence_logprobs(
                policy, chosen_batch
            )
            pol_rej_seq_logp, pol_rej_tok_logp, _ = response_sequence_logprobs(
                policy, rejected_batch
            )

            loss, diags = dpo_loss(
                pol_chosen_seq_logp,
                pol_rej_seq_logp,
                ref_chosen_seq_logp,
                ref_rej_seq_logp,
                beta=beta,
            )

            all_losses.append(loss.item())
            dpo_margin = diags["dpo_margin_mean"].item()
            all_margins.append(dpo_margin)
            all_accuracies.append(diags["preference_accuracy"].item())

            # Sampled token KL for chosen and rejected responses
            chosen_kl = sampled_kl(pol_chosen_tok_logp, ref_chosen_tok_logp, chosen_mask).item()
            rej_kl = sampled_kl(pol_rej_tok_logp, ref_rej_tok_logp, rej_mask).item()
            all_chosen_kls.append(chosen_kl)
            all_rejected_kls.append(rej_kl)

    mean_loss = float(np.mean(all_losses)) if all_losses else 0.0
    mean_margin = float(np.mean(all_margins)) if all_margins else 0.0
    mean_acc = float(np.mean(all_accuracies)) if all_accuracies else 0.0
    mean_chosen_kl = float(np.mean(all_chosen_kls)) if all_chosen_kls else 0.0
    mean_rej_kl = float(np.mean(all_rejected_kls)) if all_rejected_kls else 0.0
    mean_kl = 0.5 * (mean_chosen_kl + mean_rej_kl)

    # 2. Generation, Response Length, and Reward-Model Scoring
    gen_results = {}
    qualitative_samples = []
    if generate_eval:
        gen_rows = rows if max_gen_examples is None else rows[: int(max_gen_examples)]
        prompts = [prompt_messages_from_preference(r) for r in gen_rows]

        gen_batch_size = 4
        all_lengths = []
        all_rewards = []
        all_responses = []

        max_new_tokens = int(cfg.get("max_generation_tokens", 256))
        temp = float(cfg.get("generation", {}).get("temperature", 0.7))
        top_p = float(cfg.get("generation", {}).get("top_p", 0.9))
        do_sample = bool(cfg.get("generation", {}).get("do_sample", True))

        for i in range(0, len(prompts), gen_batch_size):
            batch_prompts = prompts[i : i + gen_batch_size]
            batch_out = batch_generate(
                policy,
                tokenizer,
                batch_prompts,
                max_prompt_length=max_seq_len,
                max_new_tokens=max_new_tokens,
                temperature=temp,
                top_p=top_p,
                do_sample=do_sample,
            )
            all_responses.extend(batch_out["responses"])
            all_lengths.extend(batch_out["response_lengths"])

            if rm_model is not None and rm_tokenizer is not None:
                rewards = score_reward_pairs(
                    rm_model, rm_tokenizer, batch_prompts, batch_out["responses"]
                )
                all_rewards.extend([float(r) for r in rewards.cpu().numpy()])

        for idx, (p, resp, lgt) in enumerate(zip(prompts, all_responses, all_lengths)):
            rw = all_rewards[idx] if all_rewards else None
            sample = {
                "prompt": p,
                "response": resp,
                "length_tokens": int(lgt),
                "reward_score": round(float(rw), 4) if rw is not None else None,
            }
            qualitative_samples.append(sample)

        gen_results = {
            "n_generated": len(all_responses),
            "mean_response_length": float(np.mean(all_lengths)) if all_lengths else 0.0,
            "std_response_length": float(np.std(all_lengths)) if all_lengths else 0.0,
            "median_response_length": float(np.median(all_lengths)) if all_lengths else 0.0,
            "mean_reward_score": float(np.mean(all_rewards)) if all_rewards else None,
            "std_reward_score": float(np.std(all_rewards)) if all_rewards else None,
        }

    summary = {
        "name": name,
        "adapter": adapter,
        "beta": beta,
        "n_eval_pairs": len(rows),
        "held_out_dpo_loss": round(mean_loss, 4),
        "held_out_dpo_margin": round(mean_margin, 4),
        "held_out_preference_accuracy": round(mean_acc, 4),
        "held_out_kl": round(mean_kl, 5),
        "held_out_chosen_kl": round(mean_chosen_kl, 5),
        "held_out_rejected_kl": round(mean_rej_kl, 5),
        **gen_results,
    }

    print("\n--- Evaluation Summary ---")
    print(f"Preference Accuracy: {summary['held_out_preference_accuracy']:.2%}")
    print(f"DPO Loss: {summary['held_out_dpo_loss']:.4f} | Margin: {summary['held_out_dpo_margin']:.4f}")
    print(f"Sampled KL: {summary['held_out_kl']:.5f}")
    if "mean_reward_score" in summary and summary["mean_reward_score"] is not None:
        print(f"Mean Reward: {summary['mean_reward_score']:.4f} ± {summary['std_reward_score']:.4f}")
    if "mean_response_length" in summary:
        print(f"Mean Response Length: {summary['mean_response_length']:.1f} ± {summary['std_response_length']:.1f} tokens")

    # Save to results directory
    results_dir = repo_path(cfg.get("results_dir", "results/task1_dpo"))
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / f"{name}_eval.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary,
                "qualitative_samples": qualitative_samples[:20],
            },
            f,
            indent=2,
        )
    print(f"Saved evaluation results to {out_file}")

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--max-examples", type=int)
    ap.add_argument("--no-generate", action="store_true")
    ap.add_argument("--no-reward", action="store_true")
    args = ap.parse_args()

    evaluate_dpo_policy(
        config_path=args.config,
        adapter=args.adapter,
        name=args.name,
        eval_dataset_path=args.dataset,
        max_eval_examples=args.max_examples,
        generate_eval=not args.no_generate,
        load_reward=not args.no_reward,
    )


if __name__ == "__main__":
    main()

