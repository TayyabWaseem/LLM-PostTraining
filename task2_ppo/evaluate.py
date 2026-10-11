import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import (
    batch_generate,
    response_token_logprobs,
    score_reward_pairs,
)
from common.metrics import sample_entropy, sampled_kl
from common.models import (
    clear_gpu,
    load_policy,
    load_reward_model,
    load_tokenizer,
    reference_mode,
)


def load_evaluation_bundle(config_path: str, adapter: str | None = None, load_reward: bool = True):
    cfg = load_yaml(config_path)
    eval_rows = read_jsonl(cfg["paths"]["rl_prompt_eval"])
    tokenizer = load_tokenizer(cfg["base_model"])
    target_adapter = adapter
    if target_adapter is None:
        target_adapter = cfg.get("output", "outputs/task2_ppo/standard")
    elif str(target_adapter).lower() == "none":
        target_adapter = None
    policy = load_policy(cfg, adapter_path=target_adapter, trainable=False)
    rm_bundle = load_reward_model(cfg) if load_reward else (None, None)

    return {
        "cfg": cfg,
        "rows": eval_rows,
        "tokenizer": tokenizer,
        "policy": policy,
        "reward": rm_bundle,
        "adapter_path": target_adapter,
    }


def evaluate_ppo_policy(
    config_path: str,
    adapter: str | None = None,
    name: str = "standard",
    max_eval_examples: int | None = None,
    load_reward: bool = True,
    bundle: dict | None = None,
):
    if bundle is None:
        bundle = load_evaluation_bundle(config_path, adapter, load_reward=load_reward)

    cfg = bundle["cfg"]
    policy = bundle["policy"]
    tokenizer = bundle["tokenizer"]
    rm_model, rm_tokenizer = bundle["reward"]
    rows = bundle["rows"]
    if max_eval_examples is not None:
        rows = rows[: int(max_eval_examples)]

    results_dir = repo_path(cfg.get("results_dir", "results/task2_ppo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    max_prompt_len = int(cfg.get("max_prompt_length", 256))
    max_resp_len = int(cfg.get("eval_max_response_length", 768))
    missing_eos_penalty = float(cfg.get("missing_eos_penalty", 1.0))
    temp = float(cfg.get("generation", {}).get("temperature", 0.7))
    top_p = float(cfg.get("generation", {}).get("top_p", 0.9))
    do_sample = bool(cfg.get("generation", {}).get("do_sample", True))

    prompts = [prompt_messages(r) for r in rows]
    batch_size = 4

    print(f"\n--- Evaluating PPO Policy: {name} ---")
    print(f"Total evaluation prompts: {len(prompts)} | Eval response cap: {max_resp_len} tokens")

    all_responses = []
    all_lengths = []
    all_terminated = []
    all_truncated = []
    all_raw_rewards = []
    all_effective_rewards = []
    all_kls = []
    all_entropies = []
    qualitative_samples = []

    for i in range(0, len(prompts), batch_size):
        b_prompts = prompts[i : i + batch_size]
        b_rows = rows[i : i + batch_size]

        gen_out = batch_generate(
            model=policy,
            tokenizer=tokenizer,
            prompts=b_prompts,
            max_prompt_length=max_prompt_len,
            max_new_tokens=max_resp_len,
            temperature=temp,
            top_p=top_p,
            do_sample=do_sample,
        )

        seq = gen_out["sequences"]
        attn = gen_out["attention_mask"]
        p_width = gen_out["prompt_width"]
        resp_ids = gen_out["response_ids"]
        resp_mask = gen_out["response_mask"]
        resps = gen_out["responses"]
        terms = gen_out["terminated_with_eos"]
        truncs = gen_out["truncated"]
        lengths = gen_out["response_lengths"]

        all_responses.extend(resps)
        all_lengths.extend(lengths)
        all_terminated.extend(terms)
        all_truncated.extend(truncs)

        # 1. Reward model scoring
        if rm_model is not None and rm_tokenizer is not None:
            rw_raw = score_reward_pairs(
                rm_model, rm_tokenizer, b_prompts, resps, max_length=int(cfg.get("reward_max_length", 1280))
            )
            rw_effective = rw_raw.clone()
            for b_idx, term in enumerate(terms):
                if not term:
                    rw_effective[b_idx] -= missing_eos_penalty

            rw_raw_list = [float(r) for r in rw_raw.cpu().numpy()]
            rw_eff_list = [float(r) for r in rw_effective.cpu().numpy()]
            all_raw_rewards.extend(rw_raw_list)
            all_effective_rewards.extend(rw_eff_list)
        else:
            rw_raw_list = [None] * len(resps)
            rw_eff_list = [None] * len(resps)

        # 2. Token KL from reference policy and Entropy
        with torch.no_grad():
            pol_logp, _ = response_token_logprobs(policy, seq, attn, p_width, resp_ids)
            with reference_mode(policy):
                ref_logp, _ = response_token_logprobs(policy, seq, attn, p_width, resp_ids)

            for b_idx in range(seq.shape[0]):
                m = resp_mask[b_idx : b_idx + 1]
                p_lp = pol_logp[b_idx : b_idx + 1]
                r_lp = ref_logp[b_idx : b_idx + 1]
                kl_val = sampled_kl(p_lp, r_lp, m).item()
                ent_val = sample_entropy(p_lp, m).item()
                all_kls.append(kl_val)
                all_entropies.append(ent_val)

        for b_idx in range(len(resps)):
            qualitative_samples.append(
                {
                    "prompt_id": b_rows[b_idx].get("prompt_id"),
                    "prompt": b_prompts[b_idx],
                    "response": resps[b_idx],
                    "length_tokens": int(lengths[b_idx]),
                    "terminated_with_eos": bool(terms[b_idx]),
                    "raw_reward": round(rw_raw_list[b_idx], 4) if rw_raw_list[b_idx] is not None else None,
                    "effective_reward": round(rw_eff_list[b_idx], 4) if rw_eff_list[b_idx] is not None else None,
                    "token_kl": round(float(all_kls[i + b_idx]), 5),
                }
            )

    summary = {
        "name": name,
        "adapter": str(bundle.get("adapter_path") or adapter or "base_model"),
        "n_evaluated": len(prompts),
        "mean_raw_reward": float(np.mean(all_raw_rewards)) if all_raw_rewards else None,
        "std_raw_reward": float(np.std(all_raw_rewards)) if all_raw_rewards else None,
        "mean_effective_reward": float(np.mean(all_effective_rewards)) if all_effective_rewards else None,
        "std_effective_reward": float(np.std(all_effective_rewards)) if all_effective_rewards else None,
        "mean_token_kl": float(np.mean(all_kls)) if all_kls else 0.0,
        "std_token_kl": float(np.std(all_kls)) if all_kls else 0.0,
        "mean_entropy": float(np.mean(all_entropies)) if all_entropies else 0.0,
        "mean_response_length": float(np.mean(all_lengths)) if all_lengths else 0.0,
        "std_response_length": float(np.std(all_lengths)) if all_lengths else 0.0,
        "median_response_length": float(np.median(all_lengths)) if all_lengths else 0.0,
        "eos_termination_rate": float(np.mean(all_terminated)) if all_terminated else 0.0,
        "truncation_rate": float(np.mean(all_truncated)) if all_truncated else 0.0,
    }

    print("\n--- PPO Evaluation Summary ---")
    if summary["mean_effective_reward"] is not None:
        print(f"Effective Reward: {summary['mean_effective_reward']:.4f} ± {summary['std_effective_reward']:.4f}")
    print(f"Token KL vs Ref: {summary['mean_token_kl']:.5f} ± {summary['std_token_kl']:.5f}")
    print(f"Entropy: {summary['mean_entropy']:.4f}")
    print(f"Mean Response Length: {summary['mean_response_length']:.1f} tokens")
    print(f"EOS Rate: {summary['eos_termination_rate']:.1%} | Truncation Rate: {summary['truncation_rate']:.1%}")

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
    clear_gpu()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--adapter")
    ap.add_argument("--name", default="standard")
    ap.add_argument("--max-examples", type=int)
    ap.add_argument("--no-reward", action="store_true")
    args = ap.parse_args()

    evaluate_ppo_policy(
        config_path=args.config,
        adapter=args.adapter,
        name=args.name,
        max_eval_examples=args.max_examples,
        load_reward=not args.no_reward,
    )


if __name__ == "__main__":
    main()

