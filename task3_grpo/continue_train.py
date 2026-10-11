from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import (
    batch_generate,
    response_token_logprobs,
    score_reward_pairs,
)
from common.logging_utils import set_seed
from common.metrics import masked_mean, sampled_kl, sample_entropy
from common.models import (
    clear_gpu,
    load_policy,
    load_reward_model,
    load_tokenizer,
    reference_mode,
    trainable_parameters,
)
from task3_grpo.grpo import (
    group_relative_advantages,
    grpo_policy_loss,
    mask_truncated_sequences,
)


def disable_dropout(model: torch.nn.Module) -> None:
    """Disable dropout layers to guarantee numerical consistency in RL passes."""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0


def prepare_grpo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["grpo_midpoint_policy"],
        trainable=True,
    )
    disable_dropout(policy)

    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])
    optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["learning_rate"]))

    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "optimizer": optimizer,
    }


def run_grpo(
    config_path: str,
    output: str | None = None,
    updates: int | None = None,
    loss_type: str = "grpo",
    run_name: str = "standard",
):
    bundle = prepare_grpo_continuation(config_path)
    cfg = bundle["cfg"]
    policy = bundle["policy"]
    tokenizer = bundle["tokenizer"]
    reward_model = bundle["reward_model"]
    reward_tokenizer = bundle["reward_tokenizer"]
    prompt_rows = bundle["prompt_rows"]
    optimizer = bundle["optimizer"]

    total_updates = int(updates if updates is not None else cfg.get("updates", 20))
    num_generations = int(cfg.get("num_generations", 4))
    prompts_per_update = int(cfg.get("prompts_per_update", 1))
    eps = float(cfg.get("clip_epsilon", 0.20))
    beta = float(cfg.get("kl_beta", 0.10))
    policy_epochs = int(cfg.get("policy_epochs", 1))
    max_prompt_len = int(cfg.get("max_prompt_length", 256))
    max_comp_len = int(cfg.get("max_completion_length", 512))
    mask_truncated = bool(cfg.get("mask_truncated_completions", True))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
    temp = float(cfg.get("generation", {}).get("temperature", 0.7))
    top_p = float(cfg.get("generation", {}).get("top_p", 0.9))

    out_path = repo_path(output or cfg.get("output", "outputs/task3_grpo/standard"))
    out_path.mkdir(parents=True, exist_ok=True)
    results_dir = repo_path(cfg.get("results_dir", "results/task3_grpo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    device = next(policy.parameters()).device
    policy.config.use_cache = False

    print(f"\n=== Starting GRPO Continuation: {run_name} ===")
    print(f"Updates: {total_updates} | K={num_generations} per prompt | Loss: {loss_type}")
    print(f"Learning rate: {cfg['learning_rate']} | Epsilon: {eps} | Beta KL: {beta}")
    print(f"Mask truncated completions: {mask_truncated}")

    train_logs = []
    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for u in range(total_updates):
        # 1. Sample prompt(s) and replicate K times for within-group competition
        batch_slice = [
            prompt_rows[(u * prompts_per_update + i) % len(prompt_rows)]
            for i in range(prompts_per_update)
        ]
        base_prompts = [prompt_messages(r) for r in batch_slice]

        flat_prompts = []
        group_id_list = []
        for g_idx, p in enumerate(base_prompts):
            for _ in range(num_generations):
                flat_prompts.append(p)
                group_id_list.append(g_idx)
        group_ids = torch.tensor(group_id_list, device=device)

        # 2. Collect on-policy rollouts
        policy.eval()
        gen_out = batch_generate(
            model=policy,
            tokenizer=tokenizer,
            prompts=flat_prompts,
            max_prompt_length=max_prompt_len,
            max_new_tokens=max_comp_len,
            temperature=temp,
            top_p=top_p,
            do_sample=True,
        )

        sequences = gen_out["sequences"]
        attention_mask = gen_out["attention_mask"]
        prompt_width = gen_out["prompt_width"]
        response_ids = gen_out["response_ids"]
        response_mask = gen_out["response_mask"]
        responses = gen_out["responses"]
        terminated = gen_out["terminated_with_eos"]
        truncated = gen_out["truncated"]
        response_lengths = gen_out["response_lengths"]

        # 3. Score completions with frozen reward model
        if reward_model is not None and reward_tokenizer is not None:
            raw_rewards = score_reward_pairs(
                reward_model,
                reward_tokenizer,
                flat_prompts,
                responses,
                max_length=int(cfg.get("reward_max_length", 1280)),
            )
        else:
            raw_rewards = torch.zeros(len(responses), device=device)

        # 4. Mask max-length completions if configured
        if mask_truncated:
            effective_mask = mask_truncated_sequences(response_mask, truncated)
        else:
            effective_mask = response_mask

        # 5. Compute within-group relative advantages
        advantages, group_diags = group_relative_advantages(
            rewards=raw_rewards,
            group_ids=group_ids,
            eps=1e-6,
            return_diags=True,
        )

        # 6. Evaluate old and reference log-probabilities
        with torch.no_grad():
            old_logp, _ = response_token_logprobs(
                policy, sequences, attention_mask, prompt_width, response_ids
            )
            with reference_mode(policy):
                ref_logp, _ = response_token_logprobs(
                    policy, sequences, attention_mask, prompt_width, response_ids
                )

        # 7. GRPO Policy Optimization
        policy.train()
        epoch_losses = []
        epoch_kls = []
        epoch_clips = []
        epoch_affected = []
        epoch_entropies = []
        epoch_gnorms = []

        for p_epoch in range(policy_epochs):
            optimizer.zero_grad()
            new_logp, _ = response_token_logprobs(
                policy, sequences, attention_mask, prompt_width, response_ids
            )

            loss, loss_diags = grpo_policy_loss(
                new_logp=new_logp,
                old_logp=old_logp,
                seq_adv=advantages,
                token_mask=effective_mask,
                ref_logp=ref_logp,
                eps=eps,
                beta=beta,
                loss_type=loss_type,
                max_completion_length=max_comp_len,
            )

            loss.backward()

            if max_grad_norm > 0:
                gnorm = torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), max_grad_norm)
                gnorm_val = float(gnorm.item() if hasattr(gnorm, "item") else gnorm)
            else:
                gnorm_val = 0.0

            if torch.isfinite(torch.tensor(gnorm_val)):
                optimizer.step()
            else:
                optimizer.zero_grad()

            epoch_losses.append(loss.item() if torch.isfinite(loss) else 0.0)
            epoch_kls.append(loss_diags["sampled_kl"].item())
            epoch_clips.append(loss_diags["clip_fraction"].item())
            epoch_affected.append(loss_diags["affected_fraction"].item())
            epoch_entropies.append(loss_diags["sample_entropy"].item())
            epoch_gnorms.append(gnorm_val)

        # 8. Record Metrics
        mean_resp_len = float(sum(response_lengths) / len(response_lengths))
        trunc_rate = float(sum(truncated) / len(truncated))
        peak_vram = (
            torch.cuda.max_memory_allocated() / (1024**3)
            if torch.cuda.is_available()
            else 0.0
        )
        elapsed_sec = time.time() - start_time

        record = {
            "update": u + 1,
            "mean_reward": round(float(raw_rewards.mean().item()), 4),
            "within_group_reward_std": round(group_diags["within_group_reward_std"], 4),
            "uninformative_group_fraction": round(group_diags["uninformative_group_fraction"], 4),
            "policy_loss": round(float(sum(epoch_losses) / len(epoch_losses)), 4),
            "sampled_kl": round(float(sum(epoch_kls) / len(epoch_kls)), 5),
            "clip_fraction": round(float(sum(epoch_clips) / len(epoch_clips)), 4),
            "affected_fraction": round(float(sum(epoch_affected) / len(epoch_affected)), 4),
            "sample_entropy": round(float(sum(epoch_entropies) / len(epoch_entropies)), 4),
            "policy_grad_norm": round(float(sum(epoch_gnorms) / len(epoch_gnorms)), 4),
            "mean_response_length": round(mean_resp_len, 2),
            "truncation_rate": round(trunc_rate, 4),
            "all_eos_terminated": bool(all(terminated)),
            "wall_clock_sec": round(elapsed_sec, 2),
            "peak_vram_gb": round(peak_vram, 3),
        }
        train_logs.append(record)

        print(
            f"Update {u+1:02d}/{total_updates} | "
            f"Reward: {record['mean_reward']:.3f} | "
            f"GroupStd: {record['within_group_reward_std']:.3f} | "
            f"Uninf%: {record['uninformative_group_fraction']:.1%} | "
            f"KL: {record['sampled_kl']:.4f} | "
            f"Loss: {record['policy_loss']:.4f} | "
            f"Len: {record['mean_response_length']:.0f}"
        )

    # 9. Save Checkpoint & Logs
    print(f"\nSaving continued GRPO policy to {out_path}...")
    policy.save_pretrained(str(out_path))
    tokenizer.save_pretrained(str(out_path))

    log_path = results_dir / f"{run_name}_train_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_name": run_name,
                "config_path": config_path,
                "loss_type": loss_type,
                "total_updates": total_updates,
                "num_generations": num_generations,
                "clip_epsilon": eps,
                "kl_beta": beta,
                "total_wall_clock_sec": round(time.time() - start_time, 2),
                "peak_vram_gb": round(
                    torch.cuda.max_memory_allocated() / (1024**3)
                    if torch.cuda.is_available()
                    else 0.0,
                    3,
                ),
                "history": train_logs,
            },
            f,
            indent=2,
        )
    print(f"GRPO continuation logs saved to {log_path}")
    clear_gpu()

    return {
        "output_path": str(out_path),
        "total_updates": total_updates,
        "history": train_logs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--loss-type", choices=["grpo", "dr_grpo"], default="grpo")
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_grpo(args.config, args.output, args.updates, args.loss_type, args.run_name)


if __name__ == "__main__":
    main()
