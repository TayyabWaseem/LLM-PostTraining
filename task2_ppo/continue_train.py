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
from common.metrics import masked_mean, sample_entropy, sampled_kl
from common.models import (
    clear_gpu,
    load_policy,
    load_reward_model,
    load_tokenizer,
    load_value_model,
    reference_mode,
    token_values,
    trainable_parameters,
    value_parameter_groups,
)
from task2_ppo.ppo import (
    compute_gae,
    explained_variance,
    normalize_advantages,
    ppo_policy_loss,
    shaped_rewards,
    value_mse_loss,
)


def disable_dropout(model: torch.nn.Module) -> None:
    """Disable dropout layers to guarantee numerical consistency between rollout and training passes."""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0


def prepare_ppo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["ppo_midpoint_policy"],
        trainable=True,
    )
    disable_dropout(policy)

    value_model = load_value_model(
        cfg,
        cfg["paths"]["ppo_midpoint_value"],
        train_mode=cfg.get("value_train_mode", "lora_head"),
    )
    disable_dropout(value_model)

    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])

    policy_optimizer = AdamW(
        trainable_parameters(policy),
        lr=float(cfg["policy_learning_rate"]),
    )
    value_optimizer = AdamW(
        value_parameter_groups(
            value_model,
            lora_lr=float(cfg["value_lora_learning_rate"]),
            head_lr=float(cfg["value_head_learning_rate"]),
        ),
        weight_decay=0.0,
    )

    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "value_model": value_model,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "policy_optimizer": policy_optimizer,
        "value_optimizer": value_optimizer,
    }


def run_ppo(
    config_path: str,
    output: str | None = None,
    updates: int | None = None,
    clip_epsilon: float | None = None,
    kl_beta: float | None = None,
    run_name: str = "standard",
):
    bundle = prepare_ppo_continuation(config_path)
    cfg = bundle["cfg"]
    policy = bundle["policy"]
    value_model = bundle["value_model"]
    tokenizer = bundle["tokenizer"]
    reward_model = bundle["reward_model"]
    reward_tokenizer = bundle["reward_tokenizer"]
    prompt_rows = bundle["prompt_rows"]
    policy_optimizer = bundle["policy_optimizer"]
    value_optimizer = bundle["value_optimizer"]

    total_updates = int(updates if updates is not None else cfg.get("updates", 20))
    eps = float(clip_epsilon if clip_epsilon is not None else cfg.get("clip_epsilon", 0.20))
    beta_kl = float(kl_beta if kl_beta is not None else cfg.get("kl_beta", 0.10))
    ppo_epochs = int(cfg.get("ppo_epochs", 2))
    prompts_per_update = int(cfg.get("prompts_per_update", 1))
    missing_eos_penalty = float(cfg.get("missing_eos_penalty", 1.0))
    value_coef = float(cfg.get("value_coef", 0.50))
    gamma = float(cfg.get("gamma", 1.0))
    lam = float(cfg.get("gae_lambda", 0.95))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))

    out_path = repo_path(output or cfg.get("output", "outputs/task2_ppo/standard"))
    out_path.mkdir(parents=True, exist_ok=True)
    results_dir = repo_path(cfg.get("results_dir", "results/task2_ppo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    device = next(policy.parameters()).device
    policy.config.use_cache = False
    value_model.config.use_cache = False

    print(f"\n=== Starting PPO Continuation: {run_name} ===")
    print(f"Updates: {total_updates} | Epsilon: {eps} | Beta KL: {beta_kl} | PPO Epochs: {ppo_epochs}")
    print(f"Policy lr: {cfg['policy_learning_rate']} | Value lr: {cfg['value_lora_learning_rate']}")

    train_logs = []
    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for u in range(total_updates):
        u_start_time = time.time()

        # 1. Sample prompt(s) for this update
        batch_slice = [
            prompt_rows[(u * prompts_per_update + i) % len(prompt_rows)]
            for i in range(prompts_per_update)
        ]
        prompts = [prompt_messages(r) for r in batch_slice]

        # 2. Collect on-policy rollout
        policy.eval()
        gen_out = batch_generate(
            model=policy,
            tokenizer=tokenizer,
            prompts=prompts,
            max_prompt_length=int(cfg.get("max_prompt_length", 256)),
            max_new_tokens=int(cfg.get("max_response_length", 512)),
            temperature=float(cfg.get("generation", {}).get("temperature", 0.7)),
            top_p=float(cfg.get("generation", {}).get("top_p", 0.9)),
            do_sample=bool(cfg.get("generation", {}).get("do_sample", True)),
        )

        sequences = gen_out["sequences"]
        attention_mask = gen_out["attention_mask"]
        prompt_width = gen_out["prompt_width"]
        response_ids = gen_out["response_ids"]
        response_mask = gen_out["response_mask"]
        responses = gen_out["responses"]
        terminated = gen_out["terminated_with_eos"]
        response_lengths = gen_out["response_lengths"]

        # 3. Score learned terminal reward
        if reward_model is not None and reward_tokenizer is not None:
            raw_rewards = score_reward_pairs(
                reward_model,
                reward_tokenizer,
                prompts,
                responses,
                max_length=int(cfg.get("reward_max_length", 1280)),
            )
        else:
            raw_rewards = torch.zeros(len(responses), device=device)

        # Apply missing EOS penalty
        effective_rewards = raw_rewards.clone()
        for b, term in enumerate(terminated):
            if not term:
                effective_rewards[b] -= missing_eos_penalty

        # 4. Old and Reference Log-Probabilities (unscaled temperature=1.0 logits)
        with torch.no_grad():
            old_logp, _ = response_token_logprobs(
                policy, sequences, attention_mask, prompt_width, response_ids
            )
            with reference_mode(policy):
                ref_logp, _ = response_token_logprobs(
                    policy, sequences, attention_mask, prompt_width, response_ids
                )

            # Token values from critic
            full_values = token_values(value_model, sequences, attention_mask)
            values = full_values[:, prompt_width : prompt_width + response_ids.shape[1]]

        # Token-level KL shaping
        token_rewards = shaped_rewards(
            task_reward=effective_rewards,
            policy_logp=old_logp,
            ref_logp=ref_logp,
            response_mask=response_mask,
            beta_kl=beta_kl,
        )

        # 5. GAE & Raw Returns calculation (whitening applied ONLY to policy advantages)
        raw_advantages, returns = compute_gae(
            rewards=token_rewards,
            values=values,
            mask=response_mask,
            gamma=gamma,
            lam=lam,
        )
        norm_advantages = normalize_advantages(raw_advantages, response_mask)

        # 6. PPO Optimization Epochs
        policy.train()
        value_model.train()

        update_pol_losses = []
        update_val_losses = []
        update_clip_fractions = []
        update_affected_fractions = []
        update_approx_kls = []
        update_entropies = []
        update_evs = []
        update_pol_gnorms = []
        update_val_gnorms = []

        for p_epoch in range(ppo_epochs):
            policy_optimizer.zero_grad()
            value_optimizer.zero_grad()

            new_logp, _ = response_token_logprobs(
                policy, sequences, attention_mask, prompt_width, response_ids
            )
            new_full_values = token_values(value_model, sequences, attention_mask)
            new_values = new_full_values[:, prompt_width : prompt_width + response_ids.shape[1]]

            pol_loss, ppo_diags = ppo_policy_loss(
                new_logp=new_logp,
                old_logp=old_logp,
                advantage=norm_advantages,
                mask=response_mask,
                eps=eps,
            )

            val_loss = value_mse_loss(new_values, returns, response_mask)

            # Separate backward passes to prevent critic instability from corrupting policy gradients
            if torch.isfinite(pol_loss):
                pol_loss.backward()
            if torch.isfinite(val_loss):
                (value_coef * val_loss).backward()

            if max_grad_norm > 0:
                pol_gnorm = torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), max_grad_norm)
                val_gnorm = torch.nn.utils.clip_grad_norm_(trainable_parameters(value_model), max_grad_norm)
                pol_gnorm_val = float(pol_gnorm.item() if hasattr(pol_gnorm, "item") else pol_gnorm)
                val_gnorm_val = float(val_gnorm.item() if hasattr(val_gnorm, "item") else val_gnorm)
            else:
                pol_gnorm_val, val_gnorm_val = 0.0, 0.0

            if torch.isfinite(torch.tensor(pol_gnorm_val)):
                policy_optimizer.step()
            else:
                policy_optimizer.zero_grad()

            if torch.isfinite(torch.tensor(val_gnorm_val)):
                value_optimizer.step()
            else:
                value_optimizer.zero_grad()

            entropy_val = sample_entropy(new_logp, response_mask).item()
            ev_val = explained_variance(new_values.detach(), returns, response_mask)

            update_pol_losses.append(pol_loss.item() if torch.isfinite(pol_loss) else 0.0)
            update_val_losses.append(val_loss.item() if torch.isfinite(val_loss) else 0.0)
            update_clip_fractions.append(ppo_diags["clip_fraction"].item())
            update_affected_fractions.append(ppo_diags["affected_fraction"].item())
            update_approx_kls.append(ppo_diags["approx_kl"].item())
            update_entropies.append(entropy_val)
            update_evs.append(ev_val)
            update_pol_gnorms.append(pol_gnorm_val)
            update_val_gnorms.append(val_gnorm_val)

        # 7. Record Update Metrics
        token_kl = sampled_kl(old_logp, ref_logp, response_mask).item()
        mean_resp_len = float(sum(response_lengths) / len(response_lengths))
        peak_vram = (
            torch.cuda.max_memory_allocated() / (1024**3)
            if torch.cuda.is_available()
            else 0.0
        )
        elapsed_sec = time.time() - start_time

        record = {
            "update": u + 1,
            "raw_reward_mean": round(float(raw_rewards.mean().item()), 4),
            "effective_reward_mean": round(float(effective_rewards.mean().item()), 4),
            "token_kl_mean": round(float(token_kl), 5),
            "policy_loss": round(float(sum(update_pol_losses) / len(update_pol_losses)), 4),
            "value_loss": round(float(sum(update_val_losses) / len(update_val_losses)), 4),
            "explained_variance": round(float(sum(update_evs) / len(update_evs)), 4),
            "entropy": round(float(sum(update_entropies) / len(update_entropies)), 4),
            "clip_fraction": round(float(sum(update_clip_fractions) / len(update_clip_fractions)), 4),
            "affected_fraction": round(float(sum(update_affected_fractions) / len(update_affected_fractions)), 4),
            "approx_kl": round(float(sum(update_approx_kls) / len(update_approx_kls)), 5),
            "policy_grad_norm": round(float(sum(update_pol_gnorms) / len(update_pol_gnorms)), 4),
            "value_grad_norm": round(float(sum(update_val_gnorms) / len(update_val_gnorms)), 4),
            "mean_response_length": round(mean_resp_len, 2),
            "all_eos_terminated": bool(all(terminated)),
            "wall_clock_sec": round(elapsed_sec, 2),
            "peak_vram_gb": round(peak_vram, 3),
        }
        train_logs.append(record)

        print(
            f"Update {u+1:02d}/{total_updates} | "
            f"Reward: {record['effective_reward_mean']:.3f} | "
            f"KL: {record['token_kl_mean']:.4f} | "
            f"PolLoss: {record['policy_loss']:.4f} | "
            f"ValLoss: {record['value_loss']:.4f} | "
            f"EV: {record['explained_variance']:.3f} | "
            f"Clip%: {record['clip_fraction']:.1%} | "
            f"Len: {record['mean_response_length']:.0f}"
        )

    # 8. Save Checkpoints & Machine-Readable History
    print(f"\nSaving continued PPO policy to {out_path}...")
    policy.save_pretrained(str(out_path))
    tokenizer.save_pretrained(str(out_path))
    val_out = out_path / "value_adapter"
    value_model.save_pretrained(str(val_out))

    log_path = results_dir / f"{run_name}_train_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_name": run_name,
                "config_path": config_path,
                "total_updates": total_updates,
                "clip_epsilon": eps,
                "kl_beta": beta_kl,
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
    print(f"PPO continuation logs saved to {log_path}")
    clear_gpu()

    return {
        "output_path": str(out_path),
        "total_updates": total_updates,
        "history": train_logs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--clip-epsilon", type=float)
    ap.add_argument("--kl-beta", type=float)
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_ppo(args.config, args.output, args.updates, args.clip_epsilon, args.kl_beta, args.run_name)


if __name__ == "__main__":
    main()
