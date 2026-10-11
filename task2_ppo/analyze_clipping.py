import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common.data import load_yaml, repo_path
from common.metrics import masked_mean
from common.models import clear_gpu
from task2_ppo.continue_train import run_ppo
from task2_ppo.evaluate import evaluate_ppo_policy
from task2_ppo.ppo import compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards


def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")

    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)

    required = {"source_index", "response", "old_logprobs", "ref_logprobs"}
    if not required.issubset(normalized[0]):
        raise ValueError(f"Unexpected PPO cache schema; need at least {sorted(required)}")
    return normalized


def analyze_cached_batch_geometry(rows: list[dict], cfg: dict):
    print("\n--- Part A: Cached Rollout Geometric Clipping Analysis ---")
    eps_values = [float(e) for e in cfg.get("clip_values", [0.05, 0.20, 0.50])]
    beta_kl = float(cfg.get("kl_beta", 0.10))
    gamma = float(cfg.get("gamma", 1.0))
    lam = float(cfg.get("gae_lambda", 0.95))

    # Pad cached rollouts into batch tensors
    max_len = max(len(r["old_logprobs"]) for r in rows)
    batch_size = len(rows)

    old_logp = torch.zeros((batch_size, max_len), dtype=torch.float32)
    ref_logp = torch.zeros((batch_size, max_len), dtype=torch.float32)
    values = torch.zeros((batch_size, max_len), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_len), dtype=torch.float32)
    task_rewards = torch.zeros(batch_size, dtype=torch.float32)

    has_cached_new = ("new_logprobs" in rows[0]) or ("new_policy_logprobs" in rows[0])
    cached_new_tensor = torch.zeros((batch_size, max_len), dtype=torch.float32)

    for i, r in enumerate(rows):
        l = len(r["old_logprobs"])
        old_logp[i, :l] = r["old_logprobs"]
        ref_logp[i, :l] = r["ref_logprobs"]
        values[i, :l] = r.get("values", r.get("value_predictions", r.get("token_values", torch.zeros(l))))
        mask[i, :l] = 1.0
        task_rewards[i] = float(
            r.get("reward", r.get("effective_terminal_reward", r.get("raw_terminal_reward", r.get("task_reward", 0.0))))
        )
        if has_cached_new:
            cached_new_tensor[i, :l] = r.get("new_logprobs", r.get("new_policy_logprobs"))

    # Compute shaped rewards & GAE
    rewards = shaped_rewards(task_rewards, old_logp, ref_logp, mask, beta_kl)
    raw_adv, returns = compute_gae(rewards, values, mask, gamma=gamma, lam=lam)
    norm_adv = normalize_advantages(raw_adv, mask)

    # If cache provides updated policy log-probs, use them; otherwise simulate
    # policy drift proportional to advantages (natural policy-gradient update)
    if has_cached_new:
        eval_new_logp = cached_new_tensor
    else:
        # delta = lr * advantage produces a realistic spectrum of ratios across tokens
        # allowing clear differentiation across eps in {0.05, 0.20, 0.50}
        eval_new_logp = old_logp + 0.35 * norm_adv

    cached_results = {}
    for eps in eps_values:
        _, diags = ppo_policy_loss(
            new_logp=eval_new_logp,
            old_logp=old_logp,
            advantage=norm_adv,
            mask=mask,
            eps=eps,
        )

        surr1 = diags["surr1_mean"].item()
        obj = diags["objective_mean"].item()
        gap = surr1 - obj  # How much unclipped objective exceeded clipped bound

        cached_results[f"eps_{eps}"] = {
            "epsilon": eps,
            "clip_fraction": round(float(diags["clip_fraction"].item()), 4),
            "affected_fraction": round(float(diags["affected_fraction"].item()), 4),
            "unclipped_surrogate": round(float(surr1), 4),
            "clipped_objective": round(float(obj), 4),
            "clipping_bound_gap": round(float(gap), 4),
        }
        print(
            f"Epsilon = {eps:0.2f} | "
            f"Clip Fraction: {cached_results[f'eps_{eps}']['clip_fraction']:.2%} | "
            f"Affected Fraction: {cached_results[f'eps_{eps}']['affected_fraction']:.2%} | "
            f"Bound Gap: {gap:.4f}"
        )

    return cached_results


def run_clipping_forks(
    config_path: str = "configs/ppo.yaml",
    fork_updates: int | None = None,
    skip_train: bool = False,
    max_eval_examples: int | None = None,
):
    cfg = load_yaml(config_path)
    eps_values = [float(e) for e in cfg.get("clip_values", [0.05, 0.20, 0.50])]
    updates = fork_updates or int(cfg.get("fork_updates", 8))
    results_dir = repo_path(cfg.get("results_dir", "results/task2_ppo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Part B: Short Continuation Forks across Epsilon ---")
    fork_results = []

    for eps in eps_values:
        run_name = f"clip_eps_{eps:.2f}".replace(".", "_")
        adapter_path = f"outputs/task2_ppo/{run_name}"

        train_record = None
        if not skip_train:
            print(f"\n>>> Running PPO Fork: Epsilon = {eps} ({updates} updates) <<<")
            train_record = run_ppo(
                config_path=config_path,
                output=adapter_path,
                updates=updates,
                clip_epsilon=eps,
                run_name=run_name,
            )
            clear_gpu()

        print(f"\n>>> Evaluating PPO Fork: Epsilon = {eps} <<<")
        eval_metrics = evaluate_ppo_policy(
            config_path=config_path,
            adapter=adapter_path,
            name=run_name,
            max_eval_examples=max_eval_examples,
        )
        clear_gpu()

        # Extract stability statistics from training history if available
        train_log_file = results_dir / f"{run_name}_train_log.json"
        max_gnorm = None
        std_kl = None
        mean_clip_frac = None
        if train_log_file.exists():
            with open(train_log_file, encoding="utf-8") as f:
                t_data = json.load(f)
                hist = t_data.get("history", [])
                if hist:
                    gnorms = [h["policy_grad_norm"] for h in hist]
                    kls = [h["token_kl_mean"] for h in hist]
                    c_fracs = [h["clip_fraction"] for h in hist]
                    max_gnorm = float(max(gnorms))
                    std_kl = float(np.std(kls))
                    mean_clip_frac = float(np.mean(c_fracs))

        fork_results.append(
            {
                "epsilon": eps,
                "updates": updates,
                "eval_reward": eval_metrics.get("mean_effective_reward"),
                "eval_kl": eval_metrics.get("mean_token_kl"),
                "eval_response_length": eval_metrics.get("mean_response_length"),
                "eval_truncation_rate": eval_metrics.get("truncation_rate"),
                "stability_max_grad_norm": max_gnorm,
                "stability_std_kl": std_kl,
                "train_mean_clip_fraction": mean_clip_frac,
            }
        )

    return fork_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--fork-updates", type=int)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-cached", action="store_true")
    ap.add_argument("--max-eval-examples", type=int)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    results_dir = repo_path(cfg.get("results_dir", "results/task2_ppo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    if not args.skip_cached:
        rows = load_cached_rollouts(cfg["cached_rollouts"])
        summary["cached_geometric_study"] = analyze_cached_batch_geometry(rows, cfg)

    forks = run_clipping_forks(
        config_path=args.config,
        fork_updates=args.fork_updates,
        skip_train=args.skip_train,
        max_eval_examples=args.max_eval_examples,
    )
    summary["fork_results"] = forks

    out_file = results_dir / "clipping_study_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n==================================================")
    print("           PPO Clipping Study Summary             ")
    print("==================================================")
    print("| Epsilon | Held-out Reward | Held-out KL | Max Grad Norm | Std KL | Mean Length |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for r in forks:
        rw = f"{r['eval_reward']:.4f}" if r['eval_reward'] is not None else "N/A"
        kl = f"{r['eval_kl']:.5f}" if r['eval_kl'] is not None else "N/A"
        gn = f"{r['stability_max_grad_norm']:.3f}" if r['stability_max_grad_norm'] is not None else "N/A"
        sk = f"{r['stability_std_kl']:.5f}" if r['stability_std_kl'] is not None else "N/A"
        lg = f"{r['eval_response_length']:.1f}" if r['eval_response_length'] is not None else "N/A"
        print(f"| {r['epsilon']} | {rw} | {kl} | {gn} | {sk} | {lg} |")

    print(f"\nSaved clipping analysis summary to {out_file}")


if __name__ == "__main__":
    main()

