from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common.data import load_yaml, repo_path
from common.models import clear_gpu
from task3_grpo.continue_train import run_grpo
from task3_grpo.evaluate import evaluate_grpo_policy


def run_normalization_study(
    config_path: str = "configs/grpo.yaml",
    fork_updates: int | None = None,
    skip_train: bool = False,
    max_eval_examples: int | None = None,
):
    cfg = load_yaml(config_path)
    updates = int(fork_updates if fork_updates is not None else cfg.get("fork_updates", 8))
    results_dir = repo_path(cfg.get("results_dir", "results/task3_grpo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    conditions = [
        {
            "name": "canonical_grpo",
            "loss_type": "grpo",
            "adapter_path": "outputs/task3_grpo/norm_canonical",
            "description": "Per-sequence 1/T_k normalization (overweights short completions)",
        },
        {
            "name": "dr_grpo",
            "loss_type": "dr_grpo",
            "adapter_path": "outputs/task3_grpo/norm_dr_grpo",
            "description": "Constant 1/T_max normalization (length-invariant token weighting)",
        },
    ]

    print("==================================================")
    print("   Task 3: Canonical vs Dr. GRPO Normalization    ")
    print("==================================================")
    print(f"Matched fork updates: {updates}")
    print(f"Max completion length T_max: {cfg.get('max_completion_length', 512)}")

    study_records = []

    for cond in conditions:
        run_name = cond["name"]
        adapter_path = cond["adapter_path"]
        loss_type = cond["loss_type"]

        # 1. Training Fork
        if not skip_train:
            print(f"\n>>> Running Continuation Fork: {run_name} (loss_type={loss_type}) <<<")
            run_grpo(
                config_path=config_path,
                output=adapter_path,
                updates=updates,
                loss_type=loss_type,
                run_name=run_name,
            )
            clear_gpu()

        # 2. Evaluation
        print(f"\n>>> Evaluating Fork: {run_name} <<<")
        eval_metrics = evaluate_grpo_policy(
            config_path=config_path,
            adapter=adapter_path,
            name=run_name,
            max_eval_examples=max_eval_examples,
        )
        clear_gpu()

        # 3. Read training logs to extract gradient norm and training dynamics
        train_log_file = results_dir / f"{run_name}_train_log.json"
        mean_gnorm = None
        max_gnorm = None
        train_mean_length = None
        if train_log_file.exists():
            with open(train_log_file, encoding="utf-8") as f:
                t_data = json.load(f)
                hist = t_data.get("history", [])
                if hist:
                    gnorms = [h["policy_grad_norm"] for h in hist]
                    lengths = [h["mean_response_length"] for h in hist]
                    mean_gnorm = round(float(np.mean(gnorms)), 4)
                    max_gnorm = round(float(max(gnorms)), 4)
                    train_mean_length = round(float(np.mean(lengths)), 2)

        # 4. Read qualitative samples to compute length-conditioned reward correlation
        eval_log_file = results_dir / f"{run_name}_eval.json"
        length_reward_corr = None
        if eval_log_file.exists():
            with open(eval_log_file, encoding="utf-8") as f:
                e_data = json.load(f)
                samples = e_data.get("qualitative_samples", [])
                valid_pairs = [
                    (s["length_tokens"], s["reward"])
                    for s in samples
                    if s.get("reward") is not None
                ]
                if len(valid_pairs) >= 5:
                    lens, rws = zip(*valid_pairs)
                    if np.std(lens) > 0 and np.std(rws) > 0:
                        corr = np.corrcoef(lens, rws)[0, 1]
                        length_reward_corr = round(float(corr), 4)

        record = {
            "condition": run_name,
            "loss_type": loss_type,
            "description": cond["description"],
            "updates": updates,
            "eval_reward_mean": eval_metrics.get("mean_reward"),
            "eval_reward_std": eval_metrics.get("std_reward"),
            "eval_token_kl": eval_metrics.get("mean_token_kl"),
            "eval_entropy": eval_metrics.get("mean_entropy"),
            "eval_mean_response_length": eval_metrics.get("mean_response_length"),
            "eval_std_response_length": eval_metrics.get("std_response_length"),
            "eval_median_response_length": eval_metrics.get("median_response_length"),
            "eval_truncation_rate": eval_metrics.get("truncation_rate"),
            "eval_eos_rate": eval_metrics.get("eos_termination_rate"),
            "train_mean_grad_norm": mean_gnorm,
            "train_max_grad_norm": max_gnorm,
            "train_mean_length": train_mean_length,
            "length_conditioned_reward_corr": length_reward_corr,
        }
        study_records.append(record)

    # 5. Theoretical Gradient Weighting Comparison Statistic
    t_max = int(cfg.get("max_completion_length", 512))
    theoretical_stats = {
        "max_completion_length": t_max,
        "token_gradient_weight_ratio_short_vs_long": {
            "short_length_100": {
                "canonical_weight": round(1.0 / 100.0, 5),
                "dr_grpo_weight": round(1.0 / float(t_max), 5),
            },
            "long_length_500": {
                "canonical_weight": round(1.0 / 500.0, 5),
                "dr_grpo_weight": round(1.0 / float(t_max), 5),
            },
            "canonical_relative_short_bias": round(500.0 / 100.0, 2),
            "dr_grpo_relative_short_bias": 1.0,
        },
    }

    out_file = results_dir / "normalization_study_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "theoretical_analysis": theoretical_stats,
                "conditions": study_records,
            },
            f,
            indent=2,
        )

    print("\n==================================================")
    print("      GRPO Normalization Comparison Summary       ")
    print("==================================================")
    print("| Method | Loss Type | Held-out Reward | Token KL | Mean Length | Truncation % | Mean Grad Norm | Length-Reward Corr |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for r in study_records:
        rw = f"{r['eval_reward_mean']:.4f}" if r['eval_reward_mean'] is not None else "N/A"
        kl = f"{r['eval_token_kl']:.5f}" if r['eval_token_kl'] is not None else "N/A"
        lg = f"{r['eval_mean_response_length']:.1f}" if r['eval_mean_response_length'] is not None else "N/A"
        tr = f"{r['eval_truncation_rate']:.1%}" if r['eval_truncation_rate'] is not None else "N/A"
        gn = f"{r['train_mean_grad_norm']:.3f}" if r['train_mean_grad_norm'] is not None else "N/A"
        cr = f"{r['length_conditioned_reward_corr']:.3f}" if r['length_conditioned_reward_corr'] is not None else "N/A"
        print(f"| {r['condition']} | {r['loss_type']} | {rw} | {kl} | {lg} | {tr} | {gn} | {cr} |")

    print(f"\nSaved Normalization Study results to {out_file}")
    return study_records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--fork-updates", type=int)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--max-eval-examples", type=int)
    args = ap.parse_args()

    run_normalization_study(
        config_path=args.config,
        fork_updates=args.fork_updates,
        skip_train=args.skip_train,
        max_eval_examples=args.max_eval_examples,
    )


if __name__ == "__main__":
    main()
