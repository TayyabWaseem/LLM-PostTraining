import argparse
import json
from pathlib import Path

from common.data import load_yaml, repo_path
from common.models import clear_gpu
from task1_dpo.evaluate import evaluate_dpo_policy
from task1_dpo.train import run_training


def run_beta_ablation(
    config_path: str = "configs/dpo.yaml",
    betas: list[float] | None = None,
    max_examples: int | None = None,
    skip_train: bool = False,
    max_eval_examples: int | None = None,
    max_gen_examples: int | None = 64,
):
    cfg = load_yaml(config_path)
    betas = betas if betas is not None else [float(b) for b in cfg.get("betas", [0.03, 0.10, 0.30])]
    max_examples = max_examples or int(cfg.get("short_ablation_examples", 600))
    results_dir = repo_path(cfg.get("results_dir", "results/task1_dpo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("      Task 1: DPO Regularization Strength Study    ")
    print("==================================================")
    print(f"Beta sweep values: {betas}")
    print(f"Examples per condition: {max_examples}")

    summary_records = []

    for beta in betas:
        run_name = f"beta_{beta:.2f}".replace(".", "_")
        adapter_path = f"outputs/task1_dpo/{run_name}"

        if not skip_train:
            print(f"\n>>> Running Training for Beta = {beta} (budget = {max_examples} ex) <<<")
            run_training(
                config_path=config_path,
                run_name=run_name,
                output_path=adapter_path,
                beta=beta,
                max_examples=max_examples,
            )
            clear_gpu()

        print(f"\n>>> Running Evaluation for Beta = {beta} <<<")
        eval_metrics = evaluate_dpo_policy(
            config_path=config_path,
            adapter=adapter_path,
            name=run_name,
            max_eval_examples=max_eval_examples,
            max_gen_examples=max_gen_examples,
        )
        clear_gpu()

        summary_records.append(
            {
                "beta": beta,
                "budget_examples": max_examples,
                "held_out_loss": eval_metrics.get("held_out_dpo_loss"),
                "held_out_margin": eval_metrics.get("held_out_dpo_margin"),
                "preference_accuracy": eval_metrics.get("held_out_preference_accuracy"),
                "kl_divergence": eval_metrics.get("held_out_kl"),
                "mean_reward": eval_metrics.get("mean_reward_score"),
                "mean_length": eval_metrics.get("mean_response_length"),
                "std_length": eval_metrics.get("std_response_length"),
            }
        )

    out_file = results_dir / "beta_ablation_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_records, f, indent=2)

    print("\n==================================================")
    print("           Beta Ablation Study Results            ")
    print("==================================================")
    print("| Beta | Pref Acc (%) | Margin | KL Divergence | Mean Reward | Mean Length |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for r in summary_records:
        acc_str = f"{r['preference_accuracy']:.2%}" if r['preference_accuracy'] is not None else "N/A"
        kl_str = f"{r['kl_divergence']:.5f}" if r['kl_divergence'] is not None else "N/A"
        rw_str = f"{r['mean_reward']:.4f}" if r['mean_reward'] is not None else "N/A"
        len_str = f"{r['mean_length']:.1f} ± {r['std_length']:.1f}" if r['mean_length'] is not None else "N/A"
        print(f"| {r['beta']} | {acc_str} | {r['held_out_margin']} | {kl_str} | {rw_str} | {len_str} |")

    print(f"\nSaved summary to {out_file}")
    return summary_records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--betas", type=float, nargs="+")
    ap.add_argument("--max-examples", type=int)
    ap.add_argument("--max-eval-examples", type=int)
    ap.add_argument("--max-gen-examples", type=int, default=64)
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    run_beta_ablation(
        config_path=args.config,
        betas=args.betas,
        max_examples=args.max_examples,
        skip_train=args.skip_train,
        max_eval_examples=args.max_eval_examples,
        max_gen_examples=args.max_gen_examples,
    )


if __name__ == "__main__":
    main()

