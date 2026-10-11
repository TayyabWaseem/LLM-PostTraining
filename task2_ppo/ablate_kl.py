import argparse
import json
from pathlib import Path

from common.data import load_yaml, repo_path
from common.models import clear_gpu
from task2_ppo.continue_train import run_ppo
from task2_ppo.evaluate import evaluate_ppo_policy


def run_kl_ablation(
    config_path: str = "configs/ppo.yaml",
    kl_values: list[float] | None = None,
    fork_updates: int | None = None,
    skip_train: bool = False,
    include_midpoint_baseline: bool = True,
    max_eval_examples: int | None = None,
):
    cfg = load_yaml(config_path)
    betas = kl_values if kl_values is not None else [float(b) for b in cfg.get("kl_values", [0.0, 0.10, 0.20])]
    updates = fork_updates or int(cfg.get("fork_updates", 8))
    results_dir = repo_path(cfg.get("results_dir", "results/task2_ppo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("      Task 2: PPO KL Pressure & Reward Study      ")
    print("==================================================")
    print(f"KL beta sweep values: {betas}")
    print(f"Updates per fork: {updates}")

    records = []

    # 1. Step 0 Baseline: Evaluate starting midpoint checkpoint
    if include_midpoint_baseline:
        print("\n>>> Evaluating Baseline Midpoint Checkpoint (Step 0) <<<")
        midpoint_eval = evaluate_ppo_policy(
            config_path=config_path,
            adapter=cfg["paths"]["ppo_midpoint_policy"],
            name="midpoint_baseline",
            max_eval_examples=max_eval_examples,
        )
        clear_gpu()
        records.append(
            {
                "condition": "midpoint_step_0",
                "kl_beta": "baseline",
                "updates": 0,
                "reward": midpoint_eval.get("mean_effective_reward"),
                "token_kl": midpoint_eval.get("mean_token_kl"),
                "entropy": midpoint_eval.get("mean_entropy"),
                "mean_response_length": midpoint_eval.get("mean_response_length"),
                "truncation_rate": midpoint_eval.get("truncation_rate"),
            }
        )

    # 2. Matched forks for each KL beta
    for b in betas:
        run_name = f"kl_beta_{b:.2f}".replace(".", "_")
        adapter_path = f"outputs/task2_ppo/{run_name}"

        if not skip_train:
            print(f"\n>>> Running PPO Continuation: Beta KL = {b} ({updates} updates) <<<")
            run_ppo(
                config_path=config_path,
                output=adapter_path,
                updates=updates,
                kl_beta=b,
                run_name=run_name,
            )
            clear_gpu()

        print(f"\n>>> Evaluating PPO Policy: Beta KL = {b} <<<")
        eval_metrics = evaluate_ppo_policy(
            config_path=config_path,
            adapter=adapter_path,
            name=run_name,
            max_eval_examples=max_eval_examples,
        )
        clear_gpu()

        records.append(
            {
                "condition": f"beta_kl_{b:.2f}",
                "kl_beta": b,
                "updates": updates,
                "reward": eval_metrics.get("mean_effective_reward"),
                "token_kl": eval_metrics.get("mean_token_kl"),
                "entropy": eval_metrics.get("mean_entropy"),
                "mean_response_length": eval_metrics.get("mean_response_length"),
                "truncation_rate": eval_metrics.get("truncation_rate"),
            }
        )

    out_file = results_dir / "kl_ablation_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print("\n==================================================")
    print("         PPO KL Pressure Study Results            ")
    print("==================================================")
    print("| Condition | Beta KL | Held-out Reward | Token KL | Entropy | Mean Length | Truncation % |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for r in records:
        rw = f"{r['reward']:.4f}" if r['reward'] is not None else "N/A"
        kl = f"{r['token_kl']:.5f}" if r['token_kl'] is not None else "N/A"
        ent = f"{r['entropy']:.4f}" if r['entropy'] is not None else "N/A"
        lg = f"{r['mean_response_length']:.1f}" if r['mean_response_length'] is not None else "N/A"
        tr = f"{r['truncation_rate']:.1%}" if r['truncation_rate'] is not None else "N/A"
        print(f"| {r['condition']} | {r['kl_beta']} | {rw} | {kl} | {ent} | {lg} | {tr} |")

    print(f"\nSaved summary to {out_file}")
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--kl-values", type=float, nargs="+")
    ap.add_argument("--fork-updates", type=int)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-midpoint", action="store_true")
    ap.add_argument("--max-eval-examples", type=int)
    args = ap.parse_args()

    run_kl_ablation(
        config_path=args.config,
        kl_values=args.kl_values,
        fork_updates=args.fork_updates,
        skip_train=args.skip_train,
        include_midpoint_baseline=not args.skip_midpoint,
        max_eval_examples=args.max_eval_examples,
    )


if __name__ == "__main__":
    main()

