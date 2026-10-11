from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from common.data import load_yaml, read_jsonl, repo_path


def load_k8_cache(path: str | Path):
    rows = read_jsonl(repo_path(path))
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)

    bad = {pid: len(group) for pid, group in by_prompt.items() if len(group) < 8}
    if bad:
        raise ValueError(f"Expected at least K=8 cached completions per prompt; short groups: {bad}")

    for group in by_prompt.values():
        group.sort(key=lambda x: int(x.get("generation_index", 0)))
    return by_prompt


def regroup_equal_generation_budget(by_prompt: dict[str, list[dict]], k: int) -> list[dict]:
    """Partition cached completions into K-sized groups at a fixed total generation budget.

    Each prompt supplies 8 completions. Partitioning each prompt's completions into (8 // k)
    disjoint non-overlapping groups of size k preserves the exact total generation budget
    across K in {2, 4, 8} without discarding prompts or biasing prompt selection.
    """
    groups = []
    for pid, completions in by_prompt.items():
        # Partition 8 completions into chunks of size k
        for start_idx in range(0, 8, k):
            chunk = completions[start_idx : start_idx + k]
            if len(chunk) == k:
                groups.append(
                    {
                        "source_index": pid,
                        "rewards": [float(c["reward"]) for c in chunk],
                        "completions": chunk,
                    }
                )
    return groups


def compute_group_metrics(groups: list[dict], tol: float = 1e-5) -> dict:
    """Compute informativeness, reward standard deviation, and relative signal variance."""
    if not groups:
        return {
            "num_groups": 0,
            "total_generations": 0,
            "informative_group_rate": 0.0,
            "uninformative_group_rate": 0.0,
            "mean_within_group_std": 0.0,
            "relative_signal_variance": 0.0,
            "centered_reward_variance": 0.0,
        }

    stds = []
    all_advantages = []
    all_centered = []
    informative_count = 0

    for g in groups:
        r = np.array(g["rewards"], dtype=float)
        mu = float(np.mean(r))
        sigma = float(np.std(r))  # population std
        stds.append(sigma)

        centered = r - mu
        all_centered.extend(centered.tolist())

        if sigma > tol:
            informative_count += 1
            adv = centered / (sigma + 1e-6)
        else:
            adv = np.zeros_like(r)
        all_advantages.extend(adv.tolist())

    return {
        "num_groups": len(groups),
        "total_generations": len(all_advantages),
        "informative_group_rate": round(float(informative_count / len(groups)), 4),
        "uninformative_group_rate": round(float(1.0 - (informative_count / len(groups))), 4),
        "mean_within_group_std": round(float(np.mean(stds)), 4),
        "relative_signal_variance": round(float(np.var(all_advantages)), 4),
        "centered_reward_variance": round(float(np.var(all_centered)), 4),
    }


def analyze_group_sizes(config_path: str = "configs/grpo.yaml"):
    cfg = load_yaml(config_path)
    cache_path = cfg.get("group_cache", "cached/grpo_k_cache.jsonl")
    group_sizes = [int(k) for k in cfg.get("group_sizes", [2, 4, 8])]
    results_dir = repo_path(cfg.get("results_dir", "results/task3_grpo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    by_prompt = load_k8_cache(cache_path)
    total_prompts = len(by_prompt)
    total_generations = total_prompts * 8

    print("==================================================")
    print("      Task 3: Equal-Generation Group-Size Study   ")
    print("==================================================")
    print(f"Total cached prompts: {total_prompts} | Total generations: {total_generations}")
    print(f"Candidate group sizes K: {group_sizes}")

    # 1. Define prompt difficulty binning rule:
    # Compute mean reward for each prompt across all 8 cached completions.
    # Split by median into "Hard Prompts" (< median) and "Easy Prompts" (>= median).
    prompt_mean_rewards = {
        pid: float(np.mean([c["reward"] for c in comps]))
        for pid, comps in by_prompt.items()
    }
    median_reward = float(np.median(list(prompt_mean_rewards.values())))
    print(f"\nPrompt Difficulty Binning (Median Split = {median_reward:.4f}):")

    hard_prompts = {pid for pid, r_bar in prompt_mean_rewards.items() if r_bar < median_reward}
    easy_prompts = {pid for pid, r_bar in prompt_mean_rewards.items() if r_bar >= median_reward}
    print(f"  Hard / Low-Reward Prompts: {len(hard_prompts)}")
    print(f"  Easy / High-Reward Prompts: {len(easy_prompts)}")

    study_summary = {
        "total_prompts": total_prompts,
        "total_generations": total_generations,
        "difficulty_binning": {
            "metric": "mean_cached_reward_per_prompt",
            "median_threshold": round(median_reward, 4),
            "num_hard_prompts": len(hard_prompts),
            "num_easy_prompts": len(easy_prompts),
        },
        "by_group_size": {},
    }

    for k in group_sizes:
        all_groups = regroup_equal_generation_budget(by_prompt, k)
        hard_groups = [g for g in all_groups if g["source_index"] in hard_prompts]
        easy_groups = [g for g in all_groups if g["source_index"] in easy_prompts]

        overall_metrics = compute_group_metrics(all_groups)
        hard_metrics = compute_group_metrics(hard_groups)
        easy_metrics = compute_group_metrics(easy_groups)

        study_summary["by_group_size"][f"K_{k}"] = {
            "K": k,
            "overall": overall_metrics,
            "hard_prompts": hard_metrics,
            "easy_prompts": easy_metrics,
        }

    out_file = results_dir / "group_size_study_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(study_summary, f, indent=2)

    # Print overall comparison table
    print("\n--- Equal-Generation Group-Size Results (Overall) ---")
    print("| Group Size K | Total Gens | Num Groups | Informative % | Uninformative % | Mean Reward Std | Advantage Var |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for k in group_sizes:
        m = study_summary["by_group_size"][f"K_{k}"]["overall"]
        print(
            f"| K = {k} | {m['total_generations']} | {m['num_groups']} | "
            f"{m['informative_group_rate']:.1%} | {m['uninformative_group_rate']:.1%} | "
            f"{m['mean_within_group_std']:.4f} | {m['relative_signal_variance']:.4f} |"
        )

    # Print difficulty breakdown table
    print("\n--- Prompt Difficulty Breakdown ---")
    print("| Group Size K | Bin | Num Groups | Informative % | Mean Reward Std | Advantage Var |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for k in group_sizes:
        h = study_summary["by_group_size"][f"K_{k}"]["hard_prompts"]
        e = study_summary["by_group_size"][f"K_{k}"]["easy_prompts"]
        print(f"| K = {k} | Hard (Low-Reward) | {h['num_groups']} | {h['informative_group_rate']:.1%} | {h['mean_within_group_std']:.4f} | {h['relative_signal_variance']:.4f} |")
        print(f"| K = {k} | Easy (High-Reward) | {e['num_groups']} | {e['informative_group_rate']:.1%} | {e['mean_within_group_std']:.4f} | {e['relative_signal_variance']:.4f} |")

    print(f"\nSaved Group-Size Study summary to {out_file}")
    return study_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    analyze_group_sizes(args.config)


if __name__ == "__main__":
    main()
