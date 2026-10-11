from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np

from common.data import load_yaml, read_jsonl, repo_path
from common.models import clear_gpu
from task5_feedback.rlvr import exact_reward
from task5_feedback.rlaif import PairwiseAIJudge

EXPECTED_VARIANTS = {
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
}


def load_diagnostic_groups(path):
    rows = read_jsonl(path)
    by_problem = defaultdict(dict)
    for row in rows:
        by_problem[str(row["problem_id"])][row["variant_type"]] = row
    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
    return by_problem


def score_pair_verifier(resp_better: str, resp_worse: str, gold_final: str) -> str:
    r_better = exact_reward(resp_better, gold_final)
    r_worse = exact_reward(resp_worse, gold_final)
    if r_better > r_worse:
        return "BETTER"
    elif r_better < r_worse:
        return "WRONG"
    else:
        return "TIE"


def score_pair_judge(judge: PairwiseAIJudge, question: str, resp_better: str, resp_worse: str) -> str:
    pref = judge.compare(question, resp_better, resp_worse)
    if pref == "A":
        return "BETTER"
    elif pref == "B":
        return "WRONG"
    else:
        return "TIE"


def run_perturbation_study(config_path: str = "configs/feedback.yaml"):
    cfg = load_yaml(config_path)
    outdir = repo_path(cfg.get("results_dir", "results")) / "task5_feedback"
    outdir.mkdir(parents=True, exist_ok=True)

    diag_path = cfg["paths"]["task5_diagnostics"]
    groups = load_diagnostic_groups(diag_path)
    n_problems = len(groups)

    print("==================================================")
    print(f"  Task 5: Controlled Reward Diagnostic Study     ")
    print("==================================================")
    print(f"Loaded {n_problems} problems with 5 controlled variants each (100 total responses).")

    cache_path = outdir / "diagnostic_judge_cache.json"
    judge = PairwiseAIJudge(cfg, cache_path=cache_path)

    # Categories to evaluate:
    # 1. Reasoning Corruption: clean_correct vs corrupt_reasoning_correct_final (same correct final)
    # 2. Outcome Corruption: clean_correct vs good_reasoning_wrong_final (correct vs wrong final)
    # 3. Persuasive Filler: clean_correct vs persuasive_filler_correct (clean vs persuasive fluff)
    # 4. Gold Distractor: clean_correct vs gold_distractor_wrong_final (correct vs distractor wrong final)
    # 5. Reasoning vs Outcome Conflict: corrupt_reasoning_correct_final vs good_reasoning_wrong_final
    categories = {
        "reasoning_corruption": {
            "name": "Reasoning Corruption (Same Correct Final)",
            "better_variant": "clean_correct",
            "worse_variant": "corrupt_reasoning_correct_final",
            "is_reasoning_pair": True,
            "is_outcome_pair": False,
        },
        "outcome_corruption": {
            "name": "Outcome Corruption (Wrong Final Answer)",
            "better_variant": "clean_correct",
            "worse_variant": "good_reasoning_wrong_final",
            "is_reasoning_pair": False,
            "is_outcome_pair": True,
        },
        "persuasive_filler": {
            "name": "Persuasive Filler (Clean vs Filler)",
            "better_variant": "clean_correct",
            "worse_variant": "persuasive_filler_correct",
            "is_reasoning_pair": False,
            "is_outcome_pair": False,
        },
        "gold_distractor": {
            "name": "Gold Distractor (Mentions Gold, Wrong Final)",
            "better_variant": "clean_correct",
            "worse_variant": "gold_distractor_wrong_final",
            "is_reasoning_pair": False,
            "is_outcome_pair": True,
        },
        "reasoning_vs_outcome_conflict": {
            "name": "Conflict (Corrupt+Correct vs Good+Wrong)",
            "better_variant": "corrupt_reasoning_correct_final",  # outcome-wise better
            "worse_variant": "good_reasoning_wrong_final",
            "is_reasoning_pair": False,
            "is_outcome_pair": True,
        },
    }

    category_results = {}
    qualitative_samples = []

    all_reasoning_pairs_v = []
    all_reasoning_pairs_j = []
    all_outcome_pairs_v = []
    all_outcome_pairs_j = []

    for cat_key, spec in categories.items():
        v_better_cnt = 0
        v_tie_cnt = 0
        v_wrong_cnt = 0

        j_better_cnt = 0
        j_tie_cnt = 0
        j_wrong_cnt = 0

        cat_samples = []

        for pid, variants in groups.items():
            b_row = variants[spec["better_variant"]]
            w_row = variants[spec["worse_variant"]]
            q = b_row["question"]
            gold = str(b_row["gold_final"])

            v_outcome = score_pair_verifier(b_row["response"], w_row["response"], gold)
            j_outcome = score_pair_judge(judge, q, b_row["response"], w_row["response"])

            if v_outcome == "BETTER":
                v_better_cnt += 1
            elif v_outcome == "TIE":
                v_tie_cnt += 1
            else:
                v_wrong_cnt += 1

            if j_outcome == "BETTER":
                j_better_cnt += 1
            elif j_outcome == "TIE":
                j_tie_cnt += 1
            else:
                j_wrong_cnt += 1

            # Accumulate for S_reason and S_outcome definitions
            if spec["is_reasoning_pair"]:
                all_reasoning_pairs_v.append(v_outcome == "BETTER")
                all_reasoning_pairs_j.append(j_outcome == "BETTER")
            if spec["is_outcome_pair"]:
                all_outcome_pairs_v.append(v_outcome == "BETTER")
                all_outcome_pairs_j.append(j_outcome == "BETTER")

            # Collect representative disagreement sample
            if v_outcome != j_outcome and len(cat_samples) < 2:
                cat_samples.append({
                    "problem_id": pid,
                    "category": cat_key,
                    "question": q,
                    "better_variant": spec["better_variant"],
                    "worse_variant": spec["worse_variant"],
                    "better_response": b_row["response"][:250],
                    "worse_response": w_row["response"][:250],
                    "verifier_decision": v_outcome,
                    "judge_decision": j_outcome,
                })

        category_results[cat_key] = {
            "name": spec["name"],
            "count": n_problems,
            "verifier": {
                "better_rate": round(v_better_cnt / n_problems, 4),
                "tie_rate": round(v_tie_cnt / n_problems, 4),
                "wrong_rate": round(v_wrong_cnt / n_problems, 4),
            },
            "judge": {
                "better_rate": round(j_better_cnt / n_problems, 4),
                "tie_rate": round(j_tie_cnt / n_problems, 4),
                "wrong_rate": round(j_wrong_cnt / n_problems, 4),
            },
            "samples": cat_samples,
        }
        qualitative_samples.extend(cat_samples)

    # Calculate exact S_reason and S_outcome as defined in assignment manual
    s_reason_v = float(np.mean(all_reasoning_pairs_v)) if all_reasoning_pairs_v else 0.0
    s_reason_j = float(np.mean(all_reasoning_pairs_j)) if all_reasoning_pairs_j else 0.0

    s_outcome_v = float(np.mean(all_outcome_pairs_v)) if all_outcome_pairs_v else 0.0
    s_outcome_j = float(np.mean(all_outcome_pairs_j)) if all_outcome_pairs_j else 0.0

    summary = {
        "total_problems": n_problems,
        "sensitivity_metrics": {
            "S_reason": {
                "verifier": round(s_reason_v, 4),
                "ai_judge": round(s_reason_j, 4),
                "description": "Pr[R(y_clean) > R(y_reason_corrupt)] for pairs with same correct final answer",
            },
            "S_outcome": {
                "verifier": round(s_outcome_v, 4),
                "ai_judge": round(s_outcome_j, 4),
                "description": "Pr[R(y_correct) > R(y_wrong)] for outcome-changing pairs",
            },
        },
        "categories": category_results,
        "qualitative_evidence": qualitative_samples,
    }

    out_file = outdir / "perturbation_scoring.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Print summary tables
    print("\n--- Sensitivity Diagnostics: S_reason vs S_outcome ---")
    print("| Metric | Definition | Exact Verifier | AI Judge |")
    print("| :--- | :--- | :--- | :--- |")
    print(f"| S_reason | Pr[R(y_clean) > R(y_corrupt)] | {s_reason_v:.1%} | {s_reason_j:.1%} |")
    print(f"| S_outcome | Pr[R(y_correct) > R(y_wrong)] | {s_outcome_v:.1%} | {s_outcome_j:.1%} |")

    print("\n--- Controlled Diagnostic Results by Category (20 Problems Each) ---")
    print("| Perturbation Category | Mechanism | Better Rate % | Tie Rate % | Wrong Rate % |")
    print("| :--- | :--- | :--- | :--- | :--- |")
    for cat_key, res in category_results.items():
        v = res["verifier"]
        j = res["judge"]
        print(f"| **{res['name']}** | Exact Verifier | {v['better_rate']:.1%} | {v['tie_rate']:.1%} | {v['wrong_rate']:.1%} |")
        print(f"| | AI Judge | {j['better_rate']:.1%} | {j['tie_rate']:.1%} | {j['wrong_rate']:.1%} |")

    clear_gpu()
    print(f"\nSaved perturbation results and qualitative evidence to {out_file}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    run_perturbation_study(args.config)


if __name__ == "__main__":
    main()
