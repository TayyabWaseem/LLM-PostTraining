from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.data import load_yaml, repo_path


def load_optional_json(path: Path) -> dict | None:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Warning: Failed to read {path}: {e}")
    return None


def run_feedback_comparison(config_path: str = "configs/feedback.yaml"):
    cfg = load_yaml(config_path)
    outdir = repo_path(cfg.get("results_dir", "results")) / "task5_feedback"
    outdir.mkdir(parents=True, exist_ok=True)

    gsm_file = outdir / "math_eval_gsm.json"
    transfer_file = outdir / "math_eval_transfer.json"
    diag_file = outdir / "perturbation_scoring.json"

    gsm_data = load_optional_json(gsm_file)
    transfer_data = load_optional_json(transfer_file)
    diag_data = load_optional_json(diag_file)

    print("==================================================")
    print("      Task 5: RLVR vs RLAIF Feedback Synthesis    ")
    print("==================================================")

    if not gsm_data and not transfer_data and not diag_data:
        raise RuntimeError(
            f"No Task 5 evaluation outputs found in {outdir}.\n"
            "Please run:\n"
            "  python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset gsm\n"
            "  python -m task5_feedback.score_perturbations --config configs/feedback.yaml\n"
            "  python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset transfer"
        )

    # 1. In-Domain & Transfer Policy Performance
    print("\n### 1. In-Domain (GSM8K) and Out-of-Domain (SVAMP) Math Performance")
    print("| Metric | Dataset | SFT Baseline | RLVR Policy | RLAIF Policy |")
    print("| :--- | :--- | :--- | :--- | :--- |")

    if gsm_data:
        gm = gsm_data.get("metrics", {})
        sft_acc = gm.get("sft", {}).get("exact_accuracy", 0.0)
        rlvr_acc = gm.get("rlvr", {}).get("exact_accuracy", 0.0)
        rlaif_acc = gm.get("rlaif", {}).get("exact_accuracy", 0.0)
        print(f"| **Exact Accuracy** | GSM8K (In-Domain) | {sft_acc:.1%} | {rlvr_acc:.1%} | {rlaif_acc:.1%} |")

        sft_wr = gm.get("sft", {}).get("win_rate_vs_sft", 0.5)
        rlvr_wr = gm.get("rlvr", {}).get("win_rate_vs_sft", 0.0)
        rlaif_wr = gm.get("rlaif", {}).get("win_rate_vs_sft", 0.0)
        print(f"| **Win Rate vs SFT** | GSM8K (In-Domain) | {sft_wr:.1%} | {rlvr_wr:.1%} | {rlaif_wr:.1%} |")

        sft_len = gm.get("sft", {}).get("mean_response_length", 0.0)
        rlvr_len = gm.get("rlvr", {}).get("mean_response_length", 0.0)
        rlaif_len = gm.get("rlaif", {}).get("mean_response_length", 0.0)
        print(f"| **Mean Length** | GSM8K (In-Domain) | {sft_len:.1f} | {rlvr_len:.1f} | {rlaif_len:.1f} |")

    if transfer_data:
        tm = transfer_data.get("metrics", {})
        sft_t_acc = tm.get("sft", {}).get("exact_accuracy", 0.0)
        rlvr_t_acc = tm.get("rlvr", {}).get("exact_accuracy", 0.0)
        rlaif_t_acc = tm.get("rlaif", {}).get("exact_accuracy", 0.0)
        print(f"| **Exact Accuracy** | SVAMP (Transfer) | {sft_t_acc:.1%} | {rlvr_t_acc:.1%} | {rlaif_t_acc:.1%} |")

        sft_t_wr = tm.get("sft", {}).get("win_rate_vs_sft", 0.5)
        rlvr_t_wr = tm.get("rlvr", {}).get("win_rate_vs_sft", 0.0)
        rlaif_t_wr = tm.get("rlaif", {}).get("win_rate_vs_sft", 0.0)
        print(f"| **Win Rate vs SFT** | SVAMP (Transfer) | {sft_t_wr:.1%} | {rlvr_t_wr:.1%} | {rlaif_t_wr:.1%} |")

        # Transfer drops
        drops = transfer_data.get("transfer_drops")
        if drops:
            rlvr_d_acc = drops.get("rlvr", {}).get("accuracy_drop", 0.0)
            rlaif_d_acc = drops.get("rlaif", {}).get("accuracy_drop", 0.0)
            sft_d_acc = drops.get("sft", {}).get("accuracy_drop", 0.0)
            print(f"| **Accuracy Drop** | GSM8K $\\to$ SVAMP | {sft_d_acc:.1%} | {rlvr_d_acc:.1%} | {rlaif_d_acc:.1%} |")

    # 2. Diagnostic Sensitivities (S_reason and S_outcome)
    print("\n### 2. Controlled Diagnostic Sensitivity Analysis")
    if diag_data:
        sens = diag_data.get("sensitivity_metrics", {})
        s_reas = sens.get("S_reason", {})
        s_out = sens.get("S_outcome", {})
        print("| Sensitivity Metric | Theoretical Target | Exact Verifier | AI Judge |")
        print("| :--- | :--- | :--- | :--- |")
        print(f"| **Reasoning Sensitivity ($S_{{reason}}$)** | Detect corrupted reasoning | {s_reas.get('verifier', 0.0):.1%} | {s_reas.get('ai_judge', 0.0):.1%} |")
        print(f"| **Outcome Sensitivity ($S_{{outcome}}$)** | Detect wrong final answers | {s_out.get('verifier', 0.0):.1%} | {s_out.get('ai_judge', 0.0):.1%} |")

        print("\n**Key Diagnostic Findings**:")
        cats = diag_data.get("categories", {})
        if "reasoning_corruption" in cats:
            rc_v = cats["reasoning_corruption"]["verifier"]
            rc_j = cats["reasoning_corruption"]["judge"]
            print(f"- **Reasoning-Only Corruption**: Verifier tie rate is {rc_v['tie_rate']:.1%} (outcome blindness); AI Judge better-preference rate is {rc_j['better_rate']:.1%}.")
        if "persuasive_filler" in cats:
            pf_j = cats["persuasive_filler"]["judge"]
            print(f"- **Persuasive Filler Vulnerability**: AI Judge preferred filler {pf_j['wrong_rate']:.1%} of the time (measuring verbosity/persuasion bias).")
        if "gold_distractor" in cats:
            gd_v = cats["gold_distractor"]["verifier"]
            gd_j = cats["gold_distractor"]["judge"]
            print(f"- **Gold Distractor Robustness**: Verifier better-rate is {gd_v['better_rate']:.1%}; AI Judge better-rate is {gd_j['better_rate']:.1%}.")
    else:
        print("(Diagnostic results pending: run score_perturbations.py first)")

    # 3. High-Level Trade-off Comparison Matrix
    comparison_dimensions = {
        "coverage": {
            "dimension": "Domain Coverage",
            "exact_verifier": "Narrow (requires ground-truth checkers, sandboxes, or unit tests; inapplicable to open-ended dialogue/creative tasks).",
            "ai_judge": "Broad (generalizes to open-ended chat, creative writing, nuanced instruction-following without formal oracles).",
        },
        "noise": {
            "dimension": "Label Noise & Variance",
            "exact_verifier": "Zero noise (100% deterministic, reproducible, independent of prompt phrasing or response order).",
            "ai_judge": "Moderate to High noise (subject to position bias, temperature jitter, and surface prompt heuristics).",
        },
        "exploitability": {
            "dimension": "Optimization Exploitability",
            "exact_verifier": "Outcome-shortcut hacking: rewards lucky or illogical reasoning that stumbles upon the correct designated final token.",
            "ai_judge": "Style/fluff hacking: vulnerable to assertive formatting, excessive verbosity, and persuasive filler phrases.",
        },
        "compute_cost": {
            "dimension": "Inference / Training Compute",
            "exact_verifier": "Near zero (regex extraction and float comparison execute on CPU in < 1 ms per problem).",
            "ai_judge": "High (requires full forward-pass generation over multiple billion parameters per candidate pair or group).",
        },
    }

    print("\n### 3. Feedback Source Trade-off Matrix")
    print("| Dimension | Exact Verifier (RLVR) | AI Preference Judge (RLAIF) |")
    print("| :--- | :--- | :--- |")
    for key, dim in comparison_dimensions.items():
        print(f"| **{dim['dimension']}** | {dim['exact_verifier']} | {dim['ai_judge']} |")

    synthesis_report = {
        "in_domain_gsm": gsm_data.get("metrics") if gsm_data else None,
        "transfer_svamp": transfer_data.get("metrics") if transfer_data else None,
        "transfer_drops": transfer_data.get("transfer_drops") if transfer_data else None,
        "diagnostic_sensitivities": diag_data.get("sensitivity_metrics") if diag_data else None,
        "tradeoff_matrix": comparison_dimensions,
    }

    out_file = outdir / "task5_feedback_synthesis.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(synthesis_report, f, indent=2)

    print(f"\nConsolidated Task 5 synthesis saved to {out_file}")
    return synthesis_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    run_feedback_comparison(args.config)


if __name__ == "__main__":
    main()
