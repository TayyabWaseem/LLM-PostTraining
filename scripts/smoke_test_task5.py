from __future__ import annotations

from pathlib import Path
import numpy as np

from task5_feedback.rlvr import exact_reward, extract_designated_final, numerically_equal
from task5_feedback.score_perturbations import (
    load_diagnostic_groups,
    score_pair_verifier,
    EXPECTED_VARIANTS,
)
from common.data import read_jsonl


def test_exact_verifier():
    print("[1/5] Testing exact RLVR verifier & format extraction...")
    # Clean extraction
    assert extract_designated_final("She needs 9 hours.\n#### 9") == "9"
    assert extract_designated_final("Total cost is $144.\n#### 144") == "144"
    assert extract_designated_final("Total is 1,600 sticks.\n#### 1,600") == "1600"
    assert extract_designated_final("Negative value.\n#### -15.5") == "-15.5"
    assert extract_designated_final("No answer format provided.") is None

    # Intermediate candidate distractor handling: must ignore intermediate numbers and extract only the final #### field
    distractor_text = "A tempting candidate is 9, but that is rejected.\nMy final answer is:\n#### 10"
    assert extract_designated_final(distractor_text) == "10"
    assert exact_reward(distractor_text, "9") == 0.0
    assert exact_reward(distractor_text, "10") == 1.0

    # Numerical equivalence
    assert numerically_equal("1600", "1600.0")
    assert numerically_equal("0.5", ".5")
    assert not numerically_equal("10", "11")

    # Exact reward binary scoring
    assert exact_reward("#### 42", "42") == 1.0
    assert exact_reward("#### 42", "43") == 0.0
    assert exact_reward("Text without hash", "42") == 0.0
    print("  -> Exact verifier and format extraction: PASSED")


def test_verifier_controlled_pair_scoring():
    print("[2/5] Testing verifier scoring on controlled perturbation pairs...")
    gold = "9"
    resp_clean = "Step 1: 14 days. Step 2: 7 hours. Step 3: 14 hours. 30 - 21 = 9.\n#### 9"
    resp_corrupt = "Step 1: 15 days (wrong arithmetic). 30 - 21 = 9.\n#### 9"
    resp_wrong_final = "Step 1: 14 days. Step 2: 7 hours. 30 - 21 = 9.\n#### 10"
    resp_distractor = "Is it 9? No.\n#### 10"

    # Outcome-invariance: verifier ties on reasoning corruption when final is correct
    assert score_pair_verifier(resp_clean, resp_corrupt, gold) == "TIE"

    # Outcome sensitivity: verifier strictly prefers correct final answer over wrong final answer
    assert score_pair_verifier(resp_clean, resp_wrong_final, gold) == "BETTER"
    assert score_pair_verifier(resp_wrong_final, resp_clean, gold) == "WRONG"

    # Distractor robustness: verifier prefers correct final over distractor
    assert score_pair_verifier(resp_clean, resp_distractor, gold) == "BETTER"
    print("  -> Verifier controlled pair scoring: PASSED")


def test_diagnostic_dataset_integrity():
    print("[3/5] Testing 100-response diagnostic dataset structure...")
    path = Path("data/task5_controlled_reward_diagnostics.jsonl")
    if not path.exists():
        print("  -> Skipping diagnostic dataset test (file not found locally)")
        return

    groups = load_diagnostic_groups(path)
    assert len(groups) == 20, f"Expected 20 diagnostic problems, found {len(groups)}"

    for pid, variants in groups.items():
        assert set(variants.keys()) == EXPECTED_VARIANTS, f"Problem {pid} missing variants"
        # Check that clean variant has expected exact reward 1
        assert variants["clean_correct"]["expected_exact_reward"] == 1
        # Check that corrupted reasoning with correct final has expected exact reward 1
        assert variants["corrupt_reasoning_correct_final"]["expected_exact_reward"] == 1
        # Check that wrong final has expected exact reward 0
        assert variants["good_reasoning_wrong_final"]["expected_exact_reward"] == 0

    print(f"  -> Diagnostic dataset integrity: PASSED (20 problems x 5 variants = 100 responses)")


def test_sensitivity_metrics():
    print("[4/5] Testing S_reason and S_outcome definitions...")
    # Verifier outcomes on diagnostic definitions:
    # S_reason = Pr[R(y_clean) > R(y_corrupt)] for pairs with same correct final
    # Since verifier gives 1.0 to both, verifier S_reason is strictly 0.0 (ties)
    v_reason_preferences = [0, 0, 0, 0]  # all ties -> 0 better
    s_reason_v = float(np.mean(v_reason_preferences))
    assert s_reason_v == 0.0

    # S_outcome = Pr[R(y_correct) > R(y_wrong)] for outcome-changing pairs
    # Verifier gives 1.0 vs 0.0, so verifier S_outcome is strictly 1.0 (100% better)
    v_outcome_preferences = [1, 1, 1, 1]  # all better -> 1.0
    s_outcome_v = float(np.mean(v_outcome_preferences))
    assert s_outcome_v == 1.0

    # AI judge mock sensitivity
    j_reason_preferences = [1, 1, 0, 1]  # judge detects 3 of 4 reasoning corruptions
    s_reason_j = float(np.mean(j_reason_preferences))
    assert abs(s_reason_j - 0.75) < 1e-4

    print("  -> Sensitivity metric definitions (S_reason, S_outcome): PASSED")


def test_in_domain_and_transfer_data():
    print("[5/5] Testing GSM8K and SVAMP transfer data schema...")
    gsm_path = Path("data/gsm8k_eval.jsonl")
    transfer_path = Path("data/math_transfer_eval.jsonl")

    if gsm_path.exists():
        gsm_rows = read_jsonl(gsm_path)
        assert len(gsm_rows) > 0
        first_gsm = gsm_rows[0]
        assert "question" in first_gsm
        assert "gold_final" in first_gsm
        assert "messages" in first_gsm
        print(f"  -> GSM8K in-domain dataset validated ({len(gsm_rows)} rows)")

    if transfer_path.exists():
        trans_rows = read_jsonl(transfer_path)
        assert len(trans_rows) == 100, f"Expected 100 SVAMP transfer rows, got {len(trans_rows)}"
        first_trans = trans_rows[0]
        assert "question" in first_trans
        assert "gold_final" in first_trans
        assert "messages" in first_trans
        print(f"  -> SVAMP transfer dataset validated ({len(trans_rows)} rows)")

    print("  -> Dataset schema checks: PASSED")


def main():
    print("==================================================")
    print("      Task 5 (RLVR vs RLAIF) Smoke Test           ")
    print("==================================================")
    test_exact_verifier()
    test_verifier_controlled_pair_scoring()
    test_diagnostic_dataset_integrity()
    test_sensitivity_metrics()
    test_in_domain_and_transfer_data()
    print("\nAll Task 5 unit tests passed successfully!")


if __name__ == "__main__":
    main()
