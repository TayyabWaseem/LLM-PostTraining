"""Smoke test suite to validate the entire Task 1 (DPO) pipeline.

Validates:
1. Mathematical correctness and gradients of dpo_loss.
2. Data encoding and batch collation.
3. Training loop execution (1 forward/backward/optimizer step) with reference_mode.
4. Evaluation pipeline (preference accuracy, KL divergence, response generation).
5. Length confounding & word-limit compliance logic.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.data import load_yaml, read_jsonl, repo_path
from common.metrics import parse_word_limit, word_count, word_limit_compliance
from common.models import clear_gpu, load_policy, load_tokenizer
from task1_dpo.dpo import dpo_loss
from task1_dpo.evaluate import evaluate_dpo_policy
from task1_dpo.train import make_collate, run_training
from task1_dpo.analyze_length import (
    evaluate_stratified_accuracy,
    evaluate_word_limit_compliance,
)


def test_dpo_loss_mathematics():
    print("[1/5] Testing dpo_loss mathematical correctness...")
    beta = 0.1
    # Example 1: Policy prefers chosen much more than reference (m_theta > 0)
    pol_chosen = torch.tensor([-1.0], requires_grad=True)
    pol_rejected = torch.tensor([-3.0], requires_grad=True)
    ref_chosen = torch.tensor([-2.0])
    ref_rejected = torch.tensor([-2.0])

    # policy_margin = (-1.0 - (-3.0)) = 2.0
    # ref_margin = (-2.0 - (-2.0)) = 0.0
    # dpo_margin = 2.0 - 0.0 = 2.0
    # logits = beta * 2.0 = 0.2
    # loss = -log(sigmoid(0.2))
    expected_loss = -math.log(1.0 / (1.0 + math.exp(-0.2)))

    loss, diags = dpo_loss(pol_chosen, pol_rejected, ref_chosen, ref_rejected, beta)
    assert abs(loss.item() - expected_loss) < 1e-5, f"Loss mismatch: {loss.item()} vs {expected_loss}"
    assert diags["preference_accuracy"].item() == 1.0, "Preference accuracy should be 1.0"
    assert abs(diags["dpo_margin_mean"].item() - 2.0) < 1e-5, "DPO margin mean mismatch"

    # Backward pass check
    loss.backward()
    assert pol_chosen.grad is not None and pol_chosen.grad.item() != 0.0, "Gradients must propagate to policy_chosen"
    assert pol_rejected.grad is not None and pol_rejected.grad.item() != 0.0, "Gradients must propagate to policy_rejected"

    # Example 2: Negative margin (m_theta < 0)
    pol_chosen_bad = torch.tensor([-4.0])
    pol_rejected_bad = torch.tensor([-1.0])
    loss_bad, diags_bad = dpo_loss(pol_chosen_bad, pol_rejected_bad, ref_chosen, ref_rejected, beta)
    assert diags_bad["preference_accuracy"].item() == 0.0, "Preference accuracy should be 0.0 for negative margin"
    assert loss_bad.item() > loss.item(), "Loss on worse margin should be higher"
    print("  -> dpo_loss mathematical check PASSED.")


def test_metrics_and_word_limit():
    print("[2/5] Testing metrics and word limit parsing...")
    prompt = "Explain gradient descent in under 35 words."
    limit = parse_word_limit(prompt)
    assert limit == 35, f"Expected 35, got {limit}"

    text_short = "Gradient descent is an optimization algorithm that iteratively adjusts parameters."
    assert word_limit_compliance(prompt, text_short) == 1.0

    text_long = "word " * 40
    assert word_limit_compliance(prompt, text_long) == 0.0
    print("  -> Word-limit parser check PASSED.")


def test_dpo_pipeline_end_to_end():
    print("[3/5] Testing DPO training loop on 2 examples...")
    smoke_adapter = "outputs/task1_dpo/smoke_test_adapter"
    smoke_run_name = "smoke_test"

    # Run 1 epoch with 2 examples (grad_accum=1 for smoke test)
    # Using small sequence length for quick test
    train_res = run_training(
        config_path="configs/smoke_dpo.yaml",
        run_name=smoke_run_name,
        output_path=smoke_adapter,
        beta=0.10,
        max_examples=2,
    )

    adapter_path = Path(train_res["output_path"])
    assert (adapter_path / "adapter_model.safetensors").exists() or (adapter_path / "adapter_model.bin").exists() or (adapter_path / "adapter_config.json").exists(), "LoRA adapter files were not saved!"
    
    # Mathematical sanity check on step 1: zero-init LoRA means policy == reference
    # Hence, m_theta == 0 -> loss == -log(0.5) ~ 0.693
    initial_loss = train_res["history"][0]["loss"]
    initial_margin = train_res["history"][0]["dpo_margin_mean"]
    print(f"  -> Initial Step 1 Loss: {initial_loss:.4f} (Expected ~0.693) | Margin: {initial_margin:.4f} (Expected ~0.0)")
    assert abs(initial_loss - 0.6931) < 0.05, f"Initial loss mismatch: {initial_loss} vs ~0.693"
    assert abs(initial_margin) < 0.05, f"Initial margin mismatch: {initial_margin} vs ~0.0"

    print("  -> DPO training smoke run PASSED. Adapter successfully created.")
    clear_gpu()

    print("[4/5] Testing DPO evaluation pipeline on 2 held-out pairs...")
    eval_res = evaluate_dpo_policy(
        config_path="configs/smoke_dpo.yaml",
        adapter=smoke_adapter,
        name="smoke_test",
        max_eval_examples=2,
        generate_eval=True,
        max_gen_examples=2,
        load_reward=False,  # Skip 8-bit RM in local smoke test to avoid GPU-only requirement
    )

    assert "held_out_preference_accuracy" in eval_res
    assert "held_out_kl" in eval_res
    assert "mean_response_length" in eval_res
    print("  -> DPO evaluation pipeline check PASSED.")
    clear_gpu()

    print("[5/5] Testing stratified accuracy and word-limit evaluation functions...")
    cfg = load_yaml("configs/smoke_dpo.yaml")
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=smoke_adapter, trainable=False)

    stratified_rows = read_jsonl(cfg["paths"]["dpo_length_eval"])[:2]
    strat_res = evaluate_stratified_accuracy(
        policy=policy,
        tokenizer=tokenizer,
        stratified_rows=stratified_rows,
        beta=0.10,
        batch_size=2,
    )
    assert "overall_accuracy" in strat_res
    assert "strata" in strat_res

    word_limit_rows = read_jsonl(cfg["paths"]["word_limit_prompts"])[:2]
    wl_res = evaluate_word_limit_compliance(
        policy=policy,
        tokenizer=tokenizer,
        prompts_rows=word_limit_rows,
        max_new_tokens=16,
    )
    assert "compliance_rate" in wl_res
    assert "mean_word_count" in wl_res
    print("  -> Stratified and word-limit functions PASSED.")
    clear_gpu(policy)


def main():
    print("==================================================")
    print("        Starting Task 1 Smoke Test Suite          ")
    print("==================================================")
    test_dpo_loss_mathematics()
    test_metrics_and_word_limit()
    test_dpo_pipeline_end_to_end()
    print("\n==================================================")
    print("     ALL TASK 1 SMOKE TESTS COMPLETED SUCCESSFULLY! ")
    print("==================================================")


if __name__ == "__main__":
    main()
