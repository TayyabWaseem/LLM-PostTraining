from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from task3_grpo.grpo import (
    group_relative_advantages,
    grpo_policy_loss,
    mask_truncated_sequences,
)
from task3_grpo.analyze_group_size import (
    load_k8_cache,
    regroup_equal_generation_budget,
    compute_group_metrics,
)


def test_within_group_normalization():
    print("[1/4] Testing within-group vs global advantage normalization...")
    # Two prompt groups:
    # Prompt 0: all completions get 1.0 (tied/easy prompt)
    # Prompt 1: completions get [0.0, 1.0, 2.0, 3.0]
    rewards = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 2.0, 3.0])
    group_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

    advs, diags = group_relative_advantages(rewards, group_ids, return_diags=True)

    # Group 0 has zero reward variance: advantages must be 0.0!
    assert torch.allclose(advs[:4], torch.zeros(4)), f"Tied group should have 0 advantage, got: {advs[:4]}"
    # Group 1 should be standardized: mean ~ 0, std ~ 1
    g1_adv = advs[4:]
    assert abs(g1_adv.mean().item()) < 1e-5, f"Group 1 mean should be 0, got: {g1_adv.mean().item()}"
    assert abs(g1_adv.std(unbiased=False).item() - 1.0) < 1e-4, f"Group 1 std should be ~1, got: {g1_adv.std().item()}"

    # Informativeness diagnostics
    assert diags["num_groups"] == 2
    assert diags["uninformative_group_fraction"] == 0.5, f"Expected 0.5 uninformative fraction, got: {diags['uninformative_group_fraction']}"
    print("  -> Within-group normalization and tied-group handling: PASSED")


def test_grpo_loss_canonical_vs_dr_grpo():
    print("[2/4] Testing GRPO loss: Canonical (1/T_k) vs Dr. GRPO (1/T_max)...")
    batch_size = 2
    max_len = 10
    # Sequence 0: length 4 tokens
    # Sequence 1: length 8 tokens
    token_mask = torch.zeros((batch_size, max_len))
    token_mask[0, :4] = 1.0
    token_mask[1, :8] = 1.0

    old_logp = torch.zeros((batch_size, max_len))
    new_logp = torch.zeros((batch_size, max_len))
    ref_logp = torch.zeros((batch_size, max_len))
    adv = torch.tensor([1.0, 1.0])

    loss_canon, diags_canon = grpo_policy_loss(
        new_logp=new_logp,
        old_logp=old_logp,
        seq_adv=adv,
        token_mask=token_mask,
        ref_logp=ref_logp,
        eps=0.2,
        beta=0.1,
        loss_type="grpo",
    )

    loss_dr, diags_dr = grpo_policy_loss(
        new_logp=new_logp,
        old_logp=old_logp,
        seq_adv=adv,
        token_mask=token_mask,
        ref_logp=ref_logp,
        eps=0.2,
        beta=0.1,
        loss_type="dr_grpo",
        max_completion_length=10,
    )

    # In canonical: per_sequence is (1*4)/4 = 1.0 and (1*8)/8 = 1.0 -> mean is 1.0 -> policy_term = -1.0
    assert abs(diags_canon["policy_term"].item() - (-1.0)) < 1e-5, f"Expected canonical policy term -1.0, got: {diags_canon['policy_term']}"
    # In Dr. GRPO: per_sequence is 4/10 = 0.4 and 8/10 = 0.8 -> mean is 0.6 -> policy_term = -0.6
    assert abs(diags_dr["policy_term"].item() - (-0.6)) < 1e-5, f"Expected Dr. GRPO policy term -0.6, got: {diags_dr['policy_term']}"
    print("  -> Canonical vs Dr. GRPO loss scaling: PASSED")


def test_truncation_masking():
    print("[3/4] Testing truncation sequence masking...")
    token_mask = torch.ones((3, 5))
    truncated = [False, True, False]

    effective = mask_truncated_sequences(token_mask, truncated)
    assert effective[0].sum().item() == 5.0
    assert effective[1].sum().item() == 0.0, "Truncated sequence must be masked to all zeros"
    assert effective[2].sum().item() == 5.0
    print("  -> Truncation masking: PASSED")


def test_group_size_equal_budget():
    print("[4/4] Testing equal-generation budget regrouping for K in {2, 4, 8}...")
    cache_path = Path("cached/grpo_k_cache.jsonl")
    if not cache_path.exists():
        print("  -> Skipping cache test (cache file not present locally)")
        return

    by_prompt = load_k8_cache(cache_path)
    total_gens = len(by_prompt) * 8

    for k in [2, 4, 8]:
        groups = regroup_equal_generation_budget(by_prompt, k)
        m = compute_group_metrics(groups)
        assert m["total_generations"] == total_gens, f"K={k} total gens {m['total_generations']} != expected {total_gens}"
        assert m["num_groups"] == total_gens // k, f"K={k} num groups {m['num_groups']} != {total_gens // k}"
        print(f"  -> K = {k}: {m['num_groups']} groups, exactly {m['total_generations']} generations. PASSED")


def main():
    print("==================================================")
    print("           Task 3 (GRPO) Smoke Test Suite         ")
    print("==================================================")
    test_within_group_normalization()
    test_grpo_loss_canonical_vs_dr_grpo()
    test_truncation_masking()
    test_group_size_equal_budget()
    print("\nAll Task 3 unit tests passed successfully!")


if __name__ == "__main__":
    main()
