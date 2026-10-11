"""Smoke test suite to validate the Task 2 (PPO) mathematical objective and clipping geometry.

Validates:
1. PPO clipped surrogate 4-quadrant clipping bounds and gradients.
2. GAE computation and value targets (using raw advantages).
3. Critic explained variance calculation.
4. Cached rollout batch geometric clipping analysis.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.data import load_yaml
from task2_ppo.ppo import (
    compute_gae,
    explained_variance,
    normalize_advantages,
    ppo_policy_loss,
    shaped_rewards,
    value_mse_loss,
)
from task2_ppo.analyze_clipping import analyze_cached_batch_geometry, load_cached_rollouts


def test_ppo_loss_4_quadrants():
    print("[1/4] Testing PPO clipped surrogate 4-quadrant geometry...")
    eps = 0.2
    mask = torch.tensor([[1.0]])

    # Helper to evaluate objective for given advantage and ratio
    def eval_pt(adv_val, ratio_val):
        old_lp = torch.tensor([[0.0]])
        new_lp = torch.tensor([[math.log(ratio_val)]], requires_grad=True)
        adv = torch.tensor([[adv_val]])
        loss, diags = ppo_policy_loss(new_lp, old_lp, adv, mask, eps=eps)
        obj = diags["objective_mean"].item()
        loss.backward()
        grad = new_lp.grad.item()
        return obj, diags["clip_fraction"].item(), diags["affected_fraction"].item(), grad

    # Quadrant 1: A = +1, ratio = 1.5 -> clipped to 1.2, zero gradient with respect to ratio
    obj, clip_f, aff_f, grad = eval_pt(+1.0, 1.5)
    assert abs(obj - 1.2) < 1e-4, f"Q1 failed: expected 1.2, got {obj}"
    assert clip_f == 1.0 and aff_f == 1.0, "Q1 should be clipped and affected"
    assert abs(grad) < 1e-4, f"Q1 gradient should be 0, got {grad}"

    # Quadrant 2: A = -1, ratio = 0.5 -> clipped to -0.8, zero gradient with respect to ratio
    obj, clip_f, aff_f, grad = eval_pt(-1.0, 0.5)
    assert abs(obj - (-0.8)) < 1e-4, f"Q2 failed: expected -0.8, got {obj}"
    assert clip_f == 1.0 and aff_f == 1.0, "Q2 should be clipped and affected"
    assert abs(grad) < 1e-4, f"Q2 gradient should be 0, got {grad}"

    # Quadrant 3: A = +1, ratio = 0.5 -> unclipped (surr1 = 0.5 < surr2 = 0.8)
    obj, clip_f, aff_f, grad = eval_pt(+1.0, 0.5)
    assert abs(obj - 0.5) < 1e-4, f"Q3 failed: expected 0.5, got {obj}"
    assert clip_f == 1.0, "Q3 ratio is < 0.8 so clip_fraction is 1.0"
    assert aff_f == 0.0, "Q3 is NOT affected by clipping (gradient flows)"
    assert grad < 0, f"Q3 should have non-zero gradient pushing probability up, got {grad}"

    # Quadrant 4: A = -1, ratio = 1.5 -> unclipped (surr1 = -1.5 < surr2 = -1.2)
    obj, clip_f, aff_f, grad = eval_pt(-1.0, 1.5)
    assert abs(obj - (-1.5)) < 1e-4, f"Q4 failed: expected -1.5, got {obj}"
    assert clip_f == 1.0, "Q4 ratio is > 1.2 so clip_fraction is 1.0"
    assert aff_f == 0.0, "Q4 is NOT affected by clipping (gradient flows)"
    assert grad > 0, f"Q4 should have non-zero gradient pushing probability down, got {grad}"

    print("  -> PPO 4-quadrant clipping and gradient flow PASSED.")


def test_gae_and_returns():
    print("[2/4] Testing GAE and raw value return targets...")
    rewards = torch.tensor([[0.0, 0.0, 2.0]])
    values = torch.tensor([[1.0, 1.5, 1.8]])
    mask = torch.tensor([[1.0, 1.0, 1.0]])

    adv, returns = compute_gae(rewards, values, mask, gamma=1.0, lam=0.95)
    # Check that returns == adv + values exactly
    assert torch.allclose(returns, adv + values), "Returns must equal raw advantages + values!"

    # Verify that normalize_advantages does not touch returns
    norm_adv = normalize_advantages(adv, mask)
    assert not torch.allclose(norm_adv, adv), "Normalized advantages should differ in scale from raw advantages"
    print("  -> GAE and returns sanity check PASSED.")


def test_explained_variance():
    print("[3/4] Testing explained variance...")
    returns = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    # Perfect predictor
    pred_perfect = returns.clone()
    mask = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    ev_perfect = explained_variance(pred_perfect, returns, mask)
    assert abs(ev_perfect - 1.0) < 1e-4, f"Expected EV=1.0, got {ev_perfect}"

    # Constant mean predictor (EV=0)
    pred_mean = torch.tensor([[2.5, 2.5, 2.5, 2.5]])
    ev_mean = explained_variance(pred_mean, returns, mask)
    assert abs(ev_mean) < 1e-4, f"Expected EV=0.0, got {ev_mean}"

    # Terrible predictor (EV < 0)
    pred_bad = torch.tensor([[-10.0, -20.0, -30.0, -40.0]])
    ev_bad = explained_variance(pred_bad, returns, mask)
    assert ev_bad < 0.0, f"Expected negative EV, got {ev_bad}"
    print("  -> Explained variance metric PASSED.")


def test_cached_batch_analysis():
    print("[4/4] Testing cached rollouts geometric clipping analysis...")
    cfg = load_yaml("configs/ppo.yaml")
    rows = load_cached_rollouts(cfg["cached_rollouts"])
    assert len(rows) == 32, f"Expected 32 rollouts, found {len(rows)}"

    res = analyze_cached_batch_geometry(rows[:4], cfg)
    assert "eps_0.05" in res and "eps_0.2" in res and "eps_0.5" in res
    assert res["eps_0.05"]["clip_fraction"] >= res["eps_0.5"]["clip_fraction"], "Tighter epsilon must clip more tokens"
    print("  -> Cached batch geometric analysis PASSED.")


def main():
    print("==================================================")
    print("        Starting Task 2 Smoke Test Suite          ")
    print("==================================================")
    test_ppo_loss_4_quadrants()
    test_gae_and_returns()
    test_explained_variance()
    test_cached_batch_analysis()
    print("\n==================================================")
    print("     ALL TASK 2 SMOKE TESTS COMPLETED SUCCESSFULLY! ")
    print("==================================================")


if __name__ == "__main__":
    main()
