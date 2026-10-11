from __future__ import annotations

import torch

from common.metrics import masked_mean


def compute_gae(rewards, values, mask, gamma=1.0, lam=0.95):
    """Token-level GAE over response positions.

    rewards, values, mask: [batch, response_steps]. Padding positions must have mask=0.
    The final valid response position bootstraps with zero.
    """
    batch, steps = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros(batch, device=rewards.device, dtype=rewards.dtype)

    for t in reversed(range(steps)):
        current_valid = mask[:, t]
        if t + 1 < steps:
            next_valid = mask[:, t + 1]
            next_value = values[:, t + 1] * next_valid
        else:
            next_valid = torch.zeros_like(current_valid)
            next_value = torch.zeros_like(last_adv)

        delta = rewards[:, t] + gamma * next_value - values[:, t]
        last_adv = delta + gamma * lam * next_valid * last_adv
        last_adv = last_adv * current_valid
        advantages[:, t] = last_adv

    returns = (advantages + values) * mask
    return advantages, returns


def shaped_rewards(task_reward, policy_logp, ref_logp, response_mask, beta_kl):
    """Sampled-action KL shaping plus terminal learned reward."""
    rewards = -float(beta_kl) * (policy_logp - ref_logp) * response_mask
    for b in range(rewards.shape[0]):
        valid = int(response_mask[b].sum().item())
        if valid > 0:
            rewards[b, valid - 1] += task_reward[b]
    return rewards


def ppo_policy_loss(new_logp, old_logp, advantage, mask, eps=0.2):
    """Return PPO clipped policy loss and diagnostics.

    Validate this starter implementation against the clipped surrogate in the assignment manual.
    """
    ratio = torch.exp(new_logp - old_logp)
    surr1 = ratio * advantage
    surr2 = ratio.clamp(1.0 - eps, 1.0 + eps) * advantage

    # Fixed PPO objective: minimum forms the pessimistic lower bound on policy updates
    objective = torch.minimum(surr1, surr2)

    loss = -masked_mean(objective, mask)
    clipped_tokens = ((ratio < (1.0 - eps)) | (ratio > (1.0 + eps))).float()
    clip_fraction = masked_mean(clipped_tokens, mask)

    # Affected-token fraction: tokens where the clipping bound actually changes the objective value
    # (advantage > 0 and ratio > 1 + eps) or (advantage < 0 and ratio < 1 - eps)
    actually_affected = (
        ((advantage > 0) & (ratio > (1.0 + eps)))
        | ((advantage < 0) & (ratio < (1.0 - eps)))
    ).float()
    affected_fraction = masked_mean(actually_affected, mask)

    log_ratio = new_logp - old_logp
    approx_kl = masked_mean(0.5 * (log_ratio ** 2), mask)

    diags = {
        "loss": loss.detach(),
        "ratio": ratio.detach(),
        "clip_fraction": clip_fraction.detach(),
        "affected_fraction": affected_fraction.detach(),
        "approx_kl": approx_kl.detach(),
        "objective_mean": masked_mean(objective, mask).detach(),
        "surr1_mean": masked_mean(surr1, mask).detach(),
        "surr2_mean": masked_mean(surr2, mask).detach(),
    }
    return loss, diags


def explained_variance(predicted_values: torch.Tensor, returns: torch.Tensor, mask: torch.Tensor) -> float:
    """Calculate explained variance of critic value predictions against GAE returns.
    
    Var_expl = 1 - Var(returns - predicted_values) / Var(returns)
    """
    valid_pred = predicted_values[mask.bool()]
    valid_ret = returns[mask.bool()]
    if valid_ret.numel() <= 1:
        return 0.0
    var_ret = torch.var(valid_ret, unbiased=False)
    if var_ret.item() == 0.0:
        return 0.0
    var_diff = torch.var(valid_ret - valid_pred, unbiased=False)
    return float((1.0 - var_diff / var_ret).item())


def value_mse_loss(predicted_values, returns, mask):
    return masked_mean((predicted_values - returns) ** 2, mask)


def normalize_advantages(advantages, mask, eps=1e-6):
    valid = advantages[mask.bool()]
    if valid.numel() <= 1:
        return advantages
    mean = valid.mean()
    std = valid.std(unbiased=False).clamp_min(eps)
    return ((advantages - mean) / std) * mask
