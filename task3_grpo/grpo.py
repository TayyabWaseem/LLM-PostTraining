from __future__ import annotations

import torch

from common.metrics import masked_mean, sampled_kl, sample_entropy


def group_relative_advantages(
    rewards: torch.Tensor,
    group_ids: torch.Tensor,
    eps: float = 1e-6,
    return_diags: bool = False,
):
    """Return one scalar advantage per sampled completion using within-prompt normalization.

    `group_ids[i]` identifies which prompt produced reward `rewards[i]`.
    For each prompt group g:
        A_k = (r_k - mu_g) / (sigma_g + eps)
    If a group has zero reward variance (sigma_g < eps), relative advantages are zero.
    """
    rewards = rewards.float()
    advantages = torch.zeros_like(rewards)
    unique_groups = torch.unique(group_ids)

    group_stds = []
    uninformative_count = 0

    for g in unique_groups:
        mask = (group_ids == g)
        group_r = rewards[mask]
        n = group_r.numel()

        if n <= 1:
            advantages[mask] = 0.0
            group_stds.append(0.0)
            uninformative_count += 1
            continue

        mu_g = group_r.mean()
        sigma_g = group_r.std(unbiased=False)

        group_stds.append(float(sigma_g.item()))
        if sigma_g.item() < eps:
            advantages[mask] = 0.0
            uninformative_count += 1
        else:
            advantages[mask] = (group_r - mu_g) / (sigma_g + eps)

    diags = {
        "within_group_reward_std": float(sum(group_stds) / max(len(group_stds), 1)),
        "uninformative_group_fraction": float(uninformative_count / max(len(unique_groups), 1)),
        "num_groups": len(unique_groups),
        "group_stds": group_stds,
    }

    if return_diags:
        return advantages, diags
    return advantages


def grpo_policy_loss(
    new_logp,
    old_logp,
    seq_adv,
    token_mask,
    ref_logp,
    eps,
    beta,
    loss_type="grpo",
    max_completion_length: int | None = None,
):
    """PPO-style clipped GRPO loss for already-sampled completions.

    `token_mask` may be all-zero for a completion that was deliberately masked because it hit the
    maximum generation length.
    """
    ratio = torch.exp(new_logp - old_logp)
    adv = seq_adv[:, None]
    s1 = ratio * adv
    s2 = ratio.clamp(1.0 - eps, 1.0 + eps) * adv
    objective = torch.minimum(s1, s2)

    token_sum = (objective * token_mask).sum(-1)
    valid_seq = (token_mask.sum(-1) > 0).float()
    if loss_type == "grpo":
        denom = token_mask.sum(-1).clamp_min(1.0)
        per_sequence = token_sum / denom
        policy_term = -(per_sequence * valid_seq).sum() / valid_seq.sum().clamp_min(1.0)
    elif loss_type == "dr_grpo":
        if max_completion_length is None:
            raise ValueError("dr_grpo requires max_completion_length")
        # Constant normalization rather than dividing by each response's realized length.
        per_sequence = token_sum / float(max_completion_length)
        policy_term = -(per_sequence * valid_seq).sum() / valid_seq.sum().clamp_min(1.0)
    else:
        raise ValueError(f"Unknown loss_type={loss_type!r}")

    log_ratio_ref_over_policy = ref_logp - new_logp
    per_token_kl = torch.exp(log_ratio_ref_over_policy) - log_ratio_ref_over_policy - 1.0
    kl = masked_mean(per_token_kl, token_mask)
    loss = policy_term + float(beta) * kl
    clipped_tokens = ((ratio < (1.0 - eps)) | (ratio > (1.0 + eps))).float()
    actually_affected = (
        ((adv > 0) & (ratio > (1.0 + eps)))
        | ((adv < 0) & (ratio < (1.0 - eps)))
    ).float()

    return loss, {
        "policy_term": policy_term.detach(),
        "sampled_kl": kl.detach(),
        "clip_fraction": masked_mean(clipped_tokens, token_mask).detach(),
        "affected_fraction": masked_mean(actually_affected, token_mask).detach(),
        "ratio_mean": masked_mean(ratio.detach(), token_mask),
        "sample_entropy": sample_entropy(new_logp.detach(), token_mask),
    }


def mask_truncated_sequences(token_mask: torch.Tensor, truncated: list[bool] | torch.Tensor):
    truncated = torch.as_tensor(truncated, device=token_mask.device, dtype=torch.bool)
    keep = (~truncated).to(token_mask.dtype)[:, None]
    return token_mask * keep
