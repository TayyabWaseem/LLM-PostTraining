from __future__ import annotations

import torch
import torch.nn.functional as F


def dpo_loss(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float,
):
    """Return scalar DPO loss plus lightweight diagnostics.

    Validate this implementation against the equation in the assignment manual before using it.
    """
    policy_margin = policy_chosen_logp - policy_rejected_logp
    ref_margin = ref_chosen_logp - ref_rejected_logp

    # Starter implementation: students must validate the objective carefully.
    dpo_margin = policy_margin - ref_margin
    logits = beta * dpo_margin

    loss = -F.logsigmoid(logits).mean()
    return loss, {
        "loss": loss.detach(),
        "logit_mean": logits.detach().mean(),
        "dpo_margin_mean": dpo_margin.detach().mean(),
        "policy_margin_mean": policy_margin.detach().mean(),
        "ref_margin_mean": ref_margin.detach().mean(),
        "preference_accuracy": (dpo_margin > 0).float().mean().detach(),
    }
