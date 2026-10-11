from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.generation import response_sequence_logprobs
from common.logging_utils import set_seed
from common.models import (
    load_policy,
    load_tokenizer,
    reference_mode,
    trainable_parameters,
)
from task1_dpo.dpo import dpo_loss


def make_collate(tokenizer, max_length):
    def collate(rows):
        chosen, rejected = [], []
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
            rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
        return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected)
    return collate


def prepare_dpo_run(config_path: str, dataset_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    path = dataset_path or cfg["paths"]["dpo_standard_train"]
    rows = read_jsonl(path)
    if max_examples is not None:
        rows = rows[: int(max_examples)]

    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, trainable=True, fresh_lora=True)
    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        collate_fn=make_collate(tokenizer, int(cfg["max_sequence_length"])),
    )
    optimizer = AdamW(
        trainable_parameters(model),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    return {
        "cfg": cfg,
        "rows": rows,
        "tokenizer": tokenizer,
        "model": model,
        "loader": loader,
        "optimizer": optimizer,
        "beta": float(cfg["beta"] if beta is None else beta),
    }


def run_training(
    config_path: str,
    run_name: str,
    dataset_path: str | None = None,
    output_path: str | None = None,
    beta: float | None = None,
    max_examples: int | None = None,
):
    bundle = prepare_dpo_run(config_path, dataset_path, beta, max_examples)
    cfg = bundle["cfg"]
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    loader = bundle["loader"]
    optimizer = bundle["optimizer"]
    beta_val = bundle["beta"]

    output = repo_path(output_path or cfg["standard_output"])
    output.mkdir(parents=True, exist_ok=True)
    results_dir = repo_path(cfg.get("results_dir", "results/task1_dpo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    grad_accum_steps = int(cfg.get("grad_accum_steps", 8))
    epochs = int(cfg.get("epochs", 1))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))

    device = next(model.parameters()).device
    model.train()
    model.config.use_cache = False
    optimizer.zero_grad()

    step = 0
    update_count = 0
    train_logs = []
    total_batches = len(loader)

    print(f"=== Starting DPO Training: {run_name} ===")
    print(f"Device: {device} | Total examples: {len(bundle['rows'])} | Batches per epoch: {total_batches}")
    print(f"Epochs: {epochs} | Batch size: {cfg['batch_size']} | Grad accum: {grad_accum_steps} | Beta: {beta_val}")

    for epoch in range(epochs):
        accum = {
            "loss": 0.0,
            "logit_mean": 0.0,
            "dpo_margin_mean": 0.0,
            "policy_margin_mean": 0.0,
            "ref_margin_mean": 0.0,
            "preference_accuracy": 0.0,
        }

        for batch_idx, (chosen_batch, rejected_batch) in enumerate(loader):
            step += 1
            is_last_batch = (batch_idx + 1 == total_batches)

            # Exact group size for accurate gradient scaling and accumulation averaging
            group_start = (batch_idx // grad_accum_steps) * grad_accum_steps
            group_size = min(grad_accum_steps, total_batches - group_start)

            chosen_batch = {k: v.to(device) for k, v in chosen_batch.items()}
            rejected_batch = {k: v.to(device) for k, v in rejected_batch.items()}

            # Reference policy forward pass with adapters disabled
            with torch.no_grad():
                with reference_mode(model):
                    ref_chosen_logp, _, _ = response_sequence_logprobs(model, chosen_batch)
                    ref_rejected_logp, _, _ = response_sequence_logprobs(model, rejected_batch)

            # Active policy forward pass
            policy_chosen_logp, _, _ = response_sequence_logprobs(model, chosen_batch)
            policy_rejected_logp, _, _ = response_sequence_logprobs(model, rejected_batch)

            loss, diags = dpo_loss(
                policy_chosen_logp,
                policy_rejected_logp,
                ref_chosen_logp,
                ref_rejected_logp,
                beta=beta_val,
            )

            scaled_loss = loss / group_size
            scaled_loss.backward()

            accum["loss"] += float(loss.item())
            for k in [
                "logit_mean",
                "dpo_margin_mean",
                "policy_margin_mean",
                "ref_margin_mean",
                "preference_accuracy",
            ]:
                accum[k] += float(diags[k].item())

            if (batch_idx + 1) % grad_accum_steps == 0 or is_last_batch:
                if max_grad_norm > 0:
                    gnorm = torch.nn.utils.clip_grad_norm_(trainable_parameters(model), max_grad_norm)
                    gnorm_val = float(gnorm.item() if hasattr(gnorm, "item") else gnorm)
                else:
                    gnorm_val = 0.0

                optimizer.step()
                optimizer.zero_grad()
                update_count += 1

                step_record = {
                    "step": step,
                    "update": update_count,
                    "epoch": epoch + 1,
                    "loss": round(accum["loss"] / group_size, 4),
                    "logit_mean": round(accum["logit_mean"] / group_size, 4),
                    "dpo_margin_mean": round(accum["dpo_margin_mean"] / group_size, 4),
                    "policy_margin_mean": round(accum["policy_margin_mean"] / group_size, 4),
                    "ref_margin_mean": round(accum["ref_margin_mean"] / group_size, 4),
                    "preference_accuracy": round(accum["preference_accuracy"] / group_size, 4),
                    "grad_norm": round(gnorm_val, 4),
                }
                train_logs.append(step_record)
                accum = {k: 0.0 for k in accum}

                if update_count % max(1, 5) == 0 or is_last_batch:
                    print(
                        f"Epoch {epoch+1} | Update {update_count} (Step {step}/{total_batches * epochs}) | "
                        f"Loss: {step_record['loss']:.4f} | Margin: {step_record['dpo_margin_mean']:.4f} | "
                        f"Pref Acc: {step_record['preference_accuracy']:.2%} | GradNorm: {step_record['grad_norm']:.3f}"
                    )

    # Save trained PEFT adapter and tokenizer
    print(f"Saving trained adapter to {output}...")
    model.save_pretrained(str(output))
    tokenizer.save_pretrained(str(output))

    log_path = results_dir / f"{run_name}_train_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_name": run_name,
                "config_path": config_path,
                "dataset_path": dataset_path or cfg["paths"]["dpo_standard_train"],
                "beta": beta_val,
                "total_examples": len(bundle["rows"]),
                "total_steps": step,
                "total_updates": update_count,
                "final_loss": train_logs[-1]["loss"] if train_logs else 0.0,
                "history": train_logs,
            },
            f,
            indent=2,
        )
    print(f"Training logs saved to {log_path}")

    return {
        "output_path": str(output),
        "total_steps": step,
        "total_updates": update_count,
        "history": train_logs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    args = ap.parse_args()
    run_training(args.config, args.run_name, args.dataset, args.output, args.beta, args.max_examples)


if __name__ == "__main__":
    main()

