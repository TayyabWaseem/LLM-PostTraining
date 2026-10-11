import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.data import (
    load_yaml,
    prompt_messages,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.generation import batch_generate, response_sequence_logprobs
from common.metrics import (
    parse_word_limit,
    preference_accuracy,
    word_count,
    word_limit_compliance,
)
from common.models import clear_gpu, load_policy, load_tokenizer, reference_mode
from task1_dpo.dpo import dpo_loss
from task1_dpo.train import make_collate, run_training


def evaluate_stratified_accuracy(
    policy,
    tokenizer,
    stratified_rows: list[dict],
    beta: float,
    batch_size: int = 2,
    max_seq_len: int = 768,
):
    device = next(policy.parameters()).device
    loader = DataLoader(
        stratified_rows,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_collate(tokenizer, max_seq_len),
    )

    all_dpo_margins = []
    stratum_margins = defaultdict(list)
    stratum_accs = defaultdict(list)

    row_idx = 0
    with torch.no_grad():
        for chosen_batch, rejected_batch in loader:
            current_bs = chosen_batch["input_ids"].shape[0]
            batch_strata = [
                stratified_rows[row_idx + i].get("length_stratum", "unknown")
                for i in range(current_bs)
            ]
            row_idx += current_bs

            chosen_batch = {k: v.to(device) for k, v in chosen_batch.items()}
            rejected_batch = {k: v.to(device) for k, v in rejected_batch.items()}

            with reference_mode(policy):
                ref_chosen_logp, _, _ = response_sequence_logprobs(policy, chosen_batch)
                ref_rej_logp, _, _ = response_sequence_logprobs(policy, rejected_batch)

            pol_chosen_logp, _, _ = response_sequence_logprobs(policy, chosen_batch)
            pol_rej_logp, _, _ = response_sequence_logprobs(policy, rejected_batch)

            pol_margin = pol_chosen_logp - pol_rej_logp
            ref_margin = ref_chosen_logp - ref_rej_logp
            dpo_margin = pol_margin - ref_margin

            for dm, stratum in zip(dpo_margin, batch_strata):
                dm_val = float(dm.item())
                acc_val = float(dm_val > 0)
                all_dpo_margins.append(dm_val)
                stratum_margins[stratum].append(dm_val)
                stratum_accs[stratum].append(acc_val)

    overall_acc = float(np.mean([m > 0 for m in all_dpo_margins])) if all_dpo_margins else 0.0
    res = {
        "overall_accuracy": round(overall_acc, 4),
        "overall_margin_mean": round(float(np.mean(all_dpo_margins)), 4) if all_dpo_margins else 0.0,
        "strata": {},
    }
    for s in ["preferred_longer", "length_matched", "rejected_longer"]:
        if s in stratum_accs:
            res["strata"][s] = {
                "count": len(stratum_accs[s]),
                "accuracy": round(float(np.mean(stratum_accs[s])), 4),
                "margin_mean": round(float(np.mean(stratum_margins[s])), 4),
            }
    return res


def evaluate_word_limit_compliance(
    policy,
    tokenizer,
    prompts_rows: list[dict],
    max_seq_len: int = 768,
    max_new_tokens: int = 128,
):
    prompts = [prompt_messages(r) for r in prompts_rows]
    out = batch_generate(
        policy,
        tokenizer,
        prompts,
        max_prompt_length=max_seq_len,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        do_sample=False,
    )

    records = []
    compliances = []
    word_counts = []
    token_lengths = out["response_lengths"]

    for row, p, resp, tlen in zip(prompts_rows, prompts, out["responses"], token_lengths):
        p_text = p[0]["content"] if p and "content" in p[0] else str(p)
        limit = parse_word_limit(p_text)
        wc = word_count(resp)
        comp = float(wc <= limit) if limit is not None else None
        if comp is not None:
            compliances.append(comp)
        word_counts.append(wc)

        records.append(
            {
                "prompt_id": row.get("prompt_id"),
                "prompt": p_text,
                "limit": limit,
                "response": resp,
                "word_count": wc,
                "token_length": int(tlen),
                "compliant": bool(comp == 1.0) if comp is not None else None,
            }
        )

    compliance_rate = float(np.mean(compliances)) if compliances else 0.0
    mean_words = float(np.mean(word_counts)) if word_counts else 0.0
    mean_tokens = float(np.mean(token_lengths)) if token_lengths else 0.0

    return {
        "compliance_rate": round(compliance_rate, 4),
        "mean_word_count": round(mean_words, 2),
        "mean_token_length": round(mean_tokens, 2),
        "samples": records,
    }


def run_length_analysis(
    config_path: str = "configs/dpo.yaml",
    skip_train: bool = False,
    standard_adapter: str | None = None,
    length_adapter: str | None = None,
    max_train_examples: int | None = None,
    max_eval_examples: int | None = None,
):
    cfg = load_yaml(config_path)
    std_adapter = standard_adapter or cfg.get("standard_output", "outputs/task1_dpo/standard")
    len_adapter = length_adapter or cfg.get("length_output", "outputs/task1_dpo/length_balanced")
    results_dir = repo_path(cfg.get("results_dir", "results/task1_dpo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("      Task 1: DPO Length-Confounding Study        ")
    print("==================================================")

    # 1. Train length-balanced model if requested
    if not skip_train:
        print(f"\n>>> Training Length-Balanced DPO Model on {cfg['paths']['dpo_length_train']} <<<")
        run_training(
            config_path=config_path,
            run_name="length_balanced",
            dataset_path=cfg["paths"]["dpo_length_train"],
            output_path=len_adapter,
            beta=float(cfg.get("beta", 0.10)),
            max_examples=max_train_examples,
        )
        clear_gpu()

    # Load shared assets for evaluation
    stratified_rows = read_jsonl(cfg["paths"]["dpo_length_eval"])
    if max_eval_examples is not None:
        stratified_rows = stratified_rows[: int(max_eval_examples)]

    word_limit_rows = read_jsonl(cfg["paths"]["word_limit_prompts"])
    tokenizer = load_tokenizer(cfg["base_model"])
    beta = float(cfg.get("beta", 0.10))
    max_seq_len = int(cfg.get("max_sequence_length", 768))

    eval_targets = [
        ("standard_dpo", std_adapter),
        ("length_balanced_dpo", len_adapter),
    ]

    analysis_results = {}

    for label, adapter_path in eval_targets:
        print(f"\n>>> Evaluating {label} ({adapter_path}) <<<")
        try:
            policy = load_policy(cfg, adapter_path=adapter_path, trainable=False)
        except Exception as e:
            print(f"Warning: could not load adapter from {adapter_path}: {e}")
            continue

        strat_res = evaluate_stratified_accuracy(
            policy=policy,
            tokenizer=tokenizer,
            stratified_rows=stratified_rows,
            beta=beta,
            max_seq_len=max_seq_len,
        )

        wl_res = evaluate_word_limit_compliance(
            policy=policy,
            tokenizer=tokenizer,
            prompts_rows=word_limit_rows,
            max_seq_len=max_seq_len,
        )

        analysis_results[label] = {
            "stratified": strat_res,
            "word_limit": wl_res,
        }
        clear_gpu(policy)

    # Output comparison table
    out_file = results_dir / "length_analysis_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(analysis_results, f, indent=2)

    print("\n==================================================")
    print("      Length Confounding: Stratified Accuracy      ")
    print("==================================================")
    print("| Model | Overall Acc | Pref. Longer Acc | Matched Acc | Rej. Longer Acc |")
    print("| :--- | :--- | :--- | :--- | :--- |")
    for label, data in analysis_results.items():
        st = data["stratified"]
        p_long = st["strata"].get("preferred_longer", {}).get("accuracy", 0.0)
        match = st["strata"].get("length_matched", {}).get("accuracy", 0.0)
        r_long = st["strata"].get("rejected_longer", {}).get("accuracy", 0.0)
        print(f"| {label} | {st['overall_accuracy']:.2%} | {p_long:.2%} | {match:.2%} | {r_long:.2%} |")

    print("\n==================================================")
    print("      Word-Limit Compliance & Length Bias        ")
    print("==================================================")
    print("| Model | Word-Limit Compliance (%) | Mean Words | Mean Response Tokens |")
    print("| :--- | :--- | :--- | :--- |")
    for label, data in analysis_results.items():
        wl = data["word_limit"]
        print(f"| {label} | {wl['compliance_rate']:.2%} | {wl['mean_word_count']} words | {wl['mean_token_length']} tokens |")

    print(f"\nSaved analysis summary to {out_file}")
    return analysis_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--standard-adapter")
    ap.add_argument("--length-adapter")
    ap.add_argument("--max-train-examples", type=int)
    ap.add_argument("--max-eval-examples", type=int)
    args = ap.parse_args()

    run_length_analysis(
        config_path=args.config,
        skip_train=args.skip_train,
        standard_adapter=args.standard_adapter,
        length_adapter=args.length_adapter,
        max_train_examples=args.max_train_examples,
        max_eval_examples=args.max_eval_examples,
    )


if __name__ == "__main__":
    main()

