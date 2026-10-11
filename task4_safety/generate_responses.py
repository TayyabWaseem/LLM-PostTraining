from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate
from common.models import clear_gpu, load_policy, load_tokenizer


def policy_specs(cfg):
    return {
        "sft": None,
        "dpo": cfg["policies"]["dpo"],
        "ppo": cfg["policies"]["ppo"],
        "grpo": cfg["policies"]["grpo"],
    }


def load_xstest(cfg):
    return pd.read_csv(repo_path(cfg["paths"]["xstest"]))


def generate_for_policy(cfg, policy_name: str, batch_size: int = 8, force: bool = False):
    outdir = repo_path(cfg.get("results_dir", "results")) / "task4_safety"
    outdir.mkdir(parents=True, exist_ok=True)
    out_file = outdir / f"generated_{policy_name}.jsonl"

    df = load_xstest(cfg)
    total_prompts = len(df)

    if out_file.exists() and not force:
        try:
            existing = read_jsonl(out_file)
            if len(existing) == total_prompts:
                print(f"\n[Skip] '{policy_name}' already fully generated ({total_prompts}/{total_prompts}) at {out_file}.")
                print("       Pass --force to regenerate.")
                return existing
        except Exception:
            pass

    specs = policy_specs(cfg)
    if policy_name not in specs:
        raise KeyError(f"Unknown policy '{policy_name}'. Valid policies: {list(specs.keys())}")

    adapter = specs[policy_name]
    print(f"\n>>> Generating XSTest responses for policy: {policy_name.upper()} <<<")
    if adapter:
        adapter_resolved = repo_path(adapter)
        if not adapter_resolved.exists():
            raise FileNotFoundError(
                f"Trained adapter for policy '{policy_name}' not found at {adapter_resolved}.\n"
                f"Please ensure Task {1 if policy_name=='dpo' else (2 if policy_name=='ppo' else 3)} "
                f"training is completed and saved to '{adapter}', or pass --policy sft."
            )
        print(f"Loading adapter: {adapter_resolved}")
    else:
        print(f"Using untouched base model: {cfg['base_model']}")

    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, adapter_path=adapter, trainable=False)

    records = []
    max_new_tokens = int(cfg.get("safety_max_new_tokens", 256))

    for start in range(0, total_prompts, batch_size):
        chunk = df.iloc[start : start + batch_size]
        prompts = [[{"role": "user", "content": str(x)}] for x in chunk["prompt"].tolist()]
        gen = batch_generate(
            model,
            tokenizer,
            prompts,
            max_prompt_length=256,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        )

        for (_, row), response, n_tok in zip(chunk.iterrows(), gen["responses"], gen["response_lengths"]):
            records.append(
                {
                    "xstest_id": int(row["xstest_id"]),
                    "policy": policy_name,
                    "prompt": str(row["prompt"]),
                    "benchmark_class": str(row["benchmark_class"]),
                    "type": str(row["type"]),
                    "response": response,
                    "response_tokens": int(n_tok),
                }
            )

        if (start // batch_size) % 10 == 0:
            print(f"  Processed {min(start + batch_size, total_prompts)}/{total_prompts} prompts...")

    write_jsonl(out_file, records)
    print(f"Saved {len(records)} responses for '{policy_name}' to {out_file}")
    clear_gpu()
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--policy", choices=["sft", "dpo", "ppo", "grpo", "all"], default="all")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="Force regenerate responses even if already present")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    outdir = repo_path(cfg.get("results_dir", "results")) / "task4_safety"
    outdir.mkdir(parents=True, exist_ok=True)

    target_policies = list(policy_specs(cfg).keys()) if args.policy == "all" else [args.policy]

    print("==================================================")
    print("      Task 4: Common Deterministic Generation     ")
    print("==================================================")
    print(f"Output directory: {outdir}")
    print(f"Policies to generate: {target_policies}")

    for pol in target_policies:
        generate_for_policy(cfg, pol, batch_size=args.batch_size, force=args.force)

    print("\nGeneration process finished for all requested policies.")


if __name__ == "__main__":
    main()
