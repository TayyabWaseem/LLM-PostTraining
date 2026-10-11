from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path


def fixed_audit_ids(base_rows: list[dict], per_class: int, seed: int) -> list[int]:
    """Deterministically sample balanced safe/unsafe prompts using fixed seed."""
    rng = np.random.default_rng(seed)
    meta = pd.DataFrame(base_rows)
    ids = []
    for label in ["SAFE", "UNSAFE"]:
        pool = meta.loc[meta["benchmark_class"] == label, "xstest_id"].to_numpy()
        if len(pool) < per_class:
            raise ValueError(f"Not enough {label} rows for audit (found {len(pool)}, need {per_class})")
        ids.extend(rng.choice(pool, size=per_class, replace=False).tolist())
    return sorted(int(x) for x in ids)


def create_audit_sheet(cfg: dict, force_overwrite: bool = False):
    outdir = repo_path(cfg.get("results_dir", "results")) / "task4_safety"
    outdir.mkdir(parents=True, exist_ok=True)

    sft_file = outdir / "generated_sft.jsonl"
    xstest_path = repo_path(cfg["paths"]["xstest"])

    if sft_file.exists():
        base_rows = read_jsonl(sft_file)
    elif xstest_path.exists():
        df_x = pd.read_csv(xstest_path)
        base_rows = df_x.to_dict(orient="records")
    else:
        raise FileNotFoundError(f"Neither {sft_file} nor {xstest_path} found. Run download_assets or generate_responses first.")

    per_class = int(cfg.get("manual_audit_per_class", 30))
    seed = int(cfg.get("seed", 6304))

    audit_ids = fixed_audit_ids(base_rows, per_class=per_class, seed=seed)
    print(f"Sampled {len(audit_ids)} fixed audit prompt IDs ({per_class} SAFE + {per_class} UNSAFE) with seed={seed}.")

    # Load existing sheet if present to prevent wiping human annotations
    sheet_csv = outdir / "manual_audit_sheet.csv"
    existing_map = {}
    if sheet_csv.exists() and not force_overwrite:
        try:
            prev_df = pd.read_csv(sheet_csv)
            for _, r in prev_df.iterrows():
                existing_map[int(r["xstest_id"])] = r.to_dict()
            print(f"Found existing audit sheet with {len(existing_map)} rows. Preserving existing human labels!")
        except Exception:
            existing_map = {}

    # Load responses for all available policies
    policies = ["sft", "dpo", "ppo", "grpo"]
    policy_data = {}
    for p in policies:
        p_file = outdir / f"generated_{p}.jsonl"
        if p_file.exists():
            rows = {r["xstest_id"]: r for r in read_jsonl(p_file)}
            policy_data[p] = rows
        else:
            policy_data[p] = {}

    audit_rows = []
    base_by_id = {int(r["xstest_id"]): r for r in base_rows}

    for x_id in audit_ids:
        base = base_by_id[x_id]
        prev_row = existing_map.get(x_id, {})
        row_dict = {
            "xstest_id": x_id,
            "benchmark_class": base["benchmark_class"],
            "type": base["type"],
            "prompt": base["prompt"],
        }
        for p in policies:
            resp_info = policy_data[p].get(x_id, {})
            # Use generated response if available, else preserve existing, else empty
            row_dict[f"{p}_response"] = resp_info.get("response", prev_row.get(f"{p}_response", ""))
            # Preserve human annotation if already present
            row_dict[f"{p}_manual_label"] = prev_row.get(f"{p}_manual_label", "")

        row_dict["manual_notes"] = prev_row.get("manual_notes", "")
        audit_rows.append(row_dict)

    df_sheet = pd.DataFrame(audit_rows)
    df_sheet.to_csv(sheet_csv, index=False, encoding="utf-8")
    print(f"Saved manual audit sheet to {sheet_csv}")

    # Also maintain simple ID tracker
    id_csv = outdir / "manual_audit_ids.csv"
    prev_id_map = {}
    if id_csv.exists() and not force_overwrite:
        try:
            prev_id_df = pd.read_csv(id_csv)
            for _, r in prev_id_df.iterrows():
                prev_id_map[int(r["xstest_id"])] = r.get("manual_label", "")
        except Exception:
            prev_id_map = {}

    id_df = pd.DataFrame({
        "xstest_id": audit_ids,
        "manual_label": [prev_id_map.get(x_id, "") for x_id in audit_ids],
    })
    id_df.to_csv(id_csv, index=False)
    print(f"Saved audit IDs list to {id_csv}")

    return df_sheet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--overwrite", action="store_true", help="Force overwrite existing human annotations")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    print("==================================================")
    print("      Task 4: Create Fixed 60-Example Audit Sheet  ")
    print("==================================================")
    create_audit_sheet(cfg, force_overwrite=args.overwrite)


if __name__ == "__main__":
    main()
