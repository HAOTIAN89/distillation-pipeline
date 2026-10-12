#!/usr/bin/env python3
"""
Prepare the code/SWE distillation dataset (v1, 20K samples).

Sources:
  1. nvidia/OpenCodeReasoning            →  5 000 samples
  2. nvidia/Nemotron-SFT-SWE-v2          → 10 000 samples  (agentless split)
  3. nvidia/Nemotron-Cascade-RL-SWE      →  5 000 samples  (filtered ≤ ~32K tokens)

Output JSONL fields per item:
  problem, prompt, answer, original_solution, source, index, metadata

Usage:
  python prepare_code_distill_dataset.py [--output PATH] [--seed N]

Environment variables:
  HF_HOME           HuggingFace cache root  (default: ~/.cache/huggingface)
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


# ─── Source 1 ─────────────────────────────────────────────────────────────────

def sample_opencodereasoning(n: int = 5000, seed: int = 42) -> List[Dict]:
    import glob
    import pandas as pd  # type: ignore

    parquet_root = os.path.join(
        os.environ["HF_HOME"],
        "hub", "datasets--nvidia--OpenCodeReasoning",
    )
    # Find the most recent snapshot automatically
    snapshots = sorted(Path(parquet_root, "snapshots").iterdir()) if Path(parquet_root, "snapshots").exists() else []
    if not snapshots:
        raise FileNotFoundError(
            f"nvidia/OpenCodeReasoning not found in HF cache ({parquet_root}). "
            "Run: python -c \"from datasets import load_dataset; load_dataset('nvidia/OpenCodeReasoning')\""
        )
    parquet_dir = str(snapshots[-1])
    files = sorted(glob.glob(f"{parquet_dir}/**/*.parquet", recursive=True))
    print(f"\n[1/3] nvidia/OpenCodeReasoning  (n={n})  — {len(files)} parquet files")

    # Load chunks until we have ~3× the required rows to allow random sampling
    chunks = []
    for f in files:
        chunks.append(pd.read_parquet(f))
        if sum(len(c) for c in chunks) >= n * 3:
            break

    df = pd.concat(chunks, ignore_index=True)
    print(f"      loaded {len(df):,} rows  →  sampling {n}")
    sample = df.sample(n=n, random_state=seed)

    results = []
    for _, row in sample.iterrows():
        problem = row["input"]
        results.append({
            "problem": problem,
            "prompt":  problem,
            "answer":  row.get("solution") or "",
            "original_solution": row.get("output") or "",
            "source":  "opencodereasoning",
            "metadata": {
                "id":          row.get("id") or "",
                "orig_source": row.get("source") or "",
                "dataset":     row.get("dataset") or "",
                "difficulty":  row.get("difficulty") or "",
                "license":     row.get("license") or "",
            },
        })
    print(f"      → {len(results)} items")
    return results


# ─── Source 2 ─────────────────────────────────────────────────────────────────

def sample_nemotron_sft_swe(n: int = 10000, seed: int = 42) -> List[Dict]:
    from datasets import load_dataset  # type: ignore

    # Schema is inconsistent across shards; streaming avoids cast errors
    print(f"\n[2/3] nvidia/Nemotron-SFT-SWE-v2 agentless  (n={n})  [streaming]")
    ds = load_dataset("nvidia/Nemotron-SFT-SWE-v2", split="agentless", streaming=True)

    rng = random.Random(seed)
    reservoir: list = []
    valid_count = 0
    for item in ds:
        msgs = item.get("messages") or []
        if len(msgs) < 2:
            continue
        if valid_count < n:
            reservoir.append(item)
        else:
            j = rng.randint(0, valid_count)
            if j < n:
                reservoir[j] = item
        valid_count += 1

    print(f"      → {len(reservoir)} items")
    results = []
    for item in reservoir:
        msgs = item["messages"]
        problem = msgs[0]["content"]
        answer  = msgs[1]["content"] if len(msgs) > 1 else ""
        results.append({
            "problem": problem,
            "prompt":  problem,
            "answer":  answer,
            "original_solution": answer,
            "source":  "nemotron_sft_swe_agentless",
            "metadata": {
                "uuid":    item.get("uuid", ""),
                "license": item.get("license", ""),
                "used_in": item.get("used_in", ""),
            },
        })
    return results


# ─── Source 3 ─────────────────────────────────────────────────────────────────

def sample_nemotron_cascade_rl_swe(
    n: int = 5000,
    seed: int = 42,
    max_chars: int = 128_000,   # ~32K tokens at 4 chars/token
) -> List[Dict]:
    from datasets import load_dataset  # type: ignore

    print(f"\n[3/3] nvidia/Nemotron-Cascade-RL-SWE  (n={n}, max_chars={max_chars:,})")
    ds = load_dataset("nvidia/Nemotron-Cascade-RL-SWE", split="train")
    print(f"      total rows: {len(ds):,}")

    keep = [i for i, item in enumerate(ds) if len(item.get("prompt", "")) <= max_chars]
    print(f"      after filter: {len(keep):,}  →  sampling {n}")

    rng = random.Random(seed)
    indices = sorted(rng.sample(keep, min(n, len(keep))))
    samples = ds.select(indices)

    results = []
    for item in samples:
        problem = item["prompt"]
        patch   = item.get("golden_patch", "")
        results.append({
            "problem": problem,
            "prompt":  problem,
            "answer":  patch,
            "original_solution": patch,
            "source":  "nemotron_cascade_rl_swe",
            "metadata": {
                "instance_id":            item.get("instance_id", ""),
                "orig_source":            item.get("source", ""),
                "relevant_file_contents": item.get("relevant_file_contents", []),
            },
        })
    print(f"      → {len(results)} items")
    return results


# ─── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build code/SWE distillation dataset (v1, 20K)")
    parser.add_argument("--output",           default="code_distill_ready.jsonl")
    parser.add_argument("--n_opencode",    type=int, default=5000)
    parser.add_argument("--n_swe_sft",     type=int, default=10000)
    parser.add_argument("--n_swe_cascade", type=int, default=5000)
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--max_chars_filter", type=int, default=128_000)
    args = parser.parse_args()

    opencode = sample_opencodereasoning(args.n_opencode, args.seed)
    swe_sft  = sample_nemotron_sft_swe(args.n_swe_sft,  args.seed)
    swe_casc = sample_nemotron_cascade_rl_swe(args.n_swe_cascade, args.seed, args.max_chars_filter)

    all_items = opencode + swe_sft + swe_casc
    for i, item in enumerate(all_items):
        item["index"] = i

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for item in all_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print(f"Output: {out}  ({len(all_items)} items)")
    print(f"  opencodereasoning          : {len(opencode)}")
    print(f"  nemotron_sft_swe_agentless : {len(swe_sft)}")
    print(f"  nemotron_cascade_rl_swe    : {len(swe_casc)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
