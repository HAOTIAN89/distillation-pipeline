#!/usr/bin/env python3
"""
Prepare the code/SWE distillation dataset (v2, 10K samples, all prompts < 8K tokens).

Sources:
  1. nvidia/OpenCodeReasoning     (cached parquet)      → 4 000 samples
  2. nvidia/Nemotron-SFT-SWE-v2  (agentless, streaming) → 1 000 samples
  3. nvidia/Nemotron-Cascade-RL-SWE                      → 1 000 samples
  4. nvidia/OpenCodeInstruct      (streaming)            → 4 000 samples

All output items include `prompt_tokens` in metadata for verification.

Usage:
  python prepare_code_distill_dataset_v2.py \
      --tokenizer_path /path/to/tokenizer \
      [--output PATH] [--seed N]

Environment variables:
  HF_HOME   HuggingFace cache root  (default: ~/.cache/huggingface)
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

MAX_TOKENS = 8000
CASCADE_MAX_TOKENS = 12000  # SWE-cascade repo context prevents <8K; 12K+16K output < 32K window
CHAR_GUARD = 36_000         # skip tokenising if char length already too long (~4.5× safety)


def _load_tokenizer(tokenizer_path: str):
    from transformers import AutoTokenizer  # type: ignore
    print(f"Loading tokenizer from {tokenizer_path} ...")
    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    print("  done.\n")
    return tok


def _token_len(tok, text: str) -> int:
    return len(tok.encode(text, add_special_tokens=False))


# ─── Source 1 ─────────────────────────────────────────────────────────────────

def sample_opencodereasoning(tok, n: int = 4000, seed: int = 42) -> List[Dict]:
    import glob
    import pandas as pd  # type: ignore

    parquet_root = os.path.join(os.environ["HF_HOME"], "hub", "datasets--nvidia--OpenCodeReasoning")
    snapshots_dir = Path(parquet_root, "snapshots")
    if not snapshots_dir.exists():
        raise FileNotFoundError(
            f"nvidia/OpenCodeReasoning not found in HF cache ({parquet_root}). "
            "Download it first: python -c \"from datasets import load_dataset; "
            "load_dataset('nvidia/OpenCodeReasoning')\""
        )
    parquet_dir = str(sorted(snapshots_dir.iterdir())[-1])
    files = sorted(glob.glob(f"{parquet_dir}/**/*.parquet", recursive=True))
    print(f"[1/4] nvidia/OpenCodeReasoning  (target={n}, max_tokens={MAX_TOKENS})")
    print(f"      {len(files)} parquet files")

    chunks = []
    for f in files:
        chunks.append(pd.read_parquet(f))
        if sum(len(c) for c in chunks) >= n * 4:
            break
    df = pd.concat(chunks, ignore_index=True)
    print(f"      loaded {len(df):,} rows")

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    results = []
    for _, row in df.iterrows():
        if len(results) >= n:
            break
        prompt = str(row.get("input") or "")
        if not prompt or len(prompt) > CHAR_GUARD:
            continue
        n_tok = _token_len(tok, prompt)
        if n_tok > MAX_TOKENS:
            continue
        results.append({
            "problem": prompt,
            "prompt":  prompt,
            "answer":  str(row.get("solution") or ""),
            "original_solution": str(row.get("output") or ""),
            "source":  "opencodereasoning",
            "metadata": {
                "id":            str(row.get("id") or ""),
                "orig_source":   str(row.get("source") or ""),
                "dataset":       str(row.get("dataset") or ""),
                "difficulty":    str(row.get("difficulty") or ""),
                "license":       str(row.get("license") or ""),
                "prompt_tokens": n_tok,
            },
        })
    print(f"      → {len(results)} items\n")
    return results


# ─── Source 2 ─────────────────────────────────────────────────────────────────

def sample_nemotron_sft_swe(tok, n: int = 1000, seed: int = 42) -> List[Dict]:
    from datasets import load_dataset  # type: ignore

    print(f"[2/4] nvidia/Nemotron-SFT-SWE-v2 agentless  (target={n}, max_tokens={MAX_TOKENS})")
    ds = load_dataset("nvidia/Nemotron-SFT-SWE-v2", split="agentless", streaming=True)

    rng = random.Random(seed)
    reservoir: List[Tuple] = []
    valid_count = 0

    for item in ds:
        msgs = item.get("messages") or []
        if len(msgs) < 2:
            continue
        prompt = msgs[0].get("content", "")
        if not prompt or len(prompt) > CHAR_GUARD:
            continue
        n_tok = _token_len(tok, prompt)
        if n_tok > MAX_TOKENS:
            continue
        if valid_count < n:
            reservoir.append((item, n_tok))
        else:
            j = rng.randint(0, valid_count)
            if j < n:
                reservoir[j] = (item, n_tok)
        valid_count += 1

    print(f"      scanned {valid_count} valid items, sampled {len(reservoir)}")

    results = []
    for item, n_tok in reservoir:
        msgs = item["messages"]
        prompt = msgs[0]["content"]
        answer = msgs[1]["content"] if len(msgs) > 1 else ""
        results.append({
            "problem": prompt,
            "prompt":  prompt,
            "answer":  answer,
            "original_solution": answer,
            "source":  "nemotron_sft_swe_agentless",
            "metadata": {
                "uuid":          item.get("uuid", ""),
                "license":       item.get("license", ""),
                "used_in":       item.get("used_in", ""),
                "prompt_tokens": n_tok,
            },
        })
    print(f"      → {len(results)} items\n")
    return results


# ─── Source 3 ─────────────────────────────────────────────────────────────────

def sample_nemotron_cascade_rl_swe(tok, n: int = 1000, seed: int = 42) -> List[Dict]:
    from datasets import load_dataset  # type: ignore

    print(f"[3/4] nvidia/Nemotron-Cascade-RL-SWE  (target={n}, max_tokens={CASCADE_MAX_TOKENS})")
    ds = load_dataset("nvidia/Nemotron-Cascade-RL-SWE", split="train")
    print(f"      total={len(ds):,}")

    cascade_char_guard = CASCADE_MAX_TOKENS * 5

    candidates = [i for i, item in enumerate(ds)
                  if item.get("prompt") and len(item["prompt"]) <= cascade_char_guard]
    print(f"      char-filtered candidates: {len(candidates):,}")

    rng = random.Random(seed)
    rng.shuffle(candidates)

    valid: List[Tuple] = []
    for idx in candidates:
        item = ds[idx]
        n_tok = _token_len(tok, item["prompt"])
        if n_tok <= CASCADE_MAX_TOKENS:
            valid.append((idx, n_tok))
        if len(valid) >= n * 3:
            break

    print(f"      token-filtered valid: {len(valid):,}  → sampling {n}")
    selected = rng.sample(valid, min(n, len(valid)))

    results = []
    for idx, n_tok in selected:
        item = ds[idx]
        prompt = item["prompt"]
        patch  = item.get("golden_patch", "")
        results.append({
            "problem": prompt,
            "prompt":  prompt,
            "answer":  patch,
            "original_solution": patch,
            "source":  "nemotron_cascade_rl_swe",
            "metadata": {
                "instance_id":   item.get("instance_id", ""),
                "orig_source":   item.get("source", ""),
                "prompt_tokens": n_tok,
            },
        })
    print(f"      → {len(results)} items\n")
    return results


# ─── Source 4 ─────────────────────────────────────────────────────────────────

def sample_opencodeinstruct(tok, n: int = 4000, seed: int = 42) -> List[Dict]:
    from datasets import load_dataset  # type: ignore

    print(f"[4/4] nvidia/OpenCodeInstruct  (target={n}, max_tokens={MAX_TOKENS}, streaming)")
    ds = load_dataset("nvidia/OpenCodeInstruct", split="train", streaming=True)

    first = next(iter(ds))
    print(f"      fields: {list(first.keys())}")
    ds = load_dataset("nvidia/OpenCodeInstruct", split="train", streaming=True)

    def get_prompt(item):
        return (item.get("instruction") or item.get("prompt") or
                item.get("question") or item.get("input") or
                item.get("problem") or "")

    def get_answer(item):
        return (item.get("output") or item.get("response") or
                item.get("solution") or item.get("answer") or "")

    STRIDE = 5
    rng = random.Random(seed)
    reservoir: List[Tuple] = []
    valid_count = 0
    scanned = 0

    for item in ds:
        scanned += 1
        if scanned % STRIDE != 0:
            continue
        prompt = get_prompt(item)
        if not prompt or len(prompt) > CHAR_GUARD:
            continue
        n_tok = _token_len(tok, prompt)
        if n_tok > MAX_TOKENS:
            continue
        if valid_count < n:
            reservoir.append((item, n_tok))
        else:
            j = rng.randint(0, valid_count)
            if j < n:
                reservoir[j] = (item, n_tok)
        valid_count += 1
        if valid_count >= n * 3:
            break

    print(f"      scanned {scanned}, {valid_count} valid, sampled {len(reservoir)}")

    results = []
    for item, n_tok in reservoir[:n]:
        prompt = get_prompt(item)
        answer = get_answer(item)
        results.append({
            "problem": prompt,
            "prompt":  prompt,
            "answer":  answer,
            "original_solution": answer,
            "source":  "opencodeinstruct",
            "metadata": {
                "id":            str(item.get("id") or item.get("idx") or ""),
                "language":      item.get("language", ""),
                "difficulty":    item.get("difficulty", ""),
                "prompt_tokens": n_tok,
            },
        })
    print(f"      → {len(results)} items\n")
    return results


# ─── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build code/SWE distillation dataset (v2, 10K, prompt < 8K tokens)"
    )
    parser.add_argument("--output",               default="code_distill_ready_v2.jsonl")
    parser.add_argument("--tokenizer_path",        required=True,
                        help="Path to tokenizer for prompt-token counting")
    parser.add_argument("--n_opencode",    type=int, default=4000)
    parser.add_argument("--n_swe_sft",     type=int, default=1000)
    parser.add_argument("--n_swe_cascade", type=int, default=1000)
    parser.add_argument("--n_opencodeinstruct", type=int, default=4000)
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()

    tok = _load_tokenizer(args.tokenizer_path)

    opencode = sample_opencodereasoning(tok,       args.n_opencode,         args.seed)
    swe_sft  = sample_nemotron_sft_swe(tok,        args.n_swe_sft,          args.seed)
    swe_casc = sample_nemotron_cascade_rl_swe(tok, args.n_swe_cascade,      args.seed)
    oci      = sample_opencodeinstruct(tok,         args.n_opencodeinstruct, args.seed)

    all_items = opencode + swe_sft + swe_casc + oci
    random.Random(args.seed).shuffle(all_items)
    for i, item in enumerate(all_items):
        item["index"] = i

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for item in all_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    over_8k  = sum(1 for it in all_items if it["source"] != "nemotron_cascade_rl_swe"
                   and it["metadata"].get("prompt_tokens", 0) > MAX_TOKENS)
    over_12k = sum(1 for it in all_items if it["source"] == "nemotron_cascade_rl_swe"
                   and it["metadata"].get("prompt_tokens", 0) > CASCADE_MAX_TOKENS)

    print(f"\n{'='*60}")
    print(f"Output: {out}  ({len(all_items)} items)")
    print(f"  opencodereasoning          : {len(opencode)}  (limit={MAX_TOKENS})")
    print(f"  nemotron_sft_swe_agentless : {len(swe_sft)}  (limit={MAX_TOKENS})")
    print(f"  nemotron_cascade_rl_swe    : {len(swe_casc)}  (limit={CASCADE_MAX_TOKENS})")
    print(f"  opencodeinstruct           : {len(oci)}  (limit={MAX_TOKENS})")
    print(f"  over-budget (non-cascade)  : {over_8k}  ({'OK' if over_8k == 0 else 'FAIL'})")
    print(f"  over-budget (cascade)      : {over_12k}  ({'OK' if over_12k == 0 else 'FAIL'})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
