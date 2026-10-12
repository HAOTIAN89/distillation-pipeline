# Distillation Pipeline

A concurrent, resumable knowledge-distillation pipeline for code and math reasoning tasks.
Given a JSONL dataset of problems, it spins up a local [vLLM](https://github.com/vllm-project/vllm)
server (or calls a remote OpenAI-compatible API) and collects *N* generated solutions per problem
using a teacher model.

---

## Features

- **Concurrent workers** — configurable thread pool for high-throughput generation
- **Resume from checkpoint** — incremental JSONL writes; re-run after interruption without re-doing completed items
- **Auto-retry** — optional `--retry_until_complete` flag to keep retrying until every problem has exactly *N* samples
- **Thinking mode** — uses `apply_chat_template(enable_thinking=...)` for Qwen3 / Nemotron reasoning models
- **Dual mode** — local vLLM server *or* any remote OpenAI-compatible API (e.g. cloud endpoints)
- **Code + Math** — separate dataset builders for code/SWE and math reasoning tasks

---

## Repository Layout

```
distillation-pipeline/
├── distill_solutions.py              # Core distillation engine
├── run_distill.sh                    # Orchestration script (starts vLLM + runs distillation)
├── requirements.txt
│
├── prepare_dataset/
│   ├── prepare_code_distill_dataset.py    # v1 — 20K samples (3 sources)
│   └── prepare_code_distill_dataset_v2.py # v2 — 10K samples, all prompts < 8K tokens
│
└── slurm/
    ├── slurm_distill_kuma.sh         # SLURM job template for EPFL Kuma
    └── batch_launch_distill.sh       # Submit one job per teacher model
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
# vLLM requires a CUDA-capable GPU; see https://docs.vllm.ai/en/latest/getting_started/installation.html
```

### 2. Prepare the dataset

**v1 (20K samples, code/SWE):**

```bash
python prepare_dataset/prepare_code_distill_dataset.py \
    --output data/code_distill_ready.jsonl
```

**v2 (10K samples, all prompts ≤ 8K tokens):**

```bash
python prepare_dataset/prepare_code_distill_dataset_v2.py \
    --tokenizer_path /path/to/your/tokenizer \
    --output data/code_distill_ready_v2.jsonl
```

Each item in the output JSONL contains:

| Field | Description |
|---|---|
| `problem` | Raw problem text |
| `prompt` | User message passed to the teacher (same as `problem` for code tasks) |
| `answer` | Reference answer / golden patch |
| `original_solution` | Original solution from the source dataset |
| `source` | Dataset source identifier |
| `index` | Global integer index |
| `metadata` | Source-specific metadata (difficulty, license, etc.) |

### 3. Run distillation

**Local vLLM (recommended for on-premise GPU servers):**

```bash
./run_distill.sh \
    --model_path /path/to/nvidia_Nemotron-Cascade-14B-Thinking \
    --served_model_name Nemotron-Cascade-14B-Thinking \
    --model_name Nemotron-Cascade-14B-Thinking \
    --data_path data/code_distill_ready.jsonl \
    --save_path outputs/Nemotron-Cascade-14B-Thinking/distill_output.jsonl \
    --gpu_devices "0,1" \
    --tensor_parallel 2 \
    --num_threads 48 \
    --max_tokens 16384 \
    --enable_thinking true
```

The script starts vLLM in the background, waits until it is ready, then runs `distill_solutions.py`.
vLLM is shut down automatically when distillation finishes.

**Remote API (skip vLLM):**

```bash
./run_distill.sh \
    --api_base https://your-api-endpoint/v1 \
    --api_key YOUR_API_KEY \
    --served_model_name your-model-name \
    --model_name your-model-name \
    --data_path data/code_distill_ready.jsonl \
    --save_path outputs/your-model/distill_output.jsonl \
    --num_threads 20
```

---

## Teacher Model Presets

| Model | Tensor Parallel | GPUs | Threads | Notes |
|---|---|---|---|---|
| `nvidia/Nemotron-Cascade-14B-Thinking` | 2 | `"0,1"` | 48 | |
| `nvidia/OpenReasoning-Nemotron-14B` | 2 | `"0,1"` | 48 | |
| `nvidia/Nemotron-Terminal-32B` | 4 | `"0,1,2,3"` | 24 | |
| `Qwen/Qwen3-30B-A3B-Thinking-2507` | 4 | `"0,1,2,3"` | 24 | |

All models above are used with `enable_thinking=true`, `max_model_len=32768`, and `max_tokens=16384`.

---

## Output Format

Each line of the output JSONL is one generated sample:

```json
{
  "problem": "...",
  "original_solution": "...",
  "answer": "...",
  "source": "opencodereasoning",
  "index": 42,
  "generated_solution": "<think>...</think>\n...",
  "sample": 3,
  "model": "Nemotron-Cascade-14B-Thinking"
}
```

`sample` ranges from `0` to `N_SAMPLES - 1`. Multiple output files from different teacher models
can be concatenated directly for downstream SFT/DPO training.

---

## Running on EPFL Kuma (SLURM)

### Prerequisites

1. Clone this repo to your home directory:

   ```bash
   git clone https://github.com/HAOTIAN89/distillation-pipeline.git ~/projects/distillation-pipeline
   ```

2. Set environment variables in your `~/.bashrc`:

   ```bash
   export SCRATCH=/scratch/<your_username>
   export HF_TOKEN=<your_huggingface_token>    # for gated models
   export HF_HOME="${SCRATCH}/.cache/huggingface"
   ```

3. Edit `slurm/slurm_distill_kuma.sh`:
   - Replace `<YOUR_CONDA_ENV>` with your conda environment name.
   - Adjust `#SBATCH --account` to your allocation.

4. Place (or symlink) your dataset at `${SCRATCH}/datasets/code_distill_ready.jsonl`.

### Submit a single job manually

```bash
export DISTILL_MODEL_PATH="${SCRATCH}/huggingface_models/nvidia_Nemotron-Cascade-14B-Thinking"
export DISTILL_SERVED_MODEL_NAME="Nemotron-Cascade-14B-Thinking"
export DISTILL_MODEL_NAME="Nemotron-Cascade-14B-Thinking"
export DISTILL_DATA_PATH="${SCRATCH}/datasets/code_distill_ready.jsonl"
export DISTILL_SAVE_PATH="${SCRATCH}/distillation_outputs/Nemotron-Cascade-14B-Thinking/distill_output.jsonl"
export DISTILL_NUM_THREADS=48
export DISTILL_GPU_DEVICES="0,1"
export DISTILL_TENSOR_PARALLEL=2
export DISTILL_MAX_TOKENS=16384
export DISTILL_ENABLE_THINKING=true

sbatch --job-name="distill-cascade-14b" --gpus-per-task=2 --export=ALL \
    slurm/slurm_distill_kuma.sh
```

### Submit all teacher models at once

Edit the paths in `slurm/batch_launch_distill.sh`, then:

```bash
# Dry run — print what would be submitted
bash slurm/batch_launch_distill.sh --dry-run   # (not implemented; just read the script first)

# Submit all jobs
bash slurm/batch_launch_distill.sh
```

This submits one job per entry in `MODELS_TO_RUN`. Each job is independent and writes to its own
output directory under `${SCRATCH}/distillation_outputs/`.

### Monitor jobs

```bash
squeue -u $USER
tail -f ~/slurm_logs/distill-<job-name>-<job-id>.out
```

---

## `distill_solutions.py` Reference

```
python distill_solutions.py \
    --api_base   http://0.0.0.0:8001/v1 \
    --model      Nemotron-Cascade-14B-Thinking \
    --data_path  data/code_distill_ready.jsonl \
    --save_path  outputs/distill_output.jsonl \
    [--api_key EMPTY] \
    [--tokenizer_path /path/to/model]  \  # needed for enable_thinking
    [--n_samples 8] \
    [--temperature 1.0] \
    [--top_p 0.95] \
    [--max_tokens 16384] \
    [--num_threads 48] \
    [--enable_thinking true] \
    [--include_sources opencodereasoning,nemotron_sft_swe_agentless] \
    [--retry_until_complete] \
    [--no_resume]
```

---

## License

This pipeline code is released under the MIT License.
The datasets used for building the distillation set are subject to their own licenses
(see the respective HuggingFace dataset cards).
