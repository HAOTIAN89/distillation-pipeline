#!/usr/bin/env bash
# batch_launch_distill.sh — Submit one SLURM job per teacher model.
#
# Usage:
#   ./batch_launch_distill.sh
#
# Edit the MODEL_* arrays and path variables below to match your setup.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_TEMPLATE="${SCRIPT_DIR}/slurm_distill_kuma.sh"

# ─── Paths — adjust these ────────────────────────────────────────────────────
MODEL_BASE="${SCRATCH}/huggingface_models"
DATA_PATH="${SCRATCH}/datasets/code_distill_ready.jsonl"
OUTPUT_BASE="${SCRATCH}/distillation_outputs"

# ─── Teacher models to run ────────────────────────────────────────────────────
# Keys are short aliases used for job names and output directories.
# Format: MODEL_PATHS[alias]  MODEL_NAMES[alias]  GPU config below
declare -A MODEL_PATHS=(
    [nemotron_cascade_14b]="${MODEL_BASE}/nvidia_Nemotron-Cascade-14B-Thinking"
    [openreasoning_14b]="${MODEL_BASE}/nvidia_OpenReasoning-Nemotron-14B"
    [nemotron_terminal_32b]="${MODEL_BASE}/nvidia_Nemotron-Terminal-32B"
    [qwen3_30b]="${MODEL_BASE}/Qwen_Qwen3-30B-A3B-Thinking-2507"
)

declare -A MODEL_SERVED_NAMES=(
    [nemotron_cascade_14b]="Nemotron-Cascade-14B-Thinking"
    [openreasoning_14b]="OpenReasoning-Nemotron-14B"
    [nemotron_terminal_32b]="Nemotron-Terminal-32B"
    [qwen3_30b]="Qwen3-30B-A3B-Thinking-2507"
)

declare -A MODEL_RECORD_NAMES=(
    [nemotron_cascade_14b]="Nemotron-Cascade-14B-Thinking"
    [openreasoning_14b]="OpenReasoning-Nemotron-14B"
    [nemotron_terminal_32b]="Nemotron-Terminal-32B"
    [qwen3_30b]="Qwen3-30B-A3B-Thinking-2507"
)

# GPU layout — 14B models on 2 GPUs (TP=2); 30B/32B on 4 GPUs (TP=4)
declare -A MODEL_GPU_DEVICES=(
    [nemotron_cascade_14b]="0,1"
    [openreasoning_14b]="0,1"
    [nemotron_terminal_32b]="0,1,2,3"
    [qwen3_30b]="0,1,2,3"
)

declare -A MODEL_TENSOR_PARALLEL=(
    [nemotron_cascade_14b]=2
    [openreasoning_14b]=2
    [nemotron_terminal_32b]=4
    [qwen3_30b]=4
)

declare -A MODEL_NUM_THREADS=(
    [nemotron_cascade_14b]=48
    [openreasoning_14b]=48
    [nemotron_terminal_32b]=24
    [qwen3_30b]=24
)

# ─── Generation settings (shared across all models) ──────────────────────────
N_SAMPLES=8
MAX_TOKENS=16384
ENABLE_THINKING="true"

# ─── Which models to launch (comment out to skip) ────────────────────────────
MODELS_TO_RUN=(
    nemotron_cascade_14b
    openreasoning_14b
    nemotron_terminal_32b
    qwen3_30b
)

# ─── Sanity checks ────────────────────────────────────────────────────────────
if [ ! -f "$SLURM_TEMPLATE" ]; then
    echo "ERROR: SLURM template not found: ${SLURM_TEMPLATE}"; exit 1
fi
if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: data file not found: ${DATA_PATH}"; exit 1
fi

# ─── Launch ──────────────────────────────────────────────────────────────────
total=0
for alias in "${MODELS_TO_RUN[@]}"; do
    job_name="distill-${alias}"
    save_path="${OUTPUT_BASE}/${MODEL_RECORD_NAMES[$alias]}/distill_output.jsonl"

    total=$((total + 1))
    echo "[$(printf '%02d' $total)] ${job_name}"
    echo "      model : ${MODEL_RECORD_NAMES[$alias]}"
    echo "      gpus  : ${MODEL_GPU_DEVICES[$alias]}  (TP=${MODEL_TENSOR_PARALLEL[$alias]})"
    echo "      output: ${save_path}"

    export DISTILL_MODEL_PATH="${MODEL_PATHS[$alias]}"
    export DISTILL_SERVED_MODEL_NAME="${MODEL_SERVED_NAMES[$alias]}"
    export DISTILL_MODEL_NAME="${MODEL_RECORD_NAMES[$alias]}"
    export DISTILL_DATA_PATH="${DATA_PATH}"
    export DISTILL_SAVE_PATH="${save_path}"
    export DISTILL_NUM_THREADS="${MODEL_NUM_THREADS[$alias]}"
    export DISTILL_GPU_DEVICES="${MODEL_GPU_DEVICES[$alias]}"
    export DISTILL_TENSOR_PARALLEL="${MODEL_TENSOR_PARALLEL[$alias]}"
    export DISTILL_N_SAMPLES="${N_SAMPLES}"
    export DISTILL_MAX_TOKENS="${MAX_TOKENS}"
    export DISTILL_ENABLE_THINKING="${ENABLE_THINKING}"

    # --gpus-per-task overrides the #SBATCH default in the template
    GPU_COUNT=$(echo "${MODEL_GPU_DEVICES[$alias]}" | tr ',' '\n' | wc -l)

    sbatch \
        --job-name="${job_name}" \
        --gpus-per-task="${GPU_COUNT}" \
        --export=ALL \
        "${SLURM_TEMPLATE}"

    echo ""
done

echo "Submitted ${total} job(s)."
echo "Monitor with: squeue -u \$USER"
