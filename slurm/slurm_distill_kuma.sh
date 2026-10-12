#!/bin/bash -l
# SLURM job template for running the distillation pipeline on EPFL Kuma.
#
# Required environment variables (set by batch_launch_distill.sh or manually):
#   DISTILL_MODEL_PATH        absolute path to the model weights
#   DISTILL_SERVED_MODEL_NAME model name used in API requests
#   DISTILL_MODEL_NAME        model name written into output records
#   DISTILL_DATA_PATH         absolute path to input JSONL
#   DISTILL_SAVE_PATH         absolute path to output JSONL
#   DISTILL_NUM_THREADS       concurrent worker threads
#   DISTILL_GPU_DEVICES       comma-separated GPU indices, e.g. "0,1"
#   DISTILL_TENSOR_PARALLEL   tensor-parallel size (must match GPU count)
#   DISTILL_MAX_TOKENS        max tokens per generated response
#   DISTILL_ENABLE_THINKING   true / false
#   DISTILL_N_SAMPLES         samples per problem (default 8)

# ─── Resource request ─────────────────────────────────────────────────────────
# Adjust --gpus-per-task and --time to match your model size:
#   14B models  → gpus-per-task 2, time ~48h
#   30B/32B     → gpus-per-task 4, time ~72h
#SBATCH --account nlp
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 32
#SBATCH --mem 128G
#SBATCH --time 72:00:00
#SBATCH --gpus-per-task 4
#SBATCH --partition h100
#SBATCH --output /home/%u/slurm_logs/distill-%x-%j.out
#SBATCH --error  /home/%u/slurm_logs/distill-%x-%j.err

set -euo pipefail

# ─── Environment ─────────────────────────────────────────────────────────────
module load gcc cuda

source ~/miniconda3/etc/profile.d/conda.sh
conda activate <YOUR_CONDA_ENV>      # e.g. conda activate distill

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$HOME/miniconda3/lib"

# HuggingFace — token is read from the environment; set it in your ~/.bashrc or
# pass it via: sbatch --export=ALL,HF_TOKEN=<your_token> ...
# Do NOT hard-code tokens in this file.
if [ -z "${HF_TOKEN:-}" ]; then
    echo "WARNING: HF_TOKEN is not set. Private models will not be accessible."
fi

export HF_HOME="${SCRATCH}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"

export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# ─── Project directory ────────────────────────────────────────────────────────
# Adjust this to point at your local clone of the distillation-pipeline repo.
PROJECT_DIR="${HOME}/projects/distillation-pipeline"

# ─── Defaults for optional variables ─────────────────────────────────────────
: "${DISTILL_N_SAMPLES:=8}"
: "${DISTILL_ENABLE_THINKING:=true}"
: "${DISTILL_TEMPERATURE:=1.0}"
: "${DISTILL_TOP_P:=0.95}"
: "${DISTILL_MAX_MODEL_LEN:=32768}"
: "${DISTILL_GPU_MEMORY_UTIL:=0.95}"

mkdir -p "${HOME}/slurm_logs"

echo "====================================================="
echo "  Distillation job on Kuma"
echo "  Model    : ${DISTILL_MODEL_NAME:-?}"
echo "  Data     : ${DISTILL_DATA_PATH:-?}"
echo "  Output   : ${DISTILL_SAVE_PATH:-?}"
echo "  GPUs     : ${DISTILL_GPU_DEVICES:-?}  (TP=${DISTILL_TENSOR_PARALLEL:-?})"
echo "  Threads  : ${DISTILL_NUM_THREADS:-?}"
echo "  Samples  : ${DISTILL_N_SAMPLES}"
echo "  Thinking : ${DISTILL_ENABLE_THINKING}"
echo "====================================================="

bash "${PROJECT_DIR}/run_distill.sh" \
    --model_path        "${DISTILL_MODEL_PATH}" \
    --served_model_name "${DISTILL_SERVED_MODEL_NAME}" \
    --model_name        "${DISTILL_MODEL_NAME}" \
    --data_path         "${DISTILL_DATA_PATH}" \
    --save_path         "${DISTILL_SAVE_PATH}" \
    --n_samples         "${DISTILL_N_SAMPLES}" \
    --temperature       "${DISTILL_TEMPERATURE}" \
    --top_p             "${DISTILL_TOP_P}" \
    --max_tokens        "${DISTILL_MAX_TOKENS}" \
    --enable_thinking   "${DISTILL_ENABLE_THINKING}" \
    --num_threads       "${DISTILL_NUM_THREADS}" \
    --gpu_devices       "${DISTILL_GPU_DEVICES}" \
    --tensor_parallel   "${DISTILL_TENSOR_PARALLEL}" \
    --max_model_len     "${DISTILL_MAX_MODEL_LEN}" \
    --retry_until_complete
