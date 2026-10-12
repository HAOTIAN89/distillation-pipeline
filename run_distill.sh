#!/bin/bash
# run_distill.sh — Launch a vLLM server and run distillation against it.
#
# Usage:
#   ./run_distill.sh [options]
#
# Two modes:
#   Local vLLM   — leave --api_base unset; the script starts vLLM automatically.
#   Remote API   — pass --api_base <url> [--api_key <key>]; vLLM is skipped.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISTILL_PY="${SCRIPT_DIR}/distill_solutions.py"

# ─── Paths ───────────────────────────────────────────────────────────────────
# SCRATCH and MODEL_BASE_DIR can be overridden by environment variables.
# On EPFL Kuma set SCRATCH=/scratch/<username> in your ~/.bashrc or SLURM script.
: "${SCRATCH:=${HOME}}"
: "${MODEL_BASE_DIR:=${SCRATCH}/huggingface_models}"
: "${DATA_BASE_DIR:=${SCRATCH}/datasets}"
: "${OUTPUT_BASE_DIR:=${SCRATCH}/distillation_outputs}"
: "${LOG_BASE_DIR:=${SCRIPT_DIR}/logs}"

# ─── Model configuration ─────────────────────────────────────────────────────
# Common model presets (tensor-parallel sizes):
#
#   nvidia/Nemotron-Terminal-32B           TP=4  GPUs="0,1,2,3"  threads=24
#   nvidia/OpenReasoning-Nemotron-14B      TP=2  GPUs="0,1"      threads=48
#   nvidia/Nemotron-Cascade-14B-Thinking   TP=2  GPUs="0,1"      threads=48
#   Qwen/Qwen3-30B-A3B-Thinking-2507       TP=4  GPUs="0,1,2,3"  threads=24
MODEL_PATH="${MODEL_BASE_DIR}/nvidia_Nemotron-Cascade-14B-Thinking"
SERVED_MODEL_NAME="Nemotron-Cascade-14B-Thinking"
MODEL_NAME="Nemotron-Cascade-14B-Thinking"

# ─── API ─────────────────────────────────────────────────────────────────────
API_KEY="EMPTY"
API_BASE=""        # leave empty for local vLLM mode

# ─── Data ────────────────────────────────────────────────────────────────────
DATA_PATH="${DATA_BASE_DIR}/code_distill_ready.jsonl"
SAVE_PATH=""
SAVE_PATH_SET=false

# ─── Generation ──────────────────────────────────────────────────────────────
N_SAMPLES=8
TEMPERATURE=1.0
TOP_P=0.95
MAX_TOKENS=16384          # 16K for code/SWE; use 8192 for math
ENABLE_THINKING="true"    # set to "false" to disable thinking mode

# ─── Concurrency ─────────────────────────────────────────────────────────────
NUM_THREADS=48            # 48 for 14B models; 24 for 30B/32B models

# ─── Source filter ───────────────────────────────────────────────────────────
INCLUDE_SOURCES=""        # e.g. "opencodereasoning,nemotron_sft_swe_agentless"

# ─── vLLM ────────────────────────────────────────────────────────────────────
VLLM_PORT=8001
TENSOR_PARALLEL_SIZE=2    # match number of GPUs
GPU_MEMORY_UTIL=0.95
MAX_MODEL_LEN=32768
GPU_DEVICES="0,1"

RETRY_UNTIL_COMPLETE=true

# ─── Help ─────────────────────────────────────────────────────────────────────
show_help() {
    cat << 'EOF'
Usage: run_distill.sh [options]

Options:
  --model_path PATH          Path to the model (default: $MODEL_BASE_DIR/nvidia_Nemotron-Cascade-14B-Thinking)
  --served_model_name NAME   Name used in API requests (default: Nemotron-Cascade-14B-Thinking)
  --model_name NAME          Name written into output records
  --data_path PATH           Input JSONL file
  --save_path PATH           Output JSONL file
  --api_key KEY              API key (default: EMPTY for local vLLM)
  --api_base URL             Remote API base URL; set to skip vLLM startup
  --n_samples N              Samples per problem (default: 8)
  --temperature T            Sampling temperature (default: 1.0)
  --top_p P                  Top-p nucleus sampling (default: 0.95)
  --max_tokens N             Max tokens per response (default: 16384)
  --enable_thinking BOOL     Thinking toggle — true/false (default: true)
  --num_threads N            Concurrent worker threads (default: 48)
  --port PORT                vLLM port (default: 8001)
  --gpu_devices DEVICES      GPU list, e.g. "0,1" (default: 0,1)
  --tensor_parallel SIZE     Tensor-parallel size (default: 2)
  --max_model_len N          vLLM context window (default: 32768)
  --include_sources LIST     Comma-separated source filter (default: all)
  --retry_until_complete     Keep retrying until every problem has N samples
  --skip_vllm                Skip vLLM startup (use a running server)
  --no_resume                Restart from scratch (overwrites existing output)
  -h, --help                 Show this help

Examples:
  # Nemotron-Cascade-14B-Thinking on 2 GPUs (default)
  ./run_distill.sh

  # OpenReasoning-Nemotron-14B on 2 GPUs
  ./run_distill.sh \
    --model_path ${MODEL_BASE_DIR}/nvidia_OpenReasoning-Nemotron-14B \
    --served_model_name OpenReasoning-Nemotron-14B \
    --model_name OpenReasoning-Nemotron-14B \
    --gpu_devices "0,1" --tensor_parallel 2 --num_threads 48

  # Nemotron-Terminal-32B on 4 GPUs
  ./run_distill.sh \
    --model_path ${MODEL_BASE_DIR}/nvidia_Nemotron-Terminal-32B \
    --served_model_name Nemotron-Terminal-32B \
    --model_name Nemotron-Terminal-32B \
    --gpu_devices "0,1,2,3" --tensor_parallel 4 --num_threads 24

  # Qwen3-30B-A3B on 4 GPUs
  ./run_distill.sh \
    --model_path ${MODEL_BASE_DIR}/Qwen_Qwen3-30B-A3B-Thinking-2507 \
    --served_model_name Qwen3-30B-A3B-Thinking-2507 \
    --model_name Qwen3-30B-A3B-Thinking-2507 \
    --gpu_devices "0,1,2,3" --tensor_parallel 4 --num_threads 24

  # Remote OpenAI-compatible API (skip local vLLM)
  ./run_distill.sh \
    --api_base https://your-api-endpoint/v1 \
    --api_key YOUR_API_KEY \
    --served_model_name your-model-name

  # Resume is on by default; use --no_resume to restart
  ./run_distill.sh --no_resume
EOF
    exit 0
}

# ─── Argument parsing ─────────────────────────────────────────────────────────
SKIP_VLLM=false
NO_RESUME=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)          MODEL_PATH="$2";          shift 2 ;;
        --served_model_name)   SERVED_MODEL_NAME="$2";   shift 2 ;;
        --model_name)          MODEL_NAME="$2";          shift 2 ;;
        --data_path)           DATA_PATH="$2";           shift 2 ;;
        --save_path)           SAVE_PATH="$2"; SAVE_PATH_SET=true; shift 2 ;;
        --api_key)             API_KEY="$2";             shift 2 ;;
        --api_base)            API_BASE="$2";            shift 2 ;;
        --n_samples)           N_SAMPLES="$2";           shift 2 ;;
        --temperature)         TEMPERATURE="$2";         shift 2 ;;
        --top_p)               TOP_P="$2";               shift 2 ;;
        --max_tokens)          MAX_TOKENS="$2";          shift 2 ;;
        --enable_thinking)     ENABLE_THINKING="$2";     shift 2 ;;
        --num_threads)         NUM_THREADS="$2";         shift 2 ;;
        --port)                VLLM_PORT="$2";           shift 2 ;;
        --gpu_devices)         GPU_DEVICES="$2";         shift 2 ;;
        --tensor_parallel)     TENSOR_PARALLEL_SIZE="$2"; shift 2 ;;
        --max_model_len)       MAX_MODEL_LEN="$2";       shift 2 ;;
        --include_sources)     INCLUDE_SOURCES="$2";     shift 2 ;;
        --retry_until_complete) RETRY_UNTIL_COMPLETE=true; shift ;;
        --skip_vllm)           SKIP_VLLM=true;           shift ;;
        --no_resume)           NO_RESUME=true;           shift ;;
        -h|--help)             show_help ;;
        *) echo "Unknown option: $1  (use -h for help)"; exit 1 ;;
    esac
done

# ─── Derived paths ────────────────────────────────────────────────────────────
if [ "$SAVE_PATH_SET" = false ]; then
    SAVE_PATH="${OUTPUT_BASE_DIR}/${MODEL_NAME}/distill_output.jsonl"
fi

LOG_DIR="${LOG_BASE_DIR}/${MODEL_NAME}"
VLLM_LOG="${LOG_DIR}/vllm_server.log"
DISTILL_LOG="${LOG_DIR}/distill.log"

# ─── Print configuration ──────────────────────────────────────────────────────
printf '=%.0s' {1..70}; echo
echo "  Distillation Pipeline"
printf '=%.0s' {1..70}; echo
echo ""
echo "  model_path       : ${MODEL_PATH}"
echo "  served_model     : ${SERVED_MODEL_NAME}"
echo "  recorded_model   : ${MODEL_NAME}"
echo "  data_path        : ${DATA_PATH}"
echo "  save_path        : ${SAVE_PATH}"
echo "  api_mode         : $([ -n "$API_BASE" ] && echo "remote (${API_BASE})" || echo "local vLLM")"
echo "  n_samples        : ${N_SAMPLES}"
echo "  temperature      : ${TEMPERATURE}"
echo "  top_p            : ${TOP_P}"
echo "  max_tokens       : ${MAX_TOKENS}"
echo "  enable_thinking  : ${ENABLE_THINKING:-not set}"
echo "  num_threads      : ${NUM_THREADS}"
echo "  gpu_devices      : ${GPU_DEVICES}"
echo "  tensor_parallel  : ${TENSOR_PARALLEL_SIZE}"
echo "  max_model_len    : ${MAX_MODEL_LEN}"
echo "  vllm_port        : ${VLLM_PORT}"
echo "  include_sources  : ${INCLUDE_SOURCES:-all}"
echo "  retry_until_done : ${RETRY_UNTIL_COMPLETE}"
echo "  resume           : $([ "$NO_RESUME" = true ] && echo 'no' || echo 'yes')"
echo ""
printf '=%.0s' {1..70}; echo
echo ""

# ─── Validate inputs ─────────────────────────────────────────────────────────
if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: input data file not found: ${DATA_PATH}"; exit 1
fi
if [ ! -f "$DISTILL_PY" ]; then
    echo "ERROR: distill_solutions.py not found at ${DISTILL_PY}"; exit 1
fi

mkdir -p "$LOG_DIR"
mkdir -p "$(dirname "$SAVE_PATH")"

# ─── Step 1: Start vLLM ──────────────────────────────────────────────────────
VLLM_PID=""

if [ -n "$API_BASE" ]; then
    echo "Remote API base set — skipping vLLM startup."
    SKIP_VLLM=true
elif [ "$SKIP_VLLM" = false ]; then
    echo "Starting vLLM server (log: ${VLLM_LOG}) ..."
    CUDA_VISIBLE_DEVICES=${GPU_DEVICES} \
    vllm serve "${MODEL_PATH}" \
        --served-model-name "$SERVED_MODEL_NAME" \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
        --max-model-len "$MAX_MODEL_LEN" \
        --port "$VLLM_PORT" > "$VLLM_LOG" 2>&1 &
    VLLM_PID=$!
    echo "vLLM PID: $VLLM_PID"

    echo "Waiting for vLLM to become ready ..."
    MAX_WAIT=2400
    WAITED=0
    while [ $WAITED -lt $MAX_WAIT ]; do
        if curl -s "http://localhost:${VLLM_PORT}/v1/models" > /dev/null 2>&1; then
            echo "vLLM is ready."
            break
        fi
        sleep 5
        WAITED=$((WAITED + 5))
        echo "  waited ${WAITED}s ..."
    done
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "ERROR: vLLM startup timed out. Check ${VLLM_LOG}."
        kill "$VLLM_PID" 2>/dev/null || true
        exit 1
    fi
    echo ""
else
    echo "Skipping vLLM startup (using existing server)."
    echo ""
fi

# ─── Step 2: Run distillation ────────────────────────────────────────────────
[ -z "$API_BASE" ] && API_BASE="http://0.0.0.0:${VLLM_PORT}/v1"

CMD="python3 \"${DISTILL_PY}\" \
  --api_key \"$API_KEY\" \
  --api_base \"$API_BASE\" \
  --model \"$SERVED_MODEL_NAME\" \
  --tokenizer_path \"$MODEL_PATH\" \
  --data_path \"$DATA_PATH\" \
  --save_path \"$SAVE_PATH\" \
  --model_name \"$MODEL_NAME\" \
  --n_samples $N_SAMPLES \
  --temperature $TEMPERATURE \
  --top_p $TOP_P \
  --max_tokens $MAX_TOKENS \
  --num_threads $NUM_THREADS"

[ -n "$ENABLE_THINKING"  ] && CMD="$CMD --enable_thinking \"$ENABLE_THINKING\""
[ -n "$INCLUDE_SOURCES"  ] && CMD="$CMD --include_sources \"$INCLUDE_SOURCES\""
[ "$RETRY_UNTIL_COMPLETE" = true ] && CMD="$CMD --retry_until_complete"
[ "$NO_RESUME" = true    ] && CMD="$CMD --no_resume"

echo "Running distillation (log: ${DISTILL_LOG}) ..."
if [ "$NO_RESUME" = true ]; then
    eval "$CMD" 2>&1 | tee "$DISTILL_LOG"
else
    eval "$CMD" 2>&1 | tee -a "$DISTILL_LOG"
fi
EXIT_CODE=${PIPESTATUS[0]}

echo ""
[ $EXIT_CODE -eq 0 ] && echo "Distillation succeeded." || echo "Distillation failed (exit $EXIT_CODE)."

# ─── Step 3: Summary ─────────────────────────────────────────────────────────
if [ -f "$SAVE_PATH" ]; then
    TOTAL=$(wc -l < "$SAVE_PATH")
    EXPECTED=$(($(wc -l < "$DATA_PATH") * N_SAMPLES))
    echo ""
    echo "  Generated samples : ${TOTAL}"
    echo "  Expected samples  : ${EXPECTED}"
    echo "  Output file       : ${SAVE_PATH}"
fi

# ─── Step 4: Stop vLLM ───────────────────────────────────────────────────────
if [ -n "$VLLM_PID" ]; then
    echo ""
    echo "Stopping vLLM (PID $VLLM_PID) ..."
    kill "$VLLM_PID" 2>/dev/null || true
fi

exit $EXIT_CODE
