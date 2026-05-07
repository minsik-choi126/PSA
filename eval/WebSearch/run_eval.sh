#!/usr/bin/env bash
# Web-search agent evaluation (SimpleQA + BrowseComp).
#
# Usage:
#   bash run_eval.sh --model <model_path> [options]
#
# Options:
#   --model PATH            (required) HF repo or local model directory
#   --port N                vLLM port (default 8001)
#   --tp N                  tensor parallel size (default 2)
#   --benches NAMES         comma-separated subset of {simpleqa,browsecomp} (default: both)
#   --limit_simpleqa N      smoke-test sample limit
#   --limit_browsecomp N    smoke-test sample limit
#   --n_parallel N          concurrent agent runs (default 8)
#   --temperature F         (default 0.6)
#   --max_steps N           tool-loop steps (default 6)
#   --suffix S              extra suffix on result filenames
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS="${SCRIPT_DIR}/scripts"
RESULTS="${SCRIPT_DIR}/results"
LOG_DIR="${RESULTS}/_logs"
mkdir -p "${RESULTS}" "${LOG_DIR}"

# ── Defaults ─────────────────────────────────────────────────────────────
MODEL=""
PORT=8001
TP=2
BENCHES="simpleqa,browsecomp"
LIMIT_SQA=""
LIMIT_BC=""
N_PARALLEL=8
TEMP=0.6
MAX_STEPS=6
SUFFIX=""

# ── Args ─────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)           MODEL="$2";           shift 2 ;;
        --port)            PORT="$2";            shift 2 ;;
        --tp)              TP="$2";              shift 2 ;;
        --benches)         BENCHES="$2";         shift 2 ;;
        --limit_simpleqa)  LIMIT_SQA="$2";       shift 2 ;;
        --limit_browsecomp)LIMIT_BC="$2";        shift 2 ;;
        --n_parallel)      N_PARALLEL="$2";      shift 2 ;;
        --temperature)     TEMP="$2";            shift 2 ;;
        --max_steps)       MAX_STEPS="$2";       shift 2 ;;
        --suffix)          SUFFIX="$2";          shift 2 ;;
        -h|--help)         sed -n '/^# Usage/,/^set/p' "$0" | head -n -1; exit 0 ;;
        *)                 echo "[ERROR] unknown arg: $1" >&2; exit 1 ;;
    esac
done

[[ -z "${MODEL}" ]] && { echo "[ERROR] --model is required" >&2; exit 1; }
NAME="$(basename "${MODEL%/}")${SUFFIX}"

# ── Load .env (OpenAI / Serper keys) ─────────────────────────────────────
if [[ -f "${SCRIPT_DIR}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/.env"
    set +a
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "[ERROR] OPENAI_API_KEY not set (needed for grader). Edit ${SCRIPT_DIR}/.env" >&2; exit 1
fi
if [[ -z "${SERPER_API_KEY:-}" ]]; then
    echo "[WARN] SERPER_API_KEY not set — search tool will return empty results."
    echo "       Models will fall back to closed-book QA (low scores expected)."
fi

# ── Activate eval venv ───────────────────────────────────────────────────
EVAL_VENV="${EVAL_VENV:-./.eval}"
if [[ -f "${EVAL_VENV}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${EVAL_VENV}/bin/activate"
fi

echo "════════════════════════════════════════════════════════════════"
echo "  WebSearch eval — ${NAME}    $(date)"
echo "  model:    ${MODEL}"
echo "  port:     ${PORT}   tp=${TP}   benches=${BENCHES}"
echo "  parallel: ${N_PARALLEL}   temp=${TEMP}   max_steps=${MAX_STEPS}"
echo "  serper:   $([[ -n "${SERPER_API_KEY:-}" ]] && echo SET || echo NOT_SET)"
echo "════════════════════════════════════════════════════════════════"

# ── Launch vLLM serve ────────────────────────────────────────────────────
SERVED_NAME="eval_target"
VLLM_LOG="${LOG_DIR}/${NAME}_vllm.log"
echo "[$(date)] starting vLLM serve on port ${PORT}..."
nohup vllm serve "${MODEL}" \
    --port "${PORT}" \
    --tensor-parallel-size "${TP}" \
    --served-model-name "${SERVED_NAME}" \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --reasoning-parser qwen3 \
    --dtype bfloat16 \
    > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!
echo "  vllm PID=${VLLM_PID}  log=${VLLM_LOG}"

cleanup() {
    echo "[$(date)] cleaning up vLLM..."
    kill -TERM "${VLLM_PID}" 2>/dev/null || true
    sleep 5
    kill -KILL "${VLLM_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ── Wait for /v1/models endpoint ─────────────────────────────────────────
echo "[$(date)] waiting for vLLM endpoint..."
for i in $(seq 1 90); do
    if curl -sf "http://127.0.0.1:${PORT}/v1/models" > /dev/null 2>&1; then
        echo "  vLLM ready (after ${i}×5s)."
        break
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "[ERROR] vLLM crashed during startup. tail ${VLLM_LOG}:"
        tail -40 "${VLLM_LOG}"; exit 1
    fi
    sleep 5
done
if ! curl -sf "http://127.0.0.1:${PORT}/v1/models" > /dev/null 2>&1; then
    echo "[ERROR] vLLM did not become ready in 7.5 minutes"; exit 1
fi

# ── Run SimpleQA ─────────────────────────────────────────────────────────
if [[ ",${BENCHES}," == *,simpleqa,* ]]; then
    OUT="${RESULTS}/${NAME}.simpleqa.jsonl"
    echo ""; echo "── SimpleQA: ${OUT} ──"
    python "${SCRIPTS}/run_simpleqa.py" \
        --model "${SERVED_NAME}" \
        --endpoint "http://127.0.0.1:${PORT}/v1" \
        --n_parallel "${N_PARALLEL}" \
        --temperature "${TEMP}" \
        --max_steps "${MAX_STEPS}" \
        ${LIMIT_SQA:+--limit "${LIMIT_SQA}"} \
        --output "${OUT}" \
        2>&1 | tee "${LOG_DIR}/${NAME}_simpleqa.log"
fi

# ── Run BrowseComp ───────────────────────────────────────────────────────
if [[ ",${BENCHES}," == *,browsecomp,* ]]; then
    OUT="${RESULTS}/${NAME}.browsecomp.jsonl"
    echo ""; echo "── BrowseComp: ${OUT} ──"
    python "${SCRIPTS}/run_browsecomp.py" \
        --model "${SERVED_NAME}" \
        --endpoint "http://127.0.0.1:${PORT}/v1" \
        --n_parallel "${N_PARALLEL}" \
        --temperature "${TEMP}" \
        --max_steps "$((MAX_STEPS + 2))" \
        ${LIMIT_BC:+--limit "${LIMIT_BC}"} \
        --output "${OUT}" \
        2>&1 | tee "${LOG_DIR}/${NAME}_browsecomp.log"
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  DONE  ${NAME}  $(date)"
echo "  results dir: ${RESULTS}"
echo "════════════════════════════════════════════════════════════════"
