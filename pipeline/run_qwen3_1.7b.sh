#!/usr/bin/env bash
# Qwen3-1.7B variant: 3 RL experts (if_rl / math / lucy).
# Bundled prompt files: data/qwen3_1.7b/prompts/{if_rl,math,lucy}.jsonl
#
# If per_query npz isn't present yet, this script auto-runs gen_per_query.py
# to produce it (one-time, ~10-15 min on a single GPU). Then extract_w →
# apply_wnorm → merge.
set -euo pipefail
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASE_MODEL="${BASE_MODEL:-./models/Qwen3-1.7B-base}"
EXPERT_IF_RL="${EXPERT_IF_RL:-./models/Qwen3-1.7B-if-rl}"
EXPERT_MATH="${EXPERT_MATH:-./models/Qwen3-1.7B-math-rl}"
EXPERT_LUCY="${EXPERT_LUCY:-./models/Lucy}"

DP="${DP:-2}"
ALPHA="${ALPHA:-1.0}"
ENERGY="${ENERGY:-0.90}"

# gen_per_query knobs
N_QUERIES="${N_QUERIES:-128}"
MAX_CTX="${MAX_CTX:-4096}"
MAX_CTX_PER_TASK="${MAX_CTX_PER_TASK:-lucy=8192}"
MAX_NEW="${MAX_NEW:-256}"
TEMPERATURE="${TEMPERATURE:-0.7}"

PROMPTS_DIR="${THIS_DIR}/data/qwen3_1.7b/prompts"
DATA_DIR="${THIS_DIR}/data/qwen3_1.7b/per_query"
W_OUT="${THIS_DIR}/W_col.qwen3_1.7b.npz"
W_WNORM="${THIS_DIR}/W_col.qwen3_1.7b.Wnorm.npz"
MERGE_OUT="${THIS_DIR}/merged_qwen3_1.7b"
LOG_DIR="${THIS_DIR}/logs/qwen3_1.7b"
mkdir -p "${LOG_DIR}" "${DATA_DIR}"

EXPERTS=(
    "if_rl=${EXPERT_IF_RL}"
    "math=${EXPERT_MATH}"
    "lucy=${EXPERT_LUCY}"
)

echo "════════════════════════════════════════════════════════════════"
echo "  PSA — Qwen3-1.7B variant   $(date '+%F %T')"
echo "  α=${ALPHA}  DP=${DP}  energy=${ENERGY}"
echo "════════════════════════════════════════════════════════════════"

# Verify prompt files
for t in if_rl math lucy; do
    [ -s "${PROMPTS_DIR}/${t}.jsonl" ] || { echo "[FATAL] missing ${PROMPTS_DIR}/${t}.jsonl"; exit 1; }
done

# 0. gen_per_query — only if any task missing
NEED_GEN=0
for t in if_rl math lucy; do
    [ -s "${DATA_DIR}/${t}.npz" ] || NEED_GEN=1
done
if [ "${NEED_GEN}" -eq 1 ]; then
    echo ""
    echo "════ Step 0/4: gen_per_query (one-time, ~10-15 min) ════"
    T0_0=$(date +%s.%N)
    python "${THIS_DIR}/gen_per_query.py" \
        --base_model "${BASE_MODEL}" \
        --experts "${EXPERTS[@]}" \
        --prompts_dir "${PROMPTS_DIR}" \
        --out_dir "${DATA_DIR}" \
        --n_queries "${N_QUERIES}" \
        --max_ctx "${MAX_CTX}" --max_ctx_per_task "${MAX_CTX_PER_TASK}" \
        --max_new_tokens "${MAX_NEW}" --temperature "${TEMPERATURE}" \
        --device cuda:0 \
        > "${LOG_DIR}/step0_gen_per_query.log" 2>&1
    T0=$(awk "BEGIN{printf \"%.1f\", $(date +%s.%N) - ${T0_0}}")
    echo "[step 0] wall = ${T0}s"
else
    echo "[step 0] per_query npz already present — skipping gen_per_query"
    T0=0
fi

# 1. extract_w
echo ""
echo "════ Step 1/3: extract_w (DP=${DP}, α=${ALPHA}) ════"
T1_0=$(date +%s.%N); rm -f "${W_OUT}"
python "${THIS_DIR}/extract_w.py" \
    --base_model "${BASE_MODEL}" \
    --experts "${EXPERTS[@]}" \
    --data_dir "${DATA_DIR}" \
    --out_npz "${W_OUT}" --dp "${DP}" --alpha "${ALPHA}" \
    > "${LOG_DIR}/step1_extract.log" 2>&1
T1=$(awk "BEGIN{printf \"%.1f\", $(date +%s.%N) - ${T1_0}}")
echo "[step 1] wall = ${T1}s"

# 2. apply_wnorm
echo ""
echo "════ Step 2/3: apply_wnorm ════"
T2_0=$(date +%s.%N)
python "${THIS_DIR}/apply_wnorm.py" \
    --experts "${EXPERTS[@]}" \
    --w_col_in "${W_OUT}" --w_col_out "${W_WNORM}" \
    > "${LOG_DIR}/step2_wnorm.log" 2>&1
T2=$(awk "BEGIN{printf \"%.1f\", $(date +%s.%N) - ${T2_0}}")
echo "[step 2] wall = ${T2}s"

# 3. merge
echo ""
echo "════ Step 3/3: merge (1 GPU) ════"
T3_0=$(date +%s.%N); rm -rf "${MERGE_OUT}"
python "${THIS_DIR}/merge.py" \
    --base_model "${BASE_MODEL}" \
    --experts "${EXPERTS[@]}" \
    --w_col_file "${W_WNORM}" \
    --out_dir "${MERGE_OUT}" --device cuda:0 --energy "${ENERGY}" \
    > "${LOG_DIR}/step3_merge.log" 2>&1
T3=$(awk "BEGIN{printf \"%.1f\", $(date +%s.%N) - ${T3_0}}")
echo "[step 3] wall = ${T3}s"

T_TOTAL=$(awk "BEGIN{printf \"%.1f\", ${T0} + ${T1} + ${T2} + ${T3}}")
fmt() { local s=$(printf "%.0f" "$1"); printf "%dm%02ds" $((s/60)) $((s%60)); }
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Qwen3-1.7B PIPELINE DONE"
echo "════════════════════════════════════════════════════════════════"
[ "${T0}" != "0" ] && printf "  %-26s  %12s  %s\n" "0. gen_per_query"      "${T0}" "$(fmt ${T0})"
printf "  %-26s  %12s  %s\n" "1. extract_w (DP=${DP})"  "${T1}" "$(fmt ${T1})"
printf "  %-26s  %12s  %s\n" "2. apply_wnorm"           "${T2}" "$(fmt ${T2})"
printf "  %-26s  %12s  %s\n" "3. merge"                 "${T3}" "$(fmt ${T3})"
printf "  %-26s  %12s  %s\n" "TOTAL"                    "${T_TOTAL}" "$(fmt ${T_TOTAL})"
echo "  Merged: ${MERGE_OUT}"
echo "════════════════════════════════════════════════════════════════"
