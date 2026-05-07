#!/usr/bin/env bash
# Qwen2.5-7B variant: 3 RL experts (coding / tool / memory).
# Per_query npz is bundled (data/qwen2.5_7b/per_query/{coding,tool,memory}.npz)
# so this script runs only extract_w → apply_wnorm → merge.
set -euo pipefail
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
EXPERT_CODING="${EXPERT_CODING:-Gen-Verse/ReasonFlux-Coder-7B}"
EXPERT_TOOL="${EXPERT_TOOL:-emrgnt-cmplxty/Qwen2.5-7B-Instruct-ToolRL-grpo-cold}"
EXPERT_MEMORY="${EXPERT_MEMORY:-BytedTsinghua-SIA/RL-MemoryAgent-7B}"

DP="${DP:-2}"
ALPHA="${ALPHA:-1.0}"
ENERGY="${ENERGY:-0.90}"

DATA_DIR="${THIS_DIR}/data/qwen2.5_7b/per_query"
W_OUT="${THIS_DIR}/W_col.qwen2.5_7b.npz"
W_WNORM="${THIS_DIR}/W_col.qwen2.5_7b.Wnorm.npz"
MERGE_OUT="${THIS_DIR}/merged_qwen2.5_7b"
LOG_DIR="${THIS_DIR}/logs/qwen2.5_7b"
mkdir -p "${LOG_DIR}"

EXPERTS=(
    "coding=${EXPERT_CODING}"
    "tool=${EXPERT_TOOL}"
    "memory=${EXPERT_MEMORY}"
)

echo "════════════════════════════════════════════════════════════════"
echo "  PSA — Qwen2.5-7B variant   $(date '+%F %T')"
echo "  α=${ALPHA}  DP=${DP}  energy=${ENERGY}"
echo "════════════════════════════════════════════════════════════════"

# Verify per_query npz present
for t in coding tool memory; do
    [ -s "${DATA_DIR}/${t}.npz" ] || { echo "[FATAL] missing ${DATA_DIR}/${t}.npz"; exit 1; }
done

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

T_TOTAL=$(awk "BEGIN{printf \"%.1f\", ${T1} + ${T2} + ${T3}}")
fmt() { local s=$(printf "%.0f" "$1"); printf "%dm%02ds" $((s/60)) $((s%60)); }
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Qwen2.5-7B PIPELINE DONE"
echo "════════════════════════════════════════════════════════════════"
printf "  %-26s  %12s  %s\n" "1. extract_w (DP=${DP})"  "${T1}" "$(fmt ${T1})"
printf "  %-26s  %12s  %s\n" "2. apply_wnorm"           "${T2}" "$(fmt ${T2})"
printf "  %-26s  %12s  %s\n" "3. merge"                 "${T3}" "$(fmt ${T3})"
printf "  %-26s  %12s  %s\n" "TOTAL"                    "${T_TOTAL}" "$(fmt ${T_TOTAL})"
echo "  Merged: ${MERGE_OUT}"
echo "════════════════════════════════════════════════════════════════"
