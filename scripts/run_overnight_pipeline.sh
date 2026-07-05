#!/bin/bash
# Overnight holding pipeline wrapper — snapshot + run_overnight.py
# Runs in the overnight_holding project directory.
# Usage: bash scripts/run_overnight_pipeline.sh 2026-06-30
set -euo pipefail

TRADE_DATE="${1:-$(date +%Y-%m-%d)}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATE_FMT="${TRADE_DATE//-/}"
SNAPSHOT_DIR="${PROJECT_DIR}/data/snapshots"
OUTPUT_DIR="${PROJECT_DIR}/data/output_${DATE_FMT}"
SNAPSHOT_CSV="${SNAPSHOT_DIR}/snapshot_${DATE_FMT}_tencent.csv"

cd "$PROJECT_DIR"
export PYTHONUNBUFFERED=1

echo "============================================"
echo "  一夜持股法 实盘流水线"
echo "  Trade date: ${TRADE_DATE}"
echo "  Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

# Stage 1: Generate snapshot
echo ""
echo "[Stage 1] Generating Tencent realtime snapshot..."
python3.12 scripts/generate_snapshot.py \
  --trade-date "$TRADE_DATE" \
  --out-dir "$SNAPSHOT_DIR" \
  --chunk-size 80

if [ ! -f "$SNAPSHOT_CSV" ]; then
  echo "FATAL: Snapshot generation failed — no CSV at ${SNAPSHOT_CSV}"
  exit 1
fi
echo "Snapshot: $(wc -l < "$SNAPSHOT_CSV") lines"

# Stage 2: Run multistage pipeline (Heavy + Light + Fusion)
echo ""
echo "[Stage 2] Running Heavy + Light LLM reviews + Final Fusion..."
python3.12 scripts/run_overnight.py \
  --trade-date "$TRADE_DATE" \
  --prefilter-snapshot-csv "$SNAPSHOT_CSV" \
  --final-snapshot-csv "$SNAPSHOT_CSV" \
  --out-root "$OUTPUT_DIR" \
  --heavy-top-k 50 \
  --heavy-target-top-n 15 \
  --light-top-k 15 \
  --final-top-n 5 \
  --final-candidate-pool-size 50 \
  --live-weight 0.60 \
  --heavy-weight 0.25 \
  --light-weight 0.15 \
  --heavy-model-key yuanlan4 \
  --heavy-mode deepthink \
  --light-model-key yuanlan4 \
  --light-mode chat \
  --disable-social-hot-context \
  --disable-xueqiu \
  --disable-twitter

echo ""
echo "============================================"
echo "  Pipeline complete: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Output: ${OUTPUT_DIR}"
echo "============================================"

# Print the final Top5 summary
FINAL_SUMMARY=$(find "$OUTPUT_DIR" -name "live_summary_*final_top5*.md" -type f | sort | tail -1)
if [ -n "$FINAL_SUMMARY" ] && [ -f "$FINAL_SUMMARY" ]; then
  echo ""
  echo "========== TOP 5 PICKS =========="
  cat "$FINAL_SUMMARY"
fi
