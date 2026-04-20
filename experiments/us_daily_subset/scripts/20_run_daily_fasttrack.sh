#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/projects/KDD2026}"
cd "$ROOT"

source /data/.anaconda/etc/profile.d/conda.sh
conda activate main_latest

RUN_TAG="${RUN_TAG:-010515}"
MAX_EPOCHS="${MAX_EPOCHS:-6}"
DYNAMIC_GROUP="${DYNAMIC_GROUP:-era5_core}"
LOSS_NAME="${LOSS_NAME:-hydro_skill}"
HISTORY_LENGTH="${HISTORY_LENGTH:-365}"
LUMPED_BATCH_SIZE="${LUMPED_BATCH_SIZE:-256}"
GRAPH_BATCH_SIZE="${GRAPH_BATCH_SIZE:-256}"
STATIC_GROUP="${STATIC_GROUP:-static_base}"
if [ "$#" -gt 0 ]; then
  HUCS=("$@")
else
  HUCS=(01 05 15)
fi

LUMPED_RUN_ID="us_daily_lumped_${RUN_TAG}_core_hydro_skill_h${HISTORY_LENGTH}_b${LUMPED_BATCH_SIZE}"
SEMI_RUN_ID="us_daily_semidistributed_${RUN_TAG}_core_hydro_skill_h${HISTORY_LENGTH}_gb${GRAPH_BATCH_SIZE}"

python devp_US/scripts/04_fit_scalers.py \
  --run-name "us_daily_scalers_${RUN_TAG}" \
  --regimes lumped semidistributed \
  --frequency daily \
  --hucs "${HUCS[@]}"

python devp_US/scripts/16_build_graph_batches.py \
  --run-name "us_daily_semibatches_${RUN_TAG}" \
  --regime semidistributed \
  --frequency daily \
  --dynamic-group "$DYNAMIC_GROUP" \
  --static-group "$STATIC_GROUP" \
  --history-length "$HISTORY_LENGTH" \
  --graph-batch-size "$GRAPH_BATCH_SIZE" \
  --hucs "${HUCS[@]}"

python devp_US/scripts/08_train_lumped.py \
  --run-name "$LUMPED_RUN_ID" \
  --dynamic-group "$DYNAMIC_GROUP" \
  --frequency daily \
  --static-group "$STATIC_GROUP" \
  --loss-name "$LOSS_NAME" \
  --history-length "$HISTORY_LENGTH" \
  --batch-size "$LUMPED_BATCH_SIZE" \
  --max-epochs "$MAX_EPOCHS" \
  --device cuda \
  --hucs "${HUCS[@]}" &
LUMPED_PID=$!

python devp_US/scripts/11_train_semidistributed.py \
  --run-name "$SEMI_RUN_ID" \
  --dynamic-group "$DYNAMIC_GROUP" \
  --frequency daily \
  --static-group "$STATIC_GROUP" \
  --loss-name "$LOSS_NAME" \
  --history-length "$HISTORY_LENGTH" \
  --graph-batch-size "$GRAPH_BATCH_SIZE" \
  --max-epochs "$MAX_EPOCHS" \
  --device cuda \
  --hucs "${HUCS[@]}" &
SEMI_PID=$!

wait "$LUMPED_PID"
wait "$SEMI_PID"

python devp_US/scripts/14_make_visuals.py --run-name "us_visuals_${RUN_TAG}" --run-id "$LUMPED_RUN_ID"
python devp_US/scripts/14_make_visuals.py --run-name "us_visuals_${RUN_TAG}_semi" --run-id "$SEMI_RUN_ID"
python devp_US/scripts/18_plot_context_vs_skill.py \
  --run-name "us_context_${RUN_TAG}" \
  --lumped-run-id "$LUMPED_RUN_ID" \
  --semi-run-id "$SEMI_RUN_ID"
python devp_US/scripts/13_evaluate_all.py --run-name "us_eval_${RUN_TAG}"
python devp_US/scripts/15_write_run_summary.py --run-name "us_summary_${RUN_TAG}"

echo "Completed fast-track US daily runs:"
echo "  lumped: $LUMPED_RUN_ID"
echo "  semidistributed: $SEMI_RUN_ID"
