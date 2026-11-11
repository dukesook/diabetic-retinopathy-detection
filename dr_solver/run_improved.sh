#!/usr/bin/env bash
# Improved runs: EfficientNet-B3 at 512 px, 3 seeds, ensemble, learned pair constraint (C7).
# Run from the dr_project folder:   nohup bash run_improved.sh > improved.log 2>&1 &
# Expected time on one H100: about 3 hours. Safe to re-run: preprocessing skips finished images.
set -euo pipefail

NB=/home/gojironobackup/snandhikonda
PY=$NB/conda_envs/dr/bin/python
RAW=$NB/eyepacs/train
LABELS=$NB/eyepacs/trainLabels.csv
export HF_HOME=$NB/hf_home
export CUDA_VISIBLE_DEVICES=0

echo "== tests"; $PY -m pytest -q tests

echo "== 1. preprocess at 512 px"
$PY -m src.preprocess --src $RAW --labels $LABELS --size 512 --workers 32 --out data/processed512

echo "== 2. train EfficientNet-B3, 3 seeds"
for s in 42 1 2; do
  if [ ! -f checkpoints/b3_seed$s.pt ] || [ ! -f results/train_log_b3_seed$s.json ]; then
    $PY -m src.train_cnn --seed $s --model efficientnet_b3 --size 512 --batch 32 --lr 2e-4 \
        --workers 16 --img_dir data/processed512 --tag b3_seed$s
  fi
  $PY -m src.predict --seed $s --tag b3_seed$s --tta --workers 16
done

echo "== 3. ensembles (average of 3 seeds)"
$PY -m src.ensemble --tags b3_seed42 b3_seed1 b3_seed2 --out b3_ens
$PY -m src.ensemble --tags seed42 seed1 seed2 --out b0_ens

echo "== 4. conditions C1-C7 (all tuning on validation only)"
for t in seed42 b0_ens b3_seed42 b3_seed1 b3_seed2 b3_ens; do
  $PY -m src.run_conditions --tag $t --workers 32
done
$PY -m src.run_conditions --tag b3_ens --workers 32 --select_by recall_at_spec --min_spec 0.90

echo "== 5. evaluation"
$PY -m src.evaluate --runs seed42
$PY -m src.evaluate --runs b0_ens
$PY -m src.evaluate --runs b3_seed42 b3_seed1 b3_seed2 --name b3_3seeds
$PY -m src.evaluate --runs b3_ens
$PY -m src.evaluate --runs b3_ens_recall

echo "== validation kappa per CNN run (use this, not test, to pick the model)"
for f in results/train_log_*.json; do echo "$f: $($PY -c "import json;print(round(json.load(open('$f'))['best_val_qwk'],4))")"; done

echo "== done. Main tables: results/*/main_table.md"
