#!/usr/bin/env bash
# Full pipeline. Edit the three paths, then:  bash run_all.sh
# To run in the background on a server:  nohup bash run_all.sh > run.log 2>&1 &
set -euo pipefail

RAW_DIR=/path/to/kaggle/train            # folder with 10_left.jpeg, 10_right.jpeg, ...
LABELS=/path/to/kaggle/trainLabels.csv
EXISTING_SPLIT=""                         # path to your Oct 1 split CSV, or leave empty
SIZE=384                                  # 256 if no GPU / low VRAM
SEED=42

python -m src.check_env --data .
python -m pytest -q tests                                  # solver correctness first

python -m src.preprocess --src "$RAW_DIR" --labels "$LABELS" --size $SIZE

if [ -n "$EXISTING_SPLIT" ]; then
  python -m src.make_splits --labels "$LABELS" --existing "$EXISTING_SPLIT"
else
  python -m src.make_splits --labels "$LABELS"
fi

python -m src.train_cnn  --seed $SEED --size $SIZE
python -m src.predict    --seed $SEED --tta
python -m src.run_conditions --seed $SEED
python -m src.evaluate   --seeds $SEED

echo "done. see results/main_table.md and results/summary.txt"
