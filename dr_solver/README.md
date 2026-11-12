# Solver-Guided Diabetic Retinopathy Detection (CSCI 5750)

Learned component: CNN (timm `efficientnet_b0`, ImageNet-pretrained) predicts DR grade 0-4 per eye.
Symbolic component: OR-Tools CP-SAT optimizer combines both eyes into a patient grade and a
referral decision (refer iff patient grade >= 2).

## Symbolic model (src/symbolic_cpsat.py)

| | |
|---|---|
| Variables | one-hot `x_L[g]`, `x_R[g]`; `left_level`, `right_level`, `patient_level` in 0..4; `referable_dr`, `mismatch` Boolean |
| Hard | exactly one grade per eye; `patient_level = max(left, right)`; `referable_dr <-> patient_level >= 2`; `mismatch <-> abs(left - right) >= 2` |
| Objective (min) | `sum -log p_L[g] x_L[g] + sum -log p_R[g] x_R[g] + lam * mismatch - beta * referable_dr` |

With `lam = 0` (only the Oct 1 constraints) the two eyes are independent, so the optimum equals
per-eye argmax + max, the learned-only rule. `tests/test_symbolic.py` checks this. The
bilateral term `lam * mismatch` (a soft constraint) is what couples the two eyes.

### Learned pair compatibility (C7)

Pair Booleans `y[a][b]` (true iff left = a and right = b) are linked to the one-hot grades by
hard channeling constraints. The soft term `mu * sum W[a][b] y[a][b]` uses
`W[a][b] = -log(P(a,b) / (P(a) P(b)))`, estimated from TRAINING patients (add-1 smoothing).
With `mu = 1` the solver returns the Bayes-optimal pair
`argmax p(a|left image) p(b|right image) P(a,b) / (P(a) P(b))`, assuming the two images are
independent given the two grades. `mu` is tuned on validation. Tests check CP-SAT against
brute force and against this Bayes rule.

## Conditions (same CNN, same test patients)

| | Condition |
|---|---|
| C1 | Learned-only: per-eye argmax, patient = max |
| C2 | Learned-only + referral threshold on max P(grade >= 2), tuned on val |
| C3 | Symbolic-only: CP-SAT with training class priors, no image evidence |
| C4 | Combined with Oct 1 constraints only (lam = 0, beta tuned on val) |
| C5 | Combined + bilateral consistency (lam, beta tuned on val) - main system |
| C7 | Combined + learned pair compatibility W (mu, beta tuned on val) |
| C6 | Oracle: true per-eye labels into CP-SAT (checks the rule layer) |

All tuning uses the validation split only. The test split is used once.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
# install torch for your CUDA version from pytorch.org, then:
pip install -r requirements.txt
python -m src.check_env --data /path/to/data
python -m pytest -q tests
```

## Run, step by step (each script accepts --limit N for a quick test)

```bash
# 1. crop + resize once (resumable; re-run continues where it stopped)
python -m src.preprocess --src /path/train --labels /path/trainLabels.csv --size 384 --limit 200
python -m src.preprocess --src /path/train --labels /path/trainLabels.csv --size 384

# 2. patient split (import the Oct 1 split, or make a 70/15/15 split with seed 42)
python -m src.make_splits --labels /path/trainLabels.csv --existing /path/oct1_split.csv
#   or
python -m src.make_splits --labels /path/trainLabels.csv

# 3. train CNN (smoke test first)
python -m src.train_cnn --seed 42 --limit 500 --epochs 1
python -m src.train_cnn --seed 42

# 4. per-eye probabilities for val/test (temperature fit on val)
python -m src.predict --seed 42 --tta

# 5. all conditions -> outputs/decisions_*.csv
python -m src.run_conditions --seed 42

# 6. metrics, CIs, significance, failure cases, plots
python -m src.evaluate --seeds 42
```

Optional variance across training seeds: repeat steps 3-5 with `--seed 1` and `--seed 2`,
then `python -m src.evaluate --seeds 42 1 2`.

Or run everything: edit paths in `run_all.sh`, then `nohup bash run_all.sh > run.log 2>&1 &`.

## Improved runs (EfficientNet-B3, 512 px, 3 seeds, ensemble)

```bash
nohup bash run_improved.sh > improved.log 2>&1 &
```
Results go to `results/<run>/main_table.md` for runs `seed42`, `b0_ens`, `b3_3seeds`,
`b3_ens`, and `b3_ens_recall` (threshold/weights chosen for recall at specificity >= 0.90).
Pick the CNN by validation kappa (`results/train_log_*.json`), not by test numbers, and report
every run, including the first seed-42 run.

## Outputs

| File | Content |
|---|---|
| results/<run>/main_table.md / .csv | all conditions, metrics with 95% bootstrap CI, paired differences |
| results/<run>/paired_comparisons.csv | C5 and C7 vs C1, C2, C4, C5: paired bootstrap CI of differences (kappa, F1, recall, FNR) and exact McNemar test |
| results/failure_cases.csv | 5 random patients where C5 beats C1, and 5 where C1 beats C5 |
| results/summary.txt | model, hardware, train time, chosen lam/beta/t, how many patients the solver changed, solver ms/patient |
| results/val_grid_seed42.csv | full validation grid over lam x beta |
| results/plots/ | confusion matrices C1 vs C5, referral ROC with operating points, val F1 vs lam |
| outputs/probs_*.csv, outputs/decisions_*.csv | raw outputs; step 6 can be re-run without the CNN |

## Notes

- The raw data is never modified. Preprocessed images go to `data/processed/`.
- Patients missing an eye, or with an unreadable image, are dropped (logged in `data/bad_images.csv`).
- If validation selects `lam = 0`, C5 equals C4. That is a valid (negative) result: report it.
