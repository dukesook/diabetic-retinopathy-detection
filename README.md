# Knowledge-Guided Diabetic Retinopathy Detection

### NMSU CSCI 5750: Artificial Intelligence II Group Project

## Team Members

- Sajidur Rahman
- Devon Sookhoo
- Sumanth Reddy Nandhikonda

## Setup

### Create a Virtual Environment

```
python3 -m venv .venv
```

### Activate the Environment

```sh
source .venv/bin/activate     # Linux/macOS
.venv\Scripts\activate.bat    # Windows CMD
.\.venv\Scripts\Activate.ps1  # PowerShell
```

### Install Package

Reads the pyproject.toml file

```
pip install -e .
```

### Run

```
python -m drd
```

## Important Links

Paper on _Single Domain Generalization in Diabetic Retinopathy: A Neuro-Symbolic Learning Approach_:

- https://arxiv.org/html/2509.02918v1

### Training Data

Eyepacs, Aptos, Messidor Diabetic Retinopathy

- https://www.kaggle.com/datasets/ascanipek/eyepacs-aptos-messidor-diabetic-retinopathy
- 92K JPEG images
- 22Gb
