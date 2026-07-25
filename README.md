# Bash Script Check

Bash Script Check is an encoder-only Transformer classifier for shell-script
text. It predicts whether a script is:

- `safe` — ordinary, low-risk automation;
- `risky` — potentially destructive or privileged administration;
- `malicious` — behavior associated with attacks.

For malicious predictions, a second classification head attempts to explain
the behavior with a category such as `reverse_shell`, `exfiltration`,
`persistence`, or `defense_evasion`.

The project treats every script as plain text. It does not execute dataset
records or files passed to `--script-file`.

> [!WARNING]
> This is an experimental classifier trained largely on synthetic data. Do not
> use it as the only security control, and never execute an unknown script just
> because the model labels it safe.

## Repository contents

```text
.
|-- main.py                         Three complete workflow functions
|-- bash_classifier/
|   |-- cpu_training.py             CPU-only training entry point
|   |-- gpu_training.py             CUDA GPU training entry point
|   |-- training.py                 Shared training loop
|   |-- data.py                     Dataset/tokenizer/data-loader logic
|   |-- model.py                    Transformer and checkpoint logic
|   |-- evaluation.py               Testing and evaluation
|   |-- prediction.py               Prediction-only workflow
|   `-- cli.py                      Shared command arguments
|-- generate_dataset.py             Synthetic safe/risky dataset generator
|-- scrape_dataset.py               GitHub API dataset scraper
|-- data/
|   |-- safe.jsonl
|   |-- risky.jsonl
|   |-- malicious.jsonl
|   `-- safe_risky_combined.jsonl   Default train/test dataset
|-- showcase_scripts/               Non-executable classifier examples
`-- requirements.txt
```

## Installation

Python 3.10 or newer is recommended. A local virtual environment keeps the
project dependencies isolated; `.venv/` is ignored by Git and will not be
committed.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Linux or macOS, activate it with:

```bash
source .venv/bin/activate
```

## CPU workflow

The CPU function trains a new model, tests it on the reproducible held-out
split, and then evaluates it against every row of the selected dataset:

```powershell
python main.py cpu
```

This calls `train_test_evaluate_cpu` in `main.py`. CPU training itself lives
in `bash_classifier/cpu_training.py`.

The generated files are saved locally and ignored by Git:

```text
artifacts/bash_transformer.pt
artifacts/bash_tokenizer.json
```

For a faster demonstration run:

```powershell
python main.py cpu `
  --epochs 3 `
  --context-size 256 `
  --model-dimension 128 `
  --heads 4 `
  --blocks 2 `
  --feed-forward-dimension 512
```

## Predicting showcase scripts

Predict every file inside `showcase_scripts`:

```powershell
python main.py predict
```

This calls `predict_showcase_scripts` in `main.py`. The checkpoint is loaded
only once, and each output line includes the relative filename, predicted
label, confidence, and malicious reason when available. Use another directory
if needed:

```powershell
python main.py predict --showcase-directory path\to\scripts
```

Use `--checkpoint` and `--tokenizer` with any workflow to select different
saved artifacts.

## GPU training

The GPU function performs the same train, held-out test, and full-dataset
evaluation sequence, but requires an NVIDIA CUDA GPU:

```powershell
python main.py gpu
```

This calls `train_test_evaluate_gpu` in `main.py`. GPU training itself lives
in `bash_classifier/gpu_training.py` and never silently falls back to CPU.

If the version printed by PyTorch ends in `+cpu`, activate `.venv`, open the
official [PyTorch installation selector](https://docs.pytorch.org/get-started/locally/),
choose your operating system, Pip, Python, and a supported CUDA option, and run
the command it provides. Verify the result before training:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

The second line must print `True`; training will then report `Using device:
cuda`. CUDA training also enables mixed precision and GPU-friendly data
transfers automatically.

## Code naming and function documentation

Python cannot use kebab-case names such as `train-model`, because `-` is the
subtraction operator. Functions therefore use the valid Python equivalent,
snake_case, such as `train_model`. Project-owned variables and parameters use
camelCase, such as `trainingConfig`. Constants remain `UPPER_SNAKE_CASE` so
they are visibly different from mutable variables.

Every application function has a docstring directly below its definition. The
docstring explains the function's responsibility and why that step exists in
the pipeline.

Training publishes these matching files only after training succeeds:

```text
artifacts/bash_transformer.pt
artifacts/bash_tokenizer.json
```

The tokenizer is also embedded inside every new checkpoint. Prediction loads
the checkpoint from the project `artifacts` directory by default, even when
`main.py` is launched from another working directory.

## Dataset format

The combined dataset is JSON Lines: one JSON object per line.

```json
{"id":"safe-00001","label":"safe","script":"#!/bin/bash\necho hello\n","category":"","description":""}
```

Malicious records use a non-empty `category`, which trains the model's reason
head. Safe and risky records normally leave `category` empty.

## Interpreting accuracy

Synthetic held-out accuracy is not the same as real-world accuracy. Generated
scripts can contain class-specific wording and structures that make the test
set easier than scripts written by different people.

The training split groups normalized template families together, so variants
of one synthetic template cannot appear in both training and test data. New
training runs also use a byte-level BPE tokenizer, which learns reusable shell
fragments instead of treating each complete command or address as one word.

Testing reports several complementary numbers:

- accuracy;
- balanced accuracy, which gives every primary label equal importance;
- macro F1, which exposes poor performance on one class;
- per-label precision, recall, and F1;
- expected calibration error, where lower means confidence better matches
  observed correctness;
- malicious-reason accuracy.

After the held-out test, CPU and GPU workflows evaluate the complete prediction
system against `showcase_scripts/expected_labels.json`. This independent report
includes strict accuracy, coverage, accuracy among covered predictions, and
malicious recall.

Prediction combines the statistical model with a small set of high-precision
signatures for explicit reverse shells, sensitive-file uploads, persistence,
destructive administration, and unbounded busy loops. Infinite loops without
an obvious pause or exit are classified as `risky` with reason
`infinite_loop`. Predictions below `--minimum-confidence 0.65`
are reported as `uncertain` instead of being presented as trustworthy labels.
Empty files and unsupported tabular formats are reported as invalid inputs.

Every `risky` prediction includes a reason when displayed. Recognized reasons
include `firewall_change`, `destructive_file_operation`, `infinite_loop`,
`account_security_change`, `permission_change`, `service_management`,
`package_management`, and `privileged_operation`. If the model predicts risky
without a recognizable operation, the reason is `model_detected_risk` rather
than an invented specific explanation.

CSV-style text files containing `script_name` and `script_content` columns are
handled as script containers. Each embedded script is extracted and classified
separately, so one risky row does not incorrectly label every script in the
file.

The current dataset remains weak for rare reasons: it contains only a handful
of obfuscation and privilege-escalation examples. Add diverse, reviewed records
for those categories before treating their reason metrics as meaningful.

Regenerate the synthetic datasets with:

```powershell
python generate_dataset.py --count 1500 --seed 42
```

Rebuild only the normalized malicious and combined files while keeping the
existing safe/risky files:

```powershell
python generate_dataset.py --combine-only
```

## Optional GitHub scraping

`scrape_dataset.py` uses GitHub's API with a personal access token. It does not
clone repositories or run Git commands. Keep the token in an environment
variable and never put it in source code, `.env`, output data, or a commit.

```powershell
$env:GITHUB_TOKEN = "YOUR_TOKEN"
python scrape_dataset.py --per-query 50 --out data/scraped.jsonl
Remove-Item Env:GITHUB_TOKEN
```

Review scraped code and its original license before using or distributing it.
The default scraped output is ignored by Git.

