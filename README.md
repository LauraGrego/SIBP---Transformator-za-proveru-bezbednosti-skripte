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
|-- main.py                         Transformer training and inference
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

## Training

The default command loads `data/safe_risky_combined.jsonl`, creates a
reproducible stratified 80/20 train/test split, trains the tokenizer only on
the training portion, and trains the Transformer:

```powershell
python main.py train
```

The generated files are saved locally and ignored by Git:

```text
artifacts/bash_transformer.pt
artifacts/bash_tokenizer.json
```

For a faster demonstration run:

```powershell
python main.py train `
  --epochs 3 `
  --context-size 256 `
  --model-dimension 128 `
  --heads 4 `
  --blocks 2 `
  --feed-forward-dimension 512
```

## Classifying scripts

Read a script file as text and classify it:

```powershell
python main.py predict --script-file showcase_scripts\safe_backup.bash.txt
```

Or provide literal text:

```powershell
python main.py predict --text "sudo systemctl restart example.service"
```

The output includes the predicted label, confidence, and all three label
probabilities. A malicious result also includes a predicted reason and a short
explanation.

Evaluate the saved checkpoint on the same reproducible test split:

```powershell
python main.py evaluate
```

Use `--checkpoint` and `--tokenizer` with `train`, `evaluate`, or `predict` to
choose different artifact paths.

## Dataset format

The combined dataset is JSON Lines: one JSON object per line.

```json
{"id":"safe-00001","label":"safe","script":"#!/bin/bash\necho hello\n","category":"","description":""}
```

Malicious records use a non-empty `category`, which trains the model's reason
head. Safe and risky records normally leave `category` empty.

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

