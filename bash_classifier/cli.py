import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = PROJECT_ROOT / "data" / "safe_risky_combined.jsonl"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "artifacts" / "bash_transformer.pt"
DEFAULT_TOKENIZER = PROJECT_ROOT / "artifacts" / "bash_tokenizer.json"
DEFAULT_SHOWCASE_DIRECTORY = PROJECT_ROOT / "showcase_scripts"
DEFAULT_EVALUATION_MANIFEST = DEFAULT_SHOWCASE_DIRECTORY / "expected_labels.json"


def build_argument_parser() -> argparse.ArgumentParser:
    """Define the CPU, GPU, and showcase-prediction commands and options."""
    parser = argparse.ArgumentParser(
        description="Train, test, evaluate, or use the Bash classifier."
    )
    parser.add_argument(
        "command",
        choices=("cpu", "gpu", "predict"),
    )
    parser.add_argument(
        "--dataset", type=Path, default=DEFAULT_DATASET
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT
    )
    parser.add_argument(
        "--tokenizer", type=Path, default=DEFAULT_TOKENIZER
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument(
        "--showcase-directory",
        dest="showcaseDirectory",
        type=Path,
        default=DEFAULT_SHOWCASE_DIRECTORY,
    )
    parser.add_argument(
        "--minimum-confidence", dest="minimumConfidence", type=float, default=0.65
    )
    parser.add_argument(
        "--evaluation-manifest",
        dest="evaluationManifest",
        type=Path,
        default=DEFAULT_EVALUATION_MANIFEST,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", dest="testFraction", type=float, default=0.2)
    parser.add_argument("--batch-size", dest="batchSize", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", dest="learningRate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", dest="weightDecay", type=float, default=1e-4)
    parser.add_argument("--context-size", dest="contextSize", type=int, default=384)
    parser.add_argument("--model-dimension", dest="modelDimension", type=int, default=192)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument(
        "--feed-forward-dimension", dest="feedForwardDimension", type=int, default=768
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser
