#!/usr/bin/env python3
"""Train and use an encoder-only Transformer for Bash-script classification.

The model predicts one of ``safe``, ``risky``, or ``malicious``.  A second
classification head predicts the malicious category (the reason) for samples
whose primary label is malicious.  Dataset scripts are treated strictly as
text and are never executed.

Examples:
    python main.py train
    python main.py evaluate
    python main.py predict --script-file example.sh
    python main.py predict --text "sudo rm -rf /tmp/example"
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn as nn
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from torch.utils.data import DataLoader, Dataset


LABELS = ("safe", "risky", "malicious")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
CLS_TOKEN = "[CLS]"

REASON_EXPLANATIONS = {
    "defense_evasion": "attempts to weaken, disable, or bypass security controls",
    "exfiltration": "attempts to transfer sensitive data out of the system",
    "obfuscation": "hides or encodes behavior to make it harder to inspect",
    "persistence": "attempts to retain access across sessions or reboots",
    "privilege_escalation": "attempts to obtain elevated privileges",
    "recon": "collects system, user, process, or network information",
    "reverse_shell": "attempts to open an interactive shell to a remote host",
}


@dataclass(frozen=True)
class ModelConfig:
    context_size: int = 384
    model_dimension: int = 192
    number_of_heads: int = 6
    number_of_blocks: int = 4
    feed_forward_dimension: int = 768
    dropout: float = 0.1


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    test_fraction: float = 0.2
    batch_size: int = 32
    epochs: int = 12
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    minimum_token_frequency: int = 2
    maximum_vocabulary_size: int = 30_000
    reason_loss_weight: float = 0.5


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path: Path) -> list[dict]:
    """Read and validate training rows without evaluating their script text."""
    records: list[dict] = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if record.get("label") not in LABEL_TO_ID:
                raise ValueError(
                    f"Unknown label at {path}:{line_number}: {record.get('label')!r}"
                )
            if not isinstance(record.get("script"), str) or not record["script"].strip():
                raise ValueError(f"Missing script text at {path}:{line_number}")
            records.append(record)
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def stratified_split(
    records: Sequence[dict], test_fraction: float, seed: int
) -> tuple[list[dict], list[dict]]:
    """Preserve primary labels and malicious-reason categories in both splits."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")

    grouped: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        label = record["label"]
        reason = record.get("category", "") if label == "malicious" else ""
        grouped.setdefault((label, reason), []).append(record)

    rng = random.Random(seed)
    training_records: list[dict] = []
    test_records: list[dict] = []
    for (label, reason), rows in sorted(grouped.items()):
        rng.shuffle(rows)
        # A one-row reason class cannot exist in both sets; retain it for
        # training so the reason head at least learns that category.
        test_size = max(1, round(len(rows) * test_fraction)) if len(rows) > 1 else 0
        test_size = min(test_size, len(rows) - 1)
        test_records.extend(rows[:test_size])
        training_records.extend(rows[test_size:])

    rng.shuffle(training_records)
    rng.shuffle(test_records)
    return training_records, test_records


def get_all_scripts(records: Iterable[dict]) -> Iterable[str]:
    for record in records:
        yield record["script"]


def build_tokenizer(
    training_records: Sequence[dict],
    output_path: Path,
    minimum_frequency: int,
    vocabulary_size: int,
) -> Tokenizer:
    """Build the vocabulary from training text only, preventing test leakage."""
    tokenizer = Tokenizer(WordLevel(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(
        special_tokens=[PAD_TOKEN, UNK_TOKEN, CLS_TOKEN],
        min_frequency=minimum_frequency,
        vocab_size=vocabulary_size,
    )
    tokenizer.train_from_iterator(get_all_scripts(training_records), trainer=trainer)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))
    return tokenizer


def get_reason_names(records: Sequence[dict]) -> tuple[str, ...]:
    reasons = {
        record.get("category", "").strip()
        for record in records
        if record["label"] == "malicious" and record.get("category", "").strip()
    }
    if not reasons:
        raise ValueError("Malicious records must have non-empty categories")
    return tuple(sorted(reasons))


class BashDataset(Dataset):
    """Convert script text to fixed-length encoder input tensors."""

    def __init__(
        self,
        records: Sequence[dict],
        tokenizer: Tokenizer,
        context_size: int,
        reason_to_id: dict[str, int],
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.context_size = context_size
        self.reason_to_id = reason_to_id
        self.pad_id = required_token_id(tokenizer, PAD_TOKEN)
        self.cls_id = required_token_id(tokenizer, CLS_TOKEN)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        token_ids = self.tokenizer.encode(record["script"]).ids
        token_ids = [self.cls_id] + token_ids[: self.context_size - 1]
        padding_length = self.context_size - len(token_ids)
        input_ids = token_ids + [self.pad_id] * padding_length
        attention_mask = [1] * len(token_ids) + [0] * padding_length

        reason_id = -100
        if record["label"] == "malicious":
            reason_id = self.reason_to_id.get(record.get("category", ""), -100)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
            "label": torch.tensor(LABEL_TO_ID[record["label"]], dtype=torch.long),
            "reason": torch.tensor(reason_id, dtype=torch.long),
        }


def required_token_id(tokenizer: Tokenizer, token: str) -> int:
    token_id = tokenizer.token_to_id(token)
    if token_id is None:
        raise ValueError(f"Tokenizer does not contain required token {token!r}")
    return token_id


class PositionalEncoding(nn.Module):
    def __init__(self, model_dimension: int, context_size: int) -> None:
        super().__init__()
        positions = torch.arange(context_size).unsqueeze(1)
        frequency = torch.exp(
            torch.arange(0, model_dimension, 2)
            * (-math.log(10_000.0) / model_dimension)
        )
        encoding = torch.zeros(context_size, model_dimension)
        encoding[:, 0::2] = torch.sin(positions * frequency)
        encoding[:, 1::2] = torch.cos(positions * frequency)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return embeddings + self.encoding[:, : embeddings.size(1)]


class BashTransformerClassifier(nn.Module):
    """Encoder-only Transformer with primary-label and malicious-reason heads."""

    def __init__(
        self,
        vocabulary_size: int,
        number_of_reasons: int,
        config: ModelConfig,
        pad_id: int,
    ) -> None:
        super().__init__()
        if config.model_dimension % 2:
            raise ValueError("model_dimension must be even for positional encoding")
        if config.model_dimension % config.number_of_heads:
            raise ValueError("model_dimension must be divisible by number_of_heads")
        self.model_dimension = config.model_dimension
        self.embedding = nn.Embedding(
            vocabulary_size, config.model_dimension, padding_idx=pad_id
        )
        self.position = PositionalEncoding(config.model_dimension, config.context_size)
        self.dropout = nn.Dropout(config.dropout)

        encoder_block = nn.TransformerEncoderLayer(
            d_model=config.model_dimension,
            nhead=config.number_of_heads,
            dim_feedforward=config.feed_forward_dimension,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_block,
            num_layers=config.number_of_blocks,
            norm=nn.LayerNorm(config.model_dimension),
        )
        self.label_head = nn.Linear(config.model_dimension, len(LABELS))
        self.reason_head = nn.Linear(config.model_dimension, number_of_reasons)
        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.embedding(input_ids) * math.sqrt(self.model_dimension)
        embeddings = self.dropout(self.position(embeddings))
        encoded = self.encoder(
            embeddings,
            src_key_padding_mask=~attention_mask,
        )

        # Masked mean pooling uses all non-padding code tokens, including [CLS].
        mask = attention_mask.unsqueeze(-1).to(encoded.dtype)
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.label_head(pooled), self.reason_head(pooled)


def make_model(
    tokenizer: Tokenizer,
    reason_names: Sequence[str],
    model_config: ModelConfig,
    device: torch.device,
) -> BashTransformerClassifier:
    return BashTransformerClassifier(
        vocabulary_size=tokenizer.get_vocab_size(),
        number_of_reasons=len(reason_names),
        config=model_config,
        pad_id=required_token_id(tokenizer, PAD_TOKEN),
    ).to(device)


def class_weights(records: Sequence[dict], device: torch.device) -> torch.Tensor:
    counts = Counter(record["label"] for record in records)
    total = len(records)
    weights = [total / (len(LABELS) * counts[label]) for label in LABELS]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_epoch(
    model: BashTransformerClassifier,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    label_loss_function: nn.Module,
    reason_loss_function: nn.Module,
    reason_loss_weight: float,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        reasons = batch["reason"].to(device)

        label_logits, reason_logits = model(input_ids, attention_mask)
        loss = label_loss_function(label_logits, labels)
        reason_mask = reasons.ne(-100)
        if reason_mask.any():
            reason_loss = reason_loss_function(
                reason_logits[reason_mask], reasons[reason_mask]
            )
            loss = loss + reason_loss_weight * reason_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate_model(
    model: BashTransformerClassifier,
    dataloader: DataLoader,
    device: torch.device,
    reason_names: Sequence[str],
) -> dict:
    model.eval()
    confusion = torch.zeros(len(LABELS), len(LABELS), dtype=torch.long)
    reason_correct = 0
    reason_total = 0

    for batch in dataloader:
        labels = batch["label"].to(device)
        reasons = batch["reason"].to(device)
        label_logits, reason_logits = model(
            batch["input_ids"].to(device), batch["attention_mask"].to(device)
        )
        predictions = label_logits.argmax(dim=1)
        for expected, predicted in zip(labels.cpu(), predictions.cpu()):
            confusion[expected, predicted] += 1

        reason_mask = reasons.ne(-100)
        if reason_mask.any():
            reason_predictions = reason_logits[reason_mask].argmax(dim=1)
            reason_correct += (reason_predictions == reasons[reason_mask]).sum().item()
            reason_total += reason_mask.sum().item()

    correct = confusion.diag().sum().item()
    total = confusion.sum().item()
    per_label = {}
    for label_id, label in enumerate(LABELS):
        true_positive = confusion[label_id, label_id].item()
        predicted_total = confusion[:, label_id].sum().item()
        actual_total = confusion[label_id, :].sum().item()
        precision = true_positive / predicted_total if predicted_total else 0.0
        recall = true_positive / actual_total if actual_total else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {"precision": precision, "recall": recall, "f1": f1}

    return {
        "accuracy": correct / total if total else 0.0,
        "per_label": per_label,
        "confusion_matrix": confusion.tolist(),
        "confusion_matrix_labels": list(LABELS),
        "reason_accuracy": reason_correct / reason_total if reason_total else 0.0,
        "reason_samples": reason_total,
        "reason_names": list(reason_names),
    }


def print_metrics(metrics: dict) -> None:
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    for label, values in metrics["per_label"].items():
        print(
            f"  {label:9s} precision={values['precision']:.4f} "
            f"recall={values['recall']:.4f} f1={values['f1']:.4f}"
        )
    print(
        f"Malicious-reason accuracy: {metrics['reason_accuracy']:.4f} "
        f"({metrics['reason_samples']} samples)"
    )
    print("Confusion matrix rows=true, columns=predicted")
    print("             " + " ".join(f"{label:>9s}" for label in LABELS))
    for label, row in zip(LABELS, metrics["confusion_matrix"]):
        print(f"  {label:9s} " + " ".join(f"{value:9d}" for value in row))


def save_checkpoint(
    path: Path,
    model: BashTransformerClassifier,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    reason_names: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model_config),
            "training_config": asdict(training_config),
            "labels": list(LABELS),
            "reason_names": list(reason_names),
        },
        path,
    )


def load_checkpoint(
    checkpoint_path: Path, tokenizer_path: Path, device: torch.device
) -> tuple[BashTransformerClassifier, Tokenizer, tuple[str, ...], ModelConfig]:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if tuple(checkpoint["labels"]) != LABELS:
        raise ValueError("Checkpoint label order does not match this program")
    model_config = ModelConfig(**checkpoint["model_config"])
    reason_names = tuple(checkpoint["reason_names"])
    model = make_model(tokenizer, reason_names, model_config, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, tokenizer, reason_names, model_config


def prepare_data(
    dataset_path: Path,
    tokenizer_path: Path,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    build_new_tokenizer: bool,
) -> tuple[BashDataset, BashDataset, Tokenizer, tuple[str, ...]]:
    records = load_jsonl(dataset_path)
    training_records, test_records = stratified_split(
        records, training_config.test_fraction, training_config.seed
    )
    reason_names = get_reason_names(training_records)
    reason_to_id = {name: index for index, name in enumerate(reason_names)}

    if build_new_tokenizer:
        tokenizer = build_tokenizer(
            training_records,
            tokenizer_path,
            training_config.minimum_token_frequency,
            training_config.maximum_vocabulary_size,
        )
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    training_dataset = BashDataset(
        training_records, tokenizer, model_config.context_size, reason_to_id
    )
    test_dataset = BashDataset(
        test_records, tokenizer, model_config.context_size, reason_to_id
    )
    return training_dataset, test_dataset, tokenizer, reason_names


def train_command(args: argparse.Namespace) -> None:
    model_config = ModelConfig(
        context_size=args.context_size,
        model_dimension=args.model_dimension,
        number_of_heads=args.heads,
        number_of_blocks=args.blocks,
        feed_forward_dimension=args.feed_forward_dimension,
        dropout=args.dropout,
    )
    training_config = TrainingConfig(
        seed=args.seed,
        test_fraction=args.test_fraction,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    set_seed(training_config.seed)
    device = choose_device(args.device)
    training_dataset, test_dataset, tokenizer, reason_names = prepare_data(
        args.dataset,
        args.tokenizer,
        model_config,
        training_config,
        build_new_tokenizer=True,
    )
    generator = torch.Generator().manual_seed(training_config.seed)
    training_loader = DataLoader(
        training_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        generator=generator,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=training_config.batch_size, shuffle=False
    )

    model = make_model(tokenizer, reason_names, model_config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    label_loss = nn.CrossEntropyLoss(
        weight=class_weights(training_dataset.records, device), label_smoothing=0.05
    )
    reason_loss = nn.CrossEntropyLoss(label_smoothing=0.05)

    print(f"Using device: {device}")
    print(f"Training samples: {len(training_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    for epoch in range(1, training_config.epochs + 1):
        loss = train_epoch(
            model,
            training_loader,
            optimizer,
            label_loss,
            reason_loss,
            training_config.reason_loss_weight,
            device,
        )
        print(f"Epoch {epoch:02d}/{training_config.epochs}: loss={loss:.4f}")

    save_checkpoint(
        args.checkpoint, model, model_config, training_config, reason_names
    )
    print_metrics(evaluate_model(model, test_loader, device, reason_names))
    print(f"Saved checkpoint to {args.checkpoint}")
    print(f"Saved tokenizer to {args.tokenizer}")


def evaluate_command(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    model, tokenizer, reason_names, model_config = load_checkpoint(
        args.checkpoint, args.tokenizer, device
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    training_config = TrainingConfig(**checkpoint["training_config"])
    _, test_dataset, _, split_reason_names = prepare_data(
        args.dataset,
        args.tokenizer,
        model_config,
        training_config,
        build_new_tokenizer=False,
    )
    if tuple(split_reason_names) != tuple(reason_names):
        raise ValueError("Dataset reason categories differ from the checkpoint")
    test_loader = DataLoader(
        test_dataset, batch_size=training_config.batch_size, shuffle=False
    )
    print_metrics(evaluate_model(model, test_loader, device, reason_names))


@torch.no_grad()
def classify_text(
    script: str,
    model: BashTransformerClassifier,
    tokenizer: Tokenizer,
    reason_names: Sequence[str],
    model_config: ModelConfig,
    device: torch.device,
) -> dict:
    dummy_record = {"label": "safe", "script": script, "category": ""}
    dataset = BashDataset(
        [dummy_record],
        tokenizer,
        model_config.context_size,
        {name: index for index, name in enumerate(reason_names)},
    )
    item = dataset[0]
    label_logits, reason_logits = model(
        item["input_ids"].unsqueeze(0).to(device),
        item["attention_mask"].unsqueeze(0).to(device),
    )
    label_probabilities = label_logits.softmax(dim=1)[0]
    label_id = label_probabilities.argmax().item()
    label = LABELS[label_id]
    result = {
        "label": label,
        "confidence": round(label_probabilities[label_id].item(), 6),
        "probabilities": {
            name: round(label_probabilities[index].item(), 6)
            for index, name in enumerate(LABELS)
        },
    }
    if label == "malicious":
        reason_probabilities = reason_logits.softmax(dim=1)[0]
        reason_id = reason_probabilities.argmax().item()
        reason = reason_names[reason_id]
        result["reason"] = reason
        result["reason_confidence"] = round(
            reason_probabilities[reason_id].item(), 6
        )
        result["explanation"] = REASON_EXPLANATIONS.get(
            reason, f"matches patterns associated with {reason.replace('_', ' ')}"
        )
    return result


def predict_command(args: argparse.Namespace) -> None:
    if (args.text is None) == (args.script_file is None):
        raise ValueError("Provide exactly one of --text or --script-file")
    script = args.text
    if args.script_file is not None:
        # Reading only: the file is never passed to a shell or executed.
        script = args.script_file.read_text(encoding="utf-8")

    device = choose_device(args.device)
    model, tokenizer, reason_names, model_config = load_checkpoint(
        args.checkpoint, args.tokenizer, device
    )
    result = classify_text(
        script, model, tokenizer, reason_names, model_config, device
    )
    print(json.dumps(result, indent=2))


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("train", "evaluate", "predict"), nargs="?", default="train"
    )
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/safe_risky_combined.jsonl")
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("artifacts/bash_transformer.pt")
    )
    parser.add_argument(
        "--tokenizer", type=Path, default=Path("artifacts/bash_tokenizer.json")
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--context-size", type=int, default=384)
    parser.add_argument("--model-dimension", type=int, default=192)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--feed-forward-dimension", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--text", help="literal script text to classify")
    parser.add_argument("--script-file", type=Path, help="script file to read as text")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    if args.command == "train":
        train_command(args)
    elif args.command == "evaluate":
        evaluate_command(args)
    else:
        predict_command(args)


if __name__ == "__main__":
    main()
