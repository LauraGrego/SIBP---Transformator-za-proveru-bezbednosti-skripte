from __future__ import annotations

import json
import random
import re
from functools import partial
from pathlib import Path
from typing import Iterable, Sequence

import torch
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from .config import CLS_TOKEN, LABEL_TO_ID, PAD_TOKEN, UNK_TOKEN, ModelConfig, TrainingConfig


def load_jsonl(path: Path) -> list[dict]:
    """Read and validate labeled JSONL rows without executing script text."""
    records: list[dict] = []
    with path.open(encoding="utf-8") as inputFile:
        for lineNumber, line in enumerate(inputFile, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{lineNumber}: {exc}") from exc
            if record.get("label") not in LABEL_TO_ID:
                raise ValueError(
                    f"Unknown label at {path}:{lineNumber}: {record.get('label')!r}"
                )
            if not isinstance(record.get("script"), str) or not record["script"].strip():
                raise ValueError(f"Missing script text at {path}:{lineNumber}")
            records.append(record)
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def stratified_split(
    records: Sequence[dict], testFraction: float, seed: int
) -> tuple[list[dict], list[dict]]:
    """Split template families together to prevent synthetic test leakage."""
    if not 0.0 < testFraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    grouped: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        reason = record.get("category", "") if record["label"] == "malicious" else ""
        grouped.setdefault((record["label"], reason), []).append(record)

    rng = random.Random(seed)
    trainingRecords: list[dict] = []
    testRecords: list[dict] = []
    for rows in grouped.values():
        templateGroups: dict[str, list[dict]] = {}
        for record in rows:
            fingerprint = template_fingerprint(record["script"])
            templateGroups.setdefault(fingerprint, []).append(record)
        families = list(templateGroups.values())
        rng.shuffle(families)
        targetSize = max(1, round(len(rows) * testFraction)) if len(families) > 1 else 0
        selectedSize = 0
        for familyIndex, family in enumerate(families):
            familiesLeft = len(families) - familyIndex
            if selectedSize < targetSize and familiesLeft > 1:
                testRecords.extend(family)
                selectedSize += len(family)
            else:
                trainingRecords.extend(family)
    rng.shuffle(trainingRecords)
    rng.shuffle(testRecords)
    return trainingRecords, testRecords


def template_fingerprint(script: str) -> str:
    """Normalize generated details so variants of one template stay together."""
    normalizedScript = script.lower()
    normalizedScript = re.sub(r"https?://[^\s'\"]+", " URL ", normalizedScript)
    normalizedScript = re.sub(
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b", " IP ", normalizedScript
    )
    normalizedScript = re.sub(r"(['\"]).*?\1", " STRING ", normalizedScript)
    normalizedScript = re.sub(r"\b\d+\b", " NUMBER ", normalizedScript)
    return re.sub(r"\s+", " ", normalizedScript).strip()


def get_all_scripts(records: Iterable[dict]) -> Iterable[str]:
    """Yield script text to the tokenizer trainer without copying the dataset."""
    for record in records:
        yield record["script"]


def build_tokenizer(
    trainingRecords: Sequence[dict],
    outputPath: Path | None,
    minimumFrequency: int,
    vocabularySize: int,
) -> Tokenizer:
    """Train a vocabulary from training rows and optionally save it."""
    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        special_tokens=[PAD_TOKEN, UNK_TOKEN, CLS_TOKEN],
        min_frequency=minimumFrequency,
        vocab_size=vocabularySize,
        initial_alphabet=ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(get_all_scripts(trainingRecords), trainer=trainer)
    if outputPath is not None:
        outputPath.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(outputPath))
    return tokenizer


def get_reason_names(records: Sequence[dict]) -> tuple[str, ...]:
    """Return the malicious-reason vocabulary present in training rows."""
    reasons = {
        record.get("category", "").strip()
        for record in records
        if record["label"] == "malicious" and record.get("category", "").strip()
    }
    if not reasons:
        raise ValueError("Malicious records must have non-empty categories")
    return tuple(sorted(reasons))


def required_token_id(tokenizer: Tokenizer, token: str) -> int:
    """Return a required special-token ID or fail for an invalid tokenizer."""
    tokenId = tokenizer.token_to_id(token)
    if tokenId is None:
        raise ValueError(f"Tokenizer does not contain required token {token!r}")
    return tokenId


class BashDataset(Dataset):
    """Pre-tokenize scripts once and expose variable-length training samples."""

    def __init__(
        self,
        records: Sequence[dict],
        tokenizer: Tokenizer,
        contextSize: int,
        reasonToId: dict[str, int],
    ) -> None:
        """Encode scripts and retain primary and malicious-reason targets."""
        self.records = records
        clsId = required_token_id(tokenizer, CLS_TOKEN)
        encodings = tokenizer.encode_batch([record["script"] for record in records])
        self.tokenIds = [
            [clsId, *encoding.ids[: contextSize - 1]] for encoding in encodings
        ]
        self.reasonToId = reasonToId

    def __len__(self) -> int:
        """Return the number of scripts."""
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one encoded script and its targets."""
        record = self.records[index]
        reasonId = -100
        if record["label"] == "malicious":
            reasonId = self.reasonToId.get(record.get("category", ""), -100)
        return {
            "input_ids": torch.tensor(self.tokenIds[index], dtype=torch.long),
            "label": torch.tensor(LABEL_TO_ID[record["label"]], dtype=torch.long),
            "reason": torch.tensor(reasonId, dtype=torch.long),
        }


def collate_batch(
    samples: Sequence[dict[str, torch.Tensor]], padId: int
) -> dict[str, torch.Tensor]:
    """Pad only to the longest script in the current batch."""
    inputIds = pad_sequence(
        [sample["input_ids"] for sample in samples],
        batch_first=True,
        padding_value=padId,
    )
    return {
        "input_ids": inputIds,
        "attention_mask": inputIds.ne(padId),
        "label": torch.stack([sample["label"] for sample in samples]),
        "reason": torch.stack([sample["reason"] for sample in samples]),
    }


def make_data_loader(
    dataset: BashDataset,
    tokenizer: Tokenizer,
    batchSize: int,
    device: torch.device,
    shuffle: bool = False,
    seed: int = 42,
) -> DataLoader:
    """Create a dynamically padded loader optimized for the selected device."""
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batchSize,
        shuffle=shuffle,
        generator=generator,
        pin_memory=device.type == "cuda",
        collate_fn=partial(
            collate_batch, padId=required_token_id(tokenizer, PAD_TOKEN)
        ),
    )


def move_batch_to_device(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    """Move a complete batch to CPU or GPU."""
    return {
        name: tensor.to(device, non_blocking=device.type == "cuda")
        for name, tensor in batch.items()
    }


def prepare_split_data(
    datasetPath: Path,
    tokenizerPath: Path,
    modelConfig: ModelConfig,
    trainingConfig: TrainingConfig,
    buildNewTokenizer: bool,
) -> tuple[BashDataset, BashDataset, Tokenizer, tuple[str, ...]]:
    """Prepare matching training and held-out datasets."""
    records = load_jsonl(datasetPath)
    trainingRecords, testRecords = stratified_split(
        records, trainingConfig.testFraction, trainingConfig.seed
    )
    reasonNames = get_reason_names(trainingRecords)
    reasonToId = {name: index for index, name in enumerate(reasonNames)}
    tokenizer = (
        build_tokenizer(
            trainingRecords,
            None,
            trainingConfig.minimumTokenFrequency,
            trainingConfig.maximumVocabularySize,
        )
        if buildNewTokenizer
        else Tokenizer.from_file(str(tokenizerPath))
    )
    return (
        BashDataset(trainingRecords, tokenizer, modelConfig.contextSize, reasonToId),
        BashDataset(testRecords, tokenizer, modelConfig.contextSize, reasonToId),
        tokenizer,
        reasonNames,
    )


def prepare_evaluation_data(
    datasetPath: Path,
    tokenizer: Tokenizer,
    modelConfig: ModelConfig,
    reasonNames: Sequence[str],
) -> BashDataset:
    """Encode every row of an independent labeled evaluation dataset."""
    records = load_jsonl(datasetPath)
    knownReasons = set(reasonNames)
    unknownReasons = {
        record.get("category", "").strip()
        for record in records
        if record["label"] == "malicious"
        and record.get("category", "").strip() not in knownReasons
    }
    if unknownReasons:
        raise ValueError(
            "Unknown evaluation reason categories: " + ", ".join(sorted(unknownReasons))
        )
    reasonToId = {name: index for index, name in enumerate(reasonNames)}
    return BashDataset(records, tokenizer, modelConfig.contextSize, reasonToId)
