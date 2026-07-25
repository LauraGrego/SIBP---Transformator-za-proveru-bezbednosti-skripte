import argparse
import math
import random
from collections import Counter
from contextlib import nullcontext
from typing import Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import LABELS, TrainingConfig
from .config import ModelConfig
from .data import make_data_loader, move_batch_to_device, prepare_split_data
from .model import BashTransformerClassifier, make_model, save_checkpoint


def set_seed(seed: int) -> None:
    """Seed Python and PyTorch for reproducible splitting and optimization."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def class_weights(records: Sequence[dict], device: torch.device) -> torch.Tensor:
    """Compute inverse-frequency weights for imbalanced primary labels."""
    counts = Counter(record["label"] for record in records)
    total = len(records)
    weights = [total / (len(LABELS) * counts[label]) for label in LABELS]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def reason_class_weights(
    records: Sequence[dict], reasonNames: Sequence[str], device: torch.device
) -> torch.Tensor:
    """Reduce domination by common malicious reasons using square-root weights."""
    counts = Counter(
        record.get("category", "")
        for record in records
        if record["label"] == "malicious"
    )
    total = sum(counts[name] for name in reasonNames)
    weights = [
        math.sqrt(total / (len(reasonNames) * max(counts[name], 1)))
        for name in reasonNames
    ]
    for name in reasonNames:
        if counts[name] < 20:
            print(
                f"Warning: reason '{name}' has only {counts[name]} training samples; "
                "its metrics will not be reliable."
            )
    return torch.tensor(weights, dtype=torch.float32, device=device)


def make_gradient_scaler(enabled: bool):
    """Create a CUDA loss scaler while retaining PyTorch 2.x compatibility."""
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def train_epoch(
    model: BashTransformerClassifier,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    labelLossFunction: nn.Module,
    reasonLossFunction: nn.Module,
    reasonLossWeight: float,
    device: torch.device,
    useMixedPrecision: bool,
    scaler,
) -> float:
    """Train one epoch on CPU or GPU and return mean sample loss."""
    model.train()
    totalLoss = 0.0
    totalSamples = 0
    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        labels = batch["label"]
        reasons = batch["reason"]
        precisionContext = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if useMixedPrecision
            else nullcontext()
        )
        with precisionContext:
            labelLogits, reasonLogits = model(
                batch["input_ids"], batch["attention_mask"]
            )
            loss = labelLossFunction(labelLogits, labels)
            reasonMask = reasons.ne(-100)
            if reasonMask.any():
                reasonLoss = reasonLossFunction(
                    reasonLogits[reasonMask], reasons[reasonMask]
                )
                loss = loss + reasonLossWeight * reasonLoss

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        batchSize = labels.size(0)
        totalLoss += loss.item() * batchSize
        totalSamples += batchSize
    return totalLoss / max(totalSamples, 1)


def train_model(
    model: BashTransformerClassifier,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    labelLossFunction: nn.Module,
    reasonLossFunction: nn.Module,
    trainingConfig: TrainingConfig,
    device: torch.device,
    useMixedPrecision: bool,
) -> list[float]:
    """Train all configured epochs and return the loss history."""
    scaler = make_gradient_scaler(useMixedPrecision)
    losses = []
    for epoch in range(1, trainingConfig.epochs + 1):
        loss = train_epoch(
            model,
            dataloader,
            optimizer,
            labelLossFunction,
            reasonLossFunction,
            trainingConfig.reasonLossWeight,
            device,
            useMixedPrecision,
            scaler,
        )
        losses.append(loss)
        print(f"Epoch {epoch:02d}/{trainingConfig.epochs}: loss={loss:.4f}")
    return losses


def train_new_model(
    args: argparse.Namespace,
    device: torch.device,
    useMixedPrecision: bool,
) -> None:

    modelConfig = ModelConfig(
        contextSize=args.contextSize,
        modelDimension=args.modelDimension,
        numberOfHeads=args.heads,
        numberOfBlocks=args.blocks,
        feedForwardDimension=args.feedForwardDimension,
        dropout=args.dropout,
    )
    trainingConfig = TrainingConfig(
        seed=args.seed,
        testFraction=args.testFraction,
        batchSize=args.batchSize,
        epochs=args.epochs,
        learningRate=args.learningRate,
        weightDecay=args.weightDecay,
    )
    set_seed(trainingConfig.seed)
    trainingDataset, _, tokenizer, reasonNames = prepare_split_data(
        args.dataset,
        args.tokenizer,
        modelConfig,
        trainingConfig,
        buildNewTokenizer=True,
    )
    trainingLoader = make_data_loader(
        trainingDataset,
        tokenizer,
        trainingConfig.batchSize,
        device,
        shuffle=True,
        seed=trainingConfig.seed,
    )
    model = make_model(tokenizer, reasonNames, modelConfig, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=trainingConfig.learningRate,
        weight_decay=trainingConfig.weightDecay,
    )
    labelLoss = nn.CrossEntropyLoss(
        weight=class_weights(trainingDataset.records, device),
        label_smoothing=0.05,
    )
    reasonLoss = nn.CrossEntropyLoss(
        weight=reason_class_weights(trainingDataset.records, reasonNames, device),
        label_smoothing=0.05,
    )

    print(f"Using device: {device}")
    print(f"Training samples: {len(trainingDataset)}")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    train_model(
        model,
        trainingLoader,
        optimizer,
        labelLoss,
        reasonLoss,
        trainingConfig,
        device,
        useMixedPrecision,
    )
    save_checkpoint(
        args.checkpoint,
        model,
        tokenizer,
        modelConfig,
        trainingConfig,
        reasonNames,
    )
    args.tokenizer.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(args.tokenizer))
    print(f"Saved checkpoint to {args.checkpoint}")
    print(f"Saved tokenizer to {args.tokenizer}")
