import math
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
from tokenizers import Tokenizer

from .config import (
    LABELS,
    PAD_TOKEN,
    ModelConfig,
    TrainingConfig,
    model_config_from_dict,
)
from .data import required_token_id


class PositionalEncoding(nn.Module):
    """Add sinusoidal position information to token embeddings."""

    def __init__(self, modelDimension: int, contextSize: int) -> None:
        """Precompute positions up to the maximum context length."""
        super().__init__()
        positions = torch.arange(contextSize).unsqueeze(1)
        frequency = torch.exp(
            torch.arange(0, modelDimension, 2)
            * (-math.log(10_000.0) / modelDimension)
        )
        encoding = torch.zeros(contextSize, modelDimension)
        encoding[:, 0::2] = torch.sin(positions * frequency)
        encoding[:, 1::2] = torch.cos(positions * frequency)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Add the appropriate position vectors to a batch."""
        return embeddings + self.encoding[:, : embeddings.size(1)]


class BashTransformerClassifier(nn.Module):
    """Encoder-only Transformer with label and malicious-reason heads."""

    def __init__(
        self,
        vocabularySize: int,
        numberOfReasons: int,
        config: ModelConfig,
        padId: int,
    ) -> None:
        """Construct the encoder and both output heads."""
        super().__init__()
        if config.modelDimension % 2:
            raise ValueError("model_dimension must be even")
        if config.modelDimension % config.numberOfHeads:
            raise ValueError("model_dimension must be divisible by number_of_heads")
        self.modelDimension = config.modelDimension
        self.embedding = nn.Embedding(
            vocabularySize, config.modelDimension, padding_idx=padId
        )
        self.position = PositionalEncoding(config.modelDimension, config.contextSize)
        self.dropout = nn.Dropout(config.dropout)
        encoderBlock = nn.TransformerEncoderLayer(
            d_model=config.modelDimension,
            nhead=config.numberOfHeads,
            dim_feedforward=config.feedForwardDimension,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoderBlock,
            num_layers=config.numberOfBlocks,
            norm=nn.LayerNorm(config.modelDimension),
            enable_nested_tensor=False,
        )
        self.labelHead = nn.Linear(config.modelDimension, len(LABELS))
        self.reasonHead = nn.Linear(config.modelDimension, numberOfReasons)
        self.initialize_parameters()

    def initialize_parameters(self) -> None:
        """Initialize matrix parameters with Xavier uniform values."""
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(
        self, inputIds: torch.Tensor, attentionMask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return primary-label and malicious-reason logits for a batch."""
        embeddings = self.embedding(inputIds) * math.sqrt(self.modelDimension)
        embeddings = self.dropout(self.position(embeddings))
        encoded = self.encoder(embeddings, src_key_padding_mask=~attentionMask)
        mask = attentionMask.unsqueeze(-1).to(encoded.dtype)
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.labelHead(pooled), self.reasonHead(pooled)


def make_model(
    tokenizer: Tokenizer,
    reasonNames: Sequence[str],
    modelConfig: ModelConfig,
    device: torch.device,
) -> BashTransformerClassifier:
    """Construct a model compatible with a tokenizer and move it to a device."""
    return BashTransformerClassifier(
        vocabularySize=tokenizer.get_vocab_size(),
        numberOfReasons=len(reasonNames),
        config=modelConfig,
        padId=required_token_id(tokenizer, PAD_TOKEN),
    ).to(device)


def save_checkpoint(
    path: Path,
    model: BashTransformerClassifier,
    tokenizer: Tokenizer,
    modelConfig: ModelConfig,
    trainingConfig: TrainingConfig,
    reasonNames: Sequence[str],
    optimizer: torch.optim.Optimizer | None = None,
    completedEpochs: int = 0,
    scaler=None,
    lossHistory: Sequence[float] = (),
) -> None:
    """Save inference metadata and optional state required to resume training."""
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "tokenizer_json": tokenizer.to_str(),
        "model_config": asdict(modelConfig),
        "training_config": asdict(trainingConfig),
        "labels": list(LABELS),
        "reason_names": list(reasonNames),
        "completed_epochs": completedEpochs,
        "loss_history": list(lossHistory),
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()
    torch.save(checkpoint, path)


def load_checkpoint(
    checkpointPath: Path, tokenizerPath: Path, device: torch.device
) -> tuple[BashTransformerClassifier, Tokenizer, tuple[str, ...], ModelConfig]:
    """Reconstruct a trained model and tokenizer on the requested device."""
    checkpoint = torch.load(checkpointPath, map_location=device, weights_only=True)
    if "tokenizer_json" in checkpoint:
        tokenizer = Tokenizer.from_str(checkpoint["tokenizer_json"])
    else:
        tokenizer = Tokenizer.from_file(str(tokenizerPath))
    if tuple(checkpoint["labels"]) != LABELS:
        raise ValueError("Checkpoint label order does not match this program")
    modelConfig = model_config_from_dict(checkpoint["model_config"])
    reasonNames = tuple(checkpoint["reason_names"])
    model = make_model(tokenizer, reasonNames, modelConfig, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, tokenizer, reasonNames, modelConfig
