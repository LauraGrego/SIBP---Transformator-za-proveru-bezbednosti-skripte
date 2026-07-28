from dataclasses import dataclass
from typing import Any, Mapping


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
    """Transformer architecture"""

    contextSize: int = 384
    modelDimension: int = 192
    numberOfHeads: int = 6
    numberOfBlocks: int = 4
    feedForwardDimension: int = 768
    dropout: float = 0.1


@dataclass(frozen=True)
class TrainingConfig:
    """Dataset splitting and optimizer settings."""

    seed: int = 42
    testFraction: float = 0.2
    batchSize: int = 32
    epochs: int = 12
    learningRate: float = 3e-4
    weightDecay: float = 1e-4
    minimumTokenFrequency: int = 2
    maximumVocabularySize: int = 30_000
    reasonLossWeight: float = 0.5


def model_config_from_dict(values: Mapping[str, Any]) -> ModelConfig:
    """Reconstruct model settings stored as primitive checkpoint values."""
    return ModelConfig(**dict(values))


def training_config_from_dict(values: Mapping[str, Any]) -> TrainingConfig:
    """Reconstruct training settings stored as primitive checkpoint values."""
    return TrainingConfig(**dict(values))
