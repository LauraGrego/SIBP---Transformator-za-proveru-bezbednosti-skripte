import argparse

import torch

from .training import train_new_model


def train_on_cpu(args: argparse.Namespace) -> None:
    """Train a new model entirely on the CPU without mixed precision."""
    train_new_model(
        args,
        device=torch.device("cpu"),
        useMixedPrecision=False,
    )
