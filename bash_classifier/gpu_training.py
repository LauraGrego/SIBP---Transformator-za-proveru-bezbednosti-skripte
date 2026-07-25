import argparse

import torch

from .training import train_new_model


def require_cuda_device() -> torch.device:
    """Return the CUDA device or explain why GPU training cannot start."""
    if not torch.cuda.is_available():
        build = "CPU-only" if torch.version.cuda is None else "CUDA-enabled"
        raise RuntimeError(
            "GPU training requires CUDA, but CUDA is unavailable "
            f"({build} PyTorch build). Install a CUDA-enabled PyTorch build."
        )
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    return torch.device("cuda")


def train_on_gpu(args: argparse.Namespace) -> None:
    """Train on an NVIDIA GPU using CUDA mixed precision and pinned transfers."""
    device = require_cuda_device()
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    train_new_model(args, device=device, useMixedPrecision=True)
