import sys

import torch


def choose_inference_device(requested: str) -> torch.device:
    """Choose CUDA when requested/available, with a safe automatic CPU fallback."""
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        else:
            if torch.version.cuda is None:
                print(
                    "CUDA unavailable: this is a CPU-only PyTorch build; using CPU.",
                    file=sys.stderr,
                )
            requested = "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)
