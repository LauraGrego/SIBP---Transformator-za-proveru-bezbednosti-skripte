import argparse
from copy import copy

from bash_classifier.cli import build_argument_parser
from bash_classifier.cpu_training import train_on_cpu
from bash_classifier.evaluation import evaluate_real_world_examples, test_saved_model
from bash_classifier.gpu_training import train_on_gpu
from bash_classifier.prediction import predict_showcase_directory


def train_test_evaluate_cpu(workflowArgs: argparse.Namespace) -> None:
    """Train on CPU, test the held-out split, then evaluate the full dataset."""
    cpuArgs = copy(workflowArgs)
    cpuArgs.device = "cpu"
    train_on_cpu(cpuArgs)
    test_saved_model(cpuArgs)
    evaluate_real_world_examples(cpuArgs)


def train_test_evaluate_gpu(workflowArgs: argparse.Namespace) -> None:
    """Train on CUDA GPU, then test and evaluate on the same GPU."""
    gpuArgs = copy(workflowArgs)
    gpuArgs.device = "cuda"
    train_on_gpu(gpuArgs)
    test_saved_model(gpuArgs)
    evaluate_real_world_examples(gpuArgs)


def predict_showcase_scripts(workflowArgs: argparse.Namespace) -> None:
    """Display the filename and saved-model prediction for every showcase file."""
    predict_showcase_directory(workflowArgs)


if __name__ == "__main__":
    commandArgs = build_argument_parser().parse_args()
    if commandArgs.command == "cpu":
        train_test_evaluate_cpu(commandArgs)
    elif commandArgs.command == "gpu":
        train_test_evaluate_gpu(commandArgs)
    else:
        predict_showcase_scripts(commandArgs)
