from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from .config import LABELS
from .data import make_data_loader, move_batch_to_device, prepare_evaluation_data, prepare_split_data
from .devices import choose_inference_device
from .model import BashTransformerClassifier, load_checkpoint
from .prediction import predict_text


def calculate_metrics(
    confusion: torch.Tensor,
    reasonCorrect: int,
    reasonTotal: int,
    reasonNames: tuple[str, ...],
    calibrationCounts: torch.Tensor | None = None,
    calibrationConfidenceSums: torch.Tensor | None = None,
    calibrationCorrectSums: torch.Tensor | None = None,
) -> dict:
    """Calculate class-balanced, reason, and confidence-calibration metrics."""
    confusion = confusion.cpu()
    total = confusion.sum().item()
    perLabel = {}
    recalls = []
    f1Values = []
    for labelId, label in enumerate(LABELS):
        truePositive = confusion[labelId, labelId].item()
        predictedTotal = confusion[:, labelId].sum().item()
        actualTotal = confusion[labelId, :].sum().item()
        precision = truePositive / predictedTotal if predictedTotal else 0.0
        recall = truePositive / actualTotal if actualTotal else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        perLabel[label] = {"precision": precision, "recall": recall, "f1": f1}
        recalls.append(recall)
        f1Values.append(f1)
    calibrationError = 0.0
    if calibrationCounts is not None:
        calibrationCounts = calibrationCounts.cpu()
        calibrationConfidenceSums = calibrationConfidenceSums.cpu()
        calibrationCorrectSums = calibrationCorrectSums.cpu()
        for count, confidenceSum, correctSum in zip(
            calibrationCounts, calibrationConfidenceSums, calibrationCorrectSums
        ):
            if count.item() > 0:
                binConfidence = confidenceSum.item() / count.item()
                binAccuracy = correctSum.item() / count.item()
                calibrationError += (count.item() / total) * abs(
                    binAccuracy - binConfidence
                )
    return {
        "accuracy": confusion.diag().sum().item() / total if total else 0.0,
        "balancedAccuracy": sum(recalls) / len(recalls),
        "macroF1": sum(f1Values) / len(f1Values),
        "expectedCalibrationError": calibrationError,
        "per_label": perLabel,
        "confusion_matrix": confusion.tolist(),
        "reason_accuracy": reasonCorrect / reasonTotal if reasonTotal else 0.0,
        "reason_samples": reasonTotal,
        "reason_names": list(reasonNames),
    }


@torch.inference_mode()
def test_model(
    model: BashTransformerClassifier,
    dataloader: DataLoader,
    device: torch.device,
    reasonNames: tuple[str, ...],
) -> dict:
    """Run labeled samples through a saved model and calculate metrics."""
    model.eval()
    confusion = torch.zeros(
        len(LABELS), len(LABELS), dtype=torch.long, device=device
    )
    reasonCorrect = 0
    reasonTotal = 0
    calibrationCounts = torch.zeros(10, dtype=torch.float32, device=device)
    calibrationConfidenceSums = torch.zeros(10, dtype=torch.float32, device=device)
    calibrationCorrectSums = torch.zeros(10, dtype=torch.float32, device=device)
    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        labels = batch["label"]
        reasons = batch["reason"]
        labelLogits, reasonLogits = model(
            batch["input_ids"], batch["attention_mask"]
        )
        labelProbabilities = labelLogits.softmax(dim=1)
        confidences, predictions = labelProbabilities.max(dim=1)
        indices = labels * len(LABELS) + predictions
        confusion += torch.bincount(
            indices, minlength=len(LABELS) ** 2
        ).reshape(len(LABELS), len(LABELS))
        binIds = (confidences * 10).long().clamp(max=9)
        calibrationCounts += torch.bincount(binIds, minlength=10).float()
        calibrationConfidenceSums += torch.bincount(
            binIds, weights=confidences, minlength=10
        )
        calibrationCorrectSums += torch.bincount(
            binIds,
            weights=predictions.eq(labels).float(),
            minlength=10,
        )
        reasonMask = reasons.ne(-100)
        if reasonMask.any():
            reasonPredictions = reasonLogits[reasonMask].argmax(dim=1)
            reasonCorrect += (reasonPredictions == reasons[reasonMask]).sum().item()
            reasonTotal += reasonMask.sum().item()
    return calculate_metrics(
        confusion,
        reasonCorrect,
        reasonTotal,
        reasonNames,
        calibrationCounts,
        calibrationConfidenceSums,
        calibrationCorrectSums,
    )


def print_metrics(metrics: dict) -> None:
    """Print model metrics and its confusion matrix."""
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Balanced accuracy: {metrics['balancedAccuracy']:.4f}")
    print(f"Macro F1: {metrics['macroF1']:.4f}")
    print(
        "Expected calibration error: "
        f"{metrics['expectedCalibrationError']:.4f} (lower is better)"
    )
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


def test_saved_model(args: argparse.Namespace) -> None:
    """Test a checkpoint on its reproducible held-out training split."""
    device = choose_inference_device(args.device)
    model, tokenizer, reasonNames, modelConfig = load_checkpoint(
        args.checkpoint, args.tokenizer, device
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    trainingConfig = checkpoint["training_config"]
    _, testDataset, _, splitReasons = prepare_split_data(
        args.dataset,
        args.tokenizer,
        modelConfig,
        trainingConfig,
        buildNewTokenizer=False,
    )
    if tuple(splitReasons) != tuple(reasonNames):
        raise ValueError("Dataset reason categories differ from the checkpoint")
    loader = make_data_loader(
        testDataset, tokenizer, args.batchSize, device
    )
    print(f"Using device: {device}")
    print(f"Held-out test samples: {len(testDataset)}")
    print_metrics(test_model(model, loader, device, reasonNames))


def evaluate_saved_model(args: argparse.Namespace) -> None:
    """Evaluate a checkpoint against every row in a labeled JSONL file."""
    device = choose_inference_device(args.device)
    model, tokenizer, reasonNames, modelConfig = load_checkpoint(
        args.checkpoint, args.tokenizer, device
    )
    dataset = prepare_evaluation_data(
        args.dataset, tokenizer, modelConfig, reasonNames
    )
    loader = make_data_loader(dataset, tokenizer, args.batchSize, device)
    print(f"Using device: {device}")
    print(f"Evaluation samples: {len(dataset)}")
    print_metrics(test_model(model, loader, device, reasonNames))


def evaluate_real_world_examples(args: argparse.Namespace) -> dict:
    """Score the complete prediction system on curated showcase expectations."""
    manifestPath = args.evaluationManifest
    if not manifestPath.is_file():
        raise FileNotFoundError(f"Evaluation manifest not found: {manifestPath}")
    expectedValues = json.loads(manifestPath.read_text(encoding="utf-8"))
    device = choose_inference_device(args.device)
    model, tokenizer, reasonNames, modelConfig = load_checkpoint(
        args.checkpoint, args.tokenizer, device
    )

    correctCount = 0
    coveredCount = 0
    coveredCorrect = 0
    maliciousTotal = 0
    maliciousCorrect = 0
    rows = []
    print("Real-world showcase evaluation")
    for fileName, expected in sorted(expectedValues.items()):
        scriptPath = args.showcaseDirectory / fileName
        if not scriptPath.is_file():
            raise FileNotFoundError(f"Evaluation script not found: {scriptPath}")
        prediction = predict_text(
            scriptPath.read_text(encoding="utf-8"),
            model,
            tokenizer,
            reasonNames,
            modelConfig,
            device,
            args.minimumConfidence,
        )
        predictedLabel = prediction["label"]
        expectedLabel = expected["label"]
        isCovered = predictedLabel not in {"uncertain", "invalid"}
        isCorrect = predictedLabel == expectedLabel
        correctCount += int(isCorrect)
        coveredCount += int(isCovered)
        coveredCorrect += int(isCovered and isCorrect)
        if expectedLabel == "malicious":
            maliciousTotal += 1
            maliciousCorrect += int(isCorrect)
        rows.append(
            {
                "script": fileName,
                "expected": expectedLabel,
                "predicted": predictedLabel,
                "confidence": prediction["confidence"],
                "source": prediction["source"],
            }
        )
        marker = "OK" if isCorrect else "WRONG"
        print(
            f"  [{marker:5s}] {fileName}: expected={expectedLabel} "
            f"predicted={predictedLabel} confidence={prediction['confidence']:.4f}"
        )

    sampleCount = len(rows)
    metrics = {
        "strictAccuracy": correctCount / sampleCount if sampleCount else 0.0,
        "coverage": coveredCount / sampleCount if sampleCount else 0.0,
        "coveredAccuracy": coveredCorrect / coveredCount if coveredCount else 0.0,
        "maliciousRecall": maliciousCorrect / maliciousTotal if maliciousTotal else 0.0,
        "samples": sampleCount,
        "rows": rows,
    }
    print(f"Strict accuracy: {metrics['strictAccuracy']:.4f}")
    print(f"Coverage: {metrics['coverage']:.4f}")
    print(f"Accuracy when covered: {metrics['coveredAccuracy']:.4f}")
    print(f"Malicious recall: {metrics['maliciousRecall']:.4f}")
    return metrics
