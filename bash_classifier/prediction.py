from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path
from typing import Sequence

import torch
from tokenizers import Tokenizer

from .config import LABELS, PAD_TOKEN, REASON_EXPLANATIONS, ModelConfig
from .data import BashDataset, collate_batch, move_batch_to_device, required_token_id
from .devices import choose_inference_device
from .model import BashTransformerClassifier, load_checkpoint
from .security_rules import (
    explain_risky_behavior,
    find_security_rule,
    validate_script_text,
)


@torch.inference_mode()
def predict_text(
    script: str,
    model: BashTransformerClassifier,
    tokenizer: Tokenizer,
    reasonNames: Sequence[str],
    modelConfig: ModelConfig,
    device: torch.device,
    minimumConfidence: float = 0.65,
) -> dict:
    """Combine strict security signatures with confidence-aware model output."""
    validationError = validate_script_text(script)
    if validationError is not None:
        return {
            "label": "invalid",
            "confidence": 0.0,
            "explanation": validationError,
            "source": "input_validation",
        }
    ruleResult = find_security_rule(script)
    if ruleResult is not None:
        return ruleResult

    dummyRecord = {"label": "safe", "script": script, "category": ""}
    dataset = BashDataset(
        [dummyRecord],
        tokenizer,
        modelConfig.contextSize,
        {name: index for index, name in enumerate(reasonNames)},
    )
    batch = collate_batch(
        [dataset[0]], padId=required_token_id(tokenizer, PAD_TOKEN)
    )
    batch = move_batch_to_device(batch, device)
    labelLogits, reasonLogits = model(
        batch["input_ids"], batch["attention_mask"]
    )
    probabilities = labelLogits.softmax(dim=1)[0]
    labelId = probabilities.argmax().item()
    label = LABELS[labelId]
    displayedLabel = (
        label
        if probabilities[labelId].item() >= minimumConfidence
        else "uncertain"
    )
    result = {
        "label": displayedLabel,
        "modelLabel": label,
        "confidence": round(probabilities[labelId].item(), 6),
        "probabilities": {
            name: round(probabilities[index].item(), 6)
            for index, name in enumerate(LABELS)
        },
        "source": "model",
    }
    if displayedLabel == "uncertain":
        result["explanation"] = (
            f"top model label is {label}, below the {minimumConfidence:.2f} "
            "minimum confidence"
        )
    if displayedLabel == "malicious":
        reasonProbabilities = reasonLogits.softmax(dim=1)[0]
        reasonId = reasonProbabilities.argmax().item()
        reason = reasonNames[reasonId]
        result["reason"] = reason
        result["reason_confidence"] = round(
            reasonProbabilities[reasonId].item(), 6
        )
        result["explanation"] = REASON_EXPLANATIONS.get(
            reason, f"matches patterns associated with {reason.replace('_', ' ')}"
        )
    elif displayedLabel == "risky":
        riskyExplanation = explain_risky_behavior(script)
        result.update(riskyExplanation)
    return result


def predict_showcase_directory(args: argparse.Namespace) -> list[dict]:
    """Predict normal scripts and each script embedded in a tabular text file."""
    showcaseDirectory: Path = args.showcaseDirectory
    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {args.checkpoint}. Run CPU or GPU training first."
        )
    if not showcaseDirectory.is_dir():
        raise ValueError(f"Showcase directory does not exist: {showcaseDirectory}")

    scriptSuffixes = {".bash", ".sh", ".txt"}
    scriptPaths = sorted(
        path
        for path in showcaseDirectory.rglob("*")
        if path.is_file() and path.suffix.lower() in scriptSuffixes
    )
    if not scriptPaths:
        raise ValueError(f"No script files found in {showcaseDirectory}")

    device = choose_inference_device(args.device)
    model, tokenizer, reasonNames, modelConfig = load_checkpoint(
        args.checkpoint, args.tokenizer, device
    )
    print(f"Using device: {device}")
    print(f"Model: {args.checkpoint}")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Predicting {len(scriptPaths)} showcase files:\n")

    results = []
    for scriptPath in scriptPaths:
        script = scriptPath.read_text(encoding="utf-8")
        relativeName = scriptPath.relative_to(showcaseDirectory)
        embeddedScripts = extract_tabular_scripts(script)
        if embeddedScripts is not None:
            print(f"{relativeName}: tabular dataset ({len(embeddedScripts)} scripts)")
            for embeddedScript in embeddedScripts:
                prediction = predict_text(
                    embeddedScript["script"],
                    model,
                    tokenizer,
                    reasonNames,
                    modelConfig,
                    device,
                    args.minimumConfidence,
                )
                displayName = f"{relativeName}::{embeddedScript['name']}"
                result = {"script": displayName, **prediction}
                results.append(result)
                print("  " + format_prediction_summary(displayName, prediction))
            continue

        prediction = predict_text(
            script,
            model,
            tokenizer,
            reasonNames,
            modelConfig,
            device,
            args.minimumConfidence,
        )
        result = {"script": str(relativeName), **prediction}
        results.append(result)
        print(format_prediction_summary(str(relativeName), prediction))
    return results


def extract_tabular_scripts(script: str) -> list[dict] | None:
    """Extract named scripts from a CSV container or return None for plain text."""
    firstLine = script.lstrip().splitlines()[0].lower() if script.strip() else ""
    if not firstLine.startswith("script_id,"):
        return None
    reader = csv.DictReader(io.StringIO(script))
    requiredColumns = {"script_name", "script_content"}
    if reader.fieldnames is None or not requiredColumns.issubset(reader.fieldnames):
        raise ValueError(
            "Tabular script file must contain script_name and script_content columns"
        )
    embeddedScripts = []
    for rowIndex, row in enumerate(reader, 1):
        scriptText = (row.get("script_content") or "").strip()
        scriptName = (row.get("script_name") or f"row-{rowIndex}").strip()
        if scriptText:
            embeddedScripts.append({"name": scriptName, "script": scriptText})
    if not embeddedScripts:
        raise ValueError("Tabular script file contains no non-empty scripts")
    return embeddedScripts


def format_prediction_summary(displayName: str, prediction: dict) -> str:
    """Format one normal or CSV-embedded prediction consistently for display."""
    summary = f"{displayName}: {prediction['label']}"
    if prediction["label"] != "invalid":
        summary += f" (confidence={prediction['confidence']:.4f})"
    if prediction["label"] == "uncertain":
        summary += f" | top_label={prediction['modelLabel']}"
    if "reason" in prediction:
        summary += f" | reason={prediction['reason']}"
    if prediction["label"] == "invalid":
        summary += f" | {prediction['explanation']}"
    return summary
