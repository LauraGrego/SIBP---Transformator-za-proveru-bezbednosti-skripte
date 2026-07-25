import io
import json
import math
import pickle
import random
import re
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "artifacts" / "bash_transformer.pt"
DATASET = ROOT / "data" / "safe_risky_combined.jsonl"
TOKENIZER = ROOT / "artifacts" / "bash_tokenizer.json"


class Storage:
    def __init__(self, array):
        self.array = array


class TorchArchiveUnpickler(pickle.Unpickler):
    def __init__(self, stream, archive, prefix):
        super().__init__(stream)
        self.archive = archive
        self.prefix = prefix

    def persistent_load(self, saved_id):
        kind, storage_type, key, location, size = saved_id
        if kind != "storage":
            raise ValueError(saved_id)
        dtype_name = getattr(storage_type, "dtype_name", "float32")
        dtype = np.dtype("<" + {"float32": "f4", "float64": "f8", "int64": "i8"}[dtype_name])
        raw = self.archive.read(f"{self.prefix}/data/{key}")
        return Storage(np.frombuffer(raw, dtype=dtype, count=size))

    def find_class(self, module, name):
        if module == "torch._utils" and name in {"_rebuild_tensor_v2", "_rebuild_tensor"}:
            def rebuild(storage, offset, size, stride, *rest):
                base = storage.array[offset:]
                byte_strides = tuple(int(s) * base.dtype.itemsize for s in stride)
                return np.lib.stride_tricks.as_strided(base, shape=tuple(size), strides=byte_strides).copy()
            return rebuild
        if module == "torch" and name.endswith("Storage"):
            dtype_name = {
                "FloatStorage": "float32",
                "DoubleStorage": "float64",
                "LongStorage": "int64",
            }.get(name, "float32")
            return type(name, (), {"dtype_name": dtype_name})
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        return super().find_class(module, name)


def load_checkpoint():
    with zipfile.ZipFile(CHECKPOINT) as archive:
        root = archive.namelist()[0].split("/")[0]
        stream = io.BytesIO(archive.read(f"{root}/data.pkl"))
        return TorchArchiveUnpickler(stream, archive, root).load()


def tokenize(text, vocab, unk_id):
    # Hugging Face `Whitespace` separates words and contiguous punctuation.
    pieces = re.findall(r"\w+|[^\w\s]+", text, flags=re.UNICODE)
    return [vocab.get(piece, unk_id) for piece in pieces]


def layer_norm(x, weight, bias, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * weight + bias


def linear(x, weight, bias):
    return x @ weight.T + bias


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    values = np.exp(x)
    return values / values.sum(axis=axis, keepdims=True)


def gelu(x):
    # Numerically close tanh form; sufficient for reproducing argmax metrics.
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def forward(ids, mask, weights, config):
    dim = config["modelDimension"]
    heads = config["numberOfHeads"]
    head_dim = dim // heads
    batch, length = ids.shape
    x = weights["embedding.weight"][ids] * math.sqrt(dim)
    positions = np.arange(length)[:, None]
    frequency = np.exp(np.arange(0, dim, 2) * (-math.log(10000.0) / dim))
    pe = np.zeros((length, dim), dtype=np.float32)
    pe[:, 0::2] = np.sin(positions * frequency)
    pe[:, 1::2] = np.cos(positions * frequency)
    x = x + pe[None, :, :]

    for layer in range(config["numberOfBlocks"]):
        p = f"encoder.layers.{layer}."
        n = layer_norm(x, weights[p + "norm1.weight"], weights[p + "norm1.bias"])
        qkv = linear(n, weights[p + "self_attn.in_proj_weight"], weights[p + "self_attn.in_proj_bias"])
        q, k, v = np.split(qkv, 3, axis=-1)
        q = q.reshape(batch, length, heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, length, heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, length, heads, head_dim).transpose(0, 2, 1, 3)
        scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(head_dim)
        scores = np.where(mask[:, None, None, :], scores, -1e9)
        attended = softmax(scores) @ v
        attended = attended.transpose(0, 2, 1, 3).reshape(batch, length, dim)
        x = x + linear(attended, weights[p + "self_attn.out_proj.weight"], weights[p + "self_attn.out_proj.bias"])
        n = layer_norm(x, weights[p + "norm2.weight"], weights[p + "norm2.bias"])
        ff = linear(gelu(linear(n, weights[p + "linear1.weight"], weights[p + "linear1.bias"])),
                    weights[p + "linear2.weight"], weights[p + "linear2.bias"])
        x = x + ff

    x = layer_norm(x, weights["encoder.norm.weight"], weights["encoder.norm.bias"])
    pooled = (x * mask[:, :, None]).sum(axis=1) / np.maximum(mask.sum(axis=1, keepdims=True), 1)
    return linear(pooled, weights["labelHead.weight"], weights["labelHead.bias"]), linear(
        pooled, weights["reasonHead.weight"], weights["reasonHead.bias"]
    )


def stratified_split(records, fraction, seed):
    groups = {}
    for record in records:
        reason = record.get("category", "") if record["label"] == "malicious" else ""
        groups.setdefault((record["label"], reason), []).append(record)
    rng = random.Random(seed)
    train, test = [], []
    for rows in groups.values():
        rng.shuffle(rows)
        test_size = max(1, round(len(rows) * fraction)) if len(rows) > 1 else 0
        test_size = min(test_size, len(rows) - 1)
        test.extend(rows[:test_size])
        train.extend(rows[test_size:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def main():
    checkpoint = load_checkpoint()
    weights = checkpoint["model_state_dict"]
    weights = OrderedDict(
        (("labelHead." + key[len("label_head."):]) if key.startswith("label_head.")
         else ("reasonHead." + key[len("reason_head."):]) if key.startswith("reason_head.")
         else key, value)
        for key, value in weights.items()
    )
    config = checkpoint["model_config"].copy()
    training = checkpoint["training_config"].copy()
    for old, new in {
        "context_size": "contextSize", "model_dimension": "modelDimension",
        "number_of_heads": "numberOfHeads", "number_of_blocks": "numberOfBlocks",
        "feed_forward_dimension": "feedForwardDimension",
    }.items():
        if old in config:
            config[new] = config.pop(old)
    for old, new in {"test_fraction": "testFraction"}.items():
        if old in training:
            training[new] = training.pop(old)
    labels = checkpoint["labels"]
    reasons = checkpoint["reason_names"]
    tokenizer = json.loads(TOKENIZER.read_text(encoding="utf-8"))
    vocab = tokenizer["model"]["vocab"]
    unk_id, cls_id, pad_id = vocab["[UNK]"], vocab["[CLS]"], vocab["[PAD]"]
    records = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
    _, test = stratified_split(records, training["testFraction"], training["seed"])
    reason_to_id = {name: i for i, name in enumerate(reasons)}
    confusion = np.zeros((len(labels), len(labels)), dtype=np.int64)
    reason_confusion = np.zeros((len(reasons), len(reasons)), dtype=np.int64)
    batch_size = 16
    for start in range(0, len(test), batch_size):
        chunk = test[start:start + batch_size]
        token_rows = [[cls_id] + tokenize(row["script"], vocab, unk_id)[:config["contextSize"] - 1] for row in chunk]
        length = max(map(len, token_rows))
        ids = np.full((len(chunk), length), pad_id, dtype=np.int64)
        mask = np.zeros((len(chunk), length), dtype=bool)
        for i, row_ids in enumerate(token_rows):
            ids[i, :len(row_ids)] = row_ids
            mask[i, :len(row_ids)] = True
        label_logits, reason_logits = forward(ids, mask, weights, config)
        label_predictions = label_logits.argmax(axis=1)
        reason_predictions = reason_logits.argmax(axis=1)
        for row, pred, reason_pred in zip(chunk, label_predictions, reason_predictions):
            actual = labels.index(row["label"])
            confusion[actual, pred] += 1
            if row["label"] == "malicious" and row.get("category") in reason_to_id:
                reason_confusion[reason_to_id[row["category"]], reason_pred] += 1
        print(f"evaluated {min(start + batch_size, len(test))}/{len(test)}", flush=True)
    per_label = {}
    for i, label in enumerate(labels):
        tp = confusion[i, i]
        precision = tp / confusion[:, i].sum() if confusion[:, i].sum() else 0
        recall = tp / confusion[i, :].sum() if confusion[i, :].sum() else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        per_label[label] = {"precision": float(precision), "recall": float(recall), "f1": float(f1)}
    result = {
        "checkpoint_model_config": config,
        "checkpoint_training_config": training,
        "labels": labels,
        "reasons": reasons,
        "test_samples": len(test),
        "confusion_matrix": confusion.tolist(),
        "accuracy": float(np.trace(confusion) / confusion.sum()),
        "macro_f1": float(np.mean([m["f1"] for m in per_label.values()])),
        "per_label": per_label,
        "reason_confusion_matrix": reason_confusion.tolist(),
        "reason_accuracy": float(np.trace(reason_confusion) / reason_confusion.sum()),
        "reason_samples": int(reason_confusion.sum()),
    }
    out = ROOT / "tmp" / "pdfs" / "metrics.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
