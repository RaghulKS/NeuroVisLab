from __future__ import annotations

from typing import Any

import numpy as np


def confusion_matrix(y_true: list[str], y_pred: list[str], labels: list[str] | None = None) -> np.ndarray:
    labels = labels or sorted(set(y_true) | set(y_pred))
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for truth, pred in zip(y_true, y_pred):
        if truth in index and pred in index:
            matrix[index[truth], index[pred]] += 1
    return matrix


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _binary_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    positives = int(y_true.sum())
    negatives = int(len(y_true) - positives)
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(-scores)
    sorted_true = y_true[order]
    tps = np.cumsum(sorted_true)
    fps = np.cumsum(1 - sorted_true)
    tpr = np.concatenate([[0.0], tps / positives, [1.0]])
    fpr = np.concatenate([[0.0], fps / negatives, [1.0]])
    return float(np.trapz(tpr, fpr))


def _binary_pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    positives = int(y_true.sum())
    if positives == 0:
        return None
    order = np.argsort(-scores)
    sorted_true = y_true[order]
    tps = np.cumsum(sorted_true)
    fps = np.cumsum(1 - sorted_true)
    precision = tps / np.maximum(tps + fps, 1)
    recall = tps / positives
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    return float(np.trapz(precision, recall))


def evaluate_predictions(
    y_true: list[str],
    y_pred: list[str],
    probabilities: np.ndarray | None = None,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    labels = labels or sorted(set(y_true) | set(y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels)
    total = int(matrix.sum())
    accuracy = _safe_div(float(np.trace(matrix)), float(total))
    per_class: dict[str, dict[str, float]] = {}
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    specificities: list[float] = []
    supports: list[int] = []
    for i, label in enumerate(labels):
        tp = float(matrix[i, i])
        fp = float(matrix[:, i].sum() - tp)
        fn = float(matrix[i, :].sum() - tp)
        tn = float(matrix.sum() - tp - fp - fn)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        specificity = _safe_div(tn, tn + fp)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        support = int(matrix[i, :].sum())
        per_class[label] = {
            "precision": precision,
            "recall_sensitivity": recall,
            "specificity": specificity,
            "f1": f1,
            "support": support,
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        specificities.append(specificity)
        supports.append(support)
    support_arr = np.asarray(supports, dtype=np.float32)
    weights = support_arr / max(float(support_arr.sum()), 1.0)
    metrics: dict[str, Any] = {
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precisions)) if precisions else 0.0,
        "macro_recall_sensitivity": float(np.mean(recalls)) if recalls else 0.0,
        "macro_specificity": float(np.mean(specificities)) if specificities else 0.0,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "weighted_f1": float(np.sum(np.asarray(f1s) * weights)) if f1s else 0.0,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "labels": labels,
        "n_samples": total,
    }
    if probabilities is not None and len(probabilities) == len(y_true) and labels:
        probabilities = np.asarray(probabilities, dtype=np.float32)
        aucs: list[float] = []
        pr_aucs: list[float] = []
        for i, label in enumerate(labels):
            binary_true = np.asarray([1 if y == label else 0 for y in y_true], dtype=int)
            score = probabilities[:, i]
            auc = _binary_auc(binary_true, score)
            pr_auc = _binary_pr_auc(binary_true, score)
            if auc is not None:
                aucs.append(auc)
            if pr_auc is not None:
                pr_aucs.append(pr_auc)
        metrics["roc_auc_ovr_macro"] = float(np.mean(aucs)) if aucs else None
        metrics["pr_auc_ovr_macro"] = float(np.mean(pr_aucs)) if pr_aucs else None
    return metrics


def subgroup_performance(
    y_true: list[str],
    y_pred: list[str],
    subgroup_values: list[str | None],
    labels: list[str],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for group in sorted({value or "missing" for value in subgroup_values}):
        idx = [i for i, value in enumerate(subgroup_values) if (value or "missing") == group]
        if not idx:
            continue
        output[group] = evaluate_predictions(
            [y_true[i] for i in idx],
            [y_pred[i] for i in idx],
            labels=labels,
        )
    return output

