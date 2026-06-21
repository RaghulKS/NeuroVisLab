from __future__ import annotations

import numpy as np

from app.ml.metrics import confusion_matrix, evaluate_predictions


def test_metrics_include_sensitivity_specificity_and_auc():
    labels = ["negative", "positive"]
    y_true = ["negative", "positive", "positive", "negative"]
    y_pred = ["negative", "positive", "negative", "negative"]
    probs = np.asarray([[0.8, 0.2], [0.2, 0.8], [0.55, 0.45], [0.7, 0.3]], dtype=np.float32)
    matrix = confusion_matrix(y_true, y_pred, labels)
    metrics = evaluate_predictions(y_true, y_pred, probabilities=probs, labels=labels)
    assert matrix.tolist() == [[2, 0], [1, 1]]
    assert metrics["accuracy"] == 0.75
    assert "specificity" in metrics["per_class"]["positive"]
    assert metrics["roc_auc_ovr_macro"] is not None

