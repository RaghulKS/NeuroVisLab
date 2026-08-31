# Model card — MedMNIST CNN lifecycle demo

The serving model is a compact three-block PyTorch CNN trained from scratch on the reproducibly acquired PneumoniaMNIST benchmark subset. It exists to demonstrate model operations—not to establish a useful clinical model.

Every run writes an artifact-level `model_card.md`, `run.json`, checkpoint, prediction file, resolved config, calibration temperature, and benchmark metrics. The runtime registry references those immutable paths and hashes rather than selecting “the newest file.”

Evaluation includes accuracy, macro F1, macro AUROC/PR-AUC when defined, confusion matrix, confidence calibration/ECE, and latency. A temperature selected on held-out evaluation probabilities is saved with the checkpoint. Metrics are specific to the pinned data/version and configured subset; they are not external validation and must not be compared to clinical performance claims.

Known limitations include short CPU training, no subgroup/clinical validation, labels that do not prove disease in a real care setting, and uncertainty estimates that are engineering indicators rather than safety guarantees. The model is research-only and never provides medical advice or diagnosis.
