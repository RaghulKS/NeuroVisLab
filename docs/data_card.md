# Data card — PneumoniaMNIST benchmark

NeuroVisionLab acquires the official `pneumoniamnist.npz` MedMNIST v2 benchmark archive from the URL recorded in [`configs/demo.toml`](../configs/demo.toml). MedMNIST is a standardized, preprocessed biomedical-image benchmark released under CC BY 4.0 (with the project’s documentation noting that it is not for clinical use).

Each acquisition creates `dataset.lock.json` next to the downloaded archive. The lock records the source URL, acquisition timestamp, file byte size, SHA-256 digest, split sizes, image shape, class distribution, and a content-derived `data_version`. Training refuses a file whose hash differs from that lock.

The demo uses deliberately bounded train/validation/test subsets specified in TOML so CPU training is repeatable. Those limits, preprocessing (28×28 grayscale scaled to [0,1]), seed, and class mapping are carried into every run manifest. The label names are benchmark labels only; they must never be treated as patient-specific clinical findings.

Limitations: this data is preprocessed and benchmark-oriented, not representative of a clinical workflow; no demographic or site generalization claim is made. Do not use it for diagnosis, medical advice, or clinical decision support.
