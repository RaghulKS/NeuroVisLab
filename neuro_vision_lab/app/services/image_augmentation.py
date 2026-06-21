from __future__ import annotations

import random

from PIL import Image, ImageEnhance, ImageOps


def augment_image(image: Image.Image, seed: int | None = None) -> Image.Image:
    rng = random.Random(seed)
    output = image.convert("RGB")
    if rng.random() < 0.5:
        output = ImageOps.mirror(output)
    if rng.random() < 0.25:
        output = ImageOps.flip(output)
    angle = rng.uniform(-8, 8)
    output = output.rotate(angle, resample=Image.Resampling.BILINEAR)
    output = ImageEnhance.Contrast(output).enhance(rng.uniform(0.85, 1.2))
    output = ImageEnhance.Brightness(output).enhance(rng.uniform(0.9, 1.15))
    return output

