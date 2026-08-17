from __future__ import annotations

import cv2
import numpy as np
import pytest


@pytest.fixture
def pcb_image() -> np.ndarray:
    """Synthetic board with enough texture for contour and feature tests."""

    image = np.full((240, 340, 3), 24, dtype=np.uint8)
    cv2.rectangle(image, (35, 30), (305, 210), (38, 118, 50), -1)
    cv2.rectangle(image, (35, 30), (305, 210), (180, 220, 180), 3)
    rng = np.random.default_rng(1234)
    for index in range(28):
        x = int(rng.integers(48, 280))
        y = int(rng.integers(43, 190))
        width = int(rng.integers(7, 24))
        height = int(rng.integers(5, 17))
        color = (15, 15, 15) if index % 2 == 0 else (180, 180, 195)
        cv2.rectangle(image, (x, y), (min(x + width, 297), min(y + height, 202)), color, -1)
        cv2.circle(image, (x, y), 2, (220, 220, 220), -1)
    cv2.putText(image, "AOI-103", (80, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (235, 235, 235), 2)
    return image
