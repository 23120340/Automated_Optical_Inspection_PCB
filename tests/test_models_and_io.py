from __future__ import annotations

from io import BytesIO

import cv2
import numpy as np
import pytest

from aoi_pipeline import BoundingBox, InvalidImageError, encode_image, ensure_bgr, load_image


def test_bounding_box_geometry_and_clamping() -> None:
    box = BoundingBox(-2.2, 3.1, 12.4, 20.9)
    clamped = box.clamp(10, 18)
    assert clamped.as_xyxy() == [0.0, 3.1, 10.0, 18.0]
    assert clamped.to_int() == (0, 3, 10, 18)
    assert clamped.area == pytest.approx(149.0)


def test_invalid_bounding_box_is_rejected() -> None:
    with pytest.raises(ValueError):
        BoundingBox(10, 0, 5, 2)
    with pytest.raises(ValueError):
        BoundingBox(0, 0, float("nan"), 2)


def test_step_zero_loads_bytes_filelike_and_grayscale(pcb_image: np.ndarray) -> None:
    payload = encode_image(pcb_image)
    from_bytes = load_image(payload)
    from_filelike = load_image(BytesIO(payload))
    assert np.array_equal(from_bytes, pcb_image)
    assert np.array_equal(from_filelike, pcb_image)

    gray = cv2.cvtColor(pcb_image, cv2.COLOR_BGR2GRAY)
    bgr = ensure_bgr(gray)
    assert bgr.shape == pcb_image.shape
    assert bgr.dtype == np.uint8


def test_step_zero_converts_unit_float_and_rejects_bad_payload() -> None:
    image = np.full((4, 5, 3), 0.5, dtype=np.float32)
    converted = ensure_bgr(image)
    assert converted.dtype == np.uint8
    assert int(converted[0, 0, 0]) in {127, 128}
    with pytest.raises(InvalidImageError):
        load_image(b"not an image")


def test_step_zero_preserves_uint16_dynamic_range() -> None:
    gradient = np.linspace(0, 65535, 256, dtype=np.uint16)[None, :]
    image = np.repeat(gradient, 4, axis=0)
    success, encoded = cv2.imencode(".png", image)
    assert success
    decoded = load_image(encoded.tobytes())
    assert decoded.dtype == np.uint8
    assert decoded.shape == (4, 256, 3)
    assert len(np.unique(decoded[:, :, 0])) >= 250
    assert decoded[0, 0, 0] == 0
    assert decoded[0, -1, 0] == 255
