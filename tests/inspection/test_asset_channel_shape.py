"""Golden Inspection must survive ``cv2.imread`` not being OpenCV's.

Ultralytics replaces ``cv2.imread`` at import time with its own version so that
non-ASCII paths work. Under OpenCV 5 that replacement returns a trailing channel
axis where OpenCV itself does not:

    before importing ultralytics:  imread(mask, IMREAD_GRAYSCALE) -> (83, 95)
    after  importing ultralytics:  imread(mask, IMREAD_GRAYSCALE) -> (83, 95, 1)

The app always loads a detector, so inside the app it is always the second one.
Every ``mask.shape == (height, width)`` check then failed on a mask that was
perfectly correct. Measured on a real board: ``valid_overlap_ratio`` came back
as exactly 0.0 for all 77 slots that matched their template with scores of
0.88-0.98, every one of them was called ``unmeasurable``, and because appearance
is gated on position, all 112 appearance checks reported ``not_evaluated``.
Golden Inspection reported nothing at all while every underlying measurement was
fine.

The unit tests did not catch it because they never import ultralytics.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from aoi_pipeline.imaging.image_io import drop_singleton_channel, read_asset_under_root
from aoi_pipeline.golden.position import _translation_valid_overlap


def test_a_trailing_single_channel_is_dropped() -> None:
    squeezed = drop_singleton_channel(np.zeros((8, 5, 1), np.uint8))
    assert squeezed.shape == (8, 5)


@pytest.mark.parametrize("shape", [(8, 5), (8, 5, 3), (8, 5, 4)])
def test_everything_else_is_left_alone(shape) -> None:
    """Only the (h, w, 1) case is ambiguous; squeezing a colour image would be
    a different bug."""

    image = np.zeros(shape, np.uint8)
    assert drop_singleton_channel(image).shape == shape


def test_none_survives_the_normaliser() -> None:
    """``imread`` returns None for an unreadable file and callers check for it."""

    assert drop_singleton_channel(None) is None


def test_overlap_is_full_for_a_mask_that_arrives_with_a_channel_axis() -> None:
    """The exact failure, as a number: 0.0 against a gate of 0.80."""

    valid = np.full((200, 300), 255, np.uint8)
    roi = (40, 30, 40 + 60, 30 + 50)
    flat = np.full((50, 60), 255, np.uint8)
    channelled = flat.reshape(50, 60, 1)

    assert _translation_valid_overlap(valid, flat, roi, 0.0, 0.0) == pytest.approx(1.0)
    assert _translation_valid_overlap(valid, channelled, roi, 0.0, 0.0) == pytest.approx(1.0)


def test_a_real_size_mismatch_is_still_rejected() -> None:
    """Being tolerant of the channel axis must not make the guard tolerant of a
    mask that genuinely does not fit its ROI."""

    valid = np.full((200, 300), 255, np.uint8)
    roi = (40, 30, 40 + 60, 30 + 50)
    wrong = np.full((20, 20), 255, np.uint8)
    assert _translation_valid_overlap(valid, wrong, roi, 0.0, 0.0) == 0.0


def test_reading_a_grayscale_asset_gives_two_dimensions(tmp_path) -> None:
    """Through the project's own reader, whichever ``imread`` is installed."""

    mask = np.zeros((12, 9), np.uint8)
    mask[3:9, 2:7] = 255
    ok, encoded = cv2.imencode(".png", mask)
    assert ok
    (tmp_path / "m.png").write_bytes(encoded.tobytes())

    loaded = read_asset_under_root(tmp_path, "m.png", cv2.IMREAD_GRAYSCALE)
    assert loaded is not None
    assert loaded.ndim == 2, f"đọc ra {loaded.shape}, phải là (h, w)"
    assert loaded.shape == (12, 9)


def test_the_reader_still_refuses_a_path_outside_its_root(tmp_path) -> None:
    """Normalising the shape must not have loosened the containment check."""

    assert read_asset_under_root(tmp_path, "../outside.png", cv2.IMREAD_GRAYSCALE) is None


def test_the_patched_reader_is_handled_when_ultralytics_is_installed(tmp_path) -> None:
    """The real thing rather than a simulation, when it is available.

    Importing ultralytics has the side effect this test exists for, so the
    assertion is made after the import, deliberately.
    """

    pytest.importorskip("ultralytics")

    mask = np.zeros((14, 11), np.uint8)
    mask[2:12, 3:8] = 255
    ok, encoded = cv2.imencode(".png", mask)
    assert ok
    (tmp_path / "m.png").write_bytes(encoded.tobytes())

    loaded = read_asset_under_root(tmp_path, "m.png", cv2.IMREAD_GRAYSCALE)
    assert loaded.ndim == 2, (
        f"cv2.imread hiện là {cv2.imread.__module__}; đọc ra {loaded.shape}"
    )

    roi = (5, 4, 5 + 11, 4 + 14)
    valid = np.full((60, 60), 255, np.uint8)
    assert _translation_valid_overlap(valid, loaded, roi, 0.0, 0.0) > 0.0
