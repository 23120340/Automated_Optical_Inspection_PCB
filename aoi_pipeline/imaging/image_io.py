"""Safe image import helpers for step 0.

Only local files, byte buffers, and already-decoded arrays are accepted here.
Camera/RTSP acquisition deliberately lives outside this package.
"""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import cv2
import numpy as np

from ..exceptions import InvalidImageError


ImageSource = str | Path | bytes | bytearray | memoryview | BinaryIO | np.ndarray


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    """Validate an image and return a contiguous uint8 BGR array."""

    if not isinstance(image, np.ndarray):
        raise InvalidImageError("Expected a NumPy image array")
    if image.size == 0 or image.ndim not in (2, 3):
        raise InvalidImageError("Image is empty or has an unsupported number of dimensions")

    converted = _to_uint8(image)
    if converted.ndim == 2:
        converted = cv2.cvtColor(converted, cv2.COLOR_GRAY2BGR)
    elif converted.shape[2] == 1:
        converted = cv2.cvtColor(converted[:, :, 0], cv2.COLOR_GRAY2BGR)
    elif converted.shape[2] == 4:
        converted = cv2.cvtColor(converted, cv2.COLOR_BGRA2BGR)
    elif converted.shape[2] != 3:
        raise InvalidImageError(f"Unsupported channel count: {converted.shape[2]}")
    return np.ascontiguousarray(converted)


def load_image(source: ImageSource) -> np.ndarray:
    """Decode an uploaded/local image into BGR format.

    ``source`` may be a path, bytes, a binary file-like object, or an existing
    array. Remote URLs and camera credentials are intentionally not handled.
    """

    if isinstance(source, np.ndarray):
        return ensure_bgr(source)
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser()
        if not path.is_file():
            raise InvalidImageError(f"Image file does not exist: {path}")
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise InvalidImageError(f"Cannot read image file: {path}") from exc
    elif isinstance(source, (bytes, bytearray, memoryview)):
        payload = bytes(source)
    elif hasattr(source, "read"):
        payload = source.read()
        if not isinstance(payload, (bytes, bytearray)):
            raise InvalidImageError("The uploaded file must return bytes")
    else:
        raise InvalidImageError(f"Unsupported image source: {type(source).__name__}")

    if not payload:
        raise InvalidImageError("Image payload is empty")
    buffer = np.frombuffer(payload, dtype=np.uint8)
    decoded = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise InvalidImageError("OpenCV could not decode the supplied image")
    return ensure_bgr(decoded)


def encode_image(image: np.ndarray, extension: str = ".png", jpeg_quality: int = 95) -> bytes:
    """Encode an image for ZIP export or local download."""

    bgr = ensure_bgr(image)
    extension = extension.lower()
    if extension not in {".png", ".jpg", ".jpeg"}:
        raise InvalidImageError(f"Unsupported output image extension: {extension}")
    params: list[int] = []
    if extension in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, int(np.clip(jpeg_quality, 1, 100))]
    success, encoded = cv2.imencode(extension, bgr, params)
    if not success:
        raise InvalidImageError("OpenCV failed to encode the image")
    return encoded.tobytes()


def letterbox_normalize(
    crop: np.ndarray,
    target_size: tuple[int, int] | None,
    letterbox_color: tuple[int, int, int] = (114, 114, 114),
) -> np.ndarray:
    """Resize preserving aspect ratio and pad to ``target_size``."""

    if target_size is None:
        return np.ascontiguousarray(crop.copy())
    target_width, target_height = (int(value) for value in target_size)
    if target_width <= 0 or target_height <= 0:
        raise ValueError("target_size values must be positive")
    source_height, source_width = crop.shape[:2]
    scale = min(target_width / source_width, target_height / source_height)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(
        crop, (resized_width, resized_height), interpolation=interpolation
    )
    canvas = np.full(
        (target_height, target_width, 3),
        tuple(int(np.clip(value, 0, 255)) for value in letterbox_color),
        dtype=np.uint8,
    )
    x = (target_width - resized_width) // 2
    y = (target_height - resized_height) // 2
    canvas[y : y + resized_height, x : x + resized_width] = resized
    return canvas


def _to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image.copy()
    if image.dtype == np.bool_:
        return image.astype(np.uint8) * 255
    if not np.issubdtype(image.dtype, np.number):
        raise InvalidImageError(f"Unsupported image dtype: {image.dtype}")

    values = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0)
    if np.issubdtype(image.dtype, np.unsignedinteger) and image.dtype != np.uint8:
        # Preserve the full radiometric range of common 10/12/16-bit camera
        # containers instead of clipping every value above 255 to white.
        dtype_max = float(np.iinfo(image.dtype).max)
        values *= 255.0 / dtype_max
        return np.clip(values, 0.0, 255.0).astype(np.uint8)
    minimum = float(values.min())
    maximum = float(values.max())
    if np.issubdtype(image.dtype, np.floating) and minimum >= 0.0 and maximum <= 1.0:
        values *= 255.0
    return np.clip(values, 0.0, 255.0).astype(np.uint8)


def read_asset_under_root(root, relative_path: str, mode: int):
    """Read an image that must live inside ``root``.

    The containment check is the point: a recipe carries relative asset paths,
    and one of them saying ``../../etc/passwd`` must read as "missing asset",
    not as a file read. Step-2 position matching and Golden Compare each had
    their own identical copy of this.
    """

    from pathlib import Path as _Path

    path = (_Path(root) / relative_path).resolve()
    try:
        path.relative_to(_Path(root))
    except ValueError:
        return None
    return cv2.imread(str(path), mode)
