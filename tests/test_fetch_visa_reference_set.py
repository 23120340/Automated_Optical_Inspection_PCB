"""Offline contract tests for the small public VisA reference-set fetcher."""

from __future__ import annotations

import io
import json
import sys
import urllib.parse
from pathlib import Path

import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import fetch_visa_reference_set as visa  # noqa: E402


class FakeFetcher:
    """Dataset Viewer plus signed assets, entirely in memory."""

    def __init__(
        self,
        *,
        total_rows: int = 901,
        bad_label_at: int | None = None,
        bad_dimensions_at: int | None = None,
        duplicate_at: int | None = None,
        fail_image_at: int | None = None,
    ) -> None:
        self.total_rows = total_rows
        self.bad_label_at = bad_label_at
        self.bad_dimensions_at = bad_dimensions_at
        self.duplicate_at = duplicate_at
        self.fail_image_at = fail_image_at
        self.metadata_offsets: list[int] = []
        self.image_offsets: list[int] = []
        self.images = {
            index: _png_for(index)
            for index in visa.evenly_spaced_indices(total_rows)
        }

    def fetch(self, url: str) -> bytes:
        parsed = urllib.parse.urlparse(url)
        if parsed.path == "/rows":
            query = urllib.parse.parse_qs(parsed.query)
            assert query["dataset"] == [visa.DATASET_ID]
            assert query["config"] == [visa.CONFIG]
            assert query["split"] == [visa.DEFAULT_SPLIT]
            assert query["length"] == ["1"]
            index = int(query["offset"][0])
            self.metadata_offsets.append(index)
            width, height = _dimensions(self.images[index])
            if index == self.bad_dimensions_at:
                width += 1
            image_url = (
                "https://datasets-server.huggingface.co/"
                f"cached-assets/offline/{index}/image.png?Expires=9999999999"
                "&Signature=not-persisted&Key-Pair-Id=test"
            )
            payload = {
                "features": [],
                "rows": [
                    {
                        "row_idx": index,
                        "row": {
                            "image": {
                                "src": image_url,
                                "width": width,
                                "height": height,
                            },
                            "mask": None,
                            "label": 1 if index == self.bad_label_at else 0,
                        },
                        "truncated_cells": [],
                    }
                ],
                "num_rows_total": self.total_rows,
                "num_rows_per_page": 100,
                "partial": False,
            }
            return json.dumps(payload).encode("utf-8")

        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) == 4 and parts[:2] == ["cached-assets", "offline"]:
            index = int(parts[2])
            self.image_offsets.append(index)
            if index == self.fail_image_at:
                raise ConnectionError("signed URL intentionally omitted from this error")
            source_index = 0 if index == self.duplicate_at else index
            return self.images[source_index]
        raise AssertionError(f"Unexpected offline URL path: {parsed.path}")


class ExplodingFetcher:
    def fetch(self, url: str) -> bytes:  # pragma: no cover - failure is the assertion
        raise AssertionError("An idempotent run must not touch the network")


def _png_for(index: int) -> bytes:
    # The first two channels encode the complete row index, making every SHA
    # unique without relying on metadata or filenames.
    colour = (index & 0xFF, (index >> 8) & 0xFF, (index * 73) & 0xFF)
    image = Image.new("RGB", (11, 7), colour)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _dimensions(payload: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(payload)) as image:
        return image.size


def test_integer_selection_spans_the_complete_default_split() -> None:
    indices = visa.evenly_spaced_indices(901)
    assert indices == [
        0,
        31,
        62,
        93,
        124,
        155,
        186,
        217,
        248,
        279,
        310,
        341,
        372,
        403,
        434,
        465,
        496,
        527,
        558,
        589,
        620,
        651,
        682,
        713,
        744,
        775,
        806,
        837,
        868,
        900,
    ]
    assert len(indices) == len(set(indices)) == visa.REFERENCE_COUNT
    assert set(right - left for left, right in zip(indices, indices[1:])) <= {31, 32}


def test_cli_fetches_exact_bytes_and_writes_a_portable_provenance_manifest(
    tmp_path: Path,
) -> None:
    output = tmp_path / "visa-reference"
    fetcher = FakeFetcher()

    assert visa.main(["--output", str(output)], fetcher=fetcher) == 0

    manifest_bytes = (output / "manifest.json").read_bytes()
    manifest_text = manifest_bytes.decode("utf-8")
    manifest = json.loads(manifest_text)
    expected_indices = visa.evenly_spaced_indices(901)
    assert fetcher.metadata_offsets == expected_indices
    assert fetcher.image_offsets == expected_indices
    assert manifest["source_kind"] == "dataset_preprocessed"
    assert manifest["dataset_id"] == "BrachioLab/visa"
    assert manifest["config"] == "default"
    assert manifest["split"] == "pcb2.train"
    assert manifest["license"] == "CC BY 4.0"
    assert manifest["license_url"] == "https://creativecommons.org/licenses/by/4.0/"
    assert manifest["official_provenance"]["publisher"] == "Amazon"
    assert manifest["official_provenance"]["registry_url"] == (
        "https://registry.opendata.aws/visa/"
    )
    assert manifest["selection"]["row_indices"] == expected_indices
    assert manifest["selection"]["requested_count"] == 30
    assert len(manifest["files"]) == 30

    for entry in manifest["files"]:
        path = Path(entry["path"])
        assert not path.is_absolute()
        payload = (output / path).read_bytes()
        assert payload == fetcher.images[entry["row_index"]], (
            "the downloaded image must be stored byte-for-byte, not decoded/re-encoded"
        )
        assert entry["byte_size"] == len(payload)
        assert (entry["width"], entry["height"]) == _dimensions(payload)

    lowered = manifest_text.lower()
    assert "expires=" not in lowered
    assert "signature=" not in lowered
    assert "key-pair-id=" not in lowered
    assert "cached-assets" not in lowered
    assert str(tmp_path).lower() not in lowered


def test_existing_verified_output_is_an_offline_idempotent_noop(tmp_path: Path) -> None:
    output = tmp_path / "visa-reference"
    first = visa.fetch_reference_set(output, fetcher=FakeFetcher())
    snapshot = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }

    second = visa.fetch_reference_set(output, fetcher=ExplodingFetcher())

    assert first.created is True
    assert second.created is False
    assert snapshot == {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("label", "expected label=0"),
        ("dimensions", "not declared"),
        ("duplicate", "duplicates another image"),
        ("network", "Could not download image bytes"),
    ],
)
def test_validation_failures_publish_nothing(
    tmp_path: Path,
    failure: str,
    message: str,
) -> None:
    output = tmp_path / "visa-reference"
    selected = visa.evenly_spaced_indices(901)
    kwargs = {
        "bad_label_at": selected[3] if failure == "label" else None,
        "bad_dimensions_at": selected[3] if failure == "dimensions" else None,
        "duplicate_at": selected[3] if failure == "duplicate" else None,
        "fail_image_at": selected[3] if failure == "network" else None,
    }

    with pytest.raises(visa.ReferenceSetError, match=message):
        visa.fetch_reference_set(output, fetcher=FakeFetcher(**kwargs))

    assert not output.exists()
    assert list(tmp_path.glob(".visa-reference.tmp-*")) == []


def test_invalid_existing_output_is_refused_without_overwrite_or_network(
    tmp_path: Path,
) -> None:
    output = tmp_path / "visa-reference"
    visa.fetch_reference_set(output, fetcher=FakeFetcher())
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    first_image = output / manifest["files"][0]["path"]
    first_image.write_bytes(b"tampered")

    with pytest.raises(visa.ReferenceSetError, match="byte size changed"):
        visa.fetch_reference_set(output, fetcher=ExplodingFetcher())

    assert first_image.read_bytes() == b"tampered"
    assert (output / "manifest.json").is_file()
