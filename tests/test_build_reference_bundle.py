from __future__ import annotations

import csv
from hashlib import sha256
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_reference_bundle as bundle  # noqa: E402

from aoi_pipeline.golden.enrollment import (  # noqa: E402
    ReferenceSelectionReport,
    ReferenceSelectionResult,
)
from aoi_pipeline.models import (  # noqa: E402
    AlignmentResult,
    BoardRegion,
    BoundingBox,
    Detection,
)


def _reference_set(root: Path, count: int = 3) -> list[Path]:
    images = root / "images"
    images.mkdir(parents=True)
    entries = []
    paths = []
    for index in range(count):
        image = np.full((120, 180, 3), (38, 42, 45), dtype=np.uint8)
        cv2.rectangle(image, (20, 20), (160, 100), (170 + index, 90, 25), -1)
        cv2.rectangle(image, (48, 45), (78, 70), (20, 20, 20), -1)
        cv2.rectangle(image, (108, 50), (138, 68), (220, 220, 220), -1)
        path = images / f"frame_{index:02d}.png"
        assert cv2.imwrite(str(path), image)
        payload = path.read_bytes()
        entries.append(
            {
                "row_index": index,
                "path": f"images/{path.name}",
                "label": 0,
                "width": 180,
                "height": 120,
                "byte_size": len(payload),
                "sha256": sha256(payload).hexdigest(),
            }
        )
        paths.append(path)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "source_kind": "dataset_preprocessed",
                "files": entries,
            }
        ),
        encoding="utf-8",
    )
    return paths


def _declare_upstream_alignment(root: Path) -> None:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "source_kind": "dataset_preprocessed_aligned",
            "board": {
                "upstream_registration": "registered by the upstream dataset"
            },
            "selection": {
                "same_layout_consensus_allowed": True,
                "different_layout_mixing_allowed": False,
            },
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


class _FakeRuntime:
    @property
    def identifier(self) -> str:
        return "FakeDetector:test"

    def align(self, image: np.ndarray, reference: np.ndarray) -> AlignmentResult:
        return AlignmentResult(
            image=image.copy(),
            method="orb_homography",
            success=True,
            homography=np.eye(3, dtype=np.float64),
            inliers=20,
            inlier_ratio=1.0,
        )

    def localize(self, image: np.ndarray) -> BoardRegion:
        return BoardRegion(
            bbox=BoundingBox(20, 20, 161, 101),
            polygon=[(160, 100), (20, 100), (20, 20), (160, 20)],
            confidence=0.9,
            method="test",
        )

    def detect(
        self,
        image: np.ndarray,
        board_region: BoardRegion,
        *,
        frame_id: str,
    ) -> list[Detection]:
        jitter = float(int(frame_id[-6:-4])) * 0.2
        return [
            Detection(
                "ic",
                0.93,
                BoundingBox(48 + jitter, 45, 78 + jitter, 70),
                source="ultralytics",
            ),
            Detection(
                "resistor",
                0.88,
                BoundingBox(108 + jitter, 50, 138 + jitter, 68),
                source="ultralytics",
            ),
            Detection(
                "pads",
                0.95,
                BoundingBox(18, 18, 24, 24),
                source="ultralytics",
            ),
        ]


class _UpstreamIdentityRuntime(_FakeRuntime):
    def __init__(self) -> None:
        self.align_call_count = 0
        self.detected_frame_ids: list[str] = []

    def align(self, image: np.ndarray, reference: np.ndarray) -> AlignmentResult:
        self.align_call_count += 1
        raise AssertionError("runtime.align must not run in upstream alignment mode")

    def detect(
        self,
        image: np.ndarray,
        board_region: BoardRegion,
        *,
        frame_id: str,
    ) -> list[Detection]:
        self.detected_frame_ids.append(frame_id)
        return super().detect(image, board_region, frame_id=frame_id)


class _GoldenAnchoredSubsetRuntime(_FakeRuntime):
    """Add one consensus site that is absent from the selected Golden frame."""

    def detect(
        self,
        image: np.ndarray,
        board_region: BoardRegion,
        *,
        frame_id: str,
    ) -> list[Detection]:
        detections = super().detect(image, board_region, frame_id=frame_id)
        if frame_id != "frame_01.png":
            detections.append(
                Detection(
                    "capacitor",
                    0.81,
                    BoundingBox(125, 78, 150, 94),
                    source="ultralytics",
                )
            )
        return detections


def _selector(sources, *, config):
    selected = Path(sources[1])
    payload = selected.read_bytes()
    report = ReferenceSelectionReport(
        algorithm="test_source_medoid",
        source_count=len(sources),
        image_size=(120, 180),
        diagnostic_image_size=(120, 180),
        quality_candidate_count=len(sources),
        required_peers=2,
        selected_basename=selected.name,
        selected_sha256=sha256(payload).hexdigest(),
        candidates=(),
    )
    return ReferenceSelectionResult(selected, report)


def test_provisional_registration_maps_ordered_corners_to_nominal_mm() -> None:
    unordered = [(120, 90), (10, 20), (110, 20), (20, 90)]
    matrix, registration, scale = bundle.provisional_registration(
        unordered,
        board_width_mm=50.0,
        board_height_mm=25.0,
    )
    ordered = bundle.order_board_quad(unordered)
    points = np.column_stack((ordered, np.ones(4))) @ matrix.T
    projected = points[:, :2] / points[:, 2, None]

    assert np.allclose(
        projected,
        [[0, 0], [50, 0], [50, 25], [0, 25]],
        atol=1e-5,
    )
    assert registration.ambiguous is True
    assert registration.inlier_ratio == 0.0
    assert scale[0] > 0 and scale[1] > 0


def test_reference_manifest_is_rechecked_against_file_bytes(tmp_path: Path) -> None:
    root = tmp_path / "refs"
    paths = _reference_set(root)
    inputs, _, _ = bundle.load_reference_inputs(root, expected_count=3)
    assert [item.frame_id for item in inputs] == [path.name for path in paths]

    paths[0].write_bytes(paths[0].read_bytes() + b"tamper")
    with pytest.raises(bundle.BundleError, match="SHA-256"):
        bundle.load_reference_inputs(root, expected_count=3)


def test_reference_manifest_rejects_explicit_same_layout_consensus_denial(
    tmp_path: Path,
) -> None:
    root = tmp_path / "refs"
    _reference_set(root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["selection"] = {"same_layout_consensus_allowed": False}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(bundle.BundleError, match="forbids same-layout consensus"):
        bundle.load_reference_inputs(root, expected_count=3)


def test_reference_manifest_rejects_explicit_different_layout_mixing_permission(
    tmp_path: Path,
) -> None:
    root = tmp_path / "refs"
    _reference_set(root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["selection"] = {"different_layout_mixing_allowed": True}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(bundle.BundleError, match="allows different-layout mixing"):
        bundle.load_reference_inputs(root, expected_count=3)


def test_upstream_alignment_refuses_manifest_without_registration_provenance(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "refs"
    _reference_set(reference_root)
    manifest_path = reference_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "source_kind": "dataset_preprocessed_aligned",
            "board": {"upstream_registration": "  "},
            "selection": {
                "same_layout_consensus_allowed": True,
                "different_layout_mixing_allowed": False,
            },
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        bundle.BundleError,
        match=r"non-empty board\.upstream_registration",
    ):
        bundle.build_reference_bundle(
            bundle.BundleConfig(
                reference_root=reference_root,
                output_dir=tmp_path / "output",
                model_path=tmp_path / "unused.onnx",
                board_id="TEST_BOARD",
                side="top",
                expected_count=3,
                use_upstream_alignment=True,
            ),
            selector=_selector,
            runtime=_UpstreamIdentityRuntime(),
        )


def test_upstream_alignment_uses_identity_and_detects_every_manifest_frame(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "refs"
    paths = _reference_set(reference_root)
    _declare_upstream_alignment(reference_root)
    output = tmp_path / "golden" / "upstream_pixel_v1"
    runtime = _UpstreamIdentityRuntime()

    result = bundle.build_reference_bundle(
        bundle.BundleConfig(
            reference_root=reference_root,
            output_dir=output,
            model_path=tmp_path / "unused.onnx",
            board_id="TEST_BOARD",
            side="top",
            expected_count=3,
            min_aligned_ratio=1.0,
            use_upstream_alignment=True,
        ),
        selector=_selector,
        runtime=runtime,
    )

    assert result.accepted_frame_count == 3
    assert runtime.align_call_count == 0
    assert runtime.detected_frame_ids == [path.name for path in paths]

    report = json.loads(
        (output / "alignment_report.json").read_text(encoding="utf-8")
    )
    assert report["alignment_mode"] == "upstream_dataset_identity_enrollment_only"
    assert report["alignment_provenance"] == {
        "description": "registered by the upstream dataset",
        "kind": "manifest_declared_upstream_dataset_registration",
        "production_registration_eligible": False,
        "scope": "enrollment_only",
    }
    by_frame = {row["frame_id"]: row for row in report["frames"]}
    assert by_frame[paths[1].name]["method"] == "selected_reference_identity"
    for path in (paths[0], paths[2]):
        row = by_frame[path.name]
        assert row["method"] == "upstream_dataset_identity"
        assert row["success"] is True
        assert row["accepted_for_detection"] is True
        assert np.allclose(row["homography"], np.eye(3))
        assert row["inlier_ratio"] is None
        assert row["correlation"] is None
        assert row["fit_metrics_status"] == "not_measured_upstream_identity"
        assert "enrollment only" in row["message"]
        assert "not production camera/fixture registration" in row["message"]
        assert row["production_registration_eligible"] is False

    manifest = json.loads(
        (output / "bundle_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["alignment_mode"] == "upstream_dataset_identity_enrollment_only"


def test_build_writes_review_only_recipe_consensus_pnp_and_overlay(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "refs"
    paths = _reference_set(reference_root)
    output = tmp_path / "golden" / "draft_v1"
    result = bundle.build_reference_bundle(
        bundle.BundleConfig(
            reference_root=reference_root,
            output_dir=output,
            model_path=tmp_path / "unused.onnx",
            board_id="TEST_BOARD",
            side="bottom",
            board_width_mm=50.0,
            board_height_mm=25.0,
            expected_count=3,
            min_aligned_ratio=1.0,
        ),
        selector=_selector,
        runtime=_FakeRuntime(),
    )

    assert result.selected_source == paths[1]
    assert result.accepted_frame_count == 3
    assert result.component_count == 2
    assert result.eligible_component_count == 2
    assert (reference_root / "reference_selection.json").is_file()
    for name in (
        "golden.png",
        "recipe.json",
        "consensus_components.json",
        "consensus_components.csv",
        "pnp_pixels_NEEDS_REVIEW.csv",
        "placement_draft_NEEDS_REVIEW.csv",
        "registration_draft_NEEDS_REVIEW.json",
        "overlay_consensus.png",
        "overlay_pnp_NEEDS_REVIEW.png",
        "README_FIRST.md",
        "NEEDS_REVIEW.md",
        "bundle_manifest.json",
    ):
        assert (output / name).is_file(), name

    recipe = json.loads((output / "recipe.json").read_text(encoding="utf-8"))
    assert recipe["production_eligible"] is False
    assert recipe["metrology"]["verified"] is False
    assert recipe["alignment"]["anchor_provenance"] == "demo_grid"
    assert len(recipe["slots"]) == 2

    consensus = json.loads(
        (output / "consensus_components.json").read_text(encoding="utf-8")
    )
    assert consensus["artifact_status"] == "AUTHORITATIVE_PIXEL_CONSENSUS"
    assert consensus["summary"]["excluded_observation_count"] == 3

    with (output / "placement_draft_NEEDS_REVIEW.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert all(row["Status"] == "NEEDS_REVIEW" for row in rows)
    assert all(row["Rotation"] == "" for row in rows)
    assert all("_AUTO_" in row["Designator"] for row in rows)

    registration = json.loads(
        (output / "registration_draft_NEEDS_REVIEW.json").read_text(
            encoding="utf-8"
        )
    )
    assert registration["verified"] is False
    assert registration["draft_axis_convention"]["bottom_side_mirroring_verified"] is False

    manifest = json.loads(
        (output / "bundle_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "NEEDS_REVIEW"
    assert manifest["production_eligible"] is False
    assert manifest["pnp_pixels_review"] == {
        "authoritative": False,
        "component_count": 2,
        "coordinate_space": "golden_board_pixels",
        "proposal_basis": (
            "median_consensus_cluster_with_selected_golden_observation"
        ),
        "schema_version": "aoi-pnp-pixels-review-draft/1.0",
        "selected_golden_frame_id": paths[1].name,
        "status": "NEEDS_REVIEW",
        "verified": False,
    }
    assert all(not Path(item["path"]).is_absolute() for item in manifest["artifacts"])


def test_pixel_only_build_does_not_invent_mm_or_recipe(tmp_path: Path) -> None:
    reference_root = tmp_path / "refs"
    paths = _reference_set(reference_root)
    output = tmp_path / "golden" / "pixel_bootstrap_v1"

    result = bundle.build_reference_bundle(
        bundle.BundleConfig(
            reference_root=reference_root,
            output_dir=output,
            model_path=tmp_path / "unused.onnx",
            board_id="TEST_BOARD",
            side="top",
            expected_count=3,
            min_aligned_ratio=1.0,
        ),
        selector=_selector,
        runtime=_FakeRuntime(),
    )

    assert result.selected_source == paths[1]
    for name in (
        "golden.png",
        "consensus_components.json",
        "consensus_components.csv",
        "pnp_pixels_NEEDS_REVIEW.csv",
        "alignment_report.json",
        "overlay_consensus.png",
        "overlay_pnp_NEEDS_REVIEW.png",
        "README_FIRST.md",
        "NEEDS_REVIEW.md",
        "bundle_manifest.json",
    ):
        assert (output / name).is_file(), name
    for name in (
        "recipe.json",
        "placement_draft_NEEDS_REVIEW.csv",
        "registration_draft_NEEDS_REVIEW.json",
    ):
        assert not (output / name).exists(), name

    manifest = json.loads(
        (output / "bundle_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["bundle_mode"] == "pixel_only_no_physical_calibration"
    assert manifest["recipe_emitted"] is False
    assert manifest["deferred_without_physical_calibration"] == [
        "recipe.json",
        "placement_draft_NEEDS_REVIEW.csv",
        "registration_draft_NEEDS_REVIEW.json",
    ]
    readme = (output / "README_FIRST.md").read_text(encoding="utf-8")
    assert "intentionally absent" in readme
    assert "non-authoritative proposal queue" in readme
    assert "not human-verified component ground truth or PnP" in readme
    assert "Inclusion in\n`pnp_pixels_NEEDS_REVIEW.csv`" in readme
    assert "either gate colour can occur" in readme
    assert "excluded from the default PnP rows" not in readme

    with (output / "pnp_pixels_NEEDS_REVIEW.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        assert all("mm" not in name.lower() for name in reader.fieldnames)
        rows = list(reader)
    assert len(rows) == 2
    assert all(row["artifact_status"] == "NEEDS_REVIEW" for row in rows)
    assert all(row["rotation_deg"] == "" for row in rows)
    assert all(row["footprint"] == "" for row in rows)

    assert manifest["pnp_pixels_review"]["component_count"] == len(rows)
    artifact_paths = {item["path"] for item in manifest["artifacts"]}
    assert {
        "pnp_pixels_NEEDS_REVIEW.csv",
        "overlay_pnp_NEEDS_REVIEW.png",
    } <= artifact_paths


def test_review_pnp_is_exact_selected_golden_consensus_subset(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "refs"
    paths = _reference_set(reference_root)
    output = tmp_path / "golden" / "pixel_subset_v1"

    bundle.build_reference_bundle(
        bundle.BundleConfig(
            reference_root=reference_root,
            output_dir=output,
            model_path=tmp_path / "unused.onnx",
            board_id="TEST_BOARD",
            side="top",
            expected_count=3,
            min_aligned_ratio=1.0,
        ),
        selector=_selector,
        runtime=_GoldenAnchoredSubsetRuntime(),
    )

    consensus = json.loads(
        (output / "consensus_components.json").read_text(encoding="utf-8")
    )
    assert consensus["summary"]["component_count"] == 3
    expected = {
        component["designator"]: component
        for component in consensus["components"]
        if paths[1].name in component["frame_ids"]
    }
    assert len(expected) == 2

    with (output / "pnp_pixels_NEEDS_REVIEW.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert {row["designator"] for row in rows} == set(expected)
    for row in rows:
        component = expected[row["designator"]]
        assert row["schema_version"] == "aoi-pnp-pixels-review-draft/1.0"
        assert row["artifact_status"] == "NEEDS_REVIEW"
        assert row["review_status"] == "NEEDS_REVIEW"
        assert row["coordinate_space"] == "golden_board_pixels"
        assert row["proposal_basis"] == (
            "median_consensus_cluster_with_selected_golden_observation"
        )
        assert row["selected_golden_frame_id"] == paths[1].name
        assert row["selected_golden_sha256"] == sha256(
            paths[1].read_bytes()
        ).hexdigest()
        assert "synthetic_auto_designator" in row["review_reasons"]
        assert json.loads(row["frame_ids_json"]).count(paths[1].name) == 1
        assert float(row["center_x_px"]) == pytest.approx(
            component["center_px"][0]
        )
        assert float(row["center_y_px"]) == pytest.approx(
            component["center_px"][1]
        )
        for field, value in zip(
            ("x1_px", "y1_px", "x2_px", "y2_px"),
            component["bbox_xyxy"],
            strict=True,
        ):
            assert float(row[field]) == pytest.approx(value)
        assert float(row["support_ratio"]) == pytest.approx(
            component["support_ratio"]
        )
        assert float(row["median_confidence"]) == pytest.approx(
            component["median_confidence"]
        )

    golden = cv2.imread(str(output / "golden.png"), cv2.IMREAD_COLOR)
    overlay = cv2.imread(
        str(output / "overlay_pnp_NEEDS_REVIEW.png"),
        cv2.IMREAD_COLOR,
    )
    assert golden is not None and overlay is not None
    # The non-Golden-anchored capacitor is retained in full consensus but must
    # not be drawn in the review-PnP overlay.
    assert np.array_equal(golden[77:81, 124:152], overlay[77:81, 124:152])
    assert not np.array_equal(golden[44:48, 47:80], overlay[44:48, 47:80])

    manifest = json.loads(
        (output / "bundle_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["pnp_pixels_review"]["component_count"] == len(rows)


def test_board_dimensions_must_be_supplied_as_a_pair(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be supplied together"):
        bundle.BundleConfig(
            reference_root=tmp_path / "refs",
            output_dir=tmp_path / "output",
            model_path=tmp_path / "unused.onnx",
            board_id="TEST_BOARD",
            side="top",
            board_width_mm=50.0,
        )


def test_existing_output_is_not_overwritten(tmp_path: Path) -> None:
    reference_root = tmp_path / "refs"
    _reference_set(reference_root)
    output = tmp_path / "already_here"
    output.mkdir()
    marker = output / "user.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(bundle.BundleError, match="Output already exists"):
        bundle.build_reference_bundle(
            bundle.BundleConfig(
                reference_root=reference_root,
                output_dir=output,
                model_path=tmp_path / "unused.onnx",
                board_id="TEST_BOARD",
                side="bottom",
                board_width_mm=50,
                board_height_mm=25,
                expected_count=3,
            ),
            selector=_selector,
            runtime=_FakeRuntime(),
        )
    assert marker.read_text(encoding="utf-8") == "keep"
