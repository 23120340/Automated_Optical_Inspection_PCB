from __future__ import annotations

import pytest

from aoi_pipeline.classification.package import PACKAGE_CLASS_NAMES
from aoi_pipeline.models import BoundingBox, SolderJoint
from scripts.evaluate_package_roi_gate import (
    _coverage_summary,
    _evaluation_metrics,
)


def _manifest(*, macro_recall: float = 0.86, dangerous: int = 0):
    matrix = [[0 for _ in PACKAGE_CLASS_NAMES] for _ in PACKAGE_CLASS_NAMES]
    dual = PACKAGE_CLASS_NAMES.index("ic_hai_ben")
    hidden = PACKAGE_CLASS_NAMES.index("ic_khong_chan")
    matrix[dual][hidden] = dangerous
    return {
        "evaluation": {
            "split_unit": "board",
            "test_macro_recall": macro_recall,
            "test_confusion_matrix": {
                "class_names": list(PACKAGE_CLASS_NAMES),
                "matrix": matrix,
            },
        }
    }


def _joint(box: BoundingBox) -> SolderJoint:
    return SolderJoint(
        detection_id="det",
        joint_id="joint",
        label="ic",
        kind="joint",
        bbox=box,
        terminal_geometry="dual_sided",
        position="lead_top_pin_01",
    )


def test_gate_metrics_require_board_split_and_count_dangerous_pair_both_ways() -> None:
    manifest = _manifest(macro_recall=0.91, dangerous=2)
    labels = list(PACKAGE_CLASS_NAMES)
    matrix = manifest["evaluation"]["test_confusion_matrix"]["matrix"]
    matrix[labels.index("ic_khong_chan")][labels.index("ic_hai_ben")] = 3

    assert _evaluation_metrics(manifest) == (0.91, 5, "board")

    manifest["evaluation"]["split_unit"] = "crop"
    with pytest.raises(ValueError, match="split_unit"):
        _evaluation_metrics(manifest)


def test_coverage_gate_fails_visible_when_candidate_loses_a_baseline_pad() -> None:
    truth = {
        "components": {
            "U1": {"detection_index": 0, "pads": [[0, 0, 10, 10], [20, 0, 30, 10]]}
        }
    }
    baseline = {0: [_joint(BoundingBox(0, 0, 10, 10)), _joint(BoundingBox(20, 0, 30, 10))]}
    candidate = {0: [_joint(BoundingBox(0, 0, 10, 10))]}

    summary = _coverage_summary(truth, baseline, candidate)

    assert summary["pad_count"] == 2
    assert summary["baseline_covered"] == 2
    assert summary["candidate_covered"] == 1
    assert summary["coverage_not_reduced"] is False
