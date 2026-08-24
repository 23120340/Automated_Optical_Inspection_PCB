from __future__ import annotations

import ast
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import scripts.build_notebook as notebook_builder


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = PROJECT_ROOT / "training" / "kaggle" / "pcb_solder_defect_v2_kaggle.py"
NOTEBOOK_PATH = SOURCE_PATH.with_suffix(".ipynb")


class _BetaStub:
    @staticmethod
    def ppf(probability: float, first: int, second: int) -> float:
        # The unit test exercises the exact zero-event Clopper-Pearson case:
        # Beta(1, n). Its quantile has this closed form, so scipy is not a test dep.
        if first == 1:
            return 1.0 - (1.0 - probability) ** (1.0 / second)
        raise AssertionError("Unexpected beta quantile in this focused test")


def _notebook() -> dict[str, object]:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _all_source() -> str:
    return "\n".join("".join(cell["source"]) for cell in _notebook()["cells"])


def _function_namespace(
    cell_marker: str,
    function_names: set[str],
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    cell_source = next(
        "".join(cell["source"])
        for cell in _notebook()["cells"]
        if cell["cell_type"] == "code" and cell_marker in "".join(cell["source"])
    )
    tree = ast.parse(cell_source)
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in function_names
    ]
    assert {node.name for node in nodes} == function_names
    namespace: dict[str, object] = {"np": np, "beta": _BetaStub}
    namespace.update(extra or {})
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(NOTEBOOK_PATH), "exec"),
        namespace,
    )
    return namespace


def test_generated_notebook_is_current_and_every_code_cell_parses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_copy = tmp_path / SOURCE_PATH.name
    source_copy.write_text(SOURCE_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(notebook_builder, "ROOT", tmp_path)
    expected_path = notebook_builder.build(source_copy)
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    actual = _notebook()

    assert actual == expected
    assert actual["nbformat"] == 4
    for index, cell in enumerate(actual["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"solder-v2-cell-{index}")


def test_v2_has_public_bootstrap_now_and_camera_gated_mode_later() -> None:
    source = _all_source()

    assert '"run_mode": "public_bootstrap"' in source
    assert '"joint_bootstrap_v2"' in source
    assert '"class_names": ["good", "defect"]' in source
    assert '"taxonomy_profile": "joint_gate_v2"' in source
    assert '"class_names": ["good", "defect", "unknown"]' in source
    assert "component_label_routed_out_of_joint_scope" in source
    assert '"component_placement_v2"' in source
    assert '"evaluation_role": "public_proxy_holdout"' in source
    assert '"evaluation_role": "camera_locked_test"' in source
    assert '"production_allowed": False' in source
    assert '"production_allowed": True' in source
    assert '"primary_source_kind": "public"' in source
    assert '"train_only_source_kinds": {"public"}' in source
    assert 'for split in ("val", "calibration", "test")' in source
    assert '--split-pins --joints-only' in source
    assert '"production_preprocess_id"' in source
    assert "mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection" in source
    assert '"license": "GPL-3.0"' in source
    assert '"poor_solder": "insufficient"' in source
    assert '"no public unknown/wrong-crop class is synthesized"' in source


def test_mode_taxonomy_cannot_remove_camera_unknown_gate() -> None:
    namespace = _function_namespace(
        "def validate_mode_taxonomy",
        {"validate_mode_taxonomy"},
    )
    validate = namespace["validate_mode_taxonomy"]

    validate(
        "public_bootstrap", "joint_bootstrap_v2", "joint_bootstrap_v2",
        ["good", "defect"],
    )
    validate(
        "camera_finetune", "joint_gate_v2", "joint_gate_v2",
        ["good", "defect", "unknown"],
    )
    with pytest.raises(ValueError, match="bắt buộc taxonomy_profile=joint_gate_v2"):
        validate(
            "camera_finetune", "joint_bootstrap_v2", "joint_gate_v2",
            ["good", "defect"],
        )
    with pytest.raises(ValueError, match="bắt buộc class unknown"):
        validate(
            "camera_finetune", "joint_gate_v2", "joint_gate_v2",
            ["good", "defect"],
        )


def test_soldef_scope_router_separates_placement_good_from_joint_good() -> None:
    namespace = _function_namespace(
        "def _labelme_bbox",
        {"_labelme_bbox", "soldef_annotation_task"},
        {
            "math": __import__("math"),
            "normalize_label": lambda value: re.sub(
                r"[^a-z0-9]+", "_", str(value).strip().lower()
            ).strip("_"),
        },
    )
    route = namespace["soldef_annotation_task"]
    shape = lambda label, x: {
        "label": label,
        "shape_type": "rectangle",
        "points": [[x, 0], [x + 10, 10]],
    }

    assert route([shape("good", 0)]) == "component_placement"
    assert route([shape("no_good", 0)]) == "component_placement"
    assert route([shape("good", 0), shape("good", 20)]) == "solder_joint"
    assert route([shape("good", 0), shape("poor_solder", 20)]) == "solder_joint"


def test_soldef_reader_keeps_only_joint_polygons_and_preserves_subtype(
    tmp_path: Path,
) -> None:
    normalize = lambda value: re.sub(
        r"[^a-z0-9]+", "_", str(value).strip().lower()
    ).strip("_")
    namespace = _function_namespace(
        "def public_source_group",
        {
            "public_source_group", "_find_labelme_image", "_labelme_bbox",
            "soldef_annotation_task", "validate_soldef_scope_counts",
            "read_soldef_joint_records",
        },
        {
            "Path": Path,
            "json": json,
            "math": __import__("math"),
            "hashlib": __import__("hashlib"),
            "re": re,
            "Counter": __import__("collections").Counter,
            "normalize_label": normalize,
            "IMAGE_EXTENSIONS": {".jpg", ".png"},
            "_ROBOFLOW_FILE": re.compile(
                r"^(?P<stem>.+?)_(?:jpg|png)\.rf\.[0-9a-f]{6,}$", re.I
            ),
            "SOLDEF_JOINT_LABELS": {
                "good": "good", "exc_solder": "excess",
                "poor_solder": "insufficient", "spike": "spike",
            },
            "SOLDEF_PUBLISHED_SCOPE_COUNTS": {
                "component_placement": 228, "solder_joint": 200,
            },
        },
    )

    Image.new("RGB", (80, 40), "navy").save(tmp_path / "placement.jpg")
    Image.new("RGB", (80, 40), "navy").save(tmp_path / "joint.jpg")
    rectangle = lambda label, x: {
        "label": label, "shape_type": "rectangle",
        "points": [[x, 5], [x + 20, 30]],
    }
    (tmp_path / "placement.json").write_text(
        json.dumps({"imagePath": "placement.jpg", "shapes": [rectangle("good", 5)]}),
        encoding="utf-8",
    )
    (tmp_path / "joint.json").write_text(
        json.dumps({
            "imagePath": "joint.jpg",
            "shapes": [rectangle("good", 5), rectangle("poor_solder", 45)],
        }),
        encoding="utf-8",
    )

    records, report = namespace["read_soldef_joint_records"](tmp_path)

    assert [record["defect_class"] for record in records] == ["good", "insufficient"]
    assert report["task_files"] == {"component_placement": 1, "solder_joint": 1}
    assert report["scope_count_validation"]["passed"] is None
    assert all(record["dataset_source"] == "soldef_ai" for record in records)


def test_soldef_published_release_must_match_paper_task_counts() -> None:
    namespace = _function_namespace(
        "def validate_soldef_scope_counts",
        {"validate_soldef_scope_counts"},
        {
            "SOLDEF_PUBLISHED_SCOPE_COUNTS": {
                "component_placement": 228, "solder_joint": 200,
            },
        },
    )
    validate = namespace["validate_soldef_scope_counts"]

    report = validate({"component_placement": 228, "solder_joint": 200})
    assert report["passed"] is True
    with pytest.raises(RuntimeError, match="228 placement \\+ 200 solder-joint"):
        validate({"component_placement": 227, "solder_joint": 201})


def test_joint_profile_rejects_pair_bridge_and_semantically_invalid_good() -> None:
    namespace = _function_namespace(
        "def map_target_label",
        {"normalize_label", "map_target_label"},
        {
            "re": re,
            "PROFILE": {"scope": "joint"},
            "CONFIG": {"taxonomy_profile": "joint_gate_v2"},
            "TAXONOMY_PROFILE": "joint_gate_v2",
            "CLASS_NAMES": ["good", "defect", "unknown"],
            "COMPONENT_LABELS": {
                "shift_component": "shifted", "component_misalignment": "shifted",
                "good": "ok", "ok": "ok",
            },
            "GOOD_ALIASES": {"good", "ok"},
            "UNKNOWN_ALIASES": {"unknown"},
            "AMBIGUOUS_LABELS": {""},
            "SINGLE_JOINT_SUBTYPES": {"insufficient", "excess", "cold", "missing_solder"},
        },
    )
    route = namespace["map_target_label"]

    assert route({
        "defect_class": "good", "label_status": "source_annotation",
        "roi_kind": "joint", "label_scope": "joint", "source_kind": "public",
    }) == ("good", "mapped")
    assert route({
        "defect_class": "component_misalignment", "label_status": "source_annotation",
        "roi_kind": "joint", "label_scope": "joint", "source_kind": "public",
    }) == (None, "component_label_routed_out_of_joint_scope")
    assert route({
        "defect_class": "bridge", "label_status": "verified",
        "roi_kind": "joint", "label_scope": "pair", "source_kind": "local",
    }) == (None, "bridge_is_pair_rule_not_single_joint_classifier")
    assert route({
        "defect_class": "good", "label_status": "verified",
        "roi_kind": "joint", "label_scope": "invalid", "source_kind": "local",
    }) == (None, "invalid_scope_must_be_labeled_unknown")
    assert route({
        "defect_class": "unknown", "label_status": "verified",
        "roi_kind": "joint", "label_scope": "invalid", "source_kind": "local",
    }) == ("unknown", "mapped")


def test_v2_has_four_grouped_splits_and_fail_closed_leakage_audit() -> None:
    source = _all_source()

    assert 'SPLIT_ORDER = ("train", "val", "calibration", "test")' in source
    assert "def assign_board_splits" in source
    assert "def audit_split_integrity" in source
    assert "cross_split_board_leakage" in source
    assert "cross_split_sha256_leakage" in source
    assert "cross_split_phash_leakage" in source
    assert "conflicting_labels" in source
    assert "split_assignment_conflicts" in source
    assert "locked_test_explicit" in source
    assert "public_rows_outside_train" in source
    assert 'drop_duplicates(["dedup_cluster", "target_label"]' in source
    assert 'drop_duplicates(["duplicate_cluster", "target_label"]' not in source
    assert 'valid.groupby("dedup_cluster")' in source
    assert 'valid.groupby("near_duplicate_cluster").size().max()' in source
    assert '"annotation_instance_id"' in source
    assert 'valid["duplicate_cluster"] = valid["dedup_cluster"]' in source
    assert '"cross_split_phash_candidates": crossing("near_duplicate_cluster")' in source


def test_auto_grouping_never_adds_rows_to_an_explicit_locked_test() -> None:
    namespace = _function_namespace(
        "def assign_board_splits",
        {"assign_board_splits"},
        {
            "pd": pd,
            "SPLIT_ORDER": ("train", "val", "calibration", "test"),
            "CONFIG": {"allow_research_auto_split": False},
            "PRIMARY_SOURCE_KIND": "local",
            "TRAIN_ONLY_SOURCE_KINDS": {"public"},
            "MODE_POLICY": {"allow_auto_split": False},
            "RUN_MODE": "camera_finetune",
        },
    )
    rows = []
    for class_name in ("good", "defect", "unknown"):
        rows.append({
            "source_kind": "local",
            "split_requested": "test",
            "leakage_group": f"locked-{class_name}",
            "target_label": class_name,
        })
        for index in range(6):
            rows.append({
                "source_kind": "local",
                "split_requested": "",
                "leakage_group": f"auto-{class_name}-{index}",
                "target_label": class_name,
            })
    frame = pd.DataFrame(rows)
    assigned = namespace["assign_board_splits"](
        frame,
        {"train": 0.65, "val": 0.15, "calibration": 0.10, "test": 0.10},
        seed=42,
        attempts=20,
    )

    test_groups = set(assigned.loc[assigned["split"].eq("test"), "leakage_group"])
    assert test_groups == {"locked-good", "locked-defect", "locked-unknown"}


def test_public_bootstrap_group_splits_public_across_all_proxy_partitions() -> None:
    namespace = _function_namespace(
        "def assign_board_splits",
        {"assign_board_splits"},
        {
            "pd": pd,
            "SPLIT_ORDER": ("train", "val", "calibration", "test"),
            "PRIMARY_SOURCE_KIND": "public",
            "TRAIN_ONLY_SOURCE_KINDS": {"local"},
            "MODE_POLICY": {"allow_auto_split": True},
            "RUN_MODE": "public_bootstrap",
        },
    )
    rows = [
        {
            "source_kind": "public",
            "split_requested": "",
            "leakage_group": f"{label}-{index}",
            "target_label": label,
        }
        for label in ("good", "defect")
        for index in range(20)
    ]
    assigned = namespace["assign_board_splits"](
        pd.DataFrame(rows),
        {"train": 0.65, "val": 0.15, "calibration": 0.10, "test": 0.10},
        seed=42,
        attempts=100,
    )

    assert set(assigned["split"]) == {"train", "val", "calibration", "test"}
    assert not assigned["split_origin"].eq("public_train_only").any()


def test_decision_metrics_do_not_call_good_review_a_false_reject() -> None:
    namespace = _function_namespace(
        "def compute_decision_metrics",
        {"safe_rate", "group_event_counts", "compute_decision_metrics"},
    )
    probabilities = np.asarray(
        [
            [0.05, 0.90, 0.05],  # good -> confidently defect
            [0.90, 0.05, 0.05],  # good -> auto good
            [0.90, 0.05, 0.05],  # defect -> escape
            [0.40, 0.35, 0.25],  # defect -> review
            [0.90, 0.05, 0.05],  # unknown -> invalid auto-good
        ],
        dtype=np.float64,
    )
    metrics = namespace["compute_decision_metrics"](
        np.asarray([0, 0, 1, 1, 2]),
        probabilities,
        ["good", "defect", "unknown"],
        {"good": 0.80, "defect": 0.80, "unknown": 0.80},
    )

    assert metrics["escape"] == pytest.approx(0.50)
    assert metrics["false_reject"] == pytest.approx(0.50)
    assert metrics["good_review"] == pytest.approx(0.0)
    assert metrics["defect_review"] == pytest.approx(0.50)
    assert metrics["review_rate"] == pytest.approx(0.20)
    assert metrics["invalid_good_accept"] == pytest.approx(1.0)

    clustered = namespace["compute_decision_metrics"](
        np.asarray([0, 0, 1, 1, 2]),
        probabilities,
        ["good", "defect", "unknown"],
        {"good": 0.80, "defect": 0.80, "unknown": 0.80},
        groups=np.asarray(["good-a", "good-b", "defect-board", "defect-board", "ood-a"]),
    )
    assert clustered["escape_group_events"] == 1
    assert clustered["defect_group_total"] == 1


def test_one_sided_interval_requires_about_299_zero_escape_defects() -> None:
    namespace = _function_namespace(
        "def clopper_pearson_upper",
        {"clopper_pearson_upper", "clopper_pearson_lower"},
    )
    upper = namespace["clopper_pearson_upper"]
    lower = namespace["clopper_pearson_lower"]

    assert upper(0, 298, 0.05) > 0.01
    assert upper(0, 299, 0.05) <= 0.01
    assert upper(0, 0, 0.05) == 1.0
    assert lower(0, 100, 0.05) == 0.0


def test_calibration_onnx_and_artifact_contract_are_fail_closed() -> None:
    source = _all_source()

    assert "def fit_temperature" in source
    assert 'loaders["calibration"]' in source
    assert "def choose_operating_point" in source
    assert '"runtime_accept_floor": 0.80' in source
    assert '"schema_version": "pcb-solder-defect-classifier/1.0"' in source
    assert '"type": "raw_logits"' in source
    assert '"parity_max_abs_error"' in source
    assert '"parity_pass"' in source
    assert '"decision_equal"' in source
    assert '"argmax_equal"' in source
    assert '"public_proxy_holdout_crops"' in source
    assert '"camera_locked_test_real_crops"' in source
    assert 'if gate["production_ready"]:' in source
    assert 'PRODUCTION_DIR / "best.onnx"' in source
    assert "QUALITY GATE FAIL — không tạo best.onnx" in source
    assert "pcb_solder_defect_v2_candidate_artifacts.zip" in source
    assert "pcb_solder_defect_v2_public_bootstrap_artifacts.zip" in source
    assert "pcb_solder_defect_v2_production_artifacts.zip" in source
    assert 'artifact_status="bootstrap_only"' in source
    assert '"requires_camera_finetune": RUN_MODE == "public_bootstrap"' in source
    assert '"production_ready": False' in source
    assert '"bootstrap_checkpoint.pt"' in source


def test_quality_gate_can_pass_only_with_complete_board_level_evidence() -> None:
    namespace = _function_namespace(
        "def quality_gate",
        {"quality_gate"},
        {
            "CLASS_NAMES": ["good", "defect", "unknown"],
            "GOOD_LABEL": "good",
            "ood_control_report": {"auto_good": 0},
        },
    )
    rows = []
    for split, count in (("val", 2), ("calibration", 2)):
        for label in ("good", "defect", "unknown"):
            rows.extend(
                {"split": split, "target_label": label, "physical_board": f"{split}-{label}-{i}"}
                for i in range(count)
            )
    rows.extend(
        {"split": "test", "target_label": "defect", "physical_board": f"d-{i}"}
        for i in range(299)
    )
    rows.extend(
        {"split": "test", "target_label": "unknown", "physical_board": f"u-{i}"}
        for i in range(299)
    )
    rows.extend(
        {"split": "test", "target_label": "good", "physical_board": f"g-{i}"}
        for i in range(100)
    )
    frame = pd.DataFrame(rows)
    config = {
        "onnx_parity_atol": 1e-3,
        "minimum_test_defects": 299,
        "minimum_test_good": 100,
        "minimum_test_unknown": 100,
        "minimum_test_boards": 20,
        "minimum_test_defect_boards": 299,
        "minimum_test_unknown_boards": 299,
        "minimum_test_good_boards": 30,
        "escape_target": 0.01,
        "invalid_good_accept_target": 0.01,
        "maximum_false_reject": 0.02,
        "minimum_macro_f1": 0.80,
        "minimum_defect_recall": 0.95,
        "max_good_review_rate": 0.20,
        "minimum_local_label_retained_ratio": 0.80,
        "maximum_invalid_image_ratio": 0.02,
        "minimum_groups_per_class_eval": 2,
        "minimum_subtype_boards": 59,
        "minimum_subtype_defect_recall": 0.90,
        "subtype_escape_target": 0.05,
        "minimum_per_class_f1": 0.60,
    }
    audit = {
        "conflicting_labels": [], "split_assignment_conflicts": [],
        "physical_board_id_missing": 0, "verified_label_status_missing": 0,
        "canonical_contract_incomplete": 0, "preprocess_id_mismatch": 0,
        "local_label_retained_ratio": 1.0, "invalid_image_ratio": 0.0,
    }
    split_audit = {
        "cross_split_board_leakage": [], "cross_split_sha256_leakage": [],
        "cross_split_phash_leakage": [], "public_rows_outside_train": 0,
    }
    metrics = {
        "escape_ci_upper": 0.00999, "invalid_good_accept_ci_upper": 0.00999,
        "false_reject": 0.0, "false_reject_ci_upper": 0.0199,
        "macro_f1": 1.0, "defect_recall": 1.0,
        "good_review": 0.0,
    }
    expected_subtypes = ["cold", "excess", "insufficient", "missing_solder"]
    subtype = [
        {
            "defect_class": name, "board_count": 299,
            "defect_detection_recall": 1.0, "escape_ci_upper": 0.00999,
        }
        for name in expected_subtypes
    ]
    classification = {
        name: {"f1-score": 1.0} for name in ("good", "defect", "unknown")
    }

    gate = namespace["quality_gate"](
        audit, split_audit, metrics, frame, True, 1e-6, True, True,
        config, subtype, classification, expected_subtypes,
    )

    assert gate["production_ready"] is True
    assert gate["failures"] == []

    missing_subtype_gate = namespace["quality_gate"](
        audit, split_audit, metrics, frame, True, 1e-6, True, True,
        config, subtype[:-1], classification, expected_subtypes,
    )
    assert missing_subtype_gate["production_ready"] is False
    assert "missing_locked_test_subtype[missing_solder]" in missing_subtype_gate["failures"]
