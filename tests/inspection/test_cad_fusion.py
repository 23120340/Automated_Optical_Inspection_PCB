"""CAD loading, registration and fusion with the detector-derived ROIs."""

from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from aoi_pipeline import (
    AOIPipeline,
    BoardCad,
    BoundingBox,
    CadComponent,
    CadError,
    CadPad,
    CadRegistration,
    Detection,
    FusionConfig,
    PipelineConfig,
    SolderJointConfig,
    classes_agree,
    designator_to_class,
    fuse_solder_joints,
    is_informative_label,
    load_cad,
    register_cad,
    register_from_fiducials,
    save_cad_json,
)
from aoi_pipeline.detection.detectors import MockComponentDetector
from aoi_pipeline.reporting.exporters import cad_findings_csv, solder_joints_csv
from aoi_pipeline.solder.geometry import SolderJointCropper

IMAGE_SIZE = (600, 400)  # width, height
PX_PER_MM = 8.0
ORIGIN = (40.0, 30.0)
BOARD_HEIGHT_MM = 40.0

# Deliberately asymmetric: a symmetric layout has a perfect mirrored fit, and
# tests that pass on one are not evidence about real boards.
PARTS = [
    ("R1", "resistor", (10.0, 30.0), False),
    ("R2", "resistor", (25.0, 30.0), False),
    ("C1", "capacitor", (41.0, 29.0), True),
    ("R3", "resistor", (11.0, 14.0), False),
    ("C2", "capacitor", (27.0, 15.0), False),
    ("U1", "ic", (44.0, 16.0), False),
]
PAD_OFFSET_MM = 1.0


def mm_to_px(x_mm: float, y_mm: float) -> tuple[float, float]:
    return (
        ORIGIN[0] + x_mm * PX_PER_MM,
        ORIGIN[1] + (BOARD_HEIGHT_MM - y_mm) * PX_PER_MM,
    )


def _pad_positions(centre: tuple[float, float], vertical: bool):
    cx, cy = centre
    if vertical:
        return [(cx, cy - PAD_OFFSET_MM), (cx, cy + PAD_OFFSET_MM)]
    return [(cx - PAD_OFFSET_MM, cy), (cx + PAD_OFFSET_MM, cy)]


def write_pads_csv(path: Path, parts=PARTS) -> Path:
    rows = ["designator,pin,x_mm,y_mm,width_mm,height_mm,side,footprint"]
    for designator, _, centre, vertical in parts:
        for pin, (x, y) in enumerate(_pad_positions(centre, vertical), start=1):
            rows.append(f"{designator},{pin},{x:.3f},{y:.3f},0.9,1.0,top,CHIP")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def write_placement_csv(path: Path, parts=PARTS) -> Path:
    rows = ["Designator,Mid X,Mid Y,Rotation,Layer,Footprint"]
    for designator, _, (cx, cy), vertical in parts:
        rows.append(
            f"{designator},{cx:.3f},{cy:.3f},{90 if vertical else 0},Top,CHIP"
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def make_detections(skip: set[str] = frozenset(), extra: bool = True, parts=None):
    detections = []
    for designator, part_class, centre, vertical in (parts if parts is not None else PARTS):
        if designator in skip:
            continue
        cx, cy = mm_to_px(*centre)
        half = (
            np.array([5.0, PAD_OFFSET_MM * PX_PER_MM * 0.85])
            if vertical
            else np.array([PAD_OFFSET_MM * PX_PER_MM * 0.85, 5.0])
        )
        detections.append(
            Detection(
                part_class,
                0.9,
                BoundingBox(cx - half[0], cy - half[1], cx + half[0], cy + half[1]),
            )
        )
    if extra:
        detections.append(Detection("led", 0.8, BoundingBox(500, 300, 540, 320)))
    return detections


def board_image() -> np.ndarray:
    return np.full((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), (40, 90, 40), np.uint8)


def derived_joints(detections):
    cropper = SolderJointCropper(SolderJointConfig())
    return cropper.derive(board_image(), detections)


def fuse(board, registration, detections=None, config=None):
    detections = detections if detections is not None else make_detections()
    return fuse_solder_joints(
        detections,
        derived_joints(detections),
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        board=board,
        registration=registration,
        config=config or FusionConfig(),
    )


def truth_registration() -> CadRegistration:
    cad_points = [(0.0, 0.0), (40.0, 0.0), (0.0, 30.0)]
    image_points = [mm_to_px(*point) for point in cad_points]
    return register_from_fiducials(cad_points, image_points)


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #


def test_pads_csv_groups_lands_into_components(tmp_path: Path) -> None:
    board = load_cad(write_pads_csv(tmp_path / "pads.csv"))
    assert board.source_format == "pads_csv"
    assert len(board.components) == len(PARTS)
    assert board.pad_count == 2 * len(PARTS)
    r1 = next(c for c in board.components if c.designator == "R1")
    assert r1.part_class == "resistor"
    # The placement is the centroid of its own lands.
    assert r1.x == pytest.approx(10.0)
    assert r1.y == pytest.approx(30.0)
    assert r1.pad_span_mm() == pytest.approx(2 * PAD_OFFSET_MM)


def test_pads_csv_reads_loose_header_names(tmp_path: Path) -> None:
    path = tmp_path / "alt.csv"
    path.write_text(
        "RefDes,Pad Number,X,Y\nR9,1,1.0,2.0\nR9,2,3.0,2.0\n", encoding="utf-8"
    )
    board = load_cad(path)
    assert board.components[0].designator == "R9"
    assert len(board.components[0].pads) == 2


def test_placement_csv_loads_without_lands(tmp_path: Path) -> None:
    board = load_cad(write_placement_csv(tmp_path / "cpl.csv"))
    assert board.source_format == "placement_csv"
    assert len(board.components) == len(PARTS)
    assert board.pad_count == 0
    c1 = next(c for c in board.components if c.designator == "C1")
    assert c1.rotation == pytest.approx(90.0)
    assert c1.part_class == "capacitor"


def test_ipc356_reads_pad_coordinates(tmp_path: Path) -> None:
    path = tmp_path / "netlist.ipc"
    path.write_text(
        "P  JOB TEST\n"
        "P  UNITS CUST 1\n"
        "317VCC           R1    -1  A01X+010000Y+030000X0900Y1000\n"
        "317VCC           R1    -2  A01X+012000Y+030000X0900Y1000\n"
        "999\n",
        encoding="utf-8",
    )
    board = load_cad(path)
    assert board.source_format == "ipc356"
    component = board.components[0]
    assert component.designator == "R1"
    assert len(component.pads) == 2
    # CUST 1 means 0.001 mm units.
    assert component.pads[0].x == pytest.approx(10.0)
    assert component.pads[1].x == pytest.approx(12.0)


def test_unreadable_file_raises_a_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "junk.csv"
    path.write_text("alpha,beta\n1,2\n", encoding="utf-8")
    with pytest.raises(CadError, match="could not identify"):
        load_cad(path)
    with pytest.raises(CadError, match="not found"):
        load_cad(tmp_path / "absent.csv")


def test_json_round_trip_preserves_the_board(tmp_path: Path) -> None:
    board = load_cad(write_pads_csv(tmp_path / "pads.csv"))
    saved = save_cad_json(board, tmp_path / "board.json")
    reloaded = load_cad(saved)
    assert len(reloaded.components) == len(board.components)
    assert reloaded.pad_count == board.pad_count


def test_side_filter_keeps_one_board_face(tmp_path: Path) -> None:
    path = tmp_path / "sides.csv"
    path.write_text(
        "designator,pin,x_mm,y_mm,side\n"
        "R1,1,1,1,top\nR1,2,3,1,top\nR2,1,5,5,bottom\nR2,2,7,5,bottom\n",
        encoding="utf-8",
    )
    assert [c.designator for c in load_cad(path, side="top").components] == ["R1"]
    assert [c.designator for c in load_cad(path, side="bottom").components] == ["R2"]


@pytest.mark.parametrize(
    "designator,expected",
    [
        ("R12", "resistor"),
        ("C7", "capacitor"),
        ("U3", "ic"),
        ("LED4", "led"),   # longest prefix wins over "L"
        ("L4", "inductor"),
        ("XTAL1", "clock"),
        ("ZZ9", None),
    ],
)
def test_designator_class_prior(designator: str, expected: str | None) -> None:
    assert designator_to_class(designator) == expected


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #


def test_fiducial_registration_recovers_the_board_transform() -> None:
    registration = truth_registration()
    assert registration.scale_px_per_mm == pytest.approx(PX_PER_MM, rel=1e-3)
    assert registration.residual_px < 1e-6
    projected = registration.to_image([[25.0, 15.0]])[0]
    assert projected == pytest.approx(mm_to_px(25.0, 15.0), abs=1e-6)


def test_auto_registration_finds_scale_rotation_and_the_y_flip(tmp_path: Path) -> None:
    board = load_cad(write_pads_csv(tmp_path / "pads.csv"))
    detections = make_detections()
    registration = register_cad(board, detections, IMAGE_SIZE)
    assert registration is not None
    assert registration.scale_px_per_mm == pytest.approx(PX_PER_MM, rel=0.02)
    assert registration.residual_px < 3.0
    # CAD y grows up, image y grows down, so the fit must be mirrored.
    assert registration.y_flipped is True
    for designator, _, centre, _ in PARTS:
        component = next(c for c in board.components if c.designator == designator)
        projected = registration.to_image([[component.x, component.y]])[0]
        assert projected == pytest.approx(mm_to_px(*centre), abs=4.0)


def test_auto_registration_is_deterministic(tmp_path: Path) -> None:
    """RANSAC sampling is seeded; a registration that changes between identical
    runs could never be reviewed or signed off."""

    board = load_cad(write_pads_csv(tmp_path / "pads.csv"))
    detections = make_detections()
    first = register_cad(board, detections, IMAGE_SIZE)
    second = register_cad(board, detections, IMAGE_SIZE)
    assert first is not None and second is not None
    assert np.allclose(first.matrix, second.matrix)


def test_auto_registration_abstains_without_enough_agreement(tmp_path: Path) -> None:
    board = load_cad(write_pads_csv(tmp_path / "pads.csv"))
    lonely = [Detection("resistor", 0.9, BoundingBox(10, 10, 30, 20))]
    assert register_cad(board, lonely, IMAGE_SIZE) is None


def test_registration_round_trips_through_a_dict() -> None:
    registration = truth_registration()
    restored = CadRegistration.from_dict(json.loads(json.dumps(registration.to_dict())))
    assert np.allclose(restored.matrix, registration.matrix)
    assert restored.scale_px_per_mm == pytest.approx(registration.scale_px_per_mm)


# --------------------------------------------------------------------------- #
# Fusion
# --------------------------------------------------------------------------- #


def test_without_cad_the_derived_rois_pass_through_untouched() -> None:
    detections = make_detections()
    derived = derived_joints(detections)
    result = fuse_solder_joints(
        detections, derived, IMAGE_SIZE[0], IMAGE_SIZE[1], board=None, registration=None
    )
    assert result.used_cad is False
    assert result.joints == derived
    assert result.findings == []


def test_cad_lands_place_rois_and_agreement_is_recorded(tmp_path: Path) -> None:
    board = load_cad(write_pads_csv(tmp_path / "pads.csv"))
    result = fuse(board, truth_registration())
    assert result.used_cad is True

    joints = [joint for joint in result.joints if joint.kind == "joint"]
    r1_pins = [j for j in joints if j.designator == "R1"]
    assert len(r1_pins) == 2
    assert {j.pin for j in r1_pins} == {"1", "2"}
    # Both sources put a ROI on the same land, which is the point of fusing.
    assert all(j.source == "cad+derived" for j in r1_pins)
    for pin, (x_mm, y_mm) in zip(
        sorted(r1_pins, key=lambda j: j.pin), _pad_positions((10.0, 30.0), False)
    ):
        centre_x = (pin.bbox.x1 + pin.bbox.x2) / 2.0
        centre_y = (pin.bbox.y1 + pin.bbox.y2) / 2.0
        expected = mm_to_px(x_mm, y_mm)
        assert (centre_x, centre_y) == pytest.approx(expected, abs=2.0)


def test_a_cad_part_with_no_detection_is_reported_missing(tmp_path: Path) -> None:
    board = load_cad(write_pads_csv(tmp_path / "pads.csv"))
    result = fuse(board, truth_registration(), detections=make_detections(skip={"R3"}))
    missing = [f for f in result.findings if f.kind == "missing_component"]
    assert [f.designator for f in missing] == ["R3"]
    assert missing[0].severity == "defect"
    # Its lands still get ROIs: whether anything is soldered there is exactly
    # what step 6.2 should decide.
    assert any(j.designator == "R3" for j in result.joints)


def test_a_detection_with_no_cad_keeps_its_derived_rois(tmp_path: Path) -> None:
    board = load_cad(write_pads_csv(tmp_path / "pads.csv"))
    result = fuse(board, truth_registration())
    unexpected = [f for f in result.findings if f.kind == "unexpected_component"]
    assert len(unexpected) == 1
    assert unexpected[0].observed_class == "led"
    assert unexpected[0].severity == "info"
    led = [j for j in result.joints if j.label == "led"]
    assert led and all(j.source == "derived" for j in led)


def test_a_displaced_part_is_reported_with_its_offset_in_mm(tmp_path: Path) -> None:
    board = load_cad(write_pads_csv(tmp_path / "pads.csv"))
    detections = make_detections()
    # Push R2 one millimetre off its CAD placement.
    moved = []
    for detection in detections:
        if detection.label == "resistor" and abs(detection.bbox.x1 - mm_to_px(24.0, 30.0)[0]) < 6:
            box = detection.bbox
            shift = 1.0 * PX_PER_MM
            detection = Detection(
                detection.label,
                detection.confidence,
                BoundingBox(box.x1 + shift, box.y1, box.x2 + shift, box.y2),
                detection_id=detection.detection_id,
            )
        moved.append(detection)
    result = fuse(board, truth_registration(), detections=moved)
    shifted = [f for f in result.findings if f.kind == "shifted_component"]
    assert len(shifted) == 1
    assert shifted[0].shift_mm == pytest.approx(1.0, abs=0.15)


def test_local_refinement_follows_the_part_not_the_nominal_placement(tmp_path: Path) -> None:
    """CAD gives the footprint, the detector gives where this one actually
    landed; the ROI has to follow the part or it inspects bare board."""

    board = load_cad(write_pads_csv(tmp_path / "pads.csv"))
    detections = make_detections(extra=False)
    shift_px = 1.0 * PX_PER_MM
    moved = [
        Detection(
            detection.label,
            detection.confidence,
            BoundingBox(
                detection.bbox.x1 + shift_px,
                detection.bbox.y1,
                detection.bbox.x2 + shift_px,
                detection.bbox.y2,
            ),
            detection_id=detection.detection_id,
        )
        for detection in detections
    ]
    registration = truth_registration()
    # merge_mode="cad" keeps the CAD box as-is, so the measurement isolates the
    # refinement instead of also picking up the union with the derived box.
    refined = fuse(
        board, registration, detections=moved, config=FusionConfig(merge_mode="cad")
    )
    nominal = fuse(
        board,
        registration,
        detections=moved,
        config=FusionConfig(merge_mode="cad", local_refine=False),
    )

    def r1_centre(result: object) -> float:
        # Only the CAD-placed ROIs: an unmatched derived ROI is legitimately
        # kept alongside them and would blur the measurement.
        pins = [
            j
            for j in result.joints
            if j.designator == "R1" and j.kind == "joint" and j.source.startswith("cad")
        ]
        assert len(pins) == 2
        return float(np.mean([(j.bbox.x1 + j.bbox.x2) / 2.0 for j in pins]))

    nominal_x = mm_to_px(10.0, 30.0)[0]
    assert r1_centre(nominal) == pytest.approx(nominal_x, abs=1.0)
    assert r1_centre(refined) == pytest.approx(nominal_x + shift_px, abs=1.0)


def test_placement_only_cad_reanchors_the_derived_geometry(tmp_path: Path) -> None:
    """A pick-and-place file has no lands, so the derived ROI shape is kept but
    rebuilt on the CAD centre and rotation."""

    board = load_cad(write_placement_csv(tmp_path / "cpl.csv"))
    assert board.pad_count == 0
    result = fuse(board, truth_registration())
    assert result.used_cad is True
    fused = [j for j in result.joints if j.designator == "R1" and j.kind == "joint"]
    assert len(fused) == 2
    assert all(j.source == "cad+derived" for j in fused)
    assert all(j.terminal_geometry == "two_terminal" for j in fused)


def test_a_poor_registration_is_refused_rather_than_applied(tmp_path: Path) -> None:
    board = load_cad(write_pads_csv(tmp_path / "pads.csv"))
    bad = truth_registration()
    bad.inlier_ratio = 0.05
    bad.residual_px = 90.0
    result = fuse(board, bad)
    assert result.used_cad is False
    assert result.warnings and "registration rejected" in result.warnings[0]
    assert all(joint.source == "derived" for joint in result.joints)


def test_fusion_can_be_switched_off(tmp_path: Path) -> None:
    board = load_cad(write_pads_csv(tmp_path / "pads.csv"))
    result = fuse(board, truth_registration(), config=FusionConfig(enabled=False))
    assert result.used_cad is False


def test_cad_pad_count_overrides_the_class_topology_guess() -> None:
    """A four-pad part labelled 'resistor' must not be treated as two-terminal."""

    board = BoardCad(
        components=[
            CadComponent(
                designator="U1",
                x=25.0,
                y=15.0,
                part_class="ic",
                pads=[
                    CadPad("U1", str(i + 1), 25.0 + dx, 15.0 + dy, 0.5, 0.5)
                    for i, (dx, dy) in enumerate(
                        [(-1, -1), (1, -1), (1, 1), (-1, 1)]
                    )
                ],
            )
        ]
    )
    centre = mm_to_px(25.0, 15.0)
    detections = [
        Detection(
            "resistor",
            0.9,
            BoundingBox(centre[0] - 8, centre[1] - 8, centre[0] + 8, centre[1] + 8),
        )
    ]
    result = fuse(board, truth_registration(), detections=detections)
    pins = [j for j in result.joints if j.kind == "joint" and j.source.startswith("cad")]
    assert len(pins) == 4


# --------------------------------------------------------------------------- #
# Pipeline and export
# --------------------------------------------------------------------------- #


def test_pipeline_without_cad_is_unchanged() -> None:
    detections = make_detections()
    pipeline = AOIPipeline(PipelineConfig(), detector=MockComponentDetector(detections))
    run = pipeline.run(board_image(), source_name="plain.png")
    assert run.fusion is not None and run.fusion.used_cad is False
    assert run.solder_crops
    assert all(crop.joint.source == "derived" for crop in run.solder_crops)


def test_pipeline_auto_registers_and_exports_cad_findings(tmp_path: Path) -> None:
    cad_path = write_pads_csv(tmp_path / "pads.csv")
    detections = make_detections(skip={"R3"})
    config = PipelineConfig()
    config.cad.path = str(cad_path)
    pipeline = AOIPipeline(config, detector=MockComponentDetector(detections))
    run = pipeline.run(board_image(), source_name="board.png")

    assert run.fusion.used_cad is True
    assert run.fusion.stats["missing"] == 1
    sources = {crop.joint.source for crop in run.solder_crops}
    assert "cad+derived" in sources

    rows = list(csv.DictReader(io.StringIO(solder_joints_csv(run))))
    assert any(row["designator"] == "R1" and row["source"].startswith("cad") for row in rows)
    findings = list(csv.DictReader(io.StringIO(cad_findings_csv(run))))
    assert any(row["kind"] == "missing_component" and row["designator"] == "R3" for row in findings)

    archive_path = pipeline.export_zip(run, tmp_path / "run.zip")
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
    assert "cad/cad_findings.csv" in names
    assert "cad/registration.json" in names


def test_a_broken_cad_path_warns_and_keeps_the_run_going(tmp_path: Path) -> None:
    """Inspection without CAD is the supported baseline, so a bad board file
    must not turn into a failed run."""

    config = PipelineConfig()
    config.cad.path = str(tmp_path / "does_not_exist.csv")
    pipeline = AOIPipeline(config, detector=MockComponentDetector(make_detections()))
    run = pipeline.run(board_image(), source_name="board.png")
    assert run.solder_crops
    assert any("CAD not loaded" in warning for warning in run.warnings)


# --------------------------------------------------------------------------- #
# Knowing when the alignment is not determined
# --------------------------------------------------------------------------- #


def _blind(detections):
    """The OpenCV proposal mode labels everything the same; that names no class."""

    return [
        Detection("component_candidate", d.confidence, d.bbox, detection_id=d.detection_id)
        for d in detections
    ]


def test_uninformative_labels_are_not_treated_as_contradictions() -> None:
    assert classes_agree("resistor", "component_candidate") is True
    assert classes_agree("resistor", "") is True
    assert classes_agree("resistor", "capacitor") is False
    assert classes_agree("resistor", "resistor") is True
    assert is_informative_label("component_candidate") is False
    assert is_informative_label("resistor") is True


def test_class_labels_resolve_the_alignment(tmp_path: Path) -> None:
    board = load_cad(write_pads_csv(tmp_path / "pads.csv"))
    registration = register_cad(board, make_detections(), IMAGE_SIZE)
    assert registration is not None
    assert registration.ambiguous is False


# A regular grid of identical parts: mirrored and rotated fits map it onto
# itself perfectly, so geometry alone cannot pick one.
SYMMETRIC_PARTS = [
    (f"R{index + 1}", "resistor", (x, y), False)
    for index, (x, y) in enumerate(
        [(10.0, 30.0), (25.0, 30.0), (40.0, 30.0),
         (10.0, 15.0), (25.0, 15.0), (40.0, 15.0)]
    )
]


def test_an_asymmetric_board_stays_determined_without_class_labels(tmp_path: Path) -> None:
    board = load_cad(write_pads_csv(tmp_path / "pads.csv"))
    registration = register_cad(board, _blind(make_detections()), IMAGE_SIZE)
    assert registration is not None
    assert registration.ambiguous is False


def test_a_symmetric_board_without_class_labels_is_flagged_ambiguous(tmp_path: Path) -> None:
    """Geometry alone cannot separate a symmetric board from its own mirror;
    saying so is the only honest outcome, because both score the same residual."""

    board = load_cad(write_pads_csv(tmp_path / "sym.csv", SYMMETRIC_PARTS))
    detections = _blind(make_detections(extra=False, parts=SYMMETRIC_PARTS))
    registration = register_cad(board, detections, IMAGE_SIZE)
    assert registration is not None
    assert registration.ambiguous is True

    result = fuse(board, registration, detections=detections)
    assert result.used_cad is True
    assert any("ambiguous" in warning for warning in result.warnings)


def test_fusion_warns_when_no_class_can_corroborate_the_fit(tmp_path: Path) -> None:
    board = load_cad(write_pads_csv(tmp_path / "pads.csv"))
    detections = _blind(make_detections())
    registration = register_cad(board, detections, IMAGE_SIZE)
    result = fuse(board, registration, detections=detections)
    assert result.used_cad is True
    assert any("corroborate" in warning for warning in result.warnings)


def test_uninformative_labels_do_not_raise_class_mismatch_findings(tmp_path: Path) -> None:
    board = load_cad(write_pads_csv(tmp_path / "pads.csv"))
    detections = _blind(make_detections(extra=False))
    result = fuse(board, truth_registration(), detections=detections)
    assert [f for f in result.findings if f.kind == "class_mismatch"] == []


def test_ambiguity_survives_a_registration_round_trip() -> None:
    registration = truth_registration()
    registration.ambiguous = True
    restored = CadRegistration.from_dict(json.loads(json.dumps(registration.to_dict())))
    assert restored.ambiguous is True


# --------------------------------------------------------------------------- #
# App layer: the config path a Streamlit upload takes, and the step-5.5 view
# --------------------------------------------------------------------------- #


def test_bridge_config_carries_cad_through_to_the_pipeline(tmp_path: Path) -> None:
    """The sidebar only writes a path into the config dict; everything else has
    to follow from that, or an uploaded board would be quietly ignored."""

    from app.pipeline_bridge import PipelineBridge

    cad_path = write_pads_csv(tmp_path / "pads.csv")
    bridge = PipelineBridge(
        config={
            "cad": {"path": str(cad_path), "side": "top"},
            "fusion": {"enabled": True},
        }
    )
    assert bridge.engine is not None
    assert bridge.engine.cad is not None
    assert len(bridge.engine.cad.components) == len(PARTS)


def test_bridge_reports_cad_fusion_output_to_the_ui(tmp_path: Path) -> None:
    from app.pipeline_bridge import DetectionRecord, PipelineBridge

    cad_path = write_pads_csv(tmp_path / "pads.csv")
    bridge = PipelineBridge(config={"cad": {"path": str(cad_path), "side": "top"}})
    detections = make_detections(skip={"R3"})
    records = [
        DetectionRecord(
            detection_id=detection.detection_id,
            label=detection.label,
            confidence=detection.confidence,
            bbox=tuple(int(v) for v in detection.bbox.to_int()),
            source="model",
            raw=detection,
        )
        for detection in detections
    ]
    result = bridge.make_solder_crops(board_image(), records)
    assert result.used_cad is True
    assert result.mode == "CAD FUSION"
    assert any(item["kind"] == "missing_component" for item in result.findings)
    assert result.registration is not None
    assert any(crop.designator for crop in result.crops)


def test_step_five_five_view_shows_provenance() -> None:
    from app.pipeline_bridge import SolderCropRecord
    from app.streamlit_app import (
        SOLDER_SOURCE_COLORS,
        _draw_solder_overlay,
        _findings_frame,
        _solder_frame,
    )

    pixels = np.zeros((8, 8, 3), np.uint8)
    crops = [
        SolderCropRecord(
            "R1_pin1", "d1", "resistor", "joint", "pin1", "cad_pad", pixels,
            (4, 4, 20, 18), 0.9, source="cad+derived", designator="R1", pin="1", net="VCC",
        ),
        SolderCropRecord(
            "L1", "d2", "led", "joint", "terminal_a", "two_terminal", pixels,
            (40, 40, 56, 54), 0.8, source="derived",
        ),
    ]
    frame = _solder_frame(crops)
    assert list(frame["source"]) == ["cad+derived", "derived"]
    assert list(frame["designator"]) == ["R1", ""]
    assert frame["defect_class"].eq("").all()

    image = np.zeros((80, 80, 3), np.uint8)
    by_kind = _draw_solder_overlay(image, crops, True, by_source=False)
    by_source = _draw_solder_overlay(image, crops, True, by_source=True)
    assert image.sum() == 0
    assert not np.array_equal(by_kind, by_source)
    assert set(SOLDER_SOURCE_COLORS) == {"cad", "cad+derived", "derived"}

    findings = _findings_frame(
        [
            {
                "kind": "missing_component",
                "severity": "defect",
                "designator": "R3",
                "expected_class": "resistor",
                "observed_class": None,
                "shift_mm": None,
                "message": "R3 missing",
            }
        ]
    )
    assert list(findings["kind"]) == ["missing_component"]
    assert list(findings["severity"]) == ["defect"]
