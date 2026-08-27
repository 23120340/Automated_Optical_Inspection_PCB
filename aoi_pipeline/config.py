"""Configuration dataclasses for steps 1 through 6.1."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from collections.abc import Mapping
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class PadProfile:
    """Outward crop margin expressed as a fraction of the box side it extends.

    ``long_axis`` is applied at both ends of the longer box side and
    ``short_axis`` at both ends of the shorter side. Solder fillets sit outside
    the labelled component body, so a two-lead chip needs a much larger margin
    along its long axis (where the terminals are) than across it.
    """

    long_axis: float
    short_axis: float


# Terminal topology per detector class. It decides where the solder joints of a
# component are, which no bounding box alone can express.
#
# ``two_terminal``  joints at both ends of the long axis (chip/axial parts)
# ``multi_pin``     leads along the box edges (ICs, connectors, TO packages)
# ``pad_only``      the detection already is the joint/land
TWO_TERMINAL_CLASSES: frozenset[str] = frozenset(
    {"capacitor", "resistor", "diode", "led", "inductor", "fuse"}
)
PAD_ONLY_CLASSES: frozenset[str] = frozenset({"pads"})
# Everything else — ic, connector, transistor, relay, switch, display, clock,
# buzzer, battery, potentiometer, button, transformer, transducer, heatsink,
# pins — is treated as multi-pin. Defaulting unknown labels to the perimeter
# policy keeps recall: a wrong-but-present ROI can be reviewed, a missing one
# cannot.
DEFAULT_TERMINAL_GEOMETRY = "multi_pin"


def terminal_geometry(label: str) -> str:
    """Map a detector class name to its terminal topology."""

    normalized = str(label).strip().lower()
    if normalized in TWO_TERMINAL_CLASSES:
        return "two_terminal"
    if normalized in PAD_ONLY_CLASSES:
        return "pad_only"
    return DEFAULT_TERMINAL_GEOMETRY


def _default_pad_profiles() -> dict[str, PadProfile]:
    return {
        "two_terminal": PadProfile(long_axis=0.35, short_axis=0.22),
        "multi_pin": PadProfile(long_axis=0.20, short_axis=0.20),
        "pad_only": PadProfile(long_axis=0.10, short_axis=0.10),
    }


@dataclass(slots=True)
class PreprocessConfig:
    """Image enhancement options used by step 1.

    The defaults are intentionally conservative. They improve uneven illumination
    without aggressively destroying tiny silkscreen or component edges.
    """

    undistort: bool = False
    calibration_profile: dict[str, Any] | None = None
    undistort_alpha: float = 0.0
    calibration_aspect_tolerance: float = 0.01
    # Preserve enough detail for tiled component detection. The detector still
    # receives bounded tiles, so this does not enlarge its model input tensor.
    max_side: int | None = 4096
    denoise: bool = True
    denoise_method: Literal["nlmeans", "bilateral", "gaussian"] = "nlmeans"
    denoise_strength: int = 5
    white_balance: bool = True
    clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_grid_size: tuple[int, int] = (8, 8)
    normalize: bool = True
    normalize_low_percentile: float = 1.0
    normalize_high_percentile: float = 99.0
    sharpen: bool = True
    sharpen_amount: float = 0.35


@dataclass(slots=True)
class AlignmentConfig:
    """ORB/homography settings and the ECC fallback policy for step 2."""

    enabled: bool = True
    orb_features: int = 4000
    ratio_test: float = 0.75
    min_good_matches: int = 12
    ransac_reprojection_threshold: float = 4.0
    min_inliers: int = 8
    min_inlier_ratio: float = 0.25
    use_ecc_fallback: bool = True
    ecc_iterations: int = 120
    ecc_epsilon: float = 1e-5
    min_ecc_correlation: float = 0.65
    strict: bool = False


@dataclass(slots=True)
class BoardConfig:
    """Contour-based PCB localization settings for step 3."""

    min_area_ratio: float = 0.08
    max_area_ratio: float = 0.995
    min_rectangularity: float = 0.45
    morphology_kernel_ratio: float = 0.015
    fallback_to_full_image: bool = True
    padding_ratio: float = 0.005


@dataclass(slots=True)
class FiducialConfig:
    """Dò fiducial mark để neo board, thay cho việc bắt buộc có ảnh golden.

    Fiducial theo IPC-7351 là pad đồng tròn đường kính ~1 mm với cửa sổ mask
    rộng gấp 2–3 lần. Ở 46 µm/px (thang đo được của dự án) thì 1 mm là ~22 px
    đường kính; ở 25 µm/px là ~40 px; ở 92 µm/px là ~11 px. Dải bán kính mặc
    định phủ cả ba, nên không phải chỉnh khi đổi camera.
    """

    enabled: bool = True
    #: Bán kính tính bằng pixel. Rộng có chủ ý — hẹp lại thì mỗi lần đổi ống
    #: kính là một lần dò hụt, và dò hụt thì im lặng.
    min_radius_px: float = 4.0
    max_radius_px: float = 30.0
    #: So với nền CỤC BỘ: vành fiducial sáng so với vùng quanh nó, không so
    #: với cả ảnh. Đo được trên board thật: vành ~150-180 trong khi phân vị 99
    #: của cả ảnh là 239, nên ngưỡng toàn cục bỏ qua nó hoàn toàn.
    background_kernel: int = 31
    local_lift: float = 0.06        # phần của 255
    #: 4πA/P²; 1.0 là tròn hoàn hảo.
    min_circularity: float = 0.70
    max_aspect: float = 1.35
    #: Đốm phải nổi hơn vành xung quanh — đây là thứ phân biệt fiducial với
    #: một mảng đồng lớn cũng sáng.
    min_contrast: float = 0.05
    ring_ratio: float = 3.0
    #: Fiducial là VÀNH sáng quanh lõi TỐI. Đốm loé trên linh kiện thì sáng
    #: đặc ở giữa — đây là thứ phân biệt hai loại, và bỏ nó đi thì bộ dò nhận
    #: toàn đốm loé.
    require_dark_core: bool = True
    min_ring_lift: float = 0.04
    close_kernel: int = 3
    max_count: int = 8
    #: Ba đốm chụm một góc thường là một linh kiện bóng, không phải ba mốc.
    min_spread: float = 0.25
    #: Fiducial nằm LÙI VÀO so với mép board, nên bao lồi của chúng nhỏ hơn
    #: board. Phần nới này là ước lượng, không phải đo.
    board_margin_ratio: float = 0.06


@dataclass(slots=True)
class CVDetectorConfig:
    """Heuristic proposal detector used before a trained model is available."""

    min_area_ratio: float = 0.00004
    max_area_ratio: float = 0.035
    min_width: int = 4
    min_height: int = 4
    max_aspect_ratio: float = 12.0
    morphology_kernel: int = 3
    nms_iou_threshold: float = 0.35
    max_detections: int = 500


@dataclass(slots=True)
class ModelDetectorConfig:
    """Ultralytics inference options for a ``.pt`` or ``.onnx`` model."""

    confidence: float = 0.25
    iou: float = 0.45
    image_size: int = 1280
    device: str | None = None
    max_detections: int = 2000
    # ``None`` preserves the head declared by arbitrary Ultralytics artifacts.
    # The local UI explicitly sets False for its Kaggle one-to-many/NMS recipe.
    end2end: bool | None = None
    # Ultralytics' built-in inference-time augmentation (multi-scale + flip,
    # averaged internally before NMS). Off by default: it roughly doubles
    # inference time and every tiling/confidence threshold in this project was
    # tuned without it. Worth trying once a fresh model is trained -- it is a
    # no-retrain accuracy lever, typically +0.5-2 mAP on small/occluded
    # objects, which is exactly where this detector struggles (pads/pins).
    tta: bool = False


@dataclass(slots=True)
class SolderDefectDetectionConfig:
    """Independent full-board solder-defect segmentation deployment.

    This is deliberately separate from :class:`SolderGradingConfig`: the
    latter classifies already-cropped joint ROIs and participates in the 6.2
    classifier/rule verdict, while this model localizes diagnostic defects on
    a complete image.  ``None`` inference values inherit the recommendations
    pinned by the detector manifest.
    """

    enabled: bool = True
    model_path: str | None = None
    manifest_path: str | None = None
    confidence: float | None = None
    iou: float | None = None
    image_size: int | None = None
    device: str | None = None
    max_detections: int | None = None
    mask_threshold: float | None = None
    tta: bool = False


@dataclass(slots=True)
class TilingConfig:
    """Adaptive high-resolution inference policy for step 4."""

    mode: Literal["auto", "on", "off"] = "auto"
    # ``tile_size`` is the upper bound. Auto mode may choose a smaller detail
    # window so a 1000px PCB is not incorrectly treated as a single 1280px tile.
    tile_size: int = 1280
    min_tile_size: int = 640
    detail_window_ratio: float = 0.64
    overlap_ratio: float = 0.20
    auto_trigger_scale: float = 1.25
    include_full_image: bool = True
    detail_confidence: float | None = 0.20
    # Detail tiles improve recall, but some visually ambiguous classes need a
    # stricter floor. In the current detector, bright solder joints are the
    # main low-confidence false positive for ``led``.
    detail_class_confidence: dict[str, float] = field(
        default_factory=lambda: {"led": 0.35}
    )
    merge_iou_threshold: float = 0.45
    # IoU alone misses duplicate partial boxes on opposite sides of a tile seam.
    # IoS (intersection / smaller box) recognizes those fragments without
    # merging ordinary neighboring components that do not overlap.
    seam_ios_threshold: float = 0.50
    # Remove a partial/tight box almost completely enclosed by another box for
    # the same class when they originate from different inference windows.
    containment_ios_threshold: float = 0.80
    class_aware_merge: bool = True
    # Different class labels may still be duplicate hypotheses for one object.
    # Use a stricter threshold than same-class NMS to preserve adjacent parts.
    cross_class_iou_threshold: float = 0.70
    edge_margin_ratio: float = 0.03
    edge_confidence_penalty: float = 0.10
    non_ownership_confidence_penalty: float = 0.10
    # A tile that only sees part of a component often labels that sliver as a
    # different, smaller class -- the right-hand end of an SOIC reads as a
    # diode. Containment de-duplication cannot catch it because it compares
    # within one class, and cross-class NMS cannot either: a box nested inside
    # a larger one has IoU <= area_small / area_large by construction, so it
    # never reaches ``cross_class_iou_threshold``. Drop the nested box only
    # when the tile that produced it did not own that area, which is exactly
    # when another window had the better view.
    drop_cross_class_edge_fragments: bool = True


@dataclass(slots=True)
class CropConfig:
    """Crop and normalization options used by step 5.

    These crops feed the step-6.1 family classifier, whose accuracy depends on
    matching the padding it was trained with. Keep ``solder_aware_padding``
    off unless the classifier is retrained on the wider recipe; solder-joint
    inspection has its own ROIs in :class:`SolderJointConfig` precisely so this
    contract does not have to move.
    """

    # These three mirror the step-6.1 training notebook exactly
    # (``pad = 0.15 * max(w, h)``, clipped, no squaring). Changing them shifts
    # the classifier's input distribution away from what it was fitted on.
    padding_ratio: float = 0.15
    padding_pixels: int = 0
    square: bool = False
    # Per-axis, per-topology margins. Off by default for the same reason.
    solder_aware_padding: bool = False
    pad_profiles: dict[str, PadProfile] = field(default_factory=_default_pad_profiles)
    target_size: tuple[int, int] | None = (224, 224)
    letterbox_color: tuple[int, int, int] = (114, 114, 114)
    image_extension: Literal[".png", ".jpg"] = ".png"
    jpeg_quality: int = 95


@dataclass(slots=True)
class SolderJointConfig:
    """Step 5.5: derive solder-joint ROIs from component boxes.

    The component detector is trained on datasets that label component bodies,
    so its boxes never contain the fillet. Rather than asking a detector to
    find joints — the current model scores ``pads`` at 0.0 recall — the joints
    are *derived* geometrically from the box plus the class topology, which is
    how window-based AOI has always placed its inspection ROIs.
    """

    enabled: bool = True

    # --- which axis holds the terminals -------------------------------------
    # A box only names its long axis when it is actually longer one way. At
    # 40x40 vs 40x41 the choice flips the two ROIs by 90 degrees on a single
    # pixel, and a square-ish part -- an SMD electrolytic can, a tantalum, a
    # tactile switch -- sits exactly there. Under this aspect ratio the axis is
    # decided from the metal in the image instead of from the box.
    terminal_axis_min_aspect: float = 1.25
    # How much more metal one candidate axis needs before it wins. Below this
    # the evidence is a coin toss, so ROIs are emitted on BOTH axes: a
    # reviewable extra ROI costs an operator seconds, a missing one ships the
    # board.
    terminal_axis_decision_margin: float = 1.30

    # --- where that axis measurement looks, and what it counts as metal ------
    # The probe used to be the ROI geometry itself: 0.45 x the LONG side,
    # entirely outside the body. Both halves of that are wrong for a part with
    # no long side. Measured on the SMD board in ``tests/data``: C231's
    # horizontal probe reached 143 px from centre, landed on the chip parts
    # either side, outscored the real tabs, and both of its joints ended up
    # uninspected. And a V-chip tab straddles the body edge -- C231's runs 9 px
    # above the box top and 9 px below -- so a purely outward band sees half of
    # it at best.
    #
    # So the probe is scaled from the SHORT side and reaches both ways across
    # the edge. It only ever runs on a part too square to name its own axis.
    axis_probe_outer_ratio: float = 0.20
    axis_probe_inner_ratio: float = 0.10
    axis_probe_cross_ratio: float = 0.60
    # The probe counts metal itself instead of calling ``segment_solder``. That
    # function is shared with grading and carries an Otsu fallback that is right
    # for a joint ROI and wrong here: on a probe band of bare laminate it splits
    # the noise and reports 64% metal, which is worse than no measurement.
    #
    # Three clauses, each with a physical reason, and every one of the 81
    # combinations swept over the values below decided all ten reference cases
    # correctly -- six synthetic (grey lands on green resist) and four
    # photographed (warm solder, white legend, green resist):
    #   bright        metal reflects the illuminator; resist and package bodies
    #                 do not.
    #   not white ink silkscreen is achromatic AND blown out. Grey solder is
    #                 achromatic too but never that bright, so the pair of
    #                 conditions removes the legend and keeps the joint.
    #   not resist    solder mask is strongly coloured green; neither solder nor
    #                 ink is.
    axis_probe_value_min: int = 120
    axis_probe_ink_saturation_max: int = 30
    axis_probe_ink_value_min: int = 235
    axis_probe_resist_hue_range: tuple[int, int] = (35, 95)
    axis_probe_resist_saturation_min: int = 70

    # --- two-terminal geometry, as fractions of the box sides ---------------
    # ``inner`` reaches back over the terminal cap on the body, ``outer``
    # reaches past the body onto the land/fillet, ``side`` widens across the
    # short axis so a slightly rotated part is still covered.
    terminal_inner_ratio: float = 0.30
    terminal_outer_ratio: float = 0.45
    terminal_side_ratio: float = 0.20

    # --- the same three, for a part with no long side -----------------------
    # Scaled from the SHORT side, and much smaller. On a chip part the lands
    # take up about half the body length, so a reach of 0.75 x length is right.
    # A V-chip electrolytic is the opposite: a big round can with two small tabs
    # at its edge. Measured on the board in ``tests/data``, its tabs run 0.29 to
    # 0.33 of the body width and sit astride the body edge, so the chip ratios
    # produce an ROI 14 to 28 times the area of the pad it is meant to inspect,
    # and a reviewer sees a box swallowing half the component.
    #
    # These values were read off the same three parts: they cover every tab with
    # margin while bringing ROI-to-pad area to 2.4..3.5, inside the 2.75 that
    # the parts nobody complained about already measured. They apply only below
    # ``terminal_axis_min_aspect`` -- the same gate that decides a box cannot
    # name its own terminal axis -- so no chip part is affected.
    compact_terminal_inner_ratio: float = 0.15
    compact_terminal_outer_ratio: float = 0.20
    compact_terminal_side_ratio: float = 0.25

    # --- multi-pin geometry, as fractions of the shorter box side -----------
    lead_inner_ratio: float = 0.14
    lead_outer_ratio: float = 0.26
    # Dải dùng để ĐỊNH VỊ hàng chân, tách khỏi dải dùng để ĐO.
    #
    # Cùng một dải đang làm cả hai việc, nên thu nó lại đổi luôn cả ba thứ:
    # chỗ tìm chân, chỗ đo, và ngưỡng tương đối của bộ lọc dải. Đo được hậu quả
    # trên board của dự án khi hạ ``lead_outer_ratio``:
    #
    #     outer   U201        D201       D202
    #     0.26    8 ROI/0     7/3        4/1
    #     0.10    9/0         4/0        11/4
    #     0.00    8/0         4/2        11/5
    #
    # U201 không đổi (P2 đã xử lý xong), còn D202 **tệ hẳn đi**: dải nông có ít
    # kim loại hơn nên ngưỡng tương đối 0,35 × đỉnh tụt xuống, nhiều dải yếu lọt
    # qua rồi bị tách nhỏ. Nên muốn "IC không mở box" thì phải thu đúng dải định
    # vị, giữ nguyên ROI đo.
    #
    # Chọn 0,20/0,10 sau khi quét 16 thiết lập trên ba con multi-pin của board
    # (U201 SOIC-8, D201 và D202 SOT-23), chấm điểm với **bỏ sót chân nặng gấp
    # 5 lần ROI thừa** vì bỏ sót là escape còn ROI thừa chỉ tốn công soát:
    #
    #     thiết lập                U201      D201      D202   sót  ROI lụa
    #     mặc định cũ           8/0/-0    7/3/-0    4/1/-0     0        4
    #     outer 0,10 inner 0,20 8/0/-0    4/0/-0    4/0/-0     0        0
    #     outer 0,00 inner 0,14 4/0/-4    3/1/-1    6/2/-0     5        3
    #
    # (ô = tổng ROI / ROI trên chữ lụa / số chân bỏ sót)
    #
    # Dòng cuối là lý do **không** đặt outer = 0 như trực giác "IC không mở
    # box": dải sâu 0 nằm trọn trong thân, mà chân gull-wing của SOIC thì nằm
    # NGOÀI đường bao thân -- nên biên dạng không thấy đủ đốm, bộ tách chân bỏ
    # cuộc và U201 mất một nửa số chân.
    #
    # Bán kính ảnh hưởng đã đo: đúng **1 trong 39** linh kiện đổi số ROI, và đó
    # chính là con đang sinh ROI giả. 38 con còn lại không đổi một pixel.
    #
    # Cảnh báo: hai tham số này chỉnh trên **3 linh kiện của 1 ảnh**. Đó là ít.
    # Đặt ``None`` để quay lại hành vi cũ (locator dùng chung dải với ROI đo).
    lead_locator_inner_ratio: float | None = 0.20
    lead_locator_outer_ratio: float | None = 0.10
    # A perimeter band with no lead metal in it is dropped. ``None`` disables
    # the check and keeps all four bands.
    lead_band_energy_ratio: float | None = 0.35
    # Tín hiệu thứ hai cho cùng câu hỏi: một hàng chân là **một cái lược** --
    # N đốm kim loại cách đều. Chữ lụa trắng thì không tuần hoàn.
    #
    # Cần đến nó vì phép đo độ phủ kim loại ở trên không tách được: đo trên
    # board của dự án, chữ lụa `HDL01` ghi 49,5% "kim loại" trong khi chân IC
    # thật chỉ 38,3% và 29,9% — mực lụa trắng cũng sáng và cũng ít bão hoà màu
    # y như thiếc.
    #
    # Chỉ phát biểu khi dải có ít nhất ``lead_band_min_runs`` đốm. Dưới ngưỡng
    # đó "độ đều" luôn bằng 1,0 và vô nghĩa — mà cạnh 2 chân là chuyện thường
    # (SOT-23, SOT-223), nên im lặng mới đúng. ``None`` để tắt hẳn.
    lead_band_evenness_min: float | None = 0.95
    lead_band_min_runs: int = 3
    # Leads sit on OPPOSITE edges. SOT-23, SOT-223, SOIC, DPAK: every one of
    # them puts its leads on one axis and nothing on the other, so the two
    # opposite pairs can be compared directly instead of each edge being judged
    # against a global threshold.
    #
    # It is the safer question to ask. Measured on the SMD board in
    # ``tests/data``, per-edge energy ratios overlap badly -- real lead edges run
    # 0.81..1.00 and bare edges 0.51..0.70, so a single cut sits in a 16% gap --
    # while the summed pairs separate by 1.38x, 1.62x and 1.67x on the three
    # multi-pin parts there.
    #
    # A quad package has leads on all four edges and scores near 1.0, which is
    # below this margin, so it keeps every edge. That is the point: the rule
    # only speaks when one pair clearly dominates, and it hands back to the
    # evenness filter when it cannot. ``None`` disables it.
    lead_band_pair_margin: float | None = 1.30
    # And the losing pair has to be weak on its own terms, not merely weaker
    # than the winner. Measured on ``tests/data``, bare edges reach 0.51..0.70
    # of the strongest edge while real lead edges start at 0.81. Without this
    # second condition the rule also fires on a low-contrast part whose four
    # edges are near-equal, where the pair ratio is reading noise.
    lead_band_pair_drop_max_ratio: float = 0.75
    # Was opt-in, on the argument that a bridge spans two adjacent pins so the
    # band is the better inspection unit. Measured on ``pcb03.jpg`` (4 ICs,
    # 50 leads) that argument does not survive: ``pin_padding_ratio`` already
    # grows each pin ROI over the gap to its neighbour, so every one of the 42
    # inter-pin gaps stays covered either way -- while ROIs holding exactly one
    # lead go from 3 to 48. Two-terminal parts are untouched (60 ROIs both
    # ways); splitting only applies to ``multi_pin`` geometry.
    split_pins: bool = True
    # Grow each pin ROI along the band by this fraction of the lead pitch so
    # the gap to the neighbour is inside the crop; a bridge lives in that gap.
    pin_padding_ratio: float = 0.35
    # Two, not three. A SOT-23, SOT-223 or DPAK puts two leads on one edge,
    # and a floor of three meant that edge could never be split -- the two
    # joints stayed inside one ROI, so one of them was never inspected on
    # its own. Measured on a real board's D201/D202.
    min_pins_per_band: int = 2
    max_pins_per_band: int = 64
    # Một hàng chân của linh kiện SMD ĐỐI XỨNG qua đường tâm thân. Đó không
    # phải quan sát may rủi mà là hệ quả của cách vẽ footprint: SOT-23, SOT-223,
    # SOD, SOIC, QFP đều đặt chân đối xứng, nên hàng lẻ chân bắt buộc có một
    # chân nằm đúng giữa.
    #
    # Đo trên board thật trong ``tests/data``, residual = min|c_j - (L - c_i)|
    # chia cho chiều dài dải:
    #   D201 hàng 2 chân   0.006      D202 hàng 2 chân   0.006
    #   U201 hàng 4 chân   0.011      D202 hàng lẻ chân  0.000
    #   D201 hàng lẻ chân  0.051
    # Chữ lụa trong góc dải -- thứ mà sàn đếm ``min_pins_per_band`` sinh ra để
    # chặn -- nằm ở 0.2..0.6. Ngưỡng đặt giữa hai vùng đó.
    #
    # Ràng buộc này an toàn theo cấu trúc: loại một phép tách KHÔNG làm mất mối
    # hàn nào, nó chỉ quay về ROI cả dải như hiện nay.
    pin_row_symmetry_max_residual: float = 0.12
    # Cho phép tách khi chỉ tìm được MỘT chân, với điều kiện chân đó nằm giữa.
    # Sàn ``min_pins_per_band = 2`` tồn tại vì một đốm sáng lẻ có thể là chữ lụa
    # hay via chứ không phải chân; tính đối xứng chính là bằng chứng còn thiếu
    # để phân biệt hai thứ đó. Đo trên D201/D202: chân lẻ thật lệch tâm 0.000 và
    # 0.051, còn ROI cả dải cho tỉ lệ diện tích 2.65 lần pad so với 1.64 của
    # hàng tách được -- tức là bỏ qua ca này chính là nguồn "box quá bự".
    split_lone_centred_pin: bool = True

    # --- pad-only geometry --------------------------------------------------
    pad_margin_ratio: float = 0.15

    # --- shared -------------------------------------------------------------
    # Trần tuyệt đối cho độ sâu một ROI mối hàn, tính bằng mm.
    #
    # Luật tỉ lệ ở trên giả định pad to lên theo linh kiện. Pad không như thế --
    # nó là một kích thước vật lý gần cố định. Đo trên board của dự án: tụ hoá
    # C239 dài 6,7 mm cho ra ROI sâu **5,0 mm**, to hơn cả con tụ, trong khi
    # điện trở chip 0805 chỉ cho 1,6 mm và hoàn toàn hợp lý.
    #
    # 2,0 chứ không phải 1,5. Ở 1,5 mm trần bắt đầu cắn cả chip 0805 -- đo được
    # 7/39 linh kiện bị cắt, trong đó có những con vốn không có gì sai. Ở 2,0 mm
    # là 6/39, đúng những con to vô lý, còn chip thường không đổi một pixel nào.
    #
    # Chỉ áp dụng khi biết ``px_per_mm``; không biết thang thì không quy đổi
    # được sang mm, và đoán bừa một thang còn tệ hơn là giữ nguyên.
    max_joint_depth_mm: float | None = 2.0
    #: px trên mm của ảnh phân tích. Hiện lấy từ đăng ký CAD/pick-and-place
    #: (``CadRegistration.scale_px_per_mm``); ``None`` nghĩa là chưa biết.
    px_per_mm: float | None = None
    min_roi_pixels: int = 6
    # Also emit one ROI per component covering body + every joint. This is the
    # "see the leads too" view; it is not a joint sample by itself.
    include_body_view: bool = True
    # Experimental: recover the true component axis with ``cv2.minAreaRect`` on
    # the box interior so a rotated part gets rotated ROIs. Off by default —
    # an unreliable angle silently mis-places every ROI, which is worse than a
    # slightly loose axis-aligned one.
    estimate_orientation: bool = False
    orientation_max_angle: float = 30.0

    # --- tighten the ROI onto the metal actually inside it ------------------
    # The ratios above say WHERE a terminal is; they cannot know how wide the
    # land under it happens to be. Measuring that from the pixels raised mean
    # IoU against known pad rectangles from 0.24 to 0.70 in the synthetic
    # benchmark, and tightened 20 of 21 ROIs on a real board.
    #
    # Deliberately confined to the already-predicted ROI. Searching a wider
    # neighbourhood finds every bright thing near the part -- traces, vias, a
    # neighbour's pads -- and was measurably worse on real boards even though it
    # scored the same on the synthetic one.
    refine_to_metal: bool = True
    # Same meaning as SolderGradingConfig.saturation_max: solder is bright and
    # close to grey. Duplicated rather than shared because step 5.5 must be able
    # to run with step 6.2 switched off entirely.
    saturation_max: int = 110
    # Ignore a metal blob smaller than this share of the ROI: below it the box
    # would be shrinking onto a speck of silkscreen.
    refine_min_metal_fraction: float = 0.02
    # Refuse a refinement that shrinks the ROI below this share of its area.
    # A joint with almost no solder must stay a big empty ROI, because that is
    # the evidence step 6.2 needs; collapsing it onto the one bright pixel
    # present would hide the defect.
    refine_min_area_fraction: float = 0.10

    # --- keep one component's ROI off its neighbour --------------------------
    # Each ROI is derived from its own box in isolation, and the outward reach
    # is a fixed fraction of that box. On a dense board the reach lands on the
    # neighbour: two chips 10 px apart produced facing ROIs overlapping 97%.
    # Overlapping ROIs make step 6.2 measure the same pixels twice and make
    # "bridge" meaningless, so they are pulled apart before anything is
    # measured on them.
    deconflict_neighbours: bool = True
    # An ROI is never cut below this share of its derived area. A joint starved
    # of solder must stay a big empty ROI; that emptiness is the evidence.
    deconflict_min_area_fraction: float = 0.25

    target_size: tuple[int, int] | None = (128, 128)
    letterbox_color: tuple[int, int, int] = (114, 114, 114)
    image_extension: Literal[".png", ".jpg"] = ".png"
    jpeg_quality: int = 95


@dataclass(slots=True)
class LeadDetectionConfig:
    """Pass 2: run a detector inside each component crop to find its leads.

    Inert until a model is named here. The measurement that keeps it inert is
    about the **component** detector (``models/active/detector``, the pass-1
    model): its ``pads`` class scores 0.072 recall because 30 of 670 training
    images carry the class at all, and on component crops of a real board it
    returned ``pads``/``pins`` in 0 of 24 configurations. Nothing about that
    figure describes any other model.

    In particular it says nothing about the step-6.2 solder **defect**
    segmenter. That is a different slot with a different job: it localises
    ``Dry_joint``/``Short_circuit``-style faults on the whole board, and it has
    no class for a sound joint, so it cannot tell this stage where the leads
    are. A pass-2 model has to find every lead, good ones included.

    ``model_path`` accepts anything the Ultralytics adapter can load whose
    class names appear in :data:`aoi_pipeline.solder.leads.LEAD_CLASSES` --
    ``pads``/``pins`` from the public datasets, or ``solder_joint`` from a model
    trained on this project's own hand-drawn boxes. A class name outside that
    set is not an error: every detection is simply dropped at fusion, silently.
    With ``model_path`` unset the stage returns nothing and step 5.5 keeps
    deriving geometry, exactly as before.
    """

    enabled: bool = True
    # Weights for the pass-2 detector. ``None`` leaves the stage inert; the
    # pipeline never borrows another role's model to fill this slot, because a
    # model trained for a different question answers a different question.
    model_path: str | None = None

    # --- what pass 2 is shown ------------------------------------------------
    # The fillet lies OUTSIDE the component silhouette, so a crop that stops at
    # the box hides the very thing pass 2 is looking for. The margin is a
    # fraction of the box's longer side so one number works for an 0402 chip
    # and for a connector.
    crop_margin_ratio: float = 0.35
    crop_margin_min_px: int = 6
    # Below this the crop carries too few pixels to be worth a forward pass.
    min_crop_px: int = 24

    # --- what comes back is believed ----------------------------------------
    confidence: float = 0.25
    # A "lead" a few pixels across is noise, not a joint.
    min_lead_px: int = 3

    # --- how many crops go in at once ---------------------------------------
    # Four. Measured, not guessed: 64 crops through yolo11n at imgsz 256 on
    # this machine's CPU took 28.9 ms/crop one at a time, 19.6 ms at batch 4,
    # then got *worse* -- 28.6 ms at 16, 33.2 ms at 64. Small batches amortise
    # the per-call setup; large ones lose more to memory traffic than they win,
    # because a CPU backend already parallelises across threads within one
    # image. The 1.47x at batch 4 is worth taking and is not the bottleneck:
    # dropping imgsz from 640 to 256 was 5.2x on its own.
    #
    # A GPU would move this optimum a long way up. Re-measure before raising it.
    batch_size: int = 4


@dataclass(slots=True)
class CadConfig:
    """Where the board's CAD data comes from and how it is registered.

    Every field is optional. With ``path`` unset the pipeline runs exactly as it
    does with no CAD at all, so this can stay in place until a board file is
    actually available.
    """

    path: str | None = None
    # ``auto`` sniffs the format; otherwise name one of ``aoi_pipeline.golden.inspector.cad.CAD_LOADERS``.
    fmt: str = "auto"
    units: str = "mm"
    # Boards are inspected one side at a time; ``None`` keeps every component.
    side: str | None = "top"

    # Registration. A saved matrix is reused as-is, which is the normal steady
    # state: register once per SKU/fixture, then never again.
    registration_path: str | None = None
    registration: dict[str, Any] | None = None
    # Correspondences in board mm and analysis-image pixels, when the operator
    # picks fiducials by hand instead of relying on auto-registration.
    fiducials_mm: list[tuple[float, float]] = field(default_factory=list)
    fiducials_px: list[tuple[float, float]] = field(default_factory=list)
    fiducial_perspective: bool = False
    # Fall back to matching CAD placements against the detections themselves.
    auto_register: bool = True
    auto_min_matches: int = 4
    auto_match_tolerance_ratio: float = 0.10
    auto_refine_rounds: int = 3


@dataclass(slots=True)
class FusionConfig:
    """How registered CAD geometry and derived ROIs are combined.

    The defaults keep every ROI either source produced. Losing a joint is worse
    than inspecting one twice, so nothing is dropped just because the two
    disagree -- the disagreement is reported instead.
    """

    enabled: bool = True

    # Registration quality gates. Below these the CAD half is abandoned rather
    # than applied, because mis-registered ROIs are indistinguishable from
    # correct ones by eye.
    min_inlier_ratio: float = 0.35
    max_residual_px: float = 25.0

    # Pairing CAD placements to detections.
    match_tolerance_mm: float = 3.0
    class_mismatch_penalty: float = 0.5

    # Per-component correction: CAD gives the footprint, the detector gives
    # where this part actually landed. This is what makes a globally
    # approximate registration locally accurate.
    local_refine: bool = True
    max_local_shift_mm: float = 2.0

    # Findings.
    max_shift_mm: float = 0.5
    report_unexpected: bool = True
    emit_cad_only_rois: bool = True

    # ROI construction from CAD lands.
    pad_margin_ratio: float = 0.35
    pitch_pad_ratio: float = 0.55
    fallback_pad_mm: float = 0.8
    min_roi_pixels: int = 6
    include_body_view: bool = True

    # Reconciling the two ROI sets.
    agreement_ios: float = 0.35
    merge_mode: Literal["cad", "union", "intersect"] = "union"
    keep_unmatched_derived: bool = True

    # Placement-only CAD (pick-and-place): rebuild the derived ROI geometry on
    # the CAD centre and rotation instead of the detector's axis-aligned box.
    reanchor_on_placement: bool = True
    solder: SolderJointConfig = field(default_factory=SolderJointConfig)


@dataclass(slots=True)
class ClassificationConfig:
    """Runtime policy for the step-6.1 component-family classifier.

    Preprocessing details and the default confidence policy live in the model
    manifest. Non-``None`` values below are explicit deployment overrides.
    """

    batch_size: int = 32
    top_k: int = 3
    device: Literal["cpu", "cuda", "auto"] = "cpu"
    accept_threshold: float | None = None
    review_threshold: float | None = None
    temperature: float | None = None
    # Average softmax over the same 4 flip views (identity, horizontal,
    # vertical, both) the training notebook used to measure macro recall.
    # Measured there: 0.9292 -> 0.9417 on the same weights, so this is a
    # no-retrain accuracy gain, at the cost of 4x classifier inference time
    # (crops are small, so this is still cheap in absolute terms).
    tta: bool = False


@dataclass(slots=True)
class LeadFusionConfig:
    """Step 5.5: prefer detected lead/pad boxes over derived ones.

    Inert until the detector actually reports ``pads``/``pins``. On the shipped
    model those classes score 0.00 and 0.14-0.21 recall, so the derived geometry
    still carries the pipeline; this only takes over where a real detection
    exists.
    """

    enabled: bool = True
    # A lead box is only trusted above this. Lead classes are the weakest part
    # of the detector, so the bar sits higher than the component threshold.
    min_lead_confidence: float = 0.35
    # Distance from the lead centre to the body box, as a multiple of the body's
    # shorter side. Leads sit just outside the body, so a small multiple is
    # right; too large and a neighbouring part's lead gets adopted.
    max_lead_distance_ratio: float = 1.2
    # Overlap at which a detected lead is taken to stand in for a derived ROI.
    # Below it, both are kept -- they are evidently looking at different things.
    replace_ios: float = 0.30
    # A lead detection that belongs to no component body still describes a real
    # inspectable joint: a test point, an unpopulated footprint, a pad whose
    # component the detector missed. Refusing to *attach* it to a wrong body is
    # right; throwing it away is not. Measured on the shipped detector, ``pads``
    # scores 0.712 precision against 0.072 recall -- when it does fire it is
    # usually correct, so every firing is worth keeping.
    keep_unassigned_leads: bool = True


@dataclass(slots=True)
class SolderGradingConfig:
    """Step 6.2: how solder ROIs are measured, judged and fused.

    Works with no model at all -- the rule layer alone produces verdicts. The
    model fields stay empty until a trained artifact is dropped in, and nothing
    else has to change when it is.
    """

    # Chấm điểm trên ảnh CHƯA tăng cường quang học, khi có.
    #
    # Đo trên board của dự án: chuỗi CLAHE + normalize + sharpen làm tỉ lệ pixel
    # cháy sáng trong ROI mối hàn đi từ 19 % lên 48 %, và độ phủ kim loại đo
    # được từ 44 % lên 63 %. Hệ quả là 10 mối hàn bình thường bị gọi thành
    # ``bridge``: trên ảnh nguồn chúng có 35 % pixel cháy, sau tăng cường là
    # 71 % -- hai pad cạnh nhau nhoè vào nhau và luật đọc ra là cầu chì.
    #
    #     ảnh dùng để chấm     bridge   insufficient   phải xem tay
    #     đã tăng cường            12              1             16
    #     nguồn                     2              6             10
    #
    # Lưu ý hướng ngược lại: **khoanh ROI thì ảnh tăng cường lại tốt hơn** (0 so
    # với 3 ROI rơi trên chữ lụa), nên chỉ khâu chấm điểm đổi ảnh, còn hình học
    # giữ nguyên. Hai khâu hỏng vì hai lý do khác nhau.
    #
    # Ảnh radiometric được chụp sau undistort/resize và được warp bằng chính
    # homography của bước căn chỉnh, nên luôn phải đi kèm ảnh phân tích trong
    # cùng kết quả stage. Nếu transform không áp dụng được, pipeline bỏ nhánh
    # này và fail-safe về ảnh phân tích thay vì cắt pixel lệch tọa độ.
    prefer_radiometric_image: bool = True
    enabled: bool = True

    # --- model artifacts (optional) ----------------------------------------
    model_path: str | None = None
    manifest_path: str | None = None
    batch_size: int = 32
    device: Literal["cpu", "cuda", "auto"] = "cpu"
    # Deployment overrides for the manifest's own policy; None keeps the
    # manifest value so the trained contract stays authoritative.
    accept_threshold: float | None = None
    review_threshold: float | None = None
    temperature: float | None = None

    # --- segmentation -------------------------------------------------------
    saturation_max: int = 110
    specular_percentile: float = 99.0

    # --- joint rule thresholds ---------------------------------------------
    # Fractions of the ROI, so they survive a change of lens or magnification.
    missing_solder_ratio: float = 0.04
    insufficient_solder_ratio: float = 0.18
    excess_solder_ratio: float = 0.80
    insufficient_span_ratio: float = 0.45
    cold_specular_ratio: float = 0.010
    cold_contrast: float = 0.06
    max_centroid_offset_ratio: float = 0.55
    bridge_edge_contact: float = 0.35
    bridge_span_ratio: float = 0.85

    # --- component rule thresholds -----------------------------------------
    missing_component_ratio: float = 0.02

    # Same 4-flip-view averaging as ClassificationConfig.tta. The solder
    # notebook only validated random-flip *training* augmentation, not this
    # eval-time averaging, so treat the accuracy gain here as expected by the
    # same reasoning rather than a number measured on this specific model.
    tta: bool = False

    # --- fusion -------------------------------------------------------------
    # A model verdict below this is never accepted on its own.
    model_accept_probability: float = 0.80
    # Hard physical floor. However confident the model is that a joint is good,
    # this much solder has to actually be there; an escape ships a bad board,
    # a false call costs an operator ten seconds.
    escape_guard_solder_ratio: float = 0.10
    escape_guard_enabled: bool = True
    # When the two layers disagree, send it to review rather than picking a
    # winner. Turning this off makes the model authoritative.
    disagreement_is_review: bool = True
    # With no model loaded, should a rule-only defect be a hard reject or a
    # review queue item? Review is the safe default while thresholds settle.
    rules_only_defect_decision: Literal["reject", "review"] = "review"


@dataclass(slots=True)
class PipelineConfig:
    """Top-level configuration consumed by :class:`AOIPipeline`."""

    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    board: BoardConfig = field(default_factory=BoardConfig)
    fiducials: FiducialConfig = field(default_factory=FiducialConfig)
    cv_detector: CVDetectorConfig = field(default_factory=CVDetectorConfig)
    model_detector: ModelDetectorConfig = field(default_factory=ModelDetectorConfig)
    tiling: TilingConfig = field(default_factory=TilingConfig)
    crop: CropConfig = field(default_factory=CropConfig)
    solder: SolderJointConfig = field(default_factory=SolderJointConfig)
    lead_fusion: LeadFusionConfig = field(default_factory=LeadFusionConfig)
    lead_detection: LeadDetectionConfig = field(
        default_factory=LeadDetectionConfig
    )
    cad: CadConfig = field(default_factory=CadConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    solder_grading: SolderGradingConfig = field(default_factory=SolderGradingConfig)
    solder_defect_detection: SolderDefectDetectionConfig = field(
        default_factory=SolderDefectDetectionConfig
    )
    detector_mode: Literal["auto", "cv"] = "auto"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> PipelineConfig:
        """Translate permissive UI dictionaries into the typed core config.

        Unknown UI-only keys are ignored so the facade can evolve independently
        from Streamlit controls. Direct ``PipelineConfig`` use remains preferred
        for scripts and tests.
        """

        if not values:
            return cls()
        config = cls()
        preprocess_values = _section(values, "preprocess")
        alignment_values = _section(values, "alignment")
        board_values = _section(values, "board")
        component_values = _section(values, "components", "cv_detector")
        model_values = _section(values, "model_detector", "model")
        tiling_values = _section(values, "tiling")
        crop_values = _section(values, "crops", "crop")
        solder_values = _section(values, "solder", "solder_joints")
        lead_values = _section(values, "lead_fusion", "leads")
        lead_detection_values = _section(
            values, "lead_detection", "pass2", "second_pass"
        )
        cad_values = _section(values, "cad", "board_cad")
        fusion_values = _section(values, "fusion", "cad_fusion")
        grading_values = _section(values, "solder_grading", "solder_inspection")
        solder_defect_values = _section(
            values,
            "solder_defect_detection",
            "solder_detector",
            "solder_segmentation",
        )
        classification_values = _section(values, "classification", "classifier")

        _assign_known(
            config.preprocess,
            preprocess_values,
            aliases={"clahe_clip": "clahe_clip_limit"},
        )
        if preprocess_values.get("resize_enabled") is False:
            config.preprocess.max_side = None
        denoise = preprocess_values.get("denoise")
        if isinstance(denoise, str):
            normalized_denoise = denoise.strip().lower().replace(" ", "")
            config.preprocess.denoise = normalized_denoise not in {"none", "off", "false"}
            method_aliases = {
                "nlmeans": "nlmeans",
                "fastnlmeans": "nlmeans",
                "bilateral": "bilateral",
                "gaussian": "gaussian",
            }
            if normalized_denoise in method_aliases:
                config.preprocess.denoise_method = method_aliases[normalized_denoise]
        sharpen = preprocess_values.get("sharpen")
        if isinstance(sharpen, (int, float)) and not isinstance(sharpen, bool):
            config.preprocess.sharpen = sharpen > 0
            config.preprocess.sharpen_amount = float(sharpen)

        _assign_known(
            config.alignment,
            alignment_values,
            aliases={
                "features": "orb_features",
                "match_ratio": "ratio_test",
                "ransac_threshold": "ransac_reprojection_threshold",
            },
        )
        _assign_known(config.board, board_values)
        _assign_known(
            config.cv_detector,
            component_values,
            aliases={
                "max_candidates": "max_detections",
                "iou": "nms_iou_threshold",
            },
        )
        _assign_known(
            config.model_detector,
            model_values,
            aliases={"conf": "confidence", "imgsz": "image_size", "max_det": "max_detections"},
        )
        # The local UI deliberately exposes shared detector controls in its
        # ``components`` section. Apply those values to learned-model inference
        # as well as the OpenCV proposal detector.
        _assign_known(
            config.model_detector,
            component_values,
            aliases={
                "conf": "confidence",
                "imgsz": "image_size",
                "max_det": "max_detections",
                "max_candidates": "max_detections",
            },
        )
        if config.model_detector.device == "auto":
            config.model_detector.device = None
        _assign_known(config.tiling, tiling_values)
        _assign_known(
            config.tiling,
            component_values,
            aliases={
                "tiling_mode": "mode",
                "tile_overlap": "overlap_ratio",
                "tile_trigger_scale": "auto_trigger_scale",
                "full_image_pass": "include_full_image",
                "tile_confidence": "detail_confidence",
                "merge_iou": "merge_iou_threshold",
                "seam_ios": "seam_ios_threshold",
                "containment_ios": "containment_ios_threshold",
                "cross_class_iou": "cross_class_iou_threshold",
            },
        )
        if "tile_led_confidence" in component_values:
            config.tiling.detail_class_confidence["led"] = float(
                component_values["tile_led_confidence"]
            )
        _assign_known(
            config.crop,
            crop_values,
            aliases={"padding": "padding_pixels"},
        )
        target_size = crop_values.get("target_size")
        if isinstance(target_size, (int, float)):
            side = int(target_size)
            config.crop.target_size = (side, side) if side > 0 else None
        if crop_values.get("normalize") is False:
            config.crop.target_size = None
        _assign_known(
            config.solder,
            solder_values,
            aliases={"joint_size": "target_size", "pins": "split_pins"},
        )
        solder_target = solder_values.get("target_size")
        if isinstance(solder_target, (int, float)):
            side = int(solder_target)
            config.solder.target_size = (side, side) if side > 0 else None
        # ``cad``/``fusion`` are only read when the caller supplied that section;
        # the permissive whole-dict fallback would otherwise let unrelated keys
        # switch CAD on.
        if isinstance(values.get("cad"), Mapping) or isinstance(
            values.get("board_cad"), Mapping
        ):
            _assign_known(
                config.cad,
                cad_values,
                aliases={"file": "path", "format": "fmt", "board_side": "side"},
            )
        if isinstance(values.get("fusion"), Mapping) or isinstance(
            values.get("cad_fusion"), Mapping
        ):
            _assign_known(config.fusion, fusion_values)
        # Step 5.5 geometry is shared: re-anchored ROIs must match the ones the
        # detector-only path would have produced.
        config.fusion.solder = config.solder
        if isinstance(values.get("lead_fusion"), Mapping) or isinstance(
            values.get("leads"), Mapping
        ):
            _assign_known(config.lead_fusion, lead_values)
        # Only when the caller supplied the section: the permissive whole-dict
        # fallback would otherwise let unrelated keys reconfigure pass 2.
        if any(
            isinstance(values.get(name), Mapping)
            for name in ("lead_detection", "pass2", "second_pass")
        ):
            _assign_known(config.lead_detection, lead_detection_values)
        if isinstance(values.get("solder_grading"), Mapping) or isinstance(
            values.get("solder_inspection"), Mapping
        ):
            _assign_known(
                config.solder_grading,
                grading_values,
                aliases={"model": "model_path", "manifest": "manifest_path"},
            )
        if any(
            isinstance(values.get(name), Mapping)
            for name in (
                "solder_defect_detection",
                "solder_detector",
                "solder_segmentation",
            )
        ):
            _assign_known(
                config.solder_defect_detection,
                solder_defect_values,
                aliases={
                    "model": "model_path",
                    "manifest": "manifest_path",
                    "conf": "confidence",
                    "imgsz": "image_size",
                    "max_det": "max_detections",
                },
            )
            if config.solder_defect_detection.device == "auto":
                config.solder_defect_detection.device = None
        _assign_known(config.classification, classification_values)
        detector_mode = values.get("detector_mode")
        if detector_mode in {"auto", "cv"}:
            config.detector_mode = detector_mode
        return config


def _section(values: Mapping[str, Any], *names: str) -> dict[str, Any]:
    for name in names:
        section = values.get(name)
        if isinstance(section, Mapping):
            return dict(section)
    return dict(values)


def _assign_known(
    target: Any,
    values: Mapping[str, Any],
    aliases: Mapping[str, str] | None = None,
) -> None:
    aliases = aliases or {}
    fields = target.__dataclass_fields__
    for source_name, value in values.items():
        target_name = aliases.get(source_name, source_name)
        if target_name in fields and value is not None:
            setattr(target, target_name, value)
