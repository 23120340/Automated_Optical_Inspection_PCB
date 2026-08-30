# Context triển khai Position Check và Golden Compare

## 1. Mục tiêu

Mở rộng dự án `23120340/Automated_Optical_Inspection_PCB` để kiểm tra một PCBA theo Golden Image với hai kết quả độc lập:

1. **Position Check**: đo độ lệch `dx`, `dy`, góc xoay của từng linh kiện so với vị trí trên Golden; trả cả pixel và mm.
2. **Golden Compare**: kiểm tra bất thường ngoại hình của từng slot sau khi đã bù pose cục bộ; không dùng điểm giống nhau để suy ra độ lệch vị trí.

Phần cứng được giả định đã hoàn chỉnh và ổn định: camera, lens, ánh sáng, focus, exposure, đồ gá và độ phân giải không thay đổi trong một recipe. Phạm vi tài liệu này tập trung vào phần mềm.

Snapshot được khảo sát: nhánh `main`, commit `b477c65314aca5afbdaa0bc2abd7652313a4847d` ngày 2026-08-17.

## 2. Trạng thái repo hiện tại

Luồng hiện có trong `aoi_pipeline/pipeline.py`:

```text
Import
→ preprocessing/undistort
→ Golden alignment tùy chọn
→ board localization
→ component detection
→ crop theo detection bbox
→ family classification
```

Các thành phần có thể tái sử dụng:

- `aoi_pipeline/imaging/calibration.py`: camera intrinsic calibration và undistort.
- `aoi_pipeline/imaging/alignment.py`: ORB/homography và ECC fallback.
- `aoi_pipeline/detection/detectors.py`: `best.onnx`/`best.pt` detector adapter và CV demo.
- `aoi_pipeline/detection/tiling.py`: adaptive tiled inference và merge detection.
- `aoi_pipeline/detection/cropping.py`: crop cho classifier.
- `aoi_pipeline/classification/family.py`: family classifier ONNX.
- `app/pipeline_bridge.py`, `app/streamlit_app.py`: UI bridge và Streamlit workflow.
- `aoi_pipeline/reporting/exporters.py`: JSON/ZIP hiện tại.

Các thiếu hụt đối với inspection:

- `main` chưa có `aoi_pipeline/golden/recipe.py` hoặc persistent slot ID.
- `Detection.detection_id` được sinh ngẫu nhiên, không đại diện cho một vị trí cố định trên board.
- Alignment hiện cho phép `resize_fallback`; kết quả đó không hợp lệ cho đo lường.
- ORB trên toàn board có thể lấy feature từ chính linh kiện lỗi và làm lệch phép biến đổi.
- Crop hiện tại bám theo bbox ảnh test và có thể letterbox về 224×224; không được dùng để đo position hoặc Golden Compare.
- UI Difference mới chỉ hiển thị `cv2.absdiff`, chưa có mask, threshold, quality gate hoặc PASS/NG.
- `PipelineRun` chưa có position, appearance và board inspection decision.
- Camera calibration hiện sửa méo lens nhưng chưa cung cấp contract pixel-to-mm cho inspection.

## 3. Quyết định kiến trúc

### 3.1. `best.onnx` và `AOIInspector` không thay thế nhau

`best.onnx` tiếp tục được sử dụng bên trong hệ thống:

- Tự động đề xuất linh kiện/slot khi tạo recipe từ Golden.
- Hỗ trợ kiểm tra presence, missing, extra hoặc tìm kiếm lệch lớn ở runtime.
- Cung cấp class hint và confidence.

Không dùng tâm bbox của `best.onnx` làm phép đo cuối vì bbox có thể rung do resize, letterbox, NMS, label noise và confidence. `AOIInspector` là lớp điều phối, không phải model mới. Nó tái sử dụng detector và chịu trách nhiệm về alignment, slot matching, metrology, Golden Compare, tolerance và kết luận.

### 3.2. Giữ `AOIPipeline`, thêm `AOIInspector`

- `AOIPipeline` tiếp tục phục vụ discovery/detection/crop/classification và giữ tương thích với test hiện có.
- Thêm `AOIInspector` trong `aoi_pipeline/golden/inspector.py` cho luồng production yêu cầu recipe và strict alignment.
- Không nhồi ngay inspection state vào `AOIPipeline.run()` vì run hiện cho phép không có Golden và có fallback phục vụ demo.
- Có thể thêm facade mỏng `AOIPipeline.inspect(...)` sau này, nhưng facade phải ủy quyền cho `AOIInspector`.

### 3.3. Tách hai image domain

1. **Measurement image**: ảnh BGR ở độ phân giải đo, chỉ thực hiện decode, undistort và biến đổi hình học bắt buộc. Không letterbox, không resize tùy ý, không adaptive CLAHE/normalize độc lập mặc định.
2. **Inference image**: có thể denoise, white balance, CLAHE, normalize, sharpen và resize để chạy detector/classifier.

Mọi kết quả inspection cuối cùng phải được quy về hệ tọa độ canonical `golden_board_pixels`. `model_input_pixels` chỉ là tọa độ nội bộ và không được export như tọa độ đo.

## 4. Kiến trúc mục tiêu

```text
Golden enrollment
→ undistort measurement image
→ xác định alignment anchors
→ chạy best.onnx để đề xuất linh kiện
→ xác nhận/lọc slot
→ tạo fixed ROI, template, mask, expected pose
→ lưu InspectionRecipe

Ảnh test
→ undistort measurement image
→ strict board alignment vào golden_board_pixels
→ crop fixed ROI theo recipe
→ PositionChecker đo dx/dy/rotation
→ bù pose cục bộ cho nhánh appearance
→ GoldenComparator tính anomaly metrics
→ hợp nhất kết quả slot
→ PASS / NG / REVIEW / INVALID
```

## 5. Golden enrollment và recipe

Mỗi mã board/mặt board có một recipe riêng. Cấu trúc đề xuất:

```text
golden_recipes/<board_id>/<side>/
├── recipe.json
├── golden.png
├── templates/
│   └── slot_0001.png
└── masks/
    └── slot_0001.png
```

`recipe.json` tối thiểu chứa:

```json
{
  "schema_version": "aoi-inspection-recipe/1.0",
  "board_id": "BOARD_A",
  "side": "top",
  "golden_sha256": "...",
  "coordinate_space": "golden_board_pixels",
  "image_size": {"width": 4096, "height": 3072},
  "metrology": {
    "pixels_per_mm_x": 40.0,
    "pixels_per_mm_y": 40.0
  },
  "alignment": {
    "anchors": [],
    "max_residual_px": 0.5
  },
  "slots": []
}
```

Mỗi slot chứa:

- `slot_id` ổn định như `slot_0001`; không dùng UUID của detection.
- `label_hint`, `class_id` nếu có.
- `expected_bbox_xyxy` và `fixed_roi_xyxy` trong `golden_board_pixels`.
- `expected_center_px`, `expected_angle_deg`.
- `rotation_period_deg`: `180`, `360` hoặc `null` nếu góc không đo được.
- `template_path`, `component_mask_path`, `compare_mask_path`.
- `search_margin_px`.
- Position tolerance theo X, Y và góc.
- Appearance thresholds theo slot hoặc profile footprint.

Slot được sắp xếp ổn định theo tâm từ trên xuống dưới rồi trái sang phải. Bbox vượt biên phải được clamp; bbox rỗng hoặc nằm ngoài ảnh phải bị loại và ghi lý do.

Production recipe không được tự động tạo từ nguồn `opencv_candidate` trừ khi người dùng chủ động bật demo. Ưu tiên detector `best.onnx`; UI có thể cho phép kiểm tra/chỉnh slot một lần khi enrollment.

## 6. Strict board alignment

Alignment production phải dựa trên fiducial, lỗ board hoặc stable patches được lưu trong recipe.

Quy trình đề xuất:

1. Tìm thô anchor trong search ROI.
2. Tinh chỉnh tâm anchor ở mức sub-pixel.
3. Ước lượng similarity/partial affine bằng `cv2.estimateAffinePartial2D`.
4. Tính residual/reprojection error cho từng anchor.
5. Warp measurement image vào đúng kích thước Golden.
6. Nếu thiếu anchor, residual quá cao hoặc transform không hợp lý: trả `INVALID` và dừng inspection.

Với camera và board plane cố định, ưu tiên similarity/partial affine thay vì homography 8 tham số. Chỉ dùng homography khi có bằng chứng phối cảnh cần hiệu chỉnh. ORB/ECC toàn board có thể giữ làm fallback demo/diagnostic, không được âm thầm dùng cho production PASS/NG.

## 7. Position Check

Không crop ảnh test theo bbox detector. Mỗi slot luôn crop theo `fixed_roi_xyxy` từ Golden sau board alignment.

Pipeline đo đề xuất:

```text
fixed Golden template + fixed test ROI
→ grayscale/gradient representation
→ apply component mask
→ coarse template matching trong search margin
→ masked ECC với MOTION_EUCLIDEAN
→ dx_px, dy_px, rotation_deg
→ quality gate
→ pixel-to-mm
→ tolerance decision
```

Chỉ cho local transform translation và rotation. Không cho scale/shear vì có thể che lỗi sai kích thước hoặc biến dạng ngoại hình.

Mỗi phép đo phải có:

- `coarse_score`.
- `ecc_correlation` hoặc equivalent pose score.
- `residual`.
- `valid_overlap_ratio`.
- `measurement_status` và lý do.

Nếu score không đạt, trả `unmeasurable` hoặc `missing_candidate`; không xuất một độ lệch giả như thể phép đo hợp lệ.

Quy đổi:

```text
dx_mm = dx_px / pixels_per_mm_x
dy_mm = dy_px / pixels_per_mm_y
```

Các trục X/Y là trục canonical của Golden. Tolerance phải hỗ trợ theo từng slot hoặc footprint; không bắt buộc dùng một ngưỡng toàn board.

## 8. Golden Compare

Golden Compare chỉ đánh giá appearance, không suy ra position.

1. Nếu pose cục bộ hợp lệ, bù translation/rotation của test ROI về pose Golden.
2. Tạo valid-overlap mask sau warp.
3. Áp dụng `compare_mask`; loại nền, vùng không ổn định hoặc vùng phản xạ được đánh dấu ignore.
4. Tính tối thiểu:
   - local SSIM;
   - normalized absolute-difference ratio;
   - edge-difference ratio;
   - diện tích blob bất thường lớn nhất.
5. Áp dụng morphology/connected components để loại nhiễu nhỏ.
6. Đánh giá bằng threshold của slot/profile.

Không dùng một điểm SSIM duy nhất làm kết luận. Một linh kiện lệch nhưng ngoại hình đúng phải có thể trả `NG_POSITION` và `PASS_APPEARANCE` sau pose compensation.

Nếu pose không đo được:

- Không bù pose tùy tiện.
- Có thể chạy missing/anomaly gate riêng.
- Trả `not_evaluated` nếu không đủ dữ liệu để so sánh đáng tin cậy.

MVP chỉ nên trả `appearance_anomaly`; không tự gán `solder_defect`, `wrong_part` hoặc `polarity_error` nếu chưa có rule/model và dữ liệu xác thực cho nhãn đó.

## 9. Runtime models và quyết định

Dataclass đề xuất:

- `InspectionRecipe`
- `SlotRecipe`
- `AlignmentQuality`
- `PositionCheckResult`
- `GoldenCompareResult`
- `SlotInspectionResult`
- `InspectionRun`

Trạng thái chuẩn:

- Alignment: `valid`, `invalid`.
- Position: `pass`, `shifted`, `rotated`, `missing_candidate`, `unmeasurable`.
- Appearance: `pass`, `anomaly`, `not_evaluated`.
- Board: `pass`, `ng`, `review`, `invalid`.

Ưu tiên quyết định board:

```text
alignment invalid → INVALID
ít nhất một slot NG → NG
không có NG nhưng có unmeasurable/not_evaluated → REVIEW
tất cả slot hợp lệ và pass → PASS
```

Ví dụ kết quả slot:

```json
{
  "slot_id": "slot_0015",
  "position": {
    "dx_px": 2.35,
    "dy_px": -0.82,
    "dx_mm": 0.0588,
    "dy_mm": -0.0205,
    "rotation_deg": 0.7,
    "score": 0.96,
    "status": "shifted"
  },
  "appearance": {
    "ssim": 0.97,
    "diff_ratio": 0.011,
    "edge_diff_ratio": 0.016,
    "max_blob_area_px": 14,
    "status": "pass"
  },
  "status": "ng_position"
}
```

## 10. Kế hoạch thay đổi file

### File mới

- `aoi_pipeline/golden/recipe.py`: schema, validation, enrollment, load/save recipe.
- `aoi_pipeline/golden/position.py`: local pose estimation và tolerance decision.
- `aoi_pipeline/golden/compare.py`: pose compensation, metrics, anomaly mask.
- `aoi_pipeline/golden/inspector.py`: `AOIInspector` và board decision.
- `tests/test_recipe.py`.
- `tests/test_position.py`.
- `tests/test_golden_compare.py`.
- `tests/test_inspection.py`.

### File chỉnh sửa theo giai đoạn

- `aoi_pipeline/models.py`: runtime result dataclasses nếu không đặt trong module riêng.
- `aoi_pipeline/config.py`: recipe/alignment/position/compare configs.
- `aoi_pipeline/imaging/alignment.py`: strict anchor-based alignment; giữ API cũ tương thích.
- `aoi_pipeline/__init__.py`: public exports.
- `aoi_pipeline/reporting/exporters.py`: `positions.csv`, `appearance.csv`, overlay và anomaly masks.
- `app/pipeline_bridge.py`: adapter cho enrollment/inspection.
- `app/streamlit_app.py`: mode Build Recipe và Inspect Board.
- `README.md`: hướng dẫn và giới hạn.

Không bắt đầu bằng UI. Hoàn thành core, unit tests và JSON result trước, sau đó mới nối bridge/UI/export.

## 11. Milestone triển khai

### M1 — Contract và recipe

- Thêm dataclass/config/schema validation.
- Tạo stable slot từ Golden detections.
- Lưu/load recipe và kiểm tra Golden hash, image size, coordinate space.
- Test clamp, ordering, stable ID và malformed recipe.

### M2 — Strict alignment

- Anchor detection và partial-affine transform.
- Quality metrics và fail-closed behavior.
- Test transform đã biết, thiếu anchor, residual cao và sign convention.

### M3 — Position X/Y MVP

- Fixed ROI, coarse template và sub-pixel refinement.
- Translation only trước; thêm rotation sau khi X/Y đạt acceptance.
- Test synthetic shifts âm/dương, fractional pixel, edge slot, missing và detector bbox jitter.

### M4 — Rotation và Golden Compare

- `MOTION_EUCLIDEAN`, angle periodicity và not-applicable handling.
- Pose compensation, masks và multi-metric appearance result.
- Test không double-report shifted component thành appearance anomaly.

### M5 — Inspector integration

- `AOIInspector.inspect(test_image, recipe)`.
- Aggregate slot/board decision.
- JSON result và debug overlays.
- Giữ toàn bộ API/test cũ của `AOIPipeline`.

### M6 — UI/export/validation

- Build Recipe/Inspect Board workflow.
- Export CSV/ZIP và ảnh debug.
- Tune threshold bằng ảnh thực.

## 12. Kiểm chứng và tiêu chí chấp nhận

Tối thiểu phải có:

1. Synthetic tests với dịch chuyển `±0.25`, `±0.5`, `±1`, `±2` px và góc đã biết.
2. Cùng một board OK chụp lặp ít nhất 30 lần.
3. Mẫu hoặc ảnh kiểm chứng tại `0.5T`, `1.0T`, `1.5T`, trong đó `T` là tolerance.
4. Mẫu missing, wrong appearance và thay đổi sáng nhỏ.
5. Alignment failure test chứng minh pipeline trả `INVALID`, không tiếp tục bằng resize.

Quality targets nên biểu diễn tương đối với tolerance:

- Alignment error p95 không lớn hơn khoảng `1/4` độ lệch nhỏ nhất cần phát hiện.
- Position repeatability p95 không lớn hơn khoảng `1/3` tolerance.
- Mẫu chỉ shifted phải không tự động thành appearance NG sau pose compensation.
- Không công bố độ chính xác mm trước khi có dữ liệu repeatability và controlled-shift validation.

Golden Image xác định hình học chuẩn. Nên dùng thêm 20–30 ảnh/board OK để tune threshold appearance và đo repeatability; các ảnh này không thay thế Golden.

## 13. Thông số cần chốt trước khi tune production

- `board_id`, `side` và quy ước version recipe.
- `pixels_per_mm_x`, `pixels_per_mm_y` hoặc ma trận pixel-to-mm tương đương.
- Độ lệch nhỏ nhất cần phát hiện.
- Tolerance X/Y/góc theo slot hoặc footprint.
- Search margin tối đa.
- Loại alignment anchors và residual gate.
- Danh sách component không đo được góc hoặc có chu kỳ 180°.
- Bộ ảnh OK/NG dùng cho acceptance.

