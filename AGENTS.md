# Chỉ dẫn triển khai cho Codex

> Khi đặt file này vào root repository để Codex tự động đọc, nên đổi tên thành `AGENTS.md`. Nội dung bên dưới áp dụng cho toàn bộ repository trừ khi một `AGENTS.md` cấp sâu hơn ghi đè.

## 1. Nhiệm vụ

Triển khai Position Check và Golden Compare theo `docs/design/thiet_ke_position_va_golden_compare.md` cho dự án `Automated_Optical_Inspection_PCB`.

Kết quả mục tiêu:

- Tạo và validate Golden inspection recipe.
- Căn chỉnh board theo recipe với quality gate nghiêm ngặt.
- Đo `dx`, `dy`, rotation theo từng fixed slot ROI.
- Quy đổi pixel sang mm và áp dụng tolerance.
- Golden Compare sau local pose compensation.
- Tách rõ position status, appearance status và board status.
- Tái sử dụng `best.onnx`; không thay nó bằng `AOIInspector`.

Phần cứng được coi là hoàn chỉnh và ổn định. Không mở rộng phạm vi sang điều khiển camera, RTSP, thiết kế ánh sáng hoặc đồ gá trừ khi người dùng yêu cầu.

## 2. Việc phải làm trước khi sửa code

1. Đọc toàn bộ `docs/design/thiet_ke_position_va_golden_compare.md`.
2. Kiểm tra `git status`, branch và commit hiện tại; không giả định checkout giống snapshot trong context.
3. Đọc tối thiểu:
   - `README.md`
   - `aoi_pipeline/pipeline.py`
   - `aoi_pipeline/models.py`
   - `aoi_pipeline/config.py`
   - `aoi_pipeline/imaging/alignment.py`
   - `aoi_pipeline/detection/detectors.py`
   - `aoi_pipeline/detection/cropping.py`
   - `aoi_pipeline/reporting/exporters.py`
   - `app/pipeline_bridge.py`
   - các test liên quan.
4. Chạy test baseline và ghi nhận kết quả trước khi thay đổi.
5. Giữ nguyên mọi thay đổi không liên quan đã có trong worktree.

Nếu code hiện tại đã khác `docs/design/thiet_ke_position_va_golden_compare.md`, ưu tiên code thực tế và báo rõ khác biệt trước khi thay đổi kiến trúc lớn.

## 3. Nguyên tắc kiến trúc bắt buộc

### `best.onnx`

- Giữ detector ONNX hiện tại và adapter `UltralyticsDetector`.
- `AOIInspector` phải có thể nhận/reuse `ComponentDetector` hoặc `AOIPipeline.detector` bằng dependency injection.
- Detector dùng để đề xuất slot, presence/missing/extra và class hint.
- Không dùng detector bbox center làm kết quả position cuối.
- Không âm thầm chuyển sang `CVComponentDetector` khi model production được cấu hình nhưng load/inference thất bại.
- Không cho phép nguồn `opencv_candidate` tạo production recipe mặc định.

### `AOIPipeline` và `AOIInspector`

- Không phá behavior hiện có của `AOIPipeline.run()`.
- Thêm `AOIInspector` trong module riêng, ưu tiên `aoi_pipeline/golden/inspector.py`.
- `AOIInspector` là orchestration/service class, không phải model AI.
- Có thể thêm facade vào `AOIPipeline` sau khi core ổn định, nhưng không duplicate logic.

### Image domain và tọa độ

- Tách measurement image khỏi inference image.
- Measurement image không được letterbox hoặc resize tùy ý.
- Không dùng crop 224×224 của classifier cho metrology/compare.
- Canonical coordinate space là `golden_board_pixels`.
- Mọi transform giữa raw, undistorted, aligned và model coordinates phải được ghi rõ và test được.
- Tọa độ bbox dùng quy ước `xyxy`, right/bottom exclusive như code hiện tại.

### Alignment

- Production inspection phải fail closed.
- Nếu anchor không đủ, residual vượt gate hoặc transform không hợp lý: trả `INVALID` và dừng.
- Không dùng `resize_fallback` để đưa ra PASS/NG production.
- Ưu tiên fiducial/hole/stable-patch alignment và partial affine/similarity.
- ORB/ECC toàn board chỉ là fallback demo hoặc diagnostic nếu chưa được xác nhận riêng.

### Slot và ROI

- Slot ID phải deterministic và ổn định giữa các lần build recipe từ cùng input.
- Fixed ROI lấy từ Golden recipe; runtime không recrop theo bbox test.
- Clamp bbox/ROI vào biên ảnh.
- Loại bbox rỗng/ngoài ảnh và lưu lý do.
- Template/mask phải giữ độ phân giải gốc của measurement image.

### Position Check

- Thực hiện coarse search rồi sub-pixel refinement.
- Local warp chỉ cho translation ở MVP; sau đó Euclidean translation + rotation.
- Không cho scale/shear trong local pose refinement.
- Dùng mask/gradient để giảm ảnh hưởng nền PCB tĩnh.
- Trả quality metrics và failure reason.
- Khi phép đo không tin cậy, trả `unmeasurable`/`missing_candidate`; không tạo số đo giả.
- Quy ước dấu `dx`, `dy`, góc phải được mô tả trong docstring và khóa bằng synthetic tests.

### Golden Compare

- Chạy sau Position Check.
- Nếu pose hợp lệ, compensate pose trước khi so appearance.
- Dùng nhiều metric; không quyết định chỉ bằng một SSIM score.
- Phải hỗ trợ compare/ignore masks và valid-overlap mask.
- Không double-report một component chỉ shifted thành appearance anomaly nếu ngoại hình không đổi.
- Chỉ trả nhãn defect cụ thể khi có rule/model và test tương ứng; MVP dùng `appearance_anomaly`.

## 4. Trình tự triển khai bắt buộc

Không bắt đầu bằng UI. Thực hiện theo các vertical milestone sau:

### Phase 1 — Contract và recipe

1. Thêm recipe/config/result dataclasses.
2. Thêm schema validation và versioning.
3. Tạo stable slots từ Golden detections.
4. Thêm load/save recipe.
5. Viết `tests/test_recipe.py`.
6. Chạy toàn bộ test.

### Phase 2 — Strict alignment

1. Thêm anchor-based aligner/API mà không phá `PCBAligner.align()` hiện tại.
2. Thêm residual và transform sanity gates.
3. Thêm fail-closed inspection behavior.
4. Viết synthetic alignment tests.
5. Chạy toàn bộ test.

### Phase 3 — Position X/Y MVP

1. Thêm `aoi_pipeline/golden/position.py`.
2. Fixed ROI + template search + translation refinement.
3. Quality gate, unit conversion và tolerance.
4. Test fractional shifts, sign, boundary, missing và bbox jitter independence.
5. Chạy toàn bộ test.

### Phase 4 — Rotation và Golden Compare

1. Thêm Euclidean pose refinement và angle periodicity.
2. Thêm `aoi_pipeline/golden/compare.py`.
3. Pose compensation, masks, metrics và anomaly blobs.
4. Test shifted-only, missing, shape/color change và illumination perturbation.
5. Chạy toàn bộ test.

### Phase 5 — Inspector và output

1. Thêm `AOIInspector.inspect(test_image, recipe)`.
2. Aggregate slot và board decisions.
3. JSON serialization trước.
4. Thêm exporter CSV/ZIP và debug overlays sau khi core ổn định.
5. Chạy toàn bộ test.

### Phase 6 — UI

1. Thêm Build Recipe và Inspect Board mode.
2. UI phải hiển thị riêng alignment, position và appearance.
3. Không để UI tự tính lại logic đã có trong core.
4. Bridge chỉ normalize/forward dữ liệu; core là nguồn sự thật.
5. Thêm bridge/UI tests cần thiết và chạy toàn bộ test.

## 5. Data contract tối thiểu

Recipe phải lưu:

- schema version, board ID, side, Golden hash và image size.
- canonical coordinate space.
- pixel-to-mm calibration.
- alignment anchors và quality gates.
- stable slot ID, expected bbox/center/angle, fixed ROI.
- template/mask paths hoặc embedded asset references.
- search margin, position tolerances và appearance thresholds.

Runtime result phải lưu:

- alignment transform, residual, quality status và reason.
- position `dx_px`, `dy_px`, `dx_mm`, `dy_mm`, angle, score và status.
- appearance metrics, anomaly mask/blob summary và status.
- final slot status và final board status.
- coordinate space, recipe version/hash và model identifiers.

Không export absolute workstation paths.

## 6. Test và chất lượng

Mỗi behavior mới phải có unit test trước hoặc cùng patch. Không chỉ kiểm tra “không crash”.

Test bắt buộc:

- Stable slot ordering/IDs.
- Invalid/out-of-bounds ROI handling.
- Recipe round-trip và malformed schema.
- Known global transform và alignment failure.
- Known local shifts `±0.25`, `±0.5`, `±1`, `±2` px.
- Known rotations và 180° periodic component.
- Sign convention và pixel-to-mm conversion.
- Same image PASS.
- Shifted-only: position NG, appearance PASS sau compensation.
- Missing/wrong appearance: appearance NG hoặc review đúng policy.
- Low pose confidence không được tạo valid numeric measurement.
- Detector bbox jitter không làm position result thay đổi đáng kể.
- Existing pipeline, tiling, classifier, exporter tests không regression.

Acceptance không được dựa chỉ trên synthetic tests. Sau core implementation, yêu cầu bộ ảnh thực gồm repeated OK và controlled shifts trước khi tuyên bố độ chính xác mm.

## 7. Quy tắc thay đổi code

- Ưu tiên NumPy/OpenCV đang có; không thêm dependency nặng nếu chưa chứng minh cần thiết.
- API mới phải có type hints và docstrings cho coordinate/sign/unit conventions.
- Không duplicate image-transform logic giữa core và UI.
- Không đổi public behavior hiện có nếu không có migration/test.
- Không sửa notebook/training ngoài phạm vi inspection nếu không cần.
- Không hard-code tolerance production trong thuật toán; đặt trong config/recipe.
- Không hard-code một threshold SSIM toàn board.
- Không lưu Golden bằng JPEG; dùng PNG/TIFF lossless.
- Không commit model weights, Golden production hoặc dữ liệu nhạy cảm nếu repo policy không cho phép.
- Không commit hoặc push GitHub trừ khi người dùng yêu cầu rõ.

## 8. Cách báo cáo sau mỗi milestone

Trả lời ngắn gọn theo thứ tự:

1. Kết quả đã hoàn thành.
2. File đã thêm/sửa.
3. Behavior và contract mới.
4. Test command và số test pass/fail.
5. Giới hạn hoặc quyết định còn cần người dùng chốt.

Không tuyên bố “sub-pixel chính xác” hoặc độ chính xác mm nếu chưa có repeatability/error measurements. Phân biệt rõ độ phân giải thuật toán với độ chính xác vật lý đã được xác minh.

