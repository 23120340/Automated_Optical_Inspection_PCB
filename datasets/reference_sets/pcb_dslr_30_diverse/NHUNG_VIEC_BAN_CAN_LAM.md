# Những việc bạn cần làm với bộ 30 ảnh PCB khác layout

> Bộ này là dữ liệu khởi tạo cho kế hoạch số hóa PCB Phase 1–5. Mọi Golden,
> PnP, RefDes, footprint, góc xoay và registration sinh tự động đều là
> **bản nháp cần duyệt**, không phải ground truth sản xuất.

Các file bạn sẽ dùng sau khi bootstrap hoàn tất:

- `manifest.json`: nguồn gốc, SHA-256 và mapping của 30 ảnh.
- `boards/<board_id>/rec1.jpg`: ảnh nguồn; không chỉnh sửa trực tiếp.
- `bootstrap/contact_sheet.jpg`: xem nhanh độ nét và độ đa dạng toàn bộ.
- `bootstrap/reference_index.csv`: focus score và số proposal theo board.
- `bootstrap/references/<board_id>/golden_candidate.json`: Golden candidate
  riêng của board.
- `bootstrap/references/<board_id>/pnp_pixels_NEEDS_REVIEW.csv`: PnP pixel nháp
  để bạn hiệu chỉnh.

## 1. Hiểu đúng cấu trúc bộ dữ liệu

- [ ] Xác nhận đủ 30 ảnh và 30 layout thực sự khác nhau.
- [ ] Gán một `board_id` riêng, ổn định cho từng layout.
- [ ] Xác nhận mặt board là `top`, `bottom` hay `unknown`; không suy đoán nếu
  ảnh không đủ bằng chứng.
- [ ] Xác nhận revision nếu biết. Hai revision khác nhau phải là hai board
  profile khác nhau.

Mỗi ảnh trong bộ này phải được xử lý độc lập:

```text
1 layout
→ 1 board_id
→ 1 Golden candidate cho một board side
→ 1 PnP draft
→ 1 registration draft
→ 1 recipe riêng
```

Không được căn 30 layout vào nhau, lấy consensus giữa chúng hoặc dùng một PnP
chung. Consensus nhiều ảnh chỉ có ý nghĩa khi các ảnh là những lần chụp lặp của
**cùng SKU, revision và mặt board**.

## 2. Duyệt ảnh trước khi gán nhãn

Với từng ảnh:

- [ ] Board hiện đầy đủ, không bị cắt mất mép quan trọng.
- [ ] Không có watermark, chữ chú thích hoặc overlay che linh kiện.
- [ ] Không out-of-focus, rung, nén JPEG nặng hoặc cháy sáng trên vùng kim loại.
- [ ] Góc chụp đủ gần vuông góc để nhìn thấy hình học board.
- [ ] RefDes, mép linh kiện và pad mục tiêu còn phân biệt được khi phóng to.
- [ ] Ghi rõ vùng bị che, phản xạ mạnh hoặc không thể đọc là `ignore/unknown`.

Nếu cạnh ngắn của pad mục tiêu dưới khoảng 8 px thì không dùng ảnh đó làm
ground truth cho pad/mối hàn. Nên ưu tiên ảnh mà pad mục tiêu có kích thước
khoảng 20 px trở lên. Một ảnh có thể vẫn hữu ích cho nhận dạng board hoặc linh
kiện lớn nhưng không đủ cho kiểm tra fillet mối hàn.

## 3. Giữ provenance và ảnh nguồn bất biến

- [ ] Không resize, crop, recompress hoặc ghi đè file trong `images/`.
- [ ] Khóa mỗi file bằng SHA-256 trong manifest.
- [ ] Ghi nguồn, URL ổn định, giấy phép, ngày lấy, kích thước và định dạng ảnh.
- [ ] Ghi `source_kind=dataset_preprocessed` hoặc `source_as_received` cho ảnh
  lấy từ dataset/web; không gọi chúng là sensor RAW.
- [ ] Không lưu URL có chữ ký tạm thời, token truy cập hoặc đường dẫn tuyệt đối
  của máy làm việc trong artifact.

Mỗi dòng/record manifest tối thiểu cần có:

```text
image_id, relative_path, sha256, width_px, height_px,
source_dataset, source_url, license, source_kind,
board_id, revision, side, review_status, notes
```

Giữ nguyên tên file hoặc lưu mapping nếu đổi tên; nhãn phải ghép với ảnh bằng
`image_id` và SHA-256, không chỉ bằng tên file.

## 4. Contract chung cho nhãn

Nên tổ chức nhãn như sau:

```text
labels/
├── label_manifest.csv
├── board_geometry/
├── semantic/
├── components/
├── pads_and_joints/
└── traces/
```

`label_manifest.csv` tối thiểu có:

```text
image_id,board_id,revision,side,sha256,annotation_schema,
coordinate_space,transform_id,label_source,annotator,reviewer,
review_status,notes
```

Quy ước bắt buộc:

- `source_image_pixels`: tọa độ trên đúng ảnh nguồn chưa warp.
- `golden_board_pixels`: tọa độ sau khi đã căn về Golden của đúng board đó.
- Mọi transform phải có ID, ma trận và chiều biến đổi; không trộn hai hệ tọa độ.
- `xyxy` dùng biên phải/dưới exclusive nếu xuất bbox cho pipeline.
- `label_source` là một trong `human_verified`, `bom_or_cad`,
  `visual_transcription`, `pseudo_label` hoặc `unknown`.
- `review_status` là một trong `draft`, `reviewed`, `verified` hoặc `rejected`.
- Proposal từ detector/OCR/PnP vẫn là `pseudo_label` cho tới khi con người duyệt
  đúng từng đối tượng.

Có thể gán nhãn bằng LabelMe hoặc CVAT rồi xuất COCO/YOLO làm dữ liệu dẫn xuất.
Không thay đổi kích thước ảnh khi chuyển đổi định dạng nhãn.

## 5. Phase 1 — Hình học và phân đoạn ngữ nghĩa

### 5.1. Board geometry

Cho từng board, đánh dấu:

- [ ] Polygon biên board.
- [ ] Tâm và đường kính lỗ gá/lỗ ốc nhìn thấy được.
- [ ] Tâm fiducial nhìn thấy được.
- [ ] Các vùng ổn định có thể dùng làm alignment anchor.
- [ ] Vùng `ignore/unknown` do che khuất, phản xạ hoặc mất nét.

### 5.2. Semantic masks

Tạo các mask nhị phân riêng:

| Nhãn | Phạm vi |
|---|---|
| `substrate_visible` | Nền PCB/solder mask đang nhìn thấy |
| `copper_visible` | Chỉ phần đồng hoặc trace thực sự nhìn thấy |
| `pad_exposed` | Pad/via/kim loại lộ thiên nhìn thấy được |
| `silkscreen` | Chữ, ký hiệu và đường in lụa nhìn thấy được |
| `ignore_unknown` | Vùng không đủ bằng chứng để gán nhãn |

Không vẽ đoán trace nằm dưới solder mask, dưới linh kiện hoặc ở lớp trong.
Muốn có ground truth cho phần bị che cần bare board, Gerber/ODB++, IPC-356 hoặc
CAD/netlist do chủ board cung cấp.

## 6. Phase 2 — Linh kiện, RefDes, footprint và góc xoay

Mỗi instance linh kiện cần các trường:

```text
object_id,refdes,family,bbox_or_polygon,center_px,angle_deg,
rotation_period_deg,footprint,mount_type,side,pin1_or_polarity,
visibility,label_source,review_status
```

Checklist:

- [ ] Thêm linh kiện bị detector bỏ sót; xóa proposal nằm trên nền/chữ lụa.
- [ ] Kiểm tra kỹ connector/header, transformer, crystal, diode, IC, tụ phân
  cực và các linh kiện sát nhau.
- [ ] Đối chiếu `RefDes` với silkscreen và BOM/schematic nếu có.
- [ ] Không đổi `*_AUTO_*` thành RefDes thật chỉ dựa trên phỏng đoán.
- [ ] Ghi footprint vật lý, ví dụ `R_0603`, `SOT-23`, `SOIC-8`; không dùng chức
  năng linh kiện thay cho footprint.
- [ ] Ghi `angle_deg` theo một quy ước zero-angle đã công bố.
- [ ] Ghi `rotation_period_deg=180`, `360` hoặc `null` nếu góc không đo được.
- [ ] Không suy ra góc thật từ một axis-aligned bbox.
- [ ] Với diode, LED, IC và tụ phân cực, đánh dấu cathode/anode, pin 1 hoặc cực
  âm/dương khi nhìn thấy rõ.

Nên giữ một bảng mapping để truy vết:

```csv
board_id,Synthetic Designator,Verified RefDes,Label Source,Review Status,Comment
```

## 7. Pad, pin, mối hàn và nhãn lỗi

Mỗi pad/joint phải liên kết về linh kiện bằng `refdes` và `pin`:

```text
joint_id,refdes,pin,polygon_or_bbox,center_px,shape,
side,net,defect_label,label_level,label_source,review_status
```

Phân biệt đúng cấp nhãn:

| Cấp | Nhãn đề xuất |
|---|---|
| Một mối hàn | `good`, `insufficient`, `excess`, `cold`, `missing_solder` |
| Cặp chân/khoảng inter-pin | `bridge` |
| Cả linh kiện | `shifted_component`, `missing_component`, `wrong_component`, `polarity_error`, `tombstone`, `lifted_lead` |

- [ ] Không gán `bridge` cho một crop joint đơn nếu không thấy cặp chân và vùng
  nối giữa chúng.
- [ ] Nếu không đủ chuyên môn hoặc ảnh không đủ rõ, dùng
  `unknown/needs_expert_review`; không ép chọn một defect class.
- [ ] Trước khi có nhãn chuyên gia hoặc rule/model đã xác thực, Golden Compare
  chỉ được gọi `appearance_anomaly`, không tự khẳng định tên lỗi cụ thể.

Ảnh Internet không có nhãn `normal` đáng tin ở cấp mối hàn chỉ vì trang nguồn
không ghi lỗi. Trạng thái board tốt/xấu phải do người có chuyên môn xác nhận.

## 8. Phase 3 — Trace và netlist

- [ ] Chỉ vector hóa đoạn trace thực sự nhìn thấy.
- [ ] Ghi polyline, endpoint/pad liên quan và `net_id` chỉ khi đã đối chiếu được.
- [ ] Đánh dấu `unobservable` cho kết nối đi dưới linh kiện, solder mask, via mù
  hoặc lớp trong.
- [ ] Không tự nối hai đoạn chỉ vì chúng có vẻ cùng màu hoặc cùng hướng.

Mục tiêu khôi phục netlist đầy đủ không thể nghiệm thu chỉ bằng ảnh assembled
PCB 2D. Với bộ ảnh này, Phase 3 chỉ là bootstrap cho **kết nối bề mặt quan sát
được**.

## 9. Phase 4 — Duyệt PnP, CAD và Golden recipe

### 9.1. PnP riêng cho từng board

Mỗi board dùng một file riêng với header:

```csv
Designator,Mid X,Mid Y,Rotation,Layer,Footprint,Comment
```

- [ ] Thay `*_AUTO_*` bằng RefDes thật sau khi đối chiếu.
- [ ] Bổ sung linh kiện thiếu và xóa proposal giả.
- [ ] Xác nhận class, tâm, layer, footprint và rotation từng dòng.
- [ ] Xác nhận origin, chiều `+X/+Y`, zero-angle và quy ước mirror mặt Bottom.
- [ ] Giữ rotation/footprint trống hoặc `unknown` nếu chưa xác minh.
- [ ] Không biến tọa độ pixel thành mm bằng một kích thước board đoán từ tên
  sản phẩm hoặc ảnh trên Internet.

Khi chưa có calibration thật, file consensus/pixel là nguồn quan sát chính;
mọi PnP mm phải giữ hậu tố/trạng thái `NEEDS_REVIEW`.

### 9.2. Golden riêng cho từng board

- [ ] Chọn đúng một ảnh thật làm Golden candidate cho mỗi `board_id + side`.
- [ ] Lưu Golden dưới dạng PNG/TIFF lossless; không tạo Golden bằng ảnh median
  hoặc composite của các layout khác nhau.
- [ ] Xác nhận ảnh Golden không mờ, không cháy sáng, không cắt board và đúng
  revision/side.
- [ ] Duyệt fixed ROI, template, component mask, compare mask và ignore mask
  của từng slot.
- [ ] Giữ slot ID deterministic; không dùng UUID detection làm định danh vị trí.

Ảnh công khai không có bằng chứng board đã đạt kiểm tra điện/chất lượng nên chỉ
là **Golden candidate/reference**, chưa phải Golden production đã chứng nhận.

### 9.3. Cảnh báo bắt buộc về bản nháp

- Kích thước danh nghĩa **45 × 20 mm của HC-SR04 không liên quan tới bộ 30
  layout mới** và tuyệt đối không được dùng làm ground truth scale/PnP.
- `demo_grid` chỉ là anchor minh họa. Không được dùng nó để bật inspection
  production hoặc làm bằng chứng alignment đã đạt.
- Tolerance mẫu, threshold SSIM/diff và search margin mặc định không phải thông
  số production cho board mới.
- Recipe phải giữ `production_eligible=false` và metrology `verified=false`
  cho tới khi hoàn thành các bước hiệu chuẩn và nghiệm thu bên dưới.

## 10. Hiệu chuẩn và registration cần bổ sung

Để chuyển một board từ reference Internet thành recipe đo mm, cần board vật lý
và hệ camera thật. Với từng `board_id + revision + side + fixture`:

- [ ] Đo kích thước board thật và ít nhất một khoảng cách chuẩn để kiểm chéo.
- [ ] Chốt origin, chiều trục X/Y, đơn vị và mirror mặt Bottom.
- [ ] Cung cấp tọa độ mm của fiducial/lỗ gá hoặc CAD/PnP thật.
- [ ] Dùng ít nhất ba điểm không thẳng hàng cho affine; chỉ dùng bốn điểm trở
  lên/homography khi có bằng chứng cần sửa phối cảnh.
- [ ] Hiệu chuẩn camera intrinsic/distortion bằng checkerboard hoặc dot-grid
  đặt cùng mặt phẳng board.
- [ ] Đo `pixels_per_mm_x/y` bằng đúng camera, lens, working distance, ánh sáng
  và fixture production.
- [ ] Lưu registration theo đúng board profile; không tái sử dụng giữa hai
  layout.

Alignment production phải dùng fiducial, lỗ board hoặc stable patch được duyệt,
phân bố trên board. Thiếu anchor, residual cao hoặc transform bất hợp lý phải
trả `INVALID`; không dùng resize fallback để đưa ra PASS/NG.

## 11. Phase 5 — Dữ liệu kiểm chứng thực tế cần chụp thêm

Bộ 30 layout khác nhau không thay thế tập kiểm chứng theo một SKU. Với mỗi SKU
cần đưa vào vận hành, chụp thêm:

- ít nhất 30 ảnh repeated-OK của cùng board/revision/side;
- controlled shifts tại `0.5T`, `1.0T`, `1.5T`, với `T` là tolerance đã chốt;
- các góc rotation đã đo được;
- missing component và wrong appearance;
- thay đổi ánh sáng nhỏ trong phạm vi vận hành;
- ca alignment failure;
- mẫu lỗi hàn được chuyên gia xác nhận nếu muốn kiểm defect cụ thể.

Chia train/validation/test theo `board_id/SKU`, không chia ngẫu nhiên các crop
của cùng board sang nhiều split. Khóa test set trước khi tune threshold.

## 12. Cổng nghiệm thu Phase 1–5

Các con số dưới đây là **mục tiêu của kế hoạch**, không phải kết quả đã đạt của
bộ ảnh này:

| Phase | Cổng nghiệm thu |
|---|---|
| Phase 1 — Segmentation | Test tách theo board; mIoU mục tiêu ≥ 0,88 và báo IoU từng lớp; không tính vùng `unknown` như ground truth. |
| Phase 2 — Component/OCR | RefDes accuracy mục tiêu ≥ 92%; sai số góc ≤ 1,5° trên các mẫu có góc xác định; mọi PnP đã human-review. |
| Phase 3 — Trace/Netlist | Mục tiêu ≥ 95% chỉ trên kết nối bề mặt có ground truth; không tuyên bố phục hồi lớp trong từ ảnh 2D. |
| Phase 4 — CAD/Recipe | `pads_csv`/`placement_csv` nạp được bằng `CAD_LOADERS`; tọa độ hữu hạn, ID duy nhất, không absolute path; asset/hash recipe validate. |
| Phase 5 — Thực tế | Sai số hình học mục tiêu ≤ 0,05 mm chỉ được công bố sau calibration và đo trên fixture thật. |

Cổng Golden Inspection bổ sung:

- alignment error p95 không lớn hơn `1/4` độ dịch nhỏ nhất cần phát hiện;
- position repeatability p95 không lớn hơn `1/3` tolerance;
- mẫu chỉ shifted phải có thể `NG_POSITION` nhưng `PASS_APPEARANCE` sau pose
  compensation;
- thiếu anchor/residual cao/transform bất hợp lý phải trả `INVALID`;
- không công bố sub-pixel accuracy hoặc accuracy mm khi chưa có repeated-OK và
  controlled-shift measurements.

Ba mươi ảnh đa dạng là tập pilot tốt để chốt schema và phát hiện lỗi đường ống,
nhưng chưa đủ để tự nó chứng minh các gate train/production ở trên.

## 13. Thứ tự làm khuyến nghị

1. Duyệt 30 ảnh, thay ảnh mờ/sai phạm vi và gán `board_id`, revision, side.
2. Chọn một ảnh pilot; gán đầy đủ mọi tầng rồi duyệt schema trước khi làm 29
   ảnh còn lại.
3. Gán component, RefDes, footprint, polarity và rotation.
4. Gán board geometry, semantic masks, pad/pin/joint.
5. Chỉ gán trace/net nơi quan sát được hoặc có CAD xác nhận.
6. Duyệt Golden candidate và PnP pixel riêng cho từng board.
7. Khi có board thật, đo camera/board và tạo registration/PnP mm verified.
8. Chụp repeated-OK cùng controlled shifts/defects theo từng SKU.
9. Chạy acceptance gates; chỉ sau đó mới khóa recipe production.

## 14. Definition of Done cho phần việc gán nhãn

Một ảnh chỉ được đánh dấu hoàn tất khi:

- [ ] SHA-256 của ảnh khớp manifest.
- [ ] Board ID/revision/side đã duyệt hoặc ghi rõ `unknown`.
- [ ] Không còn proposal chưa duyệt bị ghi là ground truth.
- [ ] Mọi vùng không đủ bằng chứng đã chuyển sang `ignore/unknown`.
- [ ] Nhãn component/pad/trace dùng đúng coordinate space.
- [ ] RefDes, footprint, rotation và polarity có provenance.
- [ ] Có tên người gán nhãn, người review và trạng thái review.
- [ ] PnP/Golden/recipe vẫn giữ `NEEDS_REVIEW` nếu thiếu đo đạc vật lý.

Khi bạn hoàn thành checklist này, bộ dữ liệu mới đủ sạch để bắt đầu thử nghiệm
Phase 1–5 có kiểm soát; nó vẫn chỉ trở thành recipe production sau validation
trên board và hệ camera thực.
