# Những việc bạn cần làm với bộ 30 ảnh cùng layout

> Đây là bộ bootstrap gồm 30 lần chụp **cùng một PCB máy bơm xăng, cùng layout và
> cùng mặt linh kiện** từ MPI-PCB. Bộ này phù hợp để chọn một Golden thật và tạo
> consensus/PnP theo pixel. Nó chưa phải dữ liệu production đã xác nhận và không
> phải ảnh mặt hàn.

## Trước khi bắt đầu

- Ảnh nguồn nằm trong `images/`; không resize, crop, chỉnh màu, nén lại hoặc ghi đè.
- `manifest.json` giữ đường dẫn nguồn, CRC32, SHA-256, kích thước và mapping của
  từng ảnh.
- `ATTRIBUTION.md` giữ nguồn và giấy phép CC BY 4.0; phải đi cùng bộ dữ liệu khi
  chia sẻ hoặc phát hành artifact dẫn xuất.
- Nhãn upstream `train/good` chỉ có nghĩa là ảnh thuộc nhóm board không sửa đổi
  của dataset. Nó **không thay thế** việc bạn xác nhận board thật sự đạt chuẩn.
- Bộ 30 ảnh khác layout trong `../pcb_dslr_30_diverse/` chỉ dùng để thử schema và
  huấn luyện đa dạng. Không đưa ảnh từ bộ đó vào consensus/PnP này.

## 1. Duyệt 30 frame

Tạo `frame_review.csv` với các cột tối thiểu:

```csv
image_id,sha256,review_status,layout_ok,side_ok,focus_ok,board_complete,visible_anomaly,notes,reviewer,reviewed_at
```

Với từng ảnh, đánh dấu một trong ba trạng thái:

- `ok_verified`: đúng board/layout/mặt, đủ nét, không bị cắt, không thấy bất thường.
- `unknown`: chưa đủ bằng chứng để kết luận tốt/xấu.
- `exclude`: mờ, rung, cháy sáng nặng, bị cắt, sai layout/mặt hoặc có bất thường
  nhìn thấy được.

Checklist:

- [ ] Xác nhận đủ 30 ảnh đều là cùng PCB/layout/revision và cùng mặt linh kiện.
- [ ] Phóng to các IC, connector, điện trở và đường silkscreen để kiểm tra focus.
- [ ] Kiểm tra đủ bốn phía board; loại ảnh bị cắt mất vùng quan trọng.
- [ ] Đánh dấu vùng phản xạ mạnh, che khuất hoặc không chắc chắn là `ignore`.
- [ ] Không coi `good` của nguồn là `ok_verified` nếu chưa có người duyệt.

Nếu loại một frame, thay bằng một frame `train/good` khác của **chính board này**,
sau đó cập nhật manifest/hash. Không lấy ảnh của layout khác để bù đủ số lượng.
`frame_review.csv` là hồ sơ duyệt của người, builder hiện không tự đọc file này.
Frame `exclude`/`unknown` phải được loại hoặc thay trong `images/` và `manifest.json`,
kiểm lại hash rồi build lại; chỉ ghi trạng thái trong CSV là chưa đủ.

## 2. Duyệt Golden

Pipeline chọn đúng **một ảnh nguồn thật** và lưu lại lossless thành `golden.png`;
không dùng ảnh trung bình, median hay ảnh ghép.

- [ ] Mở `reference_selection.json` và kiểm tra frame được chọn.
- [ ] Golden phải có `review_status=ok_verified` trong `frame_review.csv`.
- [ ] Golden phải rõ nét toàn board, không cháy sáng nặng và không bị cắt.
- [ ] Đối chiếu SHA-256 để biết chính xác Golden xuất phát từ ảnh JPEG nào.
- [ ] Giữ nguyên JPEG nguồn; `golden.png` là bản lossless dùng làm canvas chuẩn.
- [ ] Ghi tên người duyệt, ngày duyệt và lý do chọn trong `golden_approval.md`.

Golden công khai này vẫn chỉ là **Golden candidate/reference** cho đến khi được
đối chiếu với board vật lý đạt chuẩn của bạn.

## 3. Duyệt alignment và consensus

Ảnh đã được MPI-PCB căn chỉnh upstream, nhưng vẫn phải kiểm lại trước khi dùng
cho AOI của bạn.

- [ ] Mở `alignment_report.json`; xác nhận đúng mode
  `upstream_dataset_identity_enrollment_only`, nhận đủ 30/30 frame và không có
  `resize_fallback`. Các fit metric của mode này là `not_measured`, không được
  diễn giải identity/overlap 1.0 thành độ chính xác registration đã đo.
- [ ] Không dùng overlay consensus/PnP để kết luận alignment từng frame. Bundle
  chưa sinh per-frame alignment overlay; hãy kiểm tra blink/difference riêng ở
  mép board, lỗ gá, fiducial và các connector xa nhau.
- [ ] Kiểm tra board quad/localization không bắt nhầm nền hoặc bóng đổ.
- [ ] Loại/thay frame có lệch hoặc double-edge khi so với Golden; không hạ gate
  chỉ để đủ 30.
- [ ] Anchor production sau này phải là fiducial, lỗ gá hoặc patch ổn định đã
  duyệt; không dùng một linh kiện có thể thiếu/sai làm anchor chính.
- [ ] Upstream alignment chỉ phục vụ enrollment/diagnostic, không chứng minh
  registration trên camera và fixture production.

Khi xem `consensus_components.json/csv` (audit detector đầy đủ),
`pnp_pixels_NEEDS_REVIEW.csv` (hàng đợi PnP thực hành) và
`overlay_pnp_NEEDS_REVIEW.png`:

- [ ] Kiểm tra `observation_count/support_ratio`: proposal đáng tin phải lặp lại trên
  phần lớn frame đã align.
- [ ] Kiểm tra `class_purity`: class không ổn định phải chuyển sang review.
- [ ] Kiểm tra `center_mad_px`; cluster có độ phân tán tâm lớn cần sửa hoặc loại.
- [ ] Thêm linh kiện detector bỏ sót và xóa box nằm trên nền/silkscreen.
- [ ] Không hạ threshold chỉ để tăng số linh kiện trên bảng.

## 4. Hiệu chỉnh PnP pixel

`pnp_pixels_NEEDS_REVIEW.csv` là hàng đợi đề xuất **không có thẩm quyền** trong hệ
tọa độ `golden_board_pixels`. Nó chỉ chứa các cluster có đúng một observation trên
Golden đã chọn; tâm và box vẫn là median của nhiều frame. `consensus_components.json/csv`
tiếp tục giữ toàn bộ detector audit, kể cả các cluster không được đưa vào hàng đợi
PnP. Trước khi có hiệu chuẩn vật lý, chỉ dùng tọa độ pixel có thể truy vết; không
tạo hoặc điền tọa độ mm suy đoán.

Snapshot của `bootstrap_pixel_v1` hiện tại: Golden là frame `0893`; detector audit
có 253 cluster nhưng chỉ 6 cluster vượt gate support/purity. Hàng đợi PnP có 32
proposal neo trên Golden, trong đó 26 dòng mang lý do `low_support`; phân bố gồm
25 IC, 4 LED, 2 capacitor và 1 connector. Nó bỏ sót rõ nhiều điện trở, diode,
capacitor, connector, buzzer/crystal và không phải PnP hoàn chỉnh hay ground truth.
Hãy dùng nó làm seed để thêm/xóa/sửa bằng mắt và BOM/CAD nếu có.

Không sửa trực tiếp artifact generated. Sao chép/hợp nhất kết quả duyệt vào file
canonical mới `pnp_pixels_REVIEWED.csv`, đủ chỗ ghi cả proposal bị reject và linh
kiện detector bỏ sót:

```csv
record_id,source_auto_id,action,verified_refdes,class,footprint,center_x_px,center_y_px,rotation_deg,polarity,review_status,label_source,notes
```

- `action` dùng một trong `accept`, `correct`, `add`, `reject`; dòng `add` để trống
  `source_auto_id`, còn dòng xuất phát từ detector phải giữ ID `*_AUTO_*` để truy vết.
- [ ] Kiểm tra box, tâm và class của từng dòng bằng overlay.
- [ ] Đối chiếu với `consensus_components.json/csv`; thêm thủ công mọi linh kiện
  Golden có thật nhưng detector bỏ sót, và reject mọi false positive.
- [ ] Giữ ID `*_AUTO_*` cho đến khi đối chiếu được RefDes thật.
- [ ] Gán RefDes từ silkscreen và ưu tiên xác minh bằng BOM/schematic/PnP/CAD.
- [ ] Gán footprint vật lý, không dùng chức năng linh kiện thay cho footprint.
- [ ] Gán rotation theo một quy ước zero-angle được ghi rõ.
- [ ] Không suy ra rotation thật từ axis-aligned bounding box.
- [ ] Với IC, diode, LED, tụ phân cực và connector có hướng, ghi pin 1/cathode/
  cực tính khi nhìn thấy rõ.
- [ ] Proposal chưa duyệt phải giữ `label_source=pseudo_label` và
  `review_status=draft`.

## 5. Dữ liệu cần đo để chuyển pixel sang mm

Không điền kích thước board giả và không lấy tỷ lệ mm từ ảnh Internet. Khi có
board vật lý/CAD thật, tạo `board_measurements.json` và `fiducials.csv`:

- [ ] Đo chiều rộng/chiều cao board và ít nhất một khoảng cách chuẩn để kiểm chéo.
- [ ] Chốt origin, chiều `+X/+Y`, đơn vị và quy ước rotation.
- [ ] Cung cấp tọa độ mm của ít nhất 3 fiducial/lỗ gá không thẳng hàng cho affine.
- [ ] Chỉ dùng homography khi có từ 4 điểm tốt trở lên và có bằng chứng cần sửa
  phối cảnh.
- [ ] Cung cấp CAD/PnP/BOM chính thức nếu có.
- [ ] Hiệu chuẩn intrinsic/distortion bằng đúng camera, lens, working distance và
  mặt phẳng board production.
- [ ] Đo scale và residual trên fixture thật; lưu version calibration và ngày đo.
- [ ] Nếu làm mặt Bottom, định nghĩa rõ phép mirror và dùng profile riêng.

Chỉ sau các bước trên mới xuất placement/PnP mm và recipe. Thiếu calibration thì
giữ `production_eligible=false` và không công bố accuracy mm/sub-pixel.

## 6. Contract gắn nhãn chung

Mọi annotation phải gắn với `image_id` và SHA-256, không chỉ dựa vào tên file.
Nên dùng cấu trúc:

```text
labels/
├── label_manifest.csv
├── board_geometry/
├── semantic/
├── components/
├── pads_and_joints/
└── traces/
```

`label_manifest.csv` tối thiểu gồm:

```csv
image_id,sha256,board_id,revision,side,annotation_schema,coordinate_space,transform_id,label_source,annotator,reviewer,review_status,notes
```

Quy ước:

- `source_image_pixels`: tọa độ trên JPEG nguồn chưa warp.
- `golden_board_pixels`: tọa độ sau khi căn về `golden.png`.
- Mỗi transform phải có ID, ma trận và chiều biến đổi rõ ràng.
- Không trộn annotation của hai coordinate space trong cùng record.
- Dùng `human_verified`, `bom_or_cad`, `visual_transcription`, `pseudo_label`
  hoặc `unknown` cho `label_source`.
- Dùng `draft`, `reviewed`, `verified` hoặc `rejected` cho `review_status`.

## 7. Phase 1 — hình học board và semantic masks

- [ ] Polygon biên board trên Golden và/hoặc từng source frame.
- [ ] Tâm/đường kính lỗ gá, lỗ ốc và fiducial nhìn thấy được.
- [ ] Vùng alignment anchor ổn định và vùng `ignore/unknown`.
- [ ] Mask `substrate_visible`, `copper_visible`, `pad_exposed`, `silkscreen`.
- [ ] Chỉ gắn nhãn phần thực sự nhìn thấy; không đoán trace dưới solder mask,
  dưới linh kiện hoặc trong lớp inner.

## 8. Phase 2 — linh kiện, RefDes, footprint và hướng

Mỗi instance nên có:

```text
object_id, refdes, family, bbox_or_polygon, center_px, angle_deg,
rotation_period_deg, footprint, mount_type, side, pin1_or_polarity,
visibility, label_source, review_status
```

- [ ] Duyệt toàn bộ proposal tự động, thêm miss và xóa false positive.
- [ ] Đối chiếu RefDes bằng silkscreen/BOM/CAD.
- [ ] Ghi footprint, mount type, góc, symmetry period và polarity.
- [ ] Chuyển vùng không đủ bằng chứng sang `unknown`, không ép gán nhãn.

## 9. Phase 3 — pad, joint, trace và net

- [ ] Liên kết pad/pin nhìn thấy được với `refdes` và số pin.
- [ ] Vector hóa chỉ những đoạn trace thực sự quan sát được.
- [ ] Chỉ ghi `net_id` khi có schematic/netlist/CAD xác nhận.
- [ ] Đánh dấu kết nối bị che hoặc lớp trong là `unobservable`.

Lưu ý quan trọng: đây là **mặt linh kiện**, không phải bộ ảnh mặt hàn chuyên dụng.
Không dùng nó làm ground truth cho solder-defect v2 hoặc các lớp bridge/cold/
insufficient solder nếu mối hàn cần đánh giá không nhìn thấy rõ. Phần đó cần bộ
ảnh solder-side riêng và người có chuyên môn xác nhận.

## 10. Phase 4 — khóa CAD/PnP/Golden draft

- [ ] Duyệt xong `pnp_pixels_REVIEWED.csv` và overlay PnP pixel.
- [ ] Import thử BOM/PnP/CAD bằng loader của dự án; kiểm ID duy nhất và số hữu hạn.
- [ ] Khóa Golden bằng SHA-256 và version bundle.
- [ ] Giữ metrology, placement mm và recipe production ở `NEEDS_REVIEW` khi thiếu
  đo đạc vật lý. Nhãn người duyệt và Golden approval vẫn có thể mang trạng thái
  `verified` với reviewer/provenance rõ ràng.
- [ ] Không bật recipe production từ `demo_grid`, resize fallback hoặc kích thước
  board suy đoán.

## 11. Phase 5 — chụp validation trên hệ thống thật

Cho đúng SKU/revision/side/camera/fixture sẽ chạy production, cần chụp thêm:

- [ ] Ít nhất 30 repeated-OK đã được chuyên gia xác nhận.
- [ ] Controlled shifts tại `0.5T`, `1.0T`, `1.5T` với tolerance `T` đã chốt.
- [ ] Các góc rotation đã đo được.
- [ ] Missing component, wrong appearance và polarity/wrong-component nếu áp dụng.
- [ ] Biến thiên ánh sáng nhỏ trong operating envelope.
- [ ] Các ca thiếu anchor, residual cao và alignment failure.
- [ ] Solder-side defects có nhãn chuyên gia trong một tập riêng nếu muốn kiểm
  lỗi hàn cụ thể.

Tách train/validation/test theo board/SKU hoặc capture session; không chia ngẫu
nhiên các crop gần giống nhau của cùng một frame sang nhiều split. Khóa test set
trước khi tune threshold.

## 12. Bộ file bạn cần bàn giao lại

- [ ] `frame_review.csv`
- [ ] `golden_approval.md`
- [ ] `pnp_pixels_REVIEWED.csv`
- [ ] `board_measurements.json`
- [ ] `fiducials.csv` và/hoặc CAD/PnP/BOM chính thức
- [ ] `labels/label_manifest.csv` cùng annotation Phase 1–3
- [ ] Overlay PnP/consensus đã duyệt
- [ ] Báo cáo validation Phase 5 và quyết định `production_eligible`

## Definition of Done cho vòng gắn nhãn đầu tiên

- [ ] 30/30 ảnh có trạng thái review và SHA-256 khớp manifest.
- [ ] Golden là một frame thật đã duyệt và truy vết được về ảnh nguồn.
- [ ] Mọi proposal component/PnP đều đã accept, sửa hoặc reject.
- [ ] RefDes/footprint/rotation/polarity có provenance; phần chưa chắc là `unknown`.
- [ ] Annotation dùng đúng coordinate space và transform version.
- [ ] Không có tọa độ mm, độ chính xác mm hoặc recipe production được suy đoán.
- [ ] Các giới hạn của mặt linh kiện và dữ liệu public được ghi rõ trong bàn giao.

Sau khi hoàn tất checklist này, bộ dữ liệu đủ sạch để bắt đầu thử nghiệm Phase
1–5 có kiểm soát. Việc bật production vẫn cần board chuẩn, camera/fixture thật và
validation riêng của hệ thống bạn.
