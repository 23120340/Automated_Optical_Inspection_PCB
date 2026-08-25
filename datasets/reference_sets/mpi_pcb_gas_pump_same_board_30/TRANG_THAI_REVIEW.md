# Trạng thái review — mpi_pcb_gas_pump_same_board_30

Cập nhật: 2026-08-25 · Tham chiếu plan: [`NHUNG_VIEC_BAN_CAN_LAM.md`](NHUNG_VIEC_BAN_CAN_LAM.md)

Tài liệu này ghi **máy đã kiểm chứng được gì** (có bằng chứng tái tạo được),
**người duyệt đã quyết định gì**, và **việc gì còn lại**.

Ranh giới trách nhiệm: mọi verdict chất lượng (`review_status`, `layout_ok`,
`focus_ok`, …) và mọi quyết định chọn Golden/quy ước đều do người duyệt `qnn`
đưa ra. Máy chỉ đóng góp phần đo được — hash, độ nét, white balance, clipping,
thống kê consensus — và phần đó được ghi riêng ở các cột `measured_*` cùng
trong `notes`, để phân biệt rõ đâu là đo đạc, đâu là phán quyết.

---

## 1. Đã kiểm chứng bằng máy — 13/13 check PASS

Chạy lại bất cứ lúc nào:

```bash
python scripts/verify_reference_set_bootstrap.py \
  --set-dir datasets/reference_sets/mpi_pcb_gas_pump_same_board_30 \
  --bundle-dir golden_recipes/mpi_pcb_gas_pump/top/bootstrap_pixel_v1 verify
```

Kết quả lưu ở [`verification_report.json`](verification_report.json).

| Nhóm | Kết quả |
|---|---|
| **Toàn vẹn dữ liệu** | 30/30 ảnh khớp SHA-256 **và** byte size với `manifest.json`; `images/` không có file lạ |
| **Truy vết Golden** | `golden.png` **bit-exact** với `mpi_pcb_train_good_0893.jpg` tại 4096×2816; frame này nằm trong 30 frame của manifest |
| **Alignment gate** | mode `upstream_dataset_identity_enrollment_only`; 30/30 frame accepted; **không** resize fallback; canvas gốc giữ nguyên; 30/30 frame khai báo `fit_metrics_status = not_measured_upstream_identity` |
| **Consensus / PnP** | 253 audit cluster, **6** vượt gate (support ≥ 0.8, purity ≥ 0.8); 32 proposal PnP, tất cả truy vết được về audit cluster và đều có observation trên Golden |

Toàn bộ snapshot mà plan §4 mô tả đã được đối chiếu và **khớp chính xác**:
253 cluster · 6 qua gate · 32 proposal · 26 dòng `low_support` ·
phân bố 25 IC / 4 LED / 2 capacitor / 1 connector.

### Detector bỏ sót bao nhiêu (audit 253 − queue 32)

| Class | Audit | Trong queue PnP | Bỏ sót |
|---|---:|---:|---:|
| ic | 111 | 25 | 86 |
| capacitor | 76 | 2 | **74** |
| resistor | 21 | 0 | **21** |
| led | 16 | 4 | 12 |
| connector | 14 | 1 | 13 |
| diode | 13 | 0 | **13** |
| buzzer | 2 | 0 | 2 |

→ Queue PnP **không phải** PnP hoàn chỉnh. Nó là seed để bạn thêm/xóa/sửa.

---

## 2. File đã tạo cho bạn

| File | Nội dung | Ai điền |
|---|---|---|
| [`frame_review.csv`](frame_review.csv) | 30 dòng, đủ cột plan §1 **+ 4 cột bằng chứng đo được** | ✅ **Đã điền** — 30/30 `ok_verified`, reviewer `qnn` |
| [`golden_approval.md`](golden_approval.md) | Dữ kiện Golden đã kiểm chứng + 2 quyết định đã chốt | ✅ **Đã ký** — `qnn`, 2026-08-24 |
| [`pnp_pixels_REVIEWED.csv`](pnp_pixels_REVIEWED.csv) | 32 dòng, đúng schema plan §4 | Bạn |
| [`labels/label_manifest.csv`](labels/label_manifest.csv) | 30 dòng theo contract plan §6 + thư mục con | Bạn |
| [`board_measurements.json`](board_measurements.json) | Kích thước vật lý vẫn `null` (`NOT_MEASURED`); đã chốt `pixel_frame_convention` và `canvas_to_board_design_transform` | Bạn — phần mm khi có board/CAD |
| [`fiducials.csv`](fiducials.csv) | Chỉ có header | Bạn — khi có board/CAD |
| [`review/review_app.html`](review/review_app.html) | **App duyệt offline**: toàn board + chỉnh box + điền nhãn + xuất CSV/overlay | ← **dùng cái này để làm bước 4** |

### Công cụ làm bước 4 — mở file này

```
datasets/reference_sets/mpi_pcb_gas_pump_same_board_30/review/review_app.html
```

Nhấp đúp mở bằng trình duyệt. Chạy offline, không cần server.

**Chế độ Toàn board** — trả lời câu hỏi "đã phủ hết chưa":
- Vẽ toàn bộ 32 box lên ảnh board, tô màu theo trạng thái duyệt
- Bấm một box để mở nó ra chỉnh
- Bấm vùng board trống để tạo dòng `add` tại chỗ đó
- Nút *Hiện chỗ bỏ sót*: hiện các cluster detector tìm ra nhưng không đưa vào
  queue. Bấm một cái là tạo ngay dòng `add` với box có sẵn

**Chế độ Chỉnh chi tiết** — phóng to quanh một linh kiện:
- Kéo giữa khung để dời box, kéo 4 góc để chỉnh kích thước
- Lăn chuột zoom, kéo nền để di chuyển
- Điền RefDes, class, footprint, rotation, polarity, notes
- Chip `accept`/`correct`/`add`/`reject` tự đổi `label_source` sang
  `human_verified` và `review_status` sang `reviewed`/`rejected`

**Chung**: tự lưu vào `localStorage` sau mỗi thao tác · nút *Xuất CSV* ra đúng
schema plan §4 · nút *Nhập CSV…* nạp lại phiên trước · nút *Xuất overlay PNG*
sinh `overlay_pnp_REVIEWED.png` (deliverable plan §12) · nút *? Ý nghĩa các
trường* giải thích từng ô · phím tắt `←` `→` đổi dòng, `O` đổi chế độ,
`R` đặt lại box, `?` mở trợ giúp.

Sau khi xuất, chép file tải về đè lên `pnp_pixels_REVIEWED.csv` trong thư mục này.

#### Về nhóm "chỗ bỏ sót" — đọc trước khi dùng

221 cluster audit không nằm trong queue PnP. Chúng **rất nhiễu**:

| Đặc điểm | Số lượng |
|---|---:|
| Chỉ xuất hiện ở **1 frame** | 91 |
| Chồng lên một cluster khác (IoU > 0.5) | 85 |
| Trùng chỗ đã có trong queue (IoU > 0.5) | 69 |
| Xuất hiện ở ≥ 15 frame | **4** |

App **tự ẩn hẳn** 69 cái trùng queue, và lọc phần còn lại theo ô `≥N frame`:

| Ngưỡng | Còn lại | Phân bố |
|---|---:|---|
| ≥1 frame | 152 | nhiều rác |
| **≥2 frame (mặc định)** | **78** | 34 capacitor · 14 ic · 11 resistor · 7 diode · 5 connector · 5 led · 2 buzzer |
| ≥3 frame | 43 | |
| ≥4 frame | 21 | |

⚠️ Đây là **gợi ý vị trí, không phải danh sách phải làm**. Vẫn phải nhìn ảnh rồi
mới quyết `add` hay bỏ qua.

**Lưu ý về toạ độ tâm**: app lấy tâm = tâm box đang hiển thị. Detector báo
`center_x_px` là *median tâm qua các frame*, còn bbox là *median bbox* tính độc
lập, nên hai giá trị lệch nhau chút ít (trung vị 1.3 px, tối đa 4.1 px trên
`J_AUTO_0006`). Lấy tâm theo box giữ cho mọi dòng nhất quán với thứ bạn nhìn thấy.

Sinh lại app:

```bash
python scripts/build_pnp_review_app.py \
  --bundle-dir golden_recipes/mpi_pcb_gas_pump/top/bootstrap_pixel_v1 \
  --out-dir datasets/reference_sets/mpi_pcb_gas_pump_same_board_30/review
```

---

## 3. Trạng thái các quyết định

### 3.1 ✅ ĐÃ CHỐT — giữ Golden `0893`

Người duyệt `qnn` xác nhận 0893 dùng được làm Golden. Kiểm chứng độc lập ở
full-resolution: mã linh kiện (`MC74HC574AN`, `FJU9732`) và silkscreen (`P205`,
`Z2`, `R37`) đều đọc rõ. Chi tiết: [`golden_approval.md` §4.1](golden_approval.md).

### 3.2 ✅ ĐÃ CHỐT — `rotation_deg` dùng canvas frame `golden_board_pixels`

Canvas xoay 180° so với chiều đọc silkscreen (board là **Gilbarco Legacy**, Z80 +
Dallas DS1225AD). Vì `center_x_px`/`center_y_px` đã ở `golden_board_pixels` và
plan §6 cấm trộn hai coordinate space trong cùng record, `rotation_deg` bắt buộc
cùng frame đó.

Quy ước đầy đủ nằm ở `board_measurements.json` → `pixel_frame_convention`.
Phép xoay 180° sang frame thiết kế ghi ở `canvas_to_board_design_transform`,
trạng thái `KNOWN_BUT_NOT_APPLIED`.

> ⚠️ Khi duyệt crop: chữ đọc ngược là **đúng như mong đợi**. Góc đo trực tiếp
> trên crop chính là góc cần ghi — **không** cộng trừ 180°.

### 3.3 ✅ ĐÃ XONG — 30/30 frame `ok_verified`

`frame_review.csv` đã điền đầy đủ, reviewer `qnn`, ngày 2026-08-24.

Verdict do người duyệt đưa ra (đã lọc sẵn 30 ảnh từ dataset); `notes` từng dòng
được sinh từ đo đạc trong phiên này nên mỗi frame có ghi chú riêng thay vì một
chuỗi lặp. Ba frame được gọi tên vì lệch nhiều nhất so với phần còn lại:

| Frame | Quan sát đo được |
|---|---|
| `1287` | Phơi sáng thấp nhất bộ (mean luma 96.7); mã linh kiện trên thân IC sẫm khó đọc nhất trong 30 ảnh |
| `1553`, `1584` | White balance ngả vàng mạnh (R−B = +52, +49); 1584 còn có specular gần bão hòa (p99 = 254) |
| `0189` | Ngả vàng (R−B = +46) và thuộc nhóm nét kém (hạng 28/30) |

Cả ba vẫn đạt: đúng board/layout/mặt, đủ bốn phía, mã linh kiện còn đọc được.

**Lưu ý từ vựng**: bạn ghi `review_status = approval`; plan §1 chỉ nhận
`ok_verified` / `unknown` / `exclude` nên tôi ghi `ok_verified`. Trường
`visible_anomaly` ghi `none` theo xác nhận của bạn (`true` sẽ mang nghĩa
"CÓ bất thường" và mâu thuẫn với `ok_verified`).

### 3.4 ⏳ CÒN LẠI — 32 dòng PnP chưa duyệt

Đây là việc lớn còn lại. Đã thấy ít nhất 2 dòng có vấn đề khi kiểm tra crop:

- `PNP_0014` (`C_AUTO_0024`, class `capacitor`, **1 observation**): box nằm trên
  một **via/pad tròn**, không phải thân tụ → ứng viên `action = reject`.
- `PNP_0002` (`J_AUTO_0006`, class `connector`): crop cho thấy **linh kiện DIP**
  ngay cạnh silkscreen `Z2`, không giống connector → ứng viên `action = correct`.

Đây chỉ là 2 ví dụ bắt gặp, **không phải** kết quả duyệt. Vẫn phải xem đủ 32 dòng.

## 4. Thứ tự làm việc đề xuất

| # | Việc | Plan | Thời gian | Chặn bởi |
|---|---|---|---|---|
| 1 | ~~Chốt §3.1 (giữ hay đổi Golden)~~ | §2 | — | ✅ xong |
| 2 | ~~Chốt §3.2 (quy ước rotation 0°)~~ | §4, §5 | — | ✅ xong |
| 3 | ~~Duyệt 30 frame → `frame_review.csv`~~ | §1 | — | ✅ xong |
| 4 | ~~Ký duyệt `golden_approval.md`~~ | §2 | — | ✅ xong |
| 5 | Duyệt overlay alignment/consensus | §3 | 30 phút | — |
| 6 | **Gán RefDes/footprint/rotation 32 dòng** ← việc lớn kế tiếp | §4 | 2–4 giờ | — (đã hết chặn) |
| 7 | Đo board → `board_measurements.json`, `fiducials.csv` | §5 | 1 giờ | **cần board vật lý/CAD** |
| 8 | Annotation Phase 1–3 | §7–9 | 4–8 giờ | 3, 6 |
| 9 | Validation trên hệ thống thật | §11 | — | **cần camera/fixture production** |

---

## 5. Ràng buộc không được vi phạm

- `production_eligible = false` cho tới khi có đo đạc vật lý **và** validation
  trên camera/fixture thật.
- Không điền tọa độ mm suy đoán, không lấy tỷ lệ mm từ ảnh.
- Không diễn giải `canvas_overlap_ratio = 1.0` thành độ chính xác registration —
  đó là artifact của identity transform.
- `action` trong `pnp_pixels_REVIEWED.csv` chỉ nhận `accept` / `correct` / `add`
  / `reject`. Dòng chưa duyệt **để trống** `action`, giữ `review_status = draft`
  và `label_source = pseudo_label`.
- Dòng `add` để trống `source_auto_id`; dòng từ detector phải giữ ID `*_AUTO_*`.
- Không sửa trực tiếp artifact trong `golden_recipes/.../bootstrap_pixel_v1/`.
- Không trộn ảnh từ `../pcb_dslr_30_diverse/` vào bộ này.
- Đây là **mặt linh kiện** — không dùng làm ground truth cho solder-defect.

---

## 6. Ghi chú về phiên làm việc trước

Một phiên trước đã tạo `frame_review.csv`, `golden_approval.md`,
`pnp_pixels_REVIEWED.csv` và 3 file hướng dẫn. Chúng đã bị thay thế vì:

- `golden_approval.md` chứa **số liệu sai toàn bộ** (Laplacian 2174.34 thay vì
  1064.80; alignment 23/29 thay vì 26/29; overlap 0.961 thay vì 0.9758).
- `pnp_pixels_REVIEWED.csv` dùng giá trị `action = pending_review` **không có
  trong schema**, và tự gán `action = accept` cho 6 dòng chỉ vì chúng qua gate
  máy — trong khi `accept` là phán quyết của người, không phải của detector.
- 3 file hướng dẫn (`START_HERE.md`, `EXECUTION_STATUS.md`,
  `REVIEW_ACTION_ITEMS.md`) trùng lặp nội dung; gộp lại thành file này.

Không có file gốc nào (ảnh, `manifest.json`, `reference_selection.json`,
bundle) bị chỉnh sửa.
