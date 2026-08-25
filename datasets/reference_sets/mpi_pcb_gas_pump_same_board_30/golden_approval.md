# Golden approval — bootstrap_pixel_v1

Trạng thái: **ĐÃ DUYỆT (APPROVED)** cho vai trò Golden candidate/reference.
Vẫn **chưa** đủ điều kiện production — xem §6.

Duyệt bởi `qnn` ngày 2026-08-24, đối chiếu độc lập bằng đo đạc trong phiên này.

Mọi con số dưới đây được sinh lại từ dữ liệu gốc bằng:

```
python scripts/verify_reference_set_bootstrap.py \
  --set-dir datasets/reference_sets/mpi_pcb_gas_pump_same_board_30 \
  --bundle-dir golden_recipes/mpi_pcb_gas_pump/top/bootstrap_pixel_v1 verify
```

Kết quả đầy đủ nằm trong `verification_report.json` (13/13 check PASS).

## 1. Frame được chọn và truy vết

| Mục | Giá trị |
|---|---|
| Frame nguồn | `mpi_pcb_train_good_0893.jpg` |
| SHA-256 JPEG nguồn | `10c5b91aa967ca44a1a576c8ecf0e8a127305be8c34e586d6e040bafda7789f4` |
| SHA-256 `golden.png` | `52e62cecf7b8e9296c6867bb84f08d4a878cd68f5d38012f500446c5ace4bb34` |
| Quan hệ hai file | **Bit-exact**: mảng pixel RGB của `golden.png` trùng khớp hoàn toàn với JPEG nguồn sau giải mã |
| Canvas | 4096 × 2816 (đúng kích thước gốc, không resize) |
| Thuật toán chọn | `aligned_source_medoid_v1`, policy `single_actual_source` |

✅ **Đã xác minh**: Golden là **một ảnh nguồn thật**, không phải ảnh trung bình,
median hay ảnh ghép. JPEG nguồn được giữ nguyên trong `images/`.

## 2. Vì sao frame này được chọn

Thuật toán chọn medoid theo **độ trung tâm về alignment**, không theo độ nét.

| Mục | Giá trị |
|---|---|
| `medoid_score` | 0.2330 — **tốt nhất** trong 8 ứng viên đủ điều kiện |
| `aligned_peer_count` | 26 / 29 peer (ratio 0.8966) |
| `required_peers` | 24 |
| Số ứng viên đủ điều kiện | 8 / 30 |

## 3. Chất lượng ảnh (đo được)

Đo trên ảnh **full-resolution**, cùng cách với cả 30 frame (nguồn: `manifest.json`):

| Mục | Golden 0893 | Bộ 30 frame |
|---|---|---|
| `focus_laplacian_variance` | **61.30** | min 51.76 · median 96.64 · max 170.69 |
| Hạng độ nét | **26 / 30** (1 = nét nhất) | — |
| `mean_luminance` | 110.92 | min 96.66 · median 115.17 · max 137.36 |

> Chỉ số alignment/quality trong `reference_selection.json` được tính trên ảnh
> chẩn đoán 1024×704, **không cùng thang** với bảng trên. Không trộn hai nguồn số.

## 4. Các điểm đã xem xét và quyết định

### 4.1 Golden nằm trong nhóm ảnh mờ nhất của bộ — ĐÃ QUYẾT: giữ 0893

Frame 0893 có độ nét xếp **26/30** toàn bộ, và **7/8** trong nhóm đủ điều kiện
(chỉ nét hơn 1090). Thuật toán chọn theo `medoid_score` (độ trung tâm alignment),
**không** theo độ nét.

So sánh 8 ứng viên đủ điều kiện (sắp theo `medoid_score`, tốt → kém):

| Frame | medoid_score | aligned peers | focus (full-res) |
|---|---|---|---|
| **0893 (đã chọn)** | **0.2330** | 26 | 61.30 |
| 0040 | 0.2426 | 26 | 72.71 |
| 0123 | 0.2557 | 25 | 81.50 |
| 0676 | 0.2621 | 25 | 71.85 |
| 0629 | 0.2675 | 25 | 61.57 |
| 1195 | 0.2724 | 25 | 62.55 |
| 1090 | 0.3071 | 24 | 59.36 |
| 1163 | 0.3095 | 24 | 68.49 |

✅ **Quyết định (qnn, 2026-08-24): giữ `0893`.**

Bằng chứng hỗ trợ quyết định: kiểm tra ở full-resolution cho thấy 0893 vẫn đọc
rõ mã linh kiện (`MC74HC574AN`, `FJU9732`) và silkscreen (`P205`, `Z2`, `R37`).
`focus_laplacian_variance` thấp ở đây phản ánh board có nhiều vùng phẳng tối, chứ
không phải ảnh bị nhòe. Ưu thế `medoid_score` tốt nhất giữ cho Golden nằm ở
trung tâm phân bố pose của bộ, có lợi cho consensus.

### 4.2 Canvas Golden xoay 180° so với chiều đọc silkscreen — ĐÃ CHỐT QUY ƯỚC

Toàn bộ chữ trên board (`GILBARCO LEGACY`, `DALLAS DS1225AD`, `ZILOG Z84C0006PEC
Z80 CPU`, các RefDes `U*`, `P*`, `R*`) đọc **ngược** trên `golden.png`; xoay 180°
thì đọc đúng. Đây là **phép xoay, không phải mirror**, và không phải lỗi.

✅ **Quyết định: `rotation_deg` được định nghĩa trong chính `golden_board_pixels`
(canvas frame), không đổi sang frame silkscreen-upright.**

Lý do: `center_x_px` / `center_y_px` đã nằm trong `golden_board_pixels`, mà plan
§6 quy định *"Không trộn annotation của hai coordinate space trong cùng record"*.
Nếu đặt góc theo frame upright còn tọa độ theo canvas thì mỗi dòng PnP mang hai
hệ quy chiếu — đúng thứ plan cấm.

Quy ước đã ghi vào `board_measurements.json` → `pixel_frame_convention`:

| Mục | Giá trị |
|---|---|
| Space | `golden_board_pixels` |
| Origin | pixel trên-trái của `golden.png` |
| +X / +Y | phải / xuống |
| Zero angle | vector từ tâm thân linh kiện tới dấu pin-1 / cathode / cực dương hướng theo **+X** |
| Chiều dương | ngược chiều kim đồng hồ **như hiển thị** trên `golden.png` |
| Miền giá trị | `[0, 360)` |

Phép chuyển sang frame thiết kế được ghi tường minh ở
`canvas_to_board_design_transform` (xoay 180° quanh tâm canvas `2048.0, 1408.0`),
trạng thái `KNOWN_BUT_NOT_APPLIED` — chỉ áp dụng như **một bước có ghi chép** khi
import/export CAD/BOM/PnP, không âm thầm áp vào tọa độ pixel đã lưu.

> ⚠️ Khi duyệt crop: chữ trên board đọc ngược là **đúng như mong đợi**. Góc bạn
> đo trực tiếp trên crop chính là góc cần ghi — không cộng trừ 180°.

### 4.3 Xác nhận bằng mắt — ĐÃ DUYỆT

- [x] Đúng board / layout / revision / mặt linh kiện
- [x] Đủ bốn phía board, không bị cắt vùng quan trọng
- [x] Đủ nét ở IC, connector, điện trở và silkscreen khi zoom 100%
- [x] Không cháy sáng nặng, không bóng đổ che linh kiện
- [ ] **Đối chiếu với board vật lý đạt chuẩn** — *chưa thực hiện*

> Vì mục cuối chưa xong, đây vẫn là **Golden candidate/reference**, chưa phải
> Golden production.

## 5. Chữ ký duyệt

| Mục | Giá trị |
|---|---|
| Người duyệt | **qnn** |
| Ngày duyệt | **2026-08-24** |
| `review_status` trong `frame_review.csv` | **`ok_verified`** (30/30 frame) |
| Phạm vi duyệt | Golden candidate/reference cho enrollment và audit pixel |
| Lý do chọn | Frame nguồn thật, truy vết bit-exact về JPEG 0893; `medoid_score` tốt nhất trong 8 ứng viên đủ điều kiện; mã linh kiện và silkscreen đọc rõ ở 100% |
| Đối chiếu độc lập | Đo đạc bằng máy trong phiên 2026-08-24: 13/13 check PASS (`verification_report.json`), so pixel 30 frame với Golden, kiểm tra focus/white-balance/clipping từng frame |
| Giới hạn | Chưa đối chiếu board vật lý; chưa đo mm; chưa validation trên camera/fixture production |

## 6. Ràng buộc còn hiệu lực

- `production_eligible = false`. Chưa có đo đạc board vật lý, chưa có
  registration fiducial, chưa có validation trên camera/fixture thật.
- Alignment upstream ở mode `upstream_dataset_identity_enrollment_only`; mọi fit
  metric là `not_measured_upstream_identity`. `canvas_overlap_ratio = 1.0` là
  **artifact của identity transform**, không phải độ chính xác registration đo được.
- Không công bố độ chính xác mm/sub-pixel từ bundle này.
