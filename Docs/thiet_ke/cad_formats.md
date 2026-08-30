# Nạp sơ đồ CAD cho bước 5.5

Tài liệu này mô tả file cần chuẩn bị để bật phần CAD của bước 5.5. Toàn bộ code
đã có sẵn; khi có file thì chỉ cần nạp vào, không phải sửa gì.

**Không có CAD thì pipeline vẫn chạy đúng như hiện tại.** ROI mối hàn được suy ra
từ box của detector cộng topology chân của class. CAD không thay thế phần đó mà
được **hợp nhất** với nó, xem [phần fusion](#hợp-nhất-chứ-không-thay-thế).

## Định dạng được hỗ trợ

Định dạng được nhận dạng tự động từ đuôi file và nội dung. Muốn ép thì truyền
`--cad-format` (script) hoặc `config.cad.fmt`.

| Tên loader | Nhận file gì | Có toạ độ land? | Ghi chú |
|---|---|---|---|
| `pads_csv` | Bảng pad dạng CSV | ✅ | Định dạng chuẩn hoá; mọi tool EDA đều xuất được |
| `placement_csv` | Pick-and-place / centroid CSV | ❌ (chỉ tâm + góc xoay) | Hay được xưởng gia công cấp nhất |
| `ipc356` | IPC-D-356A netlist | ✅ | Deliverable tiêu chuẩn, có toạ độ pad thật |
| `cad_json` | File do `save_cad_json` ghi ra | ✅ | Cache sau khi import một lần |

Muốn thêm định dạng khác (KiCad `.kicad_pcb`, ODB++, Gerber…), viết một hàm
`load_x(path, units) -> BoardCad` rồi đăng ký vào `CAD_LOADERS` trong
[cad.py](../../aoi_pipeline/solder/cad.py). Không chỗ nào khác phải sửa: cả script lẫn UI
đều đọc từ registry đó.

## `pads_csv` — định dạng chuẩn hoá

Đây là thứ nên quy về khi không có định dạng nào khớp. Xem
[cad_pads_template.csv](cad_pads_template.csv).

```csv
designator,pin,x_mm,y_mm,width_mm,height_mm,rotation_deg,shape,net,side,footprint,value
R12,1,10.500,20.000,0.90,1.00,0,rect,VCC,top,R_0603,10k
R12,2,12.100,20.000,0.90,1.00,0,rect,NET7,top,R_0603,10k
U3,1,30.000,15.000,0.30,1.20,0,rect,SDA,top,SOIC-8,PCA9548
```

Bắt buộc: `designator`, `x_mm`, `y_mm`. Còn lại tuỳ chọn.

Tên cột được so khớp lỏng, nên bảng pad xuất thẳng từ Altium hay KiCad thường
nạp được mà không cần sửa header. Ví dụ các tên đều hiểu được:

- `designator` ← `refdes`, `ref`, `reference`, `part`, `component`, `name`
- `pin` ← `pin number`, `pad`, `pad name`, `terminal`
- `x_mm` ← `x`, `mid x`, `center-x`, `pos x`, `ref x`
- `side` ← `layer`, `top/bottom`, `mounting side`

Đơn vị mặc định là mm. File dùng đơn vị khác thì truyền `--cad-units mil` /
`inch` / `um`, hoặc để nguyên hậu tố trong ô giá trị (`12.5mm`).

Vị trí và góc xoay của linh kiện được tính từ chính các land của nó, nên không
cần cột riêng.

## `placement_csv` — file pick-and-place

```csv
Designator,Mid X,Mid Y,Rotation,Layer,Footprint,Comment
R12,10.500,20.000,0,Top,R_0603,10k
C7,30.000,15.000,90,Top,C_0402,100n
```

File này không có land, nhưng cố định được **tâm thật và góc xoay thật** của mỗi
linh kiện — đã tốt hơn nhiều so với chỉ có box của detector. Bước 5.5 sẽ dựng
lại đúng hình học ROI suy ra nhưng neo trên tâm/góc của CAD thay vì trên box
axis-aligned của detector. ROI đó được đánh dấu `source=cad+derived`.

## Class linh kiện suy từ ký hiệu

Tiền tố reference designator được ánh xạ sang class của detector (`R`→resistor,
`C`→capacitor, `U`→ic, `LED`/`DS`→led, `Y`/`XTAL`→clock…). Đây chỉ là **prior**:
nó được đối chiếu với class detector đọc được chứ không ghi đè, và bất đồng
được ghi lại thành finding `class_mismatch`.

Class này còn có một vai trò kỹ thuật quan trọng: nó là thứ duy nhất phân biệt
được một board với chính ảnh phản chiếu của nó khi layout gần đối xứng.

## Căn CAD vào ảnh (registration)

CAD ở mm trên board, ảnh ở pixel. Cần một phép biến đổi giữa hai hệ. Có ba cách,
theo thứ tự tin cậy giảm dần:

**1. File registration đã lưu (khuyến nghị cho sản xuất).** Đo một lần cho mỗi
SKU + đồ gá, lưu JSON, các lần sau nạp lại. Mọi lần chạy dùng đúng một hệ toạ độ.

```powershell
# lần đầu: để app tự căn rồi lưu lại
.\.venv\Scripts\python.exe scripts\export_solder_dataset.py D:\anh_board `
  --output D:\datasets\solder_v1 --model models\detector\kaggle\best.onnx `
  --cad D:\cad\board_pads.csv --save-registration D:\cad\reg_sku01.json

# các lần sau: dùng lại
.\.venv\Scripts\python.exe scripts\export_solder_dataset.py D:\anh_board `
  --output D:\datasets\solder_v2 --model models\detector\kaggle\best.onnx `
  --cad D:\cad\board_pads.csv --cad-registration D:\cad\reg_sku01.json
```

**2. Fiducial thủ công.** Đưa vào các cặp toạ độ mm ↔ pixel qua
`config.cad.fiducials_mm` / `fiducials_px`. Ba điểm cho affine đầy đủ (tự xử lý
được chiều trục y và mặt dưới bị lật), bốn điểm trở lên kèm
`fiducial_perspective=True` cho homography khi camera không vuông góc board.

**3. Tự căn theo detection (mặc định).** RANSAC trên các cặp linh kiện: hai cặp
tương ứng là đủ xác định scale, góc xoay và tịnh tiến, sau đó tinh chỉnh bằng
nearest-neighbour. Cả hai chiều trục y đều được thử và chiều nào khớp hơn thì
thắng, vì các định dạng CAD không thống nhất y hướng lên hay xuống.

### Khi phép căn không đáng tin

Đây là phần quan trọng nhất. Một phép căn sai **trông y hệt** một phép căn đúng
nếu chỉ nhìn con số residual, nên hệ thống báo ra thay vì im lặng áp dụng:

- **`ambiguous: true`** — có một phép căn khác *khác hẳn* nhưng khớp không kém.
  Layout gần đối xứng soi bằng detector không cho class (ví dụ CV demo) luôn rơi
  vào trường hợp này: bản gốc và bản lật khớp như nhau. Phải kiểm tra overlay
  hoặc chốt bằng fiducial / file registration.
- **`inlier_ratio` thấp hoặc `residual_px` lớn** — phép căn bị **từ chối**, bước
  5.5 quay về ROI suy ra và ghi cảnh báo. Ngưỡng ở `FusionConfig.min_inlier_ratio`
  và `max_residual_px`.
- **Không có class nào đối chiếu được** — phép căn chỉ dựa trên hình học. Cảnh
  báo được ghi ra; nạp detector đã train hoặc dùng fiducial.
- **Nhiều cặp khớp nhưng không cặp nào đúng class** — chữ ký điển hình của một
  phép căn bị lật.

## Hợp nhất, chứ không thay thế

Bước 5.5 dùng cả hai nguồn và lấy cái này kiểm cái kia:

| Tình huống | ROI sinh ra | `source` |
|---|---|---|
| CAD có land + detector thấy linh kiện, hai bên chồng nhau | ROI hợp nhất | `cad+derived` |
| CAD có land, detector không có ROI tương ứng | ROI theo land CAD | `cad` |
| Detector có ROI, CAD không liệt kê land đó | Giữ ROI suy ra | `derived` |
| CAD chỉ có vị trí đặt (P&P) | Hình học suy ra, neo trên tâm/góc CAD | `cad+derived` |
| CAD có linh kiện, detector không thấy gì | ROI theo land CAD + finding `missing_component` | `cad` |
| Detector thấy linh kiện, CAD không có | Giữ ROI suy ra + finding `unexpected_component` | `derived` |

Hai chi tiết đáng chú ý:

- **Hiệu chỉnh cục bộ.** CAD cho biết footprint, detector cho biết linh kiện
  *này* thực tế nằm đâu. Mỗi linh kiện được dịch theo sai lệch giữa hai vị trí
  đó, nên một phép căn chỉ gần đúng trên toàn board vẫn cho ROI chính xác tại
  từng linh kiện. Tắt bằng `FusionConfig.local_refine=False`.
- **ROI suy ra không bị vứt.** File CAD thường thiếu thermal pad, shield và các
  land cơ khí. ROI suy ra không có land CAD tương ứng vẫn được giữ, vì mất một
  mối hàn tệ hơn là kiểm tra thừa một chỗ.

## Lỗi phát hiện được nhờ có CAD

Những lỗi này tìm ra bằng **đối chiếu**, không cần model nào, và được ghi vào
`cad_findings.csv`:

| `kind` | `severity` | Ý nghĩa |
|---|---|---|
| `missing_component` | defect | CAD có linh kiện, ảnh không thấy |
| `shifted_component` | review | Lệch quá `FusionConfig.max_shift_mm` (mặc định 0.5 mm) |
| `unexpected_component` | info | Ảnh có linh kiện, CAD không liệt kê |
| `class_mismatch` | info | CAD và detector không đồng ý về loại linh kiện |

## Trong app

Sidebar có mục **Sơ đồ CAD (tuỳ chọn)**: chọn mặt board, tải file CAD, tuỳ chọn
tải file registration. Bước 5 có tab **Đối chiếu CAD** hiển thị chất lượng phép
căn, danh sách finding và nút tải `registration.json` để dùng lại.

Trong tab **ROI overlay**, bật *Tô màu theo nguồn ROI*: xanh lá = CAD và detector
cùng đồng ý, hồng = chỉ CAD, vàng = chỉ suy ra từ detector.
