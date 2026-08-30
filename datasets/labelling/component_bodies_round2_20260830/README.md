# Component bodies — vòng 2 (2026-08-30)

Mở `label_boxes.html` trực tiếp để tiếp tục gán nhãn. App đã nhúng sẵn
`draft_boxes.json`: 16 tile đã duyệt hiện đúng trạng thái `verified`; 104 tile
còn lại nằm trong bộ lọc **chưa duyệt**.

Quy ước: khoanh sát **THÂN/gói/vỏ linh kiện**, không bao chân, pad hoặc thiếc.
Nhấn **Enter** sau khi sửa đủ mọi box trên tile. Phím **C** chỉ dành cho tile
thật sự không có thân linh kiện.

Không nạp lại `joint_boxes (3).json` vào app này: đó là checkpoint của bộ cũ và
app sẽ từ chối vì `dataset_id` khác. Khi dừng, bấm **Xuất JSON** và giữ file vừa
xuất làm checkpoint mới.

Các bất biến đã kiểm tra khi tạo vòng này:

- 120/120 tile có nội dung pixel độc nhất;
- giữ nguyên 16 record `verified`, tổng cộng 1.595 box;
- semantic SHA-256 của phần đã duyệt:
  `7a7e186cfb4944b664cbc95d980f8adde57a646556947096eec62f52b31345a4`;
- loại 29 record `unusable` và alias pixel của chúng, trừ alias có một bản
  `verified` được giữ lại;
- giữ 36 draft cũ còn hợp lệ và thêm 68 tile mới từ kho nguồn;
- 30 bo vật lý, không quá 6 tile trên mỗi bo.

Bộ cũ `datasets/labelling/component_bodies` không bị sửa hoặc xoá.

## Dựng lại bộ này

`crops/` và `label_boxes.html` bị `.gitignore` chặn (215 MB pixel), nên clone về
sẽ chỉ có `manifest.csv`, `provenance.json` và `draft_boxes.json`. Ba lệnh dưới
dựng lại đúng bộ này từ kho tile gốc — cùng checkpoint, cùng seed, cùng kết quả:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_component_labelling.py `
  --output datasets\labelling\component_bodies_round2_20260830 `
  --checkpoint "$HOME\Downloads\joint_boxes (3).json" `
  --checkpoint-root datasets\labelling\component_bodies

.\.venv\Scripts\python.exe scripts\prelabel_component_bodies.py `
  datasets\labelling\component_bodies_round2_20260830 `
  --model models\active\detector\best.onnx `
  --checkpoint "$HOME\Downloads\joint_boxes (3).json" `
  --previous-folder datasets\labelling\component_bodies `
  --base-draft datasets\labelling\component_bodies\draft_boxes.json

.\.venv\Scripts\python.exe scripts\build_joint_box_app.py `
  datasets\labelling\component_bodies_round2_20260830 `
  --classes component `
  --seed-json datasets\labelling\component_bodies_round2_20260830\draft_boxes.json
```

Hai bước đầu **từ chối ghi đè** file đã có, nên phải xoá đúng file định dựng lại
trước khi chạy. Kiểm tra kết quả bằng semantic SHA-256 ghi ở trên: khác nghĩa là
đầu vào đã khác, đừng dùng tiếp.

## Còn thiếu gì để pack được dataset train

Packer từ chối ghi khi chưa đủ **10 board vật lý đã duyệt** và khi một bucket
train/valid/test còn trống. Đo trên checkpoint 30/08: 8 board, bucket `valid`
trống. Packer tự nêu tên board cần duyệt — chạy lệnh audit trong
`datasets/public/README.md` để lấy danh sách hiện tại.
