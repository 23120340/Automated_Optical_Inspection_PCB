# Component bodies — vòng 2 (2026-08-30)

Có hai trang, và trạng thái duyệt của chúng **không thay thế nhau**:

- `label_boxes.html`: duyệt hình học thân linh kiện. App đã nhúng
  `draft_boxes.json`; 16 tile đã duyệt hiện đúng trạng thái `verified`, 104 tile
  còn lại nằm trong bộ lọc **chưa duyệt**.
- `label_packages.html`: gán một trong 7 package cho từng box thân. Tọa độ được
  giữ nguyên từ draft thân, nhưng **0 tile được coi là đã duyệt package**:
  3.847/3.855 box còn `unknown`, 8 prelabel chỉ là gợi ý bảo thủ. Phải chọn
  package 1–7 cho từng box; `unknown` chặn cả Enter và export.

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

## Cho thành viên mới vào gán nhãn cùng

**Một lệnh duy nhất**, sau khi clone repo và cài `requirements.txt`:

```powershell
.\.venv\Scripts\python.exe scripts\setup_labelling_workspace.py `
  datasets\labelling\component_bodies_round2_20260830
```

Nó chép 120 tile vào `crops/`, **đối chiếu từng file với `crops.sha256`**, rồi
dựng cả hai trang gán nhãn. Nếu chưa có kho tile, nó dừng lại và in đúng ba lệnh
cần chạy trước.

### Vì sao ảnh không nằm trong git

Không phải quên. 120 tile là **215 MB pixel** trong một repo **công khai**, và
quan trọng hơn: chúng cắt ra từ **CVL PCB-DSLR**, bộ mà chủ dữ liệu giới hạn
**nghiên cứu phi thương mại**. Điều kiện đó đi theo cả tile phái sinh, nên đăng
lại chúng ở đây là phát hành lại dữ liệu của người khác sai điều khoản. Mỗi
người tự tải nguồn theo đúng điều khoản của nguồn — xem
`datasets/test_images/ATTRIBUTION.md`.

Thứ **có** trong git là thứ làm cho việc dựng lại kiểm chứng được:

| File | Là gì |
|---|---|
| `manifest.csv` | 120 tile nào được chọn |
| `crops.sha256` | đúng pixel nào — nếu lệch, script dừng |
| `draft_boxes.json` | 16 tile đã duyệt (1.595 box) + box nháp cho phần còn lại |
| `draft_package_boxes.json` | bản nháp 7 lớp package |
| `provenance.json` | checkpoint nào sinh ra bộ này, kèm sha256 |

`crops.sha256` là chốt quan trọng nhất: 16 record đã duyệt mang **toạ độ** vẽ
trên đúng những pixel đó. Dựng ra tile khác một chút thì box sẽ trỏ sai chỗ mà
không có gì báo lỗi — nên script từ chối chạy tiếp khi hash lệch.

### Quy tắc làm việc nhiều người

- **Đừng bấm "Nạp file"** để nạp checkpoint của người khác. Trang đã seed sẵn;
  nạp thêm file của người khác sẽ đụng vào phần bạn vừa duyệt, và app sẽ **huỷ
  toàn bộ import** nếu phát hiện mâu thuẫn — an toàn, nhưng vô ích.
- **Mỗi người xuất JSON của riêng mình** (nút *Xuất JSON*) khi dừng, đặt tên
  kèm tên mình, rồi gộp sau.
- **Chia việc theo BO, không theo tile.** Tile cắt chồng nhau 256 px nên cùng
  một linh kiện xuất hiện ở nhiều tile; hai người làm hai tile chồng nhau là
  vẽ lại cùng một linh kiện. `scene_id` trong `manifest.csv` là tên bo.
- Đang cần gấp nhất: **bo `pcb_dslr:017` và `pcb_dslr:030`** — chỉ cần mỗi bo
  một tile được duyệt là packer đủ điều kiện xuất bộ train.

Quy ước khoanh và các lỗi thường gặp: xem
[docs/evaluation/danh_gia_khoanh_box_than_linh_kien.md](../../../docs/evaluation/danh_gia_khoanh_box_than_linh_kien.md).
