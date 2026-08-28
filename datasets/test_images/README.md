# Ảnh board để test pipeline

Ảnh chụp **toàn board, độ phân giải gốc**, dùng để chạy thử đường ống 0 → 6.2.
Không phải ảnh gắn nhãn để train — đó là `datasets/train/` và `datasets/public/`.

Tải lại bằng:

```bash
python scripts/fetch_test_board_images.py                 # tải + gom từ kho
python scripts/fetch_test_board_images.py --local-only    # chỉ gom, không tải
```

## "Chất lượng cao" ở đây nghĩa là gì

Không phải "ảnh nét nhất". Đây là câu hỏi về **tỉ lệ µm/px**, và dự án đã đo cả
hai kiểu sai:

- **Quá gần (macro).** SolDef_AI chụp ở 1–3 µm/px. Model train trên nó đạt Box
  mAP50 0.771 trên chính nó nhưng cho **0 box** trên board dự án ở mọi mức phóng
  đại 1×–12×. Phóng to bằng phần mềm không tạo ra chi tiết chưa từng được chụp.
- **Quá xa / đã bị nén.** Một bản export 640×640 để linh kiện lại 10 px; mối hàn
  biến mất hoàn toàn.

Board dự án chụp ở **46 µm/px** — linh kiện ~62×58 px, pad ~23 px. Một ảnh từ
**6 MP trở lên** còn dư chỗ để *hạ* về tỉ lệ đó; dưới ngưỡng thì muốn tới đó
phải *phóng lên*. Đó là ngưỡng `MIN_MEGAPIXELS` trong script.

Ngưỡng đo độ phân giải, **không** đo "đây có phải ảnh chụp không" — nên script
còn lọc theo tên để loại mask nhị phân và ảnh nhãn, vốn nằm cùng thư mục với
ảnh chụp và cùng kích thước. Bỏ bước đó thì 30 file mask lọt vào đây (đã xảy ra).

## Đã kiểm bằng cách chạy thật

Không chỉ đo megapixel — bốn ảnh được đưa qua trọn pipeline:

| Ảnh | Kích thước | Linh kiện | ROI mối hàn | Thời gian |
|---|---|---:|---:|---|
| `pcb_dslr_diverse/pcb_dslr_001__rec1.jpg` | 4928×3280 | 176 | 721 | 44,5 s |
| `mpi_pcb_gas_pump/mpi_pcb_train_good_0040.jpg` | 4096×2816 | 107 | 603 | 44,7 s |
| `cvl_pcb_dslr/pcb31__rec1.jpg` (mainboard ATX) | 4928×3280 | 172 | 790 | 33 s |
| `cvl_pcb_dslr/pcb35__rec1.jpg` (T-CON) | 4928×3280 | 45 | 333 | 41 s |

Cả bốn chạy qua đủ bước 0 → 5.5 → 6.2. Thời gian 33–45 s/ảnh là do khung phân
tích 4096 px chạy trên CPU.

## Các thư mục

| Thư mục | Ảnh | Kích thước | Nguồn |
|---|---:|---|---|
| `cvl_pcb_dslr/` | **175** | 4928×3280 (16,2 MP) | CVL PCB DSLR, TU Wien — **40 board** × rec1…rec5 |
| `pcb_dslr_diverse/` | 30 | 4928×3280 (16,2 MP) | cùng nguồn, bản `rec1` đã lấy trước đó |
| `mpi_pcb_gas_pump/` | 30 | 4096×2816 (11,5 MP) | MPI PCB, cùng một board chụp 30 lần |
| **tổng** | **235** | trung vị 16,2 MP, thấp nhất 11,5 MP | |

Đã kiểm toàn vẹn: **235/235 ảnh giải mã hết được, 0 file hỏng** (ép `Image.load()`
chứ không chỉ đọc header — ảnh tải qua mạng có thể cụt mà header vẫn đọc được).

`cvl_pcb_dslr/` phủ 40 board khác nhau: mainboard ATX dày đặc, board nguồn
xanh/vàng, T-CON xanh dương, module nhỏ — khác nhau cả màu phủ, mật độ lẫn kích
thước.

`cvl_pcb_dslr/` và `pcb_dslr_diverse/` **trùng nguồn**: thư mục sau là 30 ảnh
`rec1` lấy từ đợt trước cho bộ reference, thư mục trước là bản đầy đủ. Giữ cả hai
vì bộ reference đã được duyệt tay và có nhãn đi kèm ở
`datasets/reference_sets/pcb_dslr_30_diverse/`.

## Vì sao nhiều ảnh cùng một board

`rec1`…`rec5` là cùng một board chụp ở **vị trí và ánh sáng khác nhau**. Đó
không phải trùng lặp lãng phí — đó chính là thứ để thử bước 2 (align) và bước 3
(khoanh board): hai ảnh của cùng một board phải cho cùng một kết quả sau khi
căn chỉnh.

Cùng lý do với `mpi_pcb_gas_pump/`: 30 khung của **một** board, dùng để dựng và
kiểm golden image.

## Đã kiểm và LOẠI

### PCB-Defect (Mendeley, `vdj74sngvn`) — board TRẦN

230 ảnh, 2,57–27,7 MP, CC BY 4.0, toàn board — mọi con số đều đạt. Nhưng nhìn
ảnh thì đó là **board trần**: nền vàng, chỉ có đường đồng, **không có linh kiện
và không có mối hàn**. Sáu loại lỗi của nó (`missing pad`, `mouse bite`,
`open circuit`, `spur`, `spurious copper`) đều là lỗi đường mạch.

Nó chỉ chạy được bước 0–3; bước 4 (detect linh kiện), 6.1 và 5.5/6.2 không có gì
để làm. Đây đúng là lý do phải **mở ảnh ra xem** chứ không đọc mô tả.

### PCB-METAL — không tải công khai được

984 ảnh của 123 board có box linh kiện, đúng loại cần. Nhưng không có link công
khai; các bài dẫn lại đều ghi "not available to this publication date".

## Giấy phép

Xem `ATTRIBUTION.md`. Bộ CVL PCB DSLR giới hạn **nghiên cứu phi thương mại**;
điều kiện đó đi theo các file này. Thư mục đã bị `.gitignore` chặn.
