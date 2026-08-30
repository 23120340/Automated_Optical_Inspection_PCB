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

> ⚠️ Khi dùng một phần các ảnh/tile này để gán nhãn train, phải khử trùng ở bước
> đóng gói và chia theo **board vật lý**, không theo tên thư mục hay tile. Ví dụ
> `pcb7__rec1` và `pcb_dslr_007__rec1` là cùng board; một số tile tạo từ hai
> thư mục còn trùng byte. Giữ nguyên manifest/export đã được duyệt, ưu tiên bản
> `verified`, bỏ bản `unusable`, rồi loại exact duplicate khi pack.

> RF100 `printed-circuit-board` cũng có ảnh kiểu `pcb7rec1` khớp nguồn PCB DSLR.
> Nếu board local được giữ cho validation/test thì mọi bản RF100 của chính board
> đó phải bị loại khỏi train (hoặc đi cùng split). Không dùng public split có sẵn
> để suy ra rằng hai ảnh là độc lập.

## `tiles_1024/` — 310 tile zoom

Ảnh toàn board là đầu vào đúng cho bước 0–3 và **sai** cho việc nhìn một linh
kiện. Thư mục này là góc nhìn còn lại — đúng thứ ảnh mẫu của dự án là,
`00001__1024__1648___4120.png`: cửa sổ 1024 px lên một phần board, đọc được
từng linh kiện.

```bash
python scripts/tile_test_images.py --min-components 6
```

**Không hạ tỉ lệ, và đó là chủ ý.** Đo bằng chính detector đang dùng:

| | Linh kiện, cạnh ngắn trung vị |
|---|---|
| `00001__1024__1648___4120` (ảnh mẫu) | 22 px |
| `pcb_dslr_001__rec1` | 39 px (**1,78×**) |
| `pcb31__rec1` | 54 px (**2,43×**) |

Ảnh gốc đã mịn hơn ảnh mẫu, nên tile ở độ phân giải gốc rơi **trên** nó — linh
kiện to hơn, dễ nhìn hơn. Hạ về 22 px là vứt đi phần đó.

| | |
|---|---|
| Tile | **310**, tất cả 1024×1024, 0 file hỏng |
| Board | 58 board khác nhau |
| Linh kiện | 4.090 (trung vị 11/tile, ít nhất 6, nhiều nhất 52) |
| Dung lượng | 552 MB |

**Tile được giữ theo số linh kiện, không theo thống kê pixel.** Độ sáng và bão
hoà đã thử và không tách được: trên `pcb31` dải nền đọc 133 độ sáng còn tile
board tối nhất đọc 34. Số linh kiện thì tách được, và nó cũng là thứ quyết định
— tile không có gì thì không test được bước 4 dù đẹp đến đâu.

> ⚠️ **`components` trong manifest là số của lượt quét TOÀN BOARD**, không phải
> số linh kiện trong tile. Chạy detector thẳng trên tile ra nhiều hơn hẳn — đo
> trên hai tile: **52 → 96** và **11 → 120** — vì linh kiện to hơn và không bị
> bối cảnh cạnh tranh. Chính khoảng cách đó là lý do thư mục này tồn tại.

Board được chụp trên vải đen, nên một tile ở mép board vẫn đủ 6 linh kiện khi
chúng dồn vào một góc mà phần lớn khung là nền. Những tile đó **không bị loại**
— mép board là thứ có thật và đáng test — nhưng manifest ghi `dark_fraction` để
lọc được mà không phải đo lại: trung vị **30%**, 28% số tile trên 50%, 8% trên
70%. Muốn chặt hơn thì `--max-dark-fraction 0.5`.

## Bộ gán nhãn thân linh kiện — vòng 2

Audit ngày 2026-08-30 cho thấy 310 dòng nguồn chỉ tạo thành **170 nhóm pixel
độc nhất**; 140 dòng còn lại là exact duplicate sau khi giải mã ảnh. Bộ kế tiếp
ở `datasets/labelling/component_bodies_round2_20260830/` có 120 tile và
**120/120 hash pixel khác nhau**, phủ 30 board vật lý (tối đa 6 tile/board).
Bộ cũ `component_bodies/` được giữ nguyên làm dấu vết, không bị ghi đè.

Vòng 2 được tạo từ checkpoint `joint_boxes (3).json`: mang nguyên trạng **16
tile `verified` / 1.595 box**, loại 29 tile `unusable`, giữ 36 tile chưa duyệt
thật sự độc nhất và lấy thêm 68 tile từ phần chưa dùng của kho nguồn. Xem chi
tiết và SHA-256 checkpoint trong `provenance.json` của vòng 2.

Để tiếp tục gán nhãn, mở trực tiếp
`datasets/labelling/component_bodies_round2_20260830/label_boxes.html`. App đã
được seed: lần mở đầu tiên của dataset ID mới sẽ hiện sẵn 16 ảnh `verified` và
các box gợi ý cho ảnh còn lại; box chưa `verified` vẫn chỉ là bản nháp cần duyệt.
Sau mỗi phiên phải bấm **Xuất JSON** và giữ file xuất mới nhất làm checkpoint.

**Không bấm Nạp file để đưa checkpoint của `component_bodies/` vào app vòng
2.** Checkpoint cũ có dataset ID khác và app phải từ chối; 16 record đã duyệt đã
được chuyển sẵn, nên nạp lại vừa không cần thiết vừa dễ nhầm phiên làm việc.

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
điều kiện đó đi theo các file này và các tile/nhãn phái sinh. Việc một bản sao
downstream trên Roboflow ghi CC BY 4.0 không tự ghi đè quyền upstream; dùng
thương mại cần làm rõ provenance với chủ dữ liệu. Annotation chính thức của PCB
DSLR chỉ khoanh **IC**; không ghép nguyên ảnh vào detector một lớp `component`
như thể mọi linh kiện không được khoanh là background. Thư mục đã bị
`.gitignore` chặn.
