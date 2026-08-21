# Khảo sát dataset cho lượt 2 — detect chân/pad

> Soạn 2026-08-21. Khác với `Docs/pcb_aoi_component_datasets.md` (khảo sát nguồn
> nhận dạng **linh kiện**), file này tìm nguồn có nhãn **vị trí chân/pad/mối
> hàn** ở mức board — thứ mà lượt 2 cần.
>
> Quy ước: mục nào **đã kiểm chứng** thì ghi rõ kiểm bằng cách nào. Mục nào chỉ
> đọc được mô tả thì ghi là **chưa kiểm chứng** — đừng coi hai loại như nhau.

## Kết luận trước

**Không có dataset công khai nào đủ dùng cho lượt 2.** Đường duy nhất đi tới
model là gán nhãn trên board của chính dây chuyền. Các nguồn công khai chỉ có
giá trị làm pretrain hoặc đối chứng.

Lý do không phải "chưa tìm kỹ" mà là ràng buộc vật lý đã đo được: nhãn chân hàn
công khai hoặc **quá hiếm** (30 ảnh), hoặc **sai tỉ lệ chụp** (macro 1–3 µm/px
so với 46 µm/px của dự án).

## Đã kiểm chứng bằng số đo

### 1. `pads`/`pins` trong PCB Component Detection Consolidated

Đọc thẳng từ artifact của dự án (`models/active/detector/model_manifest.json`):

| Lớp | Số instance train | So với |
|---|---|---|
| `pads` | **186** | `capacitor` 7775 |
| `pins` | **261** | `resistor` 7135 |

`rare_image_fraction` = 0.2195 **sau khi đã nhân bản 6 lần**, tức nguyên bản chỉ
khoảng **30 ảnh** có hai lớp này. Kết quả huấn luyện: `pads` recall **0.265**,
`pins` **0.595** — và đó là sau khi notebook v2 đã tăng imgsz lên 1536, bật
`copy_paste`, oversample và kéo dài lịch train.

**Dùng được không:** làm pretrain thì có (đúng lớp, đúng tỉ lệ board). Làm nguồn
chính thì không — 30 ảnh không thể dạy một lớp.

### 2. SolDef_AI — đúng bài toán, sai tỉ lệ

Notebook `soldef-ai.ipynb` (bạn đã tải): YOLO11m-seg, 428 ảnh, val Box mAP50
**0.771**, Mask mAP50 0.766. Nhãn là **từng mối hàn**, có cả box lẫn mask — đúng
loại nhãn cần.

Nhưng đã đo: chạy `best_soldef_AI.pt` lên tile board của dự án cho **0 box** ở
conf 0.25 và 0.10; ở conf 0.05–0.01 ra 1–5 box, toàn nhãn `spike`, tức nhiễu.
Phóng to từng linh kiện **1×, 2×, 4×, 8×, 12× — vẫn 0 box**. Đối chứng: chạy
trên chính ảnh SolDef_AI ra 1–3 box mỗi ảnh, nhãn hợp lý. Model không hỏng.

Nguyên nhân: SolDef_AI chụp macro, mỗi ảnh một linh kiện, khoảng **1–3 µm/px**.
Board của dự án **46 µm/px**. Chênh khoảng 20 lần, và nội suy không tạo ra chi
tiết chưa từng được chụp.

**Dùng được không:** không, cho tới khi có camera macro. Xem
`Docs/yeu_cau_phan_cung_camera.md`.

### 3. Board của chính dự án — nguồn khả thi nhất

Chạy `scripts/bootstrap_lead_labels.py` lên một tile 1024² thật rồi áp logic
chuyển crop của notebook lượt 2:

| Chỉ số | Đo được |
|---|---|
| Linh kiện lượt 1 tìm được | 38 |
| Crop sinh ra (đều có pad) | 38 |
| Crop trống / quá nhỏ | 0 / 0 |
| Kích thước crop | trung vị **62 × 58 px** |
| **Pad, cạnh ngắn** | trung vị **23 px**, phân vị 10 là 19 px |
| Pad dưới 8 px | **2/90 (2.2%)** |

**Đây là con số đáng mừng nhất trong cả khảo sát.** Pad 23 px là học được. Ngưỡng
cảnh báo của notebook (quá nửa số pad dưới 8 px) còn rất xa.

Lưu ý: đây là box **suy ra từ hình học**, không phải ground truth — kích thước
phản ánh công thức hình học chứ không phải pad thật. Nhưng nó cho biết **thang
đo** đủ để làm việc.

## Tìm được nhưng CHƯA kiểm chứng

Ghi lại để bạn tự kiểm, tôi không xác minh được nội dung.

| Nguồn | Mô tả theo tài liệu | Vì sao chưa kiểm chứng |
|---|---|---|
| [PCB-SAID (ICCVW 2025)](https://openaccess.thecvf.com/content/ICCV2025W/VISION'25/html/Mineo_PCB-SAID_A_Low-Cost_Camera-Based_Dataset_for_Few-Shot_SMD_Assembly_Inspection_ICCVW_2025_paper.html) | 175 ảnh RGB, 66 lớp chi tiết: hàn tốt / lệch / bong / xoay / thiếu / chập. Box + polygon, quy trình gán nhãn hai bước có chuyên gia. **"Low-cost camera"** gợi ý tỉ lệ gần với dự án hơn SolDef_AI | Trang openaccess trả HTTP 403 khi fetch tự động. Không tìm thấy link GitHub/Zenodo nào. **Bạn cần tự mở bài báo để lấy link dataset và giấy phép.** Đây là ứng viên đáng xem nhất |
| [Roboflow Universe — lớp `solder`](https://universe.roboflow.com/search?q=class:solder) | Nhiều dataset cộng đồng có lớp liên quan mối hàn | Roboflow trả HTTP 403 khi fetch. Chỉ đọc được tiêu đề từ kết quả tìm kiếm |
| [smd-component-detection (Roboflow)](https://universe.roboflow.com/marco-filippozzi-siwjn/smd-component-detection) | Detect linh kiện SMD | Như trên. Nhãn nhiều khả năng là **linh kiện**, không phải chân |
| [PCBA-Dataset (GitHub)](https://github.com/ismh16/PCBA-Dataset) | Object detection cho lỗi PCBA | Chưa mở. Khảo sát cũ ghi PCBA-DET chủ yếu nhãn vít/quạt/dây/xước, không phải chân hàn |

## Nguồn đã loại, kèm lý do

- **DeepPCB, PKU PCB defect**: lỗi bare-board (`open`, `short`, `mouse bite`…),
  không có linh kiện đã gắn, không có chân hàn.
- **FPIC / FICS-PCB**: polygon cho **linh kiện** và text, không phải pad. Có ích
  cho lượt 1, không cho lượt 2.
- **PCB-Vision**: mask 3 lớp (IC / tụ / connector), mức linh kiện.
- **PCB DSLR**: chỉ bbox IC, và chỉ cho nghiên cứu phi thương mại.

## Khuyến nghị

1. **Bắt buộc:** gán nhãn board của chính bạn. `scripts/bootstrap_lead_labels.py`
   đã vẽ sẵn box (79 box trên một tile) — **sửa** nhanh hơn **vẽ** nhiều lần.
   Giá trị lớn nhất khi sửa là **thêm box ở chân mà hình học bỏ sót**; đó chính
   là thứ model cần học mà hình học không biết.
2. **Nên làm:** pretrain trên tập con `pads`/`pins` của Consolidated (30 ảnh)
   rồi fine-tune trên board của bạn. Rẻ, và đúng tỉ lệ board.
3. **Nên kiểm:** mở bài PCB-SAID, xem tỉ lệ chụp và đơn vị gán nhãn. Nếu nó gán
   nhãn **từng mối hàn** ở tỉ lệ camera thường thì đây là nguồn công khai tốt
   nhất hiện có.
4. **Chưa cần:** SolDef_AI, cho tới khi có camera macro.

## Cần bao nhiêu board?

Không có con số chắc chắn, nhưng có thể ước lượng từ chính số đo trên: một tile
1024² cho **38 crop**. Một board 5144² cho khoảng **960 crop**.

Nút thắt **không phải số crop mà là số board**. Crop từ cùng một board có cùng
ánh sáng, cùng lô hàn, thường cùng loại linh kiện. Notebook vì thế chia tập
**theo board** và chặn nếu có dưới 3 board.

Mốc thực tế để bắt đầu: **10–20 board khác nhau**, ưu tiên khác lô, khác loại
board, khác điều kiện chiếu sáng — hơn là nhiều ảnh của cùng một board.
