# Train model bước 6.2 trên Kaggle

Notebook: [pcb_solder_defect_kaggle.ipynb](pcb_solder_defect_kaggle.ipynb)
(nguồn percent-format: [pcb_solder_defect_kaggle.py](pcb_solder_defect_kaggle.py),
build lại bằng `python scripts/build_notebook.py training/kaggle/pcb_solder_defect_kaggle.py`).

Đầu ra: `best.onnx` + `model_manifest.json` trong
`/kaggle/working/pcb_solder_defect_artifacts.zip`. Tải về, nạp vào app ở sidebar
**Model kiểm tra mối hàn 6.2**, không cần đổi gì khác.

## Trước tiên: không có dataset công khai nào đủ

Khảo sát tháng 8/2026. Đây là kết luận trung thực, không phải lời mở đầu:

| Nguồn | Phủ | Trạng thái |
|---|---|---|
| [SolDef_AI](https://www.kaggle.com/datasets/mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection) | good, insufficient, excess, **shift_component** | Tốt nhất. 1150 ảnh linh kiện SMT, 3 góc nhìn, từ bài [MDPI JMMP 2024](https://doi.org/10.3390/jmmp8030117). Nguồn peer-reviewed **duy nhất** tìm được có gán nhãn lệch vị trí linh kiện |
| [ouvic215/Soldering-Data-Annotation-boarding](https://huggingface.co/datasets/ouvic215/Soldering-Data-Annotation-boarding) | bridge, excess, missing_solder | 1522 ảnh 512×512. **Không license, không nguồn gốc.** Repo anh em tên `...-ControlNet` ⇒ nghi dữ liệu sinh |
| [AndyLiu0104/Soldering-Data-Tiny-…](https://huggingface.co/datasets/AndyLiu0104/Soldering-Data-Tiny-More-Data-with-appearance-hole-micro-bridge-0801) | bridge, excess, missing_solder | 10469 ảnh nhưng chỉ **36–144 px** — dưới ngưỡng đọc được fillet. Cùng nghi vấn dữ liệu sinh |
| [Roboflow soldering-defects](https://universe.roboflow.com/search?q=class:solder) | **cold**, bridge, insufficient | Nguồn công khai duy nhất có cold solder, nhưng chỉ vài trăm ảnh, chất lượng không đồng đều |

**Bị loại thẳng — đừng nối nhầm:**

| Dataset | Vì sao loại |
|---|---|
| DeepPCB, HRIPCB / PKU-Market-PCB, DsPCBSD+, `akhatova/pcb-defects` | Lỗi **board trần** (open/short/mousebite/spur/copper/pinhole). Board chưa gắn linh kiện, không có mối hàn nào. Bài toán khác hẳn nhưng rất hay bị trích dẫn nhầm |
| AXI_PCB | Ảnh **X-quang**, không dùng được cho AOI quang học |
| [PCBSPDefect](https://github.com/cairs-project-5/PCBSPDefect) | Chưa phát hành: "will be available once the paper is published" |
| PCB-AoI (KubeEdge) | Kiểm tra **kem hàn trước reflow** (SPI), không phải mối hàn sau hàn |

Vì vậy phải **ghép nhiều nguồn**. Notebook làm việc đó và bắt buộc quá trình ghép
phải kiểm toán được.

## Chuẩn bị trên Kaggle

1. Tạo notebook mới, import `pcb_solder_defect_kaggle.ipynb`.
2. **Add Input** ít nhất một dataset. Bắt đầu bằng SolDef_AI:
   `mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection`.
3. Nguồn không có trên Kaggle (Roboflow, Hugging Face): export/tải về máy rồi
   upload thành Kaggle Dataset của bạn, sau đó Add Input như bình thường.
4. Chọn **GPU T4 x2** hoặc mới hơn. Bật **Internet** để TorchVision tải ImageNet
   weights lần đầu.
5. Sửa `SOURCES` ở cell 1: bật nguồn nào có, sửa `root` cho khớp đường dẫn
   `/kaggle/input/...`.
6. Run All.

## Ba thứ notebook cưỡng chế, đừng tắt

**1. Nhãn không map được thì bỏ và đếm, không đoán.** Gộp `solder_ball` vào
`excess` là giấu một loại lỗi model chưa từng thấy sau một nhãn đạt. Nhãn lạ sẽ
hiện ở cell 5 kèm số lượng; thêm nó vào `LABEL_MAPS` hoặc vào `IGNORE` **kèm lý
do**, rồi chạy lại.

**2. Chia tập theo board, không theo crop.** Các crop cùng một board dùng chung
ánh sáng, tiêu cự và thao tác của cùng một người. Chia theo crop đặt các mẫu gần
trùng nhau ở cả hai phía và cho ra điểm số dây chuyền không bao giờ thấy.

**3. Lớp không đủ dữ liệu bị loại khỏi `class_names`.** Cell 6 in ma trận phủ và
loại lớp dưới `min_per_class`. Một head xuất ra lớp nó chưa từng thấy sẽ cho dự
đoán tự tin mà không có gì đằng sau.

Lớp bị loại **không mất trắng**: tầng luật của bước 6.2 vẫn bắt được chúng mà
không cần model — `escape_guard` cho thiếu thiếc, luật cặp chân cho bridge, luật
so hai đầu cho dựng bia.

## Đọc kết quả

Notebook báo **escape rate** và **false call rate**, không báo accuracy. Dây
chuyền 99.5% đạt thì cứ gọi tất cả là đạt đã được 99.5% accuracy trong khi bỏ lọt
toàn bộ lỗi.

Cell 8 quét ngưỡng accept. **Chọn ngưỡng có escape chấp nhận được trước, rồi mới
xét false_call** — bỏ lọt lỗi thì giao hàng lỗi, báo nhầm chỉ tốn 10 giây của
người kiểm. Sửa `decision_thresholds` trong manifest theo lựa chọn đó.

## Sau khi tải về

```powershell
# 1. Kiểm tra artifact nạp được trước khi tin
.\.venv\Scripts\python.exe scripts\verify_solder_model.py `
  models\solder\best.onnx models\solder\model_manifest.json

# 2. Hiệu chỉnh ngưỡng tầng luật theo board của bạn
.\.venv\Scripts\python.exe scripts\calibrate_solder_thresholds.py D:\board_dat `
  --model models\detector\kaggle\best.onnx --output config\solder_thresholds.json
```

`verify_solder_model.py` nạp cặp file qua **đúng runtime app dùng**, nên pass ở
đó nghĩa là app nạp được. Nó cũng bắt hai lỗi hay gặp: export bị tách trọng số ra
file `.data` riêng (app chỉ copy `.onnx` nên sẽ hỏng), và model gọi một land trơn
là "good" với độ tự tin cao (dấu hiệu thứ tự class bị hoán vị).

## Giới hạn lớn nhất: khoảng cách miền

Model train từ dataset công khai học camera, ống kính và ánh sáng của **người
khác**. Nó là điểm khởi đầu để fine-tune trên board của bạn, không phải model
production.

Bước 6.2 hợp nhất model với tầng luật đo chính vì lý do này: khi hai tầng bất
đồng, ROI đi vào hàng đợi kiểm tra thay vì tin model. Nguồn dữ liệu giá trị nhất
vẫn là export từ chính dây chuyền của bạn — xem
`scripts/export_solder_dataset.py`, rồi bật nguồn `local_export` trong `SOURCES`.

Và nhắc lại hai nút thắt vật lý đã nêu ở mục bước 5.5: **cold solder cần đèn vòng
RGB đa góc** mới tách được khỏi mối hàn tốt, và fillet cần ~15–25 µm/px mới đọc
được hình dạng. Không dataset nào bù được hai thứ đó.
