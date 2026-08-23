# Khảo sát model "PCB defect" trên Hugging Face

> ## ⚠ ĐỌC PHẦN ĐÍNH CHÍNH Ở CUỐI TRƯỚC
>
> **Mục 1 dưới đây (loại keremberke) là SAI.** Đo lại cùng ngày cho thấy model
> đó *có* đọc được lỗi hàn thật ở đúng thang chụp của dự án. Phần đầu được giữ
> nguyên để thấy sai ở đâu, không phải để dùng.
>
> Nhảy thẳng tới **"ĐÍNH CHÍNH 2026-08-23"** ở cuối file.

> Đo 2026-08-23. Dựa trên bảng so sánh 52 model do bạn tổng hợp
> (`huggingface.co/models?search=pcb+de`).
>
> Quy ước như các khảo sát trước: mục nào **đã tải về chạy thật** thì ghi rõ đo
> bằng cách nào; mục nào chỉ đọc tài liệu thì ghi là chưa kiểm chứng.

## Kết luận trước

**Không model nào trong 52 kết quả dùng được cho dự án này.** Đã tải về và chạy
thật hai ứng viên tốt nhất; ba nhóm còn lại loại được mà không cần tải.

Điều này **không** mâu thuẫn với bảng so sánh của bạn. Bảng ấy xếp hạng đúng
theo tiêu chí "model nào có tài liệu và số đo rõ ràng nhất" — và bộ ba
keremberke đúng là nhóm duy nhất đạt tiêu chí đó. Câu hỏi khác nằm ở chỗ khác:
**số đo của họ có chuyển sang ảnh của dây chuyền này không.**

## 1. keremberke YOLOv8n/s/m — đã tải, đã chạy, KHÔNG chuyển được

Nhóm duy nhất có mAP công bố và dataset rõ ràng. Lớp `Dry_joint` và
`Short_circuit` đúng là hai thứ bước 6.2 đang thiếu, nên đây là ứng viên đáng
giá nhất.

Tải `best.pt` của bản `n` và `m`. **Nạp được bằng ultralytics 8.4.104 của dự
án** dù chúng train bằng 8.0.21/8.0.23 — không cần hạ cấp thư viện như model
card hướng dẫn (`ultralytics==8.0.23`); làm theo sẽ phá môi trường dự án.

### Đối chứng: model chạy tốt trên ảnh của chính họ

| | conf 0.25 | conf 0.10 |
|---|---|---|
| yolov8n | 5 box | 7 box |
| yolov8m | 2 box | 2 box |

Model không hỏng. Mọi kết quả dưới đây là về **ảnh**, không phải về model.

### Trên board thật của dự án (tile 1024², 46 µm/px)

| conf | yolov8n | yolov8m |
|---|---|---|
| 0.25 | 1 box | **0 box** |
| 0.10 | 2 box | 1 box |
| 0.05 | 6 box | 4 box |
| 0.01 | 33 box | 115 box |

Board này là **board chuẩn** (`golden.png`), tức mọi box đều là báo động nhầm.
0 box ở conf 0.25 nghe như kết quả đúng — nhưng đó cũng là kết quả mà một model
**không thấy gì cả** sẽ cho. Hai trường hợp này khác hẳn nhau và phải tách ra.

### Phép tách: vẽ lỗi nhân tạo lên chính board đó

Nối 6 cặp linh kiện kề nhau bằng một vệt thiếc — đúng định nghĩa lớp
`Short_circuit`.

| | board chuẩn | board có 6 chỗ chập |
|---|---|---|
| yolov8n @ 0.05 | 6 box, **0** `Short_circuit` | 6 box, **0** `Short_circuit` |
| yolov8m @ 0.05 | 4 box, **0** `Short_circuit` | 5 box, **0** `Short_circuit` |

**Đầu ra gần như không đổi khi thêm 6 mối hàn chập rõ ràng.** Các box nó cho ra
không liên quan gì tới lỗi thật.

> Giới hạn của phép thử này: vệt thiếc của tôi là đường vẽ, thiếu phản xạ bóng
> 3D của mối hàn thật. Nên nó có thể cho âm tính giả. Nhưng cộng với hai bằng
> chứng dưới đây thì kết luận vẫn đứng.

### Các box nó cho ra trông thế nào

Box lớn nhất trên board dự án (`Incorrect_installation`, conf 0.17) rộng
**730×510 px** — phủ cả biến áp và vài chục linh kiện. Đó không phải định vị
lỗi; đó là model thấy "một vật tối lớn" rồi gán nhãn. Một box cỡ đó không nói
được gì cho người vận hành.

### Vì sao — đọc từ chính dataset

Tải `train.zip` và đếm trong `_annotations.coco.json`:

| Lớp | Instance | Tỉ lệ |
|---|---|---|
| `incorrect_installation` | **220** | 67% |
| `short_circuit` | 49 | 15% |
| `dry_joint` | **44** | 13% |
| `pcb_damage` | 13 | 4% |
| **Tổng** | **326** trên **128 ảnh** | |

Ba điều giải thích toàn bộ hành vi quan sát được:

1. **Mất cân bằng 17×.** Model dự đoán `Incorrect_installation` cho gần như mọi
   thứ vì đó là 2/3 dữ liệu train. Đúng như đã thấy: chưa lần nào nó gọi
   `Dry_joint`, kể cả trên ảnh của chính họ.
2. **`Dry_joint` chỉ có 44 instance.** Cùng bậc với 30 ảnh `pads`/`pins` đã làm
   detector của dự án chỉ đạt recall 0.072. 44 instance không dạy được một lớp.
3. **Mọi ảnh đều 640×480.** Cùng con số đã loại PCB-SAID. Dataset trộn hai thang
   chụp — thumbnail cho thấy vừa ảnh macro cực gần vừa ảnh mức board chụp bằng
   webcam (tên file `WIN_2022...`) — nên model không học được một thang nhất
   quán nào.

mAP50 = 0.568 của bản `m` là số thật, nhưng là trung bình trên tập 189 ảnh
640×480 với mất cân bằng 17×, và bị lớp đa số chi phối.

**Dùng được vào việc gì:** không cho suy luận. Dataset 189 ảnh không khai giấy
phép (`license: None`, export từ Roboflow) nên cũng không dùng để train được cho
tới khi hỏi được tác giả.

## 2. Dukeb/detr-detection-PCBComponents — loại, không cần tải trọng số

Đọc `config.json`: 24 lớp, tên là **`LABEL_0` … `LABEL_23`**.

Model **không mang tên lớp**. Kể cả nó detect chính xác tuyệt đối, bạn cũng
không biết `LABEL_7` là điện trở hay connector, và bảng ánh xạ không được công
bố. Cả đường ống của dự án đều khoá theo tên lớp — hình học chân ở bước 5.5,
phân loại 6.1, đối chiếu BOM. Không có tên lớp thì không có gì để nối vào.

Loại được bằng một file 194 byte, không cần tải `pytorch_model.bin` và không
cần cài `transformers`.

## 3–6. Các nhóm còn lại

| Nhóm | Vì sao loại |
|---|---|
| DeepVisionXplain (6 backbone) | **Không có model card.** Không xác nhận được task, dataset hay hiệu năng |
| 8 model không tài liệu | Như trên |
| DeepSeek-R1-\*-PCB (8B) | **LLM văn bản**, không phải thị giác máy tính. Model card là template mặc định của Unsloth |
| gemma-pcb_product-description | Sinh mô tả bán hàng. Không liên quan phát hiện lỗi |

Bạn viết "loại nào tôi thấy cũng có thể áp dụng vô dự án được" — với nhóm 5 và
6 thì không: chúng nhận **văn bản** vào và trả **văn bản** ra, không có chỗ nào
nhận được một tấm ảnh board.

## Điều đáng rút ra

Ba khảo sát độc lập giờ chỉ về cùng một chỗ:

| Nguồn | Vì sao không dùng được |
|---|---|
| SolDef_AI | Đúng nhãn từng mối hàn, **sai tỉ lệ 20 lần** (macro 1–3 µm/px) |
| PCB-SAID | Ảnh cào web **640×480**, nhãn theo linh kiện |
| Ulger | **Đúng tỉ lệ** (~20–25 µm/px) nhưng **không có box** |
| keremberke | 189 ảnh **640×480**, mất cân bằng 17×, không chuyển được |

Không phải "chưa tìm đủ kỹ". Ràng buộc là vật lý: **ảnh của dự án là 46 µm/px,
và không nguồn công khai nào vừa đúng thang đó vừa có nhãn định vị.**

Việc cần làm không đổi, và vẫn là việc tốn công nhất: **gán nhãn board của
chính dây chuyền** (mục C1–C3 trong `Docs/tien_do_detect_2_luot.md`). Nên gán
nhãn tốt/xấu cho 6.2 **chung một lượt** với khoanh box cho lượt 2 — khi đã
khoanh box từng chân rồi thì gán thêm nhãn chất lượng rẻ hơn nhiều so với làm
hai đợt.

## Nếu muốn tự kiểm lại

Trọng số đã tải nằm trong thư mục tạm của phiên làm việc, không commit vào repo
(61 MB, và không dùng được). Tải lại:

```bash
curl -L -o yolov8m-pcb.pt \
  https://huggingface.co/keremberke/yolov8m-pcb-defect-segmentation/resolve/main/best.pt
```

Nạp bằng `ultralytics` sẵn có của dự án — **đừng** cài `ultralyticsplus` /
`ultralytics==8.0.23` theo model card, nó sẽ hạ cấp thư viện và phá bước 4.

Lưu ý `.pt` mang pickle: nạp là chạy mã. Repo này 1000+ lượt tải/tháng nên rủi
ro thấp, nhưng đó là lý do bộ chọn model của app **không liệt kê `.pt`**.

*Ghi chú: trang tìm kiếm còn 1 trang nữa (23 model) chưa khảo sát. Dựa trên
phân bố của 52 kết quả đầu, kỳ vọng hợp lý là chúng thuộc nhóm 3–6.*

---

# ĐÍNH CHÍNH 2026-08-23 — kết luận ở trên SAI về keremberke

Người dùng phản biện: "có thể là do ảnh test của tôi chứ không phải do model,
vì sau này sẽ dùng camera quét". Phản biện đúng, và khi đem kiểm chứng thì
**kết luận loại keremberke ở mục 1 là sai**.

## Ba sai lầm trong phần khảo sát ở trên

**1. Nhầm số điểm ảnh với thang chụp.** Tôi xếp keremberke chung với PCB-SAID
vì "cùng 640×480". Nhưng 640×480 nói về *số điểm ảnh*, không nói về *µm/px*.
Một khung 640×480 chụp một vùng nhỏ vẫn rất mịn.

Đo lại bằng pitch chân SOIC 1.27 mm — đúng thước đã dùng cho board dự án:

| | µm/px |
|---|---|
| keremberke | **~33** |
| Board dự án hiện tại | 46 |
| Camera mục tiêu | 25 |

Ảnh của họ **nằm giữa** hiện tại và mục tiêu, không phải macro.

**2. Đọc "0 box trên board chuẩn" thành thất bại.** `golden.png` là board
**chuẩn** — không có lỗi để tìm. Không ra box chính là hành vi *đúng*. Tôi đã
lấy nó làm bằng chứng model hỏng.

**3. Phép thử lỗi nhân tạo quá thô.** Tôi vẽ đường thẳng để giả làm mối hàn
chập. Model học từ **ảnh chụp**, chưa từng thấy nét vẽ, nên nó không phản ứng —
và điều đó không nói gì về khả năng bắt lỗi thật.

## Đo lại cho đúng

### Model chịu được thang chụp tới đâu

Lấy chính tập test của họ (79 lỗi có nhãn), thu nhỏ dần để mô phỏng camera thô
hơn. Thu nhỏ một ảnh mịn mô phỏng đúng một camera thô; phóng to thì không tạo
ra chi tiết, nên chỉ làm được một chiều.

| Hệ số | µm/px | Recall |
|---|---|---|
| 1.00 | 33 (gốc) | 0.595 |
| 0.72 | **46 (board bạn)** | **0.544** |
| 0.50 | 66 | 0.494 |
| 0.36 | 92 | 0.430 |

**Suy giảm rất từ tốn.** Ở 46 µm/px model vẫn giữ 0.544, chỉ kém gốc 9%. Nếu
thang chụp là nguyên nhân thì recall đã sụp ở đây. Nó không sụp.

### Model có bắt được lỗi THẬT ở thang của bạn không

Lấy 6 ảnh chụp thật chứa 38 lỗi thật, thu về đúng 46 µm/px, dán vào giữa board
dự án:

| Model | box | trong vùng có lỗi |
|---|---|---|
| **keremberke yolov8m** | 36 | **36 (100%)** |
| keremberke yolov8n | 31 | 25 |
| SolDef_AI | 6 | 6 (toàn `spike`) |
| detector dự án *(đối chứng)* | 280 | 41 — nó tìm linh kiện, đúng vai |

`yolov8m` đặt **toàn bộ** box vào đúng vùng có lỗi. Và nó **có** gọi
`Dry_joint` (5 box ở conf 0.25 trên một ảnh) — lớp mà mục 1 ở trên nói "chưa
lần nào xuất hiện". Câu đó sai vì tôi chỉ cho nó xem board không có lỗi.

## Kết luận đã sửa

**Thang chụp không phải rào cản** — cả 46 µm/px hiện tại lẫn 25 µm/px tương
lai. keremberke yolov8m **đọc được lỗi hàn thật ở đúng thang chụp của bạn**.

**Nhưng vẫn còn một điều chưa chứng minh được:** liệu nó có chạy trên board
*của bạn*, chụp bằng camera *của bạn*, với lỗi *thật của dây chuyền* hay không.
Mọi phép đo trên đều dùng ảnh của chính họ — cùng camera, cùng ánh sáng, cùng
loại board. Đó là câu hỏi **miền ảnh**, và không có board lỗi thật thì không
trả lời được.

Điều mất cân bằng lớp (17×, `dry_joint` chỉ 44 instance) và dataset 189 ảnh vẫn
đúng, và vẫn là lý do đừng kỳ vọng nó thay thế được một model train trên dữ
liệu của chính dây chuyền. Nhưng chúng là lý do để **kỳ vọng vừa phải**, không
phải lý do để loại.

## Việc nên làm — 10 phút, và nó quyết định

Chụp **một board có lỗi thật** bằng camera hiện tại, rồi:

```bash
curl -L -o yolov8m-pcb.pt \
  https://huggingface.co/keremberke/yolov8m-pcb-defect-segmentation/resolve/main/best.pt
```

```python
from ultralytics import YOLO
model = YOLO("yolov8m-pcb.pt")          # ultralytics 8.4.104 nạp được
for result in model.predict("board_loi.jpg", conf=0.25, imgsz=640):
    result.save("ket_qua.jpg")
```

- **Nó khoanh trúng lỗi** → có một điểm khởi đầu miễn phí, và việc gán nhãn
  chuyển từ "vẽ từ đầu" sang "sửa box có sẵn" — rẻ hơn nhiều.
- **Nó không thấy gì** → mới là lúc kết luận miền ảnh chặn, và tự gán nhãn là
  đường duy nhất.

Đừng cài `ultralyticsplus` / `ultralytics==8.0.23` theo model card — sẽ hạ cấp
thư viện và phá bước 4.

## Còn về camera

Nâng cấp camera vẫn đáng, nhưng **không phải để cứu các model này** — chúng đã
chạy được ở 46 µm/px. Lý do nâng cấp nằm ở chỗ khác, đã ghi trong
`Docs/yeu_cau_phan_cung_camera.md`: kiểm tra fillet cần 15–25 µm/px, và hàn
nguội cần **hướng chiếu sáng** đúng chứ không chỉ độ phân giải — "đèn chiếu
phẳng thì mối hàn tốt và mối hàn nguội trông như nhau ở bất kỳ độ phân giải
nào".
