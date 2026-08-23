# Thư mục model

## Làm sao biết model nào là bản nào?

Mọi artifact đều tên `best.onnx` — đó là quy ước của Ultralytics và của các
notebook, không đổi được mà không phá đường nạp. Nên **đừng nhìn tên file**.
Có ba cách xem, xếp theo độ tiện:

```bash
python scripts/list_models.py                 # bảng đầy đủ
python scripts/list_models.py --kind detector # một bước
python scripts/list_models.py --json          # cho script khác dùng
```

```
bước       nguồn      thư mục                          kiến trúc        ngày        điểm            MB
classifier đang dùng  classifier                       efficientnet_b0  2026-08-18  acc 0.958       16
classifier của bạn    classifier-convnext_base-...     convnext_base    2026-08-22  acc 0.954      351
detector   đang dùng  detector                         yolo26s          2026-08-20  mAP50 0.505     39
detector   bản cũ     detector-yolo26s-20260817        yolo26s          2026-08-17  mAP50 0.579     39
```

**Trong app**, bộ chọn ở sidebar hiện đúng những cột đó:

```
detector — yolo26s · 2026-08-20 · mAP50 0.505  (đang dùng)
detector-yolo26s-20260817 — yolo26s · 2026-08-17 · mAP50 0.579  (bản cũ)
```

**Trực tiếp**, mở `model_manifest.json` cạnh file `.onnx`. Đây là nguồn đáng
tin nhất: tên thư mục do người đặt và có thể lệch, manifest do notebook sinh ra
**cùng lúc** với trọng số và mang cả `sha256` của chính file.

## Ba thư mục, chia theo *ai sở hữu file*

| | ý nghĩa | git |
|---|---|---|
| `active/` | Model app tự nạp **khi mở phiên mới**. Một thư mục cho mỗi bước | commit kèm |
| `archive/` | Bản cũ giữ để đối chiếu. **Không bao giờ tự nạp** | commit kèm |
| `library/` | Của bạn. Thả vào là hiện trong bộ chọn | **bỏ qua** |

`active/` dùng **tên bước** (`classifier/`, `detector/`, `solder/`) chứ không
phải tên phiên bản — vì câu hỏi ở đó là "cái gì đang chạy", và app tìm mặc định
theo đúng đường dẫn `active/<bước>/best.onnx`. Đổi tên các thư mục này sẽ làm
app không tìm thấy model mặc định nữa.

Việc nạp diễn ra **một lần cho mỗi phiên**, không phải mỗi lần chạy lại. Nhờ
thế nút "Gỡ model" vẫn gỡ được: nạp lại ở mọi rerun thì bấm gỡ xong model quay
về ngay, và nút trông như hỏng. Muốn nạp lại sau khi gỡ thì chọn nó trong ô
"Chọn từ thư mục models/".

`archive/` và `library/` dùng **`<bước>-<kiến trúc>-<ngày>`**:

```
detector-yolo26s-20260817
detector-yolov8-huggingface-20260704
classifier-convnext_base-20260822
```

Đây chỉ là tiện lợi khi mở File Explorer. Bộ chọn và `list_models.py` không đọc
tên thư mục — chúng đọc manifest, nên đặt tên sai cũng không làm hỏng gì.

## Muốn thêm model của mình

Bỏ vào `library/` một thư mục chứa **hai** file:

```
models/library/<tên gì cũng được>/
    best.onnx
    model_manifest.json
```

**Thiếu manifest thì model không hiện trong bộ chọn.** Không phải khó tính vô
cớ: bước 6.1 và 6.2 cần biết thứ tự lớp, kích thước đầu vào và cách chuẩn hoá.
Đoán sai thứ tự lớp nghĩa là mọi lỗi bị ánh xạ thành "đạt" — hỏng im lặng, đúng
kiểu tệ nhất. Chào một file không manifest chỉ dời thất bại sang một cú click
sau.

File `.pt` cũng không được liệt kê: nó mang pickle, app chặn lại tới khi có
người xác nhận nguồn, và một bộ chọn mặc định sẵn nó biến việc xác nhận thành
hình thức.

### Không có manifest thì làm sao?

- **Model do notebook của dự án train:** notebook tự ghi manifest. Nếu cell xuất
  ONNX hỏng thì dựng lại từ checkpoint:
  ```bash
  python scripts/export_classifier_onnx.py best_state.pt \
      --out models/library/<tên> --temperature <giá trị ở bước 5>
  ```
- **Model tải từ ngoài về:** file ONNX của Ultralytics mang sẵn tên lớp, ngày và
  `imgsz` trong `metadata_props`. `detector-yolov8-huggingface-20260704` có
  manifest sinh theo cách đó — xem trường `provenance` trong đó, nó ghi rõ là
  **không có số đo trên tập val nào**, chỉ dùng để đối chiếu.

## Model nào đang tốt nhất?

**`Docs/xep_hang_model.md`** — xếp hạng theo từng bước, kèm dòng "nên bật cái
nào". Mọi số ở đó đo trên board thật, không chỉ đọc manifest.

Tóm tắt của lần đo lại 2026-08-23 (5 ảnh board, một lần chạy duy nhất):

- **Detector đang bật là bản kém nhất trong ba bản** — 121 box so với 203 và
  213, và **không tìm được gì trên 2/5 ảnh** trong khi hai bản kia vẫn ra 20 và
  8 box.
- **Classifier: cả hai bản đều yếu trên crop thật** — chỉ tự động chấp nhận
  4,2% và 12,2% số crop. convnext nhỉnh hơn nhưng chậm ~33 lần.
- **Model mối hàn 6.2 không phân biệt được mối hàn với mảnh board bất kỳ**:
  gọi `bridge` cho 50,2% ROI thật và 50,3% mảnh ngẫu nhiên.

Muốn kiểm lại thì chạy trên board của bạn — mọi con số trong bảng phải khớp:

```bash
python scripts/benchmark_models.py <thư-mục-kết-quả> <ảnh1> <ảnh2> ...
```

Điểm yếu còn lại của bảng: **không có nhãn người**. Thứ làm nó chính xác dần là
mục **"Đánh giá model"** ở cuối trang mỗi bước trong app — xem
`feedback/README.md`.

## Ghi chú: chế độ `end2end` của detector

Artifact do notebook của repo tạo ra dùng `end2end=False` (one-to-many + NMS),
và UI ghim đúng chế độ đó. Model dùng trực tiếp qua Python adapter vẫn để
`end2end=None` mặc định, để tôn trọng metadata/head của chính model.

Đã đo 2026-08-22 trên detector đang dùng (head end-to-end, `output [1, 300, 6]`):
đặt `end2end` thành `None`, `True` hay `False` đều cho **đúng 36 box như nhau**
— ultralytics tự nhận dạng head từ đồ thị ONNX. Nên cờ này không phải chỗ cần
chỉnh khi một detector cho kết quả kém.
