# Đánh giá model — người vận hành ghi lại chỗ model sai

Thư mục này giữ những lần người đứng máy nói *"model sai ở đây"*. Ghi từ trong
app: mở một bước (4 · 6.1 · 6.2), kéo xuống cuối trang, mục **"Đánh giá model"**.

## Vì sao nó tồn tại

Mọi bảng xếp hạng model trong `Docs/danh_gia/xep_hang_model.md` đến giờ đều đo trên **một
board chuẩn duy nhất**. Người thật sự nhìn thấy model sai ở đâu là người đứng
máy. Thư mục này là chỗ họ nói ra, và là thứ làm bảng xếp hạng chính xác dần
thay vì đứng yên.

## Định dạng: một dòng JSON cho mỗi bản ghi

`*.jsonl` — mỗi dòng là một bản ghi độc lập. Chọn kiểu này thay vì một mảng JSON
vì hai lý do đã đo được, không phải vì sở thích:

- **Ghi đồng thời.** Hai tab trình duyệt cùng bấm "Ghi nhận" là hai luồng trong
  cùng tiến trình Streamlit. Với một mảng JSON, mỗi lần ghi là đọc-sửa-ghi cả
  file và một người sẽ mất bản ghi trong im lặng.
- **Gộp git.** Hai người thêm hai dòng khác nhau thì `merge=union` gộp được máy
  móc. Một mảng JSON có dấu `]` là trạng thái chung, luôn xung đột.

Trùng lặp do merge là vô hại: mỗi bản ghi mang `entry_id` (uuid4) và bộ đọc gộp
theo id đó.

## Lưu toạ độ, không lưu ảnh

Bản ghi giữ **khung toạ độ** trong không gian `analysis_image_pixels`, cùng với
`sha256` của ảnh gốc và cấu hình tiền xử lý. Ảnh được **cắt lại khi cần xem**.

Hệ quả phải biết: **mất ảnh gốc là mất khả năng xem lại crop.** Bản ghi vẫn đọc
được (toạ độ, loại lỗi, ghi chú, model nào) nhưng không dựng lại được pixel. Nếu
ảnh đã đổi so với lúc ghi, app **từ chối** hiện crop thay vì hiện nhầm pixel —
người vận hành nhìn nhầm chỗ còn tệ hơn không nhìn gì.

Bản ghi **không** chứa đường dẫn ảnh. App giải mã ảnh thẳng vào bộ nhớ và không
ghi nó ra đĩa, nên một trường đường dẫn sẽ là bịa đặt trên mọi máy — kể cả máy
vừa ghi nó. Ngoài ra `AGENTS.md` cấm export đường dẫn tuyệt đối của máy.

## Mỗi bản ghi gắn với MỘT model cụ thể

Trường `model.sha256` lấy từ manifest. Đây là phần quan trọng nhất của định
dạng: tên file luôn là `best.onnx`, tên thư mục do người đặt, đường dẫn riêng
từng máy — chỉ `sha256` gắn với chính file trọng số.

Nhờ nó, đổi model không làm hỏng đánh giá cũ, và **hai model so được với nhau
trên cùng những lỗi đã báo**. Trong app, bản ghi thuộc model khác model đang
chạy được đánh dấu rõ.

## Muốn ghi ra chỗ khác

Đặt biến môi trường `AOI_FEEDBACK_DIR`. Test dùng đúng cách này để không ghi vào
repo.

```bash
AOI_FEEDBACK_DIR=D:/du-lieu/danh-gia streamlit run app/streamlit_app.py
```

## Đọc bằng code

```python
from aoi_pipeline.modelops.model_feedback import load_feedback, group_by_model

entries, problems = load_feedback()      # `problems` liệt kê dòng đọc không được
for key, group in group_by_model(entries).items():
    print(key[:12], len(group))
```

Một dòng hỏng (tiến trình bị giết giữa lúc ghi) được báo ở `problems` và bỏ qua,
chứ không làm cả lịch sử thành không đọc được.
