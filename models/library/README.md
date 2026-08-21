# models/library — chỗ bạn tự để model

Bỏ file `.onnx` cùng `model_manifest.json` của nó vào đây (một thư mục con cho
mỗi model là gọn nhất). App sẽ tự thấy và liệt kê nó cạnh các model đang dùng.

Git **bỏ qua** toàn bộ thư mục này trừ chính file README, nên:

- model bạn để ở đây không bị đẩy lên GitHub, không đụng giới hạn 100 MB
- `git pull` không bao giờ xung đột với model của bạn
- muốn xoá thì xoá thẳng, không cần đụng tới Git

## Ba thư mục model, khác nhau ở chỗ ai sở hữu

| Thư mục | Ai sở hữu | App tự nạp? | Có trong Git? |
|---|---|---|---|
| `models/active/` | dự án | **có** | có |
| `models/archive/` | dự án, bản cũ | không | có |
| `models/library/` | **bạn** | không, nhưng có trong danh sách chọn | không |

`archive/` cố tình không được tự nạp: một model đã bị thay thế thì không nên
chỉ cách một cú nhấp nhầm là chạy trên board thật.

## Cần gì để một model được liệt kê

Phải có `model_manifest.json` nằm cạnh. Bước 6.1 và 6.2 đều từ chối nạp khi
thiếu contract — một file ONNX không biết thứ tự lớp thì phải đoán, mà đoán sai
là mọi lỗi thành "đạt". Liệt kê file trần chỉ dẫn tới một lần thất bại muộn hơn.

File `.pt` cố tình không được liệt kê: nó chứa pickle, app chặn cho tới khi bạn
tự xác nhận nguồn, và một danh sách chọn sẵn sẽ biến xác nhận đó thành hình thức.
