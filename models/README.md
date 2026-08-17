# Component detector models

Đặt model bước 4 đã train trong thư mục này, ví dụ:

```text
models/
├── best.onnx
├── best.pt
└── model_manifest.json
```

Ứng dụng cũng cho phép upload model từ giao diện. Ưu tiên `best.onnx` khi chỉ
cần inference; chỉ mở `.pt` do chính notebook Kaggle của dự án tạo ra vì
checkpoint PyTorch là định dạng có thể thực thi mã khi nạp.

Các file trọng số được `.gitignore`; manifest, class map và metric nên được giữ
lại để truy vết phiên bản model.

Artifact do notebook của repo tạo ra dùng `end2end=False` (one-to-many + NMS),
và UI đã ghim đúng chế độ đó. Model khác dùng trực tiếp qua Python adapter vẫn để
`end2end=None` theo mặc định để tôn trọng metadata/head của chính model.
