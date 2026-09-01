# Package classifier (bước 5.2)

Thư mục này **chưa có model đã huấn luyện**. Đây là chủ ý: bước 5.2 là no-op
tuyệt đối cho tới khi có đủ dữ liệu 7 lớp và artifact vượt cổng nghiệm thu.

Một artifact hợp lệ phải đặt đồng thời hai file:

```text
models/active/package/
├── best.onnx
└── model_manifest.json
```

Manifest phải dùng schema `pcb-package-classifier/1.0`, task
`component_package_classification`, input RGB letterbox `128×128`, và đúng thứ
tự lớp:

```text
hai_chan, tru_dung, goi_nho, ic_hai_ben,
ic_bon_ben, ic_khong_chan, connector
```

Không chép model vào đây ngay sau khi train. Trước hết chạy:

```powershell
python scripts/evaluate_package_roi_gate.py `
  <thu-muc-artifact>/best.onnx `
  <thu-muc-artifact>/model_manifest.json `
  --output <thu-muc-artifact>/promotion_gate.json
```

Model chỉ đủ điều kiện để **người dùng tự promote** khi báo cáo đạt cả ba cổng:
macro recall test theo board ≥ 0,85; nhầm `ic_hai_ben ↔ ic_khong_chan` bằng 0;
và bật model không làm giảm độ phủ 28 pad đếm tay trên board thật. Registry/UI
không tự nhận model package kể cả khi có file trong `active/`.

Việc còn cần con người trước khi có thể train được ghi trong
[`docs/ke_hoach/ke_hoach_phan_nhom_package.md`](../../../docs/ke_hoach/ke_hoach_phan_nhom_package.md):
chọn nhãn 1–7 cho các box còn `unknown`, đặc biệt bổ sung bốn lớp hiện thiếu dữ
liệu là `tru_dung`, `ic_bon_ben`, `ic_khong_chan`, `connector`.
