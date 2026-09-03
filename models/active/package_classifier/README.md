# Ô model bước 5.2 — phân loại package

Trống là **đúng**. Bước 5.2 hiện quyết bằng **luật**, không bằng model:
`aoi_pipeline/classification/package_rules.py`. Đường ONNX vẫn còn nguyên và
giữ làm chỗ dự phòng — thiếu artifact ở đây là **no-op tuyệt đối**, pipeline
chạy y như khi ô này không tồn tại.

## Nếu bạn thả model vào đây

Cần **hai** file, thiếu một là ô này từ chối nạp:

```
models/active/package_classifier/
├── best.onnx
└── model_manifest.json
```

Manifest phải khai đúng bảy lớp, đúng thứ tự trong
`aoi_pipeline/classification/package.py::PACKAGE_CLASS_NAMES`:

```
hai_chan · tru_dung · goi_nho · ic_hai_ben · ic_bon_ben · ic_khong_chan · connector
```

Sai thứ tự là mọi nhãn bị hoán vị mà không có gì báo, và nhãn package đổi được
hình học ROI ở bước 5.5 — nên đây là chỗ đoán sai thì hỏng im lặng.

## Có artifact cũng KHÔNG tự bật

Ô này nằm trong `app.streamlit_app._NO_AUTO_ADOPT`. Phải chọn/nạp bằng tay
trong sidebar. Lý do: một nhãn package sai làm 5.5 dựng ROI sai chỗ, mà chuyện
đó không hiện ra cho tới khi ai đó soi ảnh mối hàn.

Cổng để bật, ghi ở `Docs/ke_hoach/ke_hoach_phan_nhom_package.md` §8:

1. nhầm `ic` ↔ thụ động phải bằng **0** trên tập kiểm;
2. bật lên phải **không giảm** độ phủ 28 pad đếm tay ở
   `tests/data/solder_geometry`;
3. mặc định TẮT cho tới khi vượt cổng 2 trên board của chính dây chuyền.
