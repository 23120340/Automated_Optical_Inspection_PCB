# Model v2 có hơn model cũ không?

> Đo 2026-08-22. Câu trả lời **khác nhau cho hai model**: detector v2 là một
> bước lùi, classifier v2 là một cải tiến thật nhưng đắt.

## Kết luận

| | Kết luận | Hành động đề xuất |
|---|---|---|
| **Detector ver2** | **Kém hơn ver1 trên mọi phép đo lấy được** | Đưa ver1 trở lại `models/active/detector/` |
| **Classifier v2** | **Tốt hơn ở mọi chỉ số chất lượng**, nhưng chậm gấp 26 lần | Đổi nếu thời gian chu kỳ cho phép; nếu không thì train lại với backbone nhỏ |

---

## 1. Detector: ver2 là bước lùi

### Chính manifest của nó đã nói

| | ver1 | ver2 |
|---|---|---|
| mAP50 | **0.5788** | 0.5052 |
| mAP50-95 | **0.2874** | 0.2310 |

Hai con số này *có thể* không cùng tập val (ver1 dùng schema cũ và không ghi
per-class), nên một mình chúng chưa đủ kết luận. Nhưng chúng chỉ cùng hướng với
phép đo dưới đây, vốn không phụ thuộc vào tập val nào.

### Trên board thật của dự án

Cùng một tile 1024² cắt giữa board:

| | ver1 | ver2 | huggingface |
|---|---|---|---|
| số linh kiện tìm được | **64** | 36 | 61 |
| confidence trung vị | 0.375 | 0.431 | 0.648 |
| thời gian | 1.79 s | 2.49 s | **0.24 s** |

Trùng box giữa ver1 và ver2: 27. Trong đó **25/27 (93%) cùng nhãn** — nên hai
bản không bất đồng về *cái chúng cùng thấy*; vấn đề là ver2 **không thấy**.

Lặp trên nhiều ảnh khác nhau:

| ảnh | ver1 | ver2 | `display` sai của ver2 |
|---|---|---|---|
| golden.png | **64** | 36 | 3 |
| sample.jpg | **67** | 49 | 6 |
| golden-image.jpg | **64** | 36 | 3 |
| 9e6aa662…jpg | **8** | 0 | 0 |

### Hạ ngưỡng không cứu được

ver2 dùng head **end-to-end** (`output [1, 300, 6]`, không NMS) còn ver1 là head
YOLO thường (33600 candidate + NMS), nên so ở cùng ngưỡng 0.25 có thể không công
bằng. Đã quét:

| conf | ver1 | ver2 | `display` của ver2 |
|---|---|---|---|
| 0.50 | **18** | 11 | 0 |
| 0.35 | **38** | 25 | 2 |
| 0.25 | **64** | 36 | 3 |
| 0.15 | **104** | 66 | 5 |
| 0.10 | **139** | 107 | 8 |
| 0.05 | **199** | 168 | **16** |
| 0.02 | 288 | 295 | **32** |

ver1 dẫn ở mọi ngưỡng cho tới khi ver2 chạm trần 300 box. Và thứ ver2 thêm vào
khi hạ ngưỡng chủ yếu là `display` sai. Cờ `end2end` (`None`/`True`/`False`)
không đổi kết quả — vẫn đúng 36 box.

### `display` là ảo giác, đã xác minh bằng mắt

`display` **chỉ ver2 nhìn thấy** — ver1 và huggingface đều 0. Vẽ ra xem thì cả
ba box nằm trên **tụ hoá và cuộn cảm** của một board nguồn 115Vac không có màn
hình nào:

```
display conf=0.47  (726,  1)-(865,  98)   -> tụ film
display conf=0.41  (895, 12)-(1023, 345)  -> tụ hoá + cuộn cảm L1
display conf=0.27  (388,897)-(513, 1024)  -> hai tụ hoá C13
```

Đáng lo hơn: 3 trong 4 box **lớn nhất** của ver2 chính là `display` sai. Trong
khi ver1 tìm được 9 linh kiện lớn (≥80 px) gồm 2 connector và 3 IC — và nhìn
overlay thì ver2 **bỏ sót cả hai transistor công suất DPAK to nhất khung**
(V1, V3).

### ver2 cũng không giúp gì cho lượt 2

Lý do hợp lý duy nhất để giữ ver2 là hai lớp `pads`/`pins` mà lượt 2 cần. Đã đo:
**cả ba detector đều tìm được 0** `pads`/`pins` trên board thật. Kết luận cũ
trong `Docs/dataset_lead_detection.md` giữ nguyên.

---

## 2. Classifier: v2 tốt hơn thật, nhưng đắt

### Đính chính

Ngày 2026-08-22 tôi từng nói manifest classifier cũ **không ghi số đo test**.
Sai — nó có, nằm ở nhánh `metrics` chứ không phải `training`:

| | cũ (efficientnet_b0) | v2 (convnext_base) |
|---|---|---|
| accuracy (test) | 0.9580 | 0.9539 |
| macro | **macro_f1** 0.8903 | **macro_recall** 0.9326 |
| số lớp | 16 | 17 |

**Hai cột này không so thẳng được**: khác tập test, khác bộ lớp, và hai chỉ số
macro là hai đại lượng khác nhau (f1 so với recall). Đừng đọc "0.9580 > 0.9539"
thành "bản cũ chính xác hơn".

### Điều so được, và nó rõ ràng

**Từ vựng khớp với detector** — đây là cải tiến lớn nhất:

| | trùng tên với detector |
|---|---|
| classifier cũ | **8/16** |
| classifier v2 | **17/17** |

Bản cũ có các lớp gộp (`discrete_semiconductor`, `magnetic`, `timing`,
`protection`, `acoustic`, `switch_control`) mà detector không có tên tương ứng,
nên **8 trong 16 lớp không đối chiếu chéo được**. Bản v2 dùng đúng tên linh kiện
của detector, nên mọi lớp đều kiểm chéo được. Đây chính là thứ trả lời câu hỏi
"detector và classifier đã hỗ trợ nhau chưa".

Bản cũ có thêm lớp `false_crop_background` mà v2 không có. Nghe như một mất mát,
nhưng đã đo: lớp đó **chỉ bắn vào vùng xám phẳng**; với nhiễu ngẫu nhiên nó gọi
`connector` 98%, với mảnh board bất kỳ nó gọi `capacitor`. Nó không phải bộ lọc
"không phải linh kiện" đáng tin.

### Trên 64 crop thật cắt từ board

| | cũ | v2 |
|---|---|---|
| dung lượng | **16 MB** | 351 MB |
| ms mỗi crop | **13.6** | 353.6 |
| confidence trung vị | 0.868 | **0.927** |
| confidence phân vị 10 | 0.468 | **0.557** |
| tự động chấp nhận (≥0.85) | 33/64 (52%) | **42/64 (66%)** |
| đồng ý với detector | 43/64 (67%) | **46/64 (72%)** |

Xem các ca bất đồng thì v2 hợp lý hơn rõ:

| detector nói | cũ nói | v2 nói |
|---|---|---|
| `ic` | `relay` 0.46 | **`ic` 0.98** |
| `ic` | `display` 0.51 | **`ic` 0.99** |
| `diode` | `discrete_semiconductor` 0.51 | **`diode` 0.74** |

### Cái giá: thời gian chu kỳ

Board này có **108 linh kiện** (đếm bằng tile chồng lấn):

| | mỗi board |
|---|---|
| classifier cũ | **1.5 s** |
| classifier v2 | **38.2 s** |

Đo trên CPU của máy này. GPU sẽ rút ngắn nhiều, nhưng tỉ lệ 26× giữa hai model
thì không đổi. **Đây là con số quyết định có đổi hay không**, không phải accuracy.

---

## 3. Đề xuất

1. **Đưa detector ver1 về `models/active/detector/`.** ver2 kém hơn ở mọi phép
   đo, sinh ảo giác `display` trên tụ, và không cho thêm gì cho lượt 2.
2. **Classifier v2: đổi nếu 38 s/board chấp nhận được.** Chất lượng hơn thật, và
   việc khớp từ vựng với detector mở ra kiểm chéo trên toàn bộ lớp.
3. **Nếu 38 s là quá lâu:** train lại notebook v2 với `convnext_tiny` hoặc
   `efficientnet_v2_s` thay cho `convnext_base`. Giữ nguyên taxonomy 17 lớp —
   đó mới là phần giá trị, không phải backbone. Notebook hỗ trợ resume nên chỉ
   cần đổi `CONFIG["model_name"]`.
4. **Trước khi train lại detector:** nguyên nhân ver2 kém chưa rõ (imgsz 1536,
   150 epoch, oversample rare 6× — đáng lẽ phải tốt hơn). Nên xem lại tập val
   của run đó trước khi tiêu thêm giờ GPU.
