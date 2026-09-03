# Promote detector thân linh kiện — 2026-09-03

Đổi ô `models/active/detector/` từ detector **22 lớp** sang detector **1 lớp**
chỉ khoanh thân linh kiện. Đây không phải bản nâng cấp cắm-là-chạy: nó đổi hợp
đồng nhãn của cả pipeline, và phải vá một chỗ trước khi promote được.

Bản 22 lớp lưu ở `models/archive/detector-yolo26s-consolidated22-ver1/`.

---

## 1. Model mới tốt hơn ở đâu

| | val | test |
|---|---:|---:|
| mAP50 | 0,711 | **0,840** |
| mAP50-95 | 0,372 | 0,465 |
| precision | 0,833 | 0,804 |
| recall | 0,672 | **0,844** |

Ba cổng của notebook đều ĐẠT, gồm cổng cận dưới:

> **cận dưới recall theo bo = 0,744 > detector đang chạy 0,54**

Cận dưới là con số đáng tin ở đây, không phải 0,844: test chỉ có **3 bo vật
lý**, nên khoảng tin cậy khi lấy mẫu lại theo bo là 0,744–0,935.

Recall từng bo — test: 0,744 / 0,827 / 0,935. Val: 0,455 / 0,856 / 0,926. Bo
kém nhất của val (`board017`) chiếm 49,5% thước đo val, nên val (0,672) thấp
hơn test (0,844) là do thành phần bo, không phải do model tệ trên val.

Chạy chính `best.onnx` trên 11 tile test: 640 box thật → 776 detection, không
tile nào chạm trần.

---

## 2. Vì sao KHÔNG phải bản thay thế cắm-là-chạy

Model cũ có 22 lớp; model mới có **một** lớp `component`. Mà bước 5.5 lấy hình
học chân từ `terminal_geometry(detection.label)`, và `terminal_geometry("component")`
rơi vào nhánh mặc định `multi_pin`.

Đo trên fixture 28 pad đếm tay (`tests/data/solder_geometry`), **giữ nguyên 39
box và chỉ đổi nhãn** — nên phép đo này tách riêng đúng tác động của hợp đồng
nhãn, không lẫn chất lượng box:

| | pad phủ ≥50% | ROI dựng ra | hình học chân |
|---|---:|---:|---|
| nhãn thật, 22 lớp | **28/28 = 100%** | 90 | 36 `two_terminal`, 3 `multi_pin` |
| ép 1 lớp `component` | **21/28 = 75%** | 132 | 39 `multi_pin` |
| 1 lớp + `apply_family_labels` | **28/28 = 100%** | 90 | như nhãn thật |

Mất 7 trong 28 pad **trong khi dựng nhiều ROI hơn 47%**: nó dựng dải quanh cả
4 cạnh của linh kiện chỉ có 2 chân rồi trượt khỏi land thật.

---

## 3. Bản vá: nhãn đến từ 6.1

`AOIPipeline.apply_family_labels()` cho họ do bước 6.1 trả về thay nhãn detector,
với hai chốt hẹp:

1. **Chỉ khi nhãn detector nằm trong `GENERIC_DETECTOR_LABELS`**
   (`component`, `components`, `object`, `part`, `smd`). Detector 22 lớp giữ
   nguyên nhãn của nó — nó được train cho đúng việc đó, và đảo ưu tiên sang 6.1
   là đổi hành vi của đường đang chạy mà không ai yêu cầu.
2. **Chỉ khi 6.1 `accept`.** `review`/`unknown` thì giữ `multi_pin`, vì đó là
   mặc định AN TOÀN: dựng thừa ROI thì xem lại được, thiếu thì không.
   `false_crop_background` không bao giờ thành nhãn.

Nhãn cũ ghi vào `metadata["detector_label"]` và nguồn ghi vào
`metadata["label_source"]`, để báo cáo truy được nhãn từ đâu ra.

Việc này chạy được là nhờ `99b962f` đã đưa 6.1 lên trước 5.5.

**Bẫy vận hành đã chặn:** nạp detector 1 lớp mà quên nạp 6.1 thì ROI vẫn dựng
ra, chỉ là sai nhánh — hỏng im lặng. `run()` giờ đếm và cảnh báo thẳng số thân
chưa giải được nhãn.

10 test mới, trong đó test cổng đo lại đúng 28 pad ở trên. Đã kiểm ngược bằng
đột biến: bỏ chốt `accept`, thay nhãn cả detector 22 lớp, và bỏ `component`
khỏi tập nhãn chung chung — mỗi đột biến làm đỏ đúng test canh nó.

---

## 4. Hai điều còn lại, chưa xử lý

**ONNX khoá cứng 300 detection.** `output0 [1, 300, 6]` — YOLO26 là kiến trúc
NMS-free nên top-k nằm trong graph, và `max_det` lúc export mặc định 300. Model
cũ khai `max_det: 2000`. Truyền `max_det` lớn lúc chạy **không** lấy lại được
box mà graph không hề sinh. Trên 11 tile test cao nhất mới 237 nên chưa cắn,
nhưng tile train từng chạm **358**. Sửa được bằng cách export lại với
`max_det=600`, không phải train lại.

**`tiling.tile_size = 1280` trong khi artifact khoá ở 1024.** Tile 1280 sẽ bị
letterbox xuống 1024, tức linh kiện nhỏ đi 0,8× so với lúc train, và một tile
1280 cũng chứa nhiều linh kiện hơn nên dễ chạm trần 300 hơn. Chưa đổi vì
**chưa đo được**: mọi fixture trong repo đều ≤1024 px nên tiling không hề kích
hoạt, không có gì để so.

---

## 5. Giới hạn do chính manifest khai

> "Ảnh train là CVL PCB-DSLR + RF100 + Winnies, **KHÔNG phải camera dây
> chuyền**. Phải fine-tune trên ảnh thật trước khi tin số đo ở production."

Và: chỉ 28 bo vật lý; tile chồng lấn nhân số box train lên ~1,85 lần (cùng linh
kiện nhìn từ hai khung, không phải dữ liệu mới); recall ở nhóm box nhỏ nên xem
riêng chứ đừng đọc mAP tổng.
