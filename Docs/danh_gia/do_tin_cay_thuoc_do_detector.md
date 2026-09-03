# Thước đo của detector thân linh kiện chính xác đến đâu?

Câu hỏi: **valid 10 ảnh / test 11 ảnh có quá ít không?** Trả lời ngắn: có, và
dưới đây là con số cụ thể cho "quá ít" nghĩa là gì. Nhưng nguyên nhân gốc
không sửa được bằng cách chia lại tập — chỉ có 30 bo trong toàn bộ PCB-DSLR.
Thứ phải sửa là **cách đọc con số**, cộng với một khiếm khuyết train riêng
biệt mà lần train đầu vừa lộ ra.

Mọi số trong tài liệu này đo từ log lần train thật đầu tiên
(`componentdetector.ipynb`, 19/150 epoch trên Kaggle T4) và từ chính bộ
`component_detect_v1_tiled`.

---

## 1. Cỡ mẫu thật không phải 10 và 11 ảnh — mà là 3 và 3 bo

Box trong cùng một bo không độc lập: cùng ánh sáng, cùng lô linh kiện, cùng
nhà sản xuất, cùng một lần chụp. Đơn vị độc lập là **bo vật lý**.

| tập   | bo canonical | tile | box   |
|-------|--------------|------|-------|
| train | 22           | 74   | 7.546 |
| valid | 3            | 10   | 1.300 |
| test  | 3            | 11   |   640 |

Việc gộp bo đang chạy **đúng**: `pcb15` và `pcb_dslr_015` là cùng một bo vật lý
dưới hai tên, và cả hai đều rơi vào `test` — không rò rỉ sang tập khác.

Và thước đo bị một bo duy nhất chi phối:

| tập   | bo lớn nhất | tỷ trọng box |
|-------|-------------|--------------|
| valid | board017    | **49,5%**    |
| test  | board009    | **42,2%**    |

Nói cách khác, "mAP50 trên valid" gần như là "model làm tốt đến đâu trên
board017".

---

## 2. Nhiễu rộng hơn cả tín hiệu — đo được, không phải phỏng đoán

18 epoch có số val trong log. Sau khi trừ xu hướng học (khớp tuyến tính), phần
dư chính là nhiễu của phép đo:

| chỉ số | biên độ 18 epoch | xu hướng học | nhiễu (sd) | dải 95% |
|--------|------------------|--------------|------------|---------|
| mAP50  | 0,127 – 0,292    | **+0,071**   | 0,038      | **±0,077** |
| recall | 0,285 – 0,598    | +0,172       | 0,071      | ±0,142  |

**Dải nhiễu của mAP50 (±0,077) rộng hơn toàn bộ tiến bộ sau 18 epoch (+0,071).**
Không thể xếp hạng epoch 17 (0,292) trên epoch 8 (0,289) hay epoch 6 (0,235).

Một phép đo thứ hai, hoàn toàn độc lập, ra cùng kết luận: bootstrap lấy mẫu lại
ở mức **bo** cho khoảng tin cậy recall rộng **0,150** với 3 bo — khớp với
sd = 0,071 ở trên. Hai đường khác nhau, cùng một con số.

---

## 3. Hệ quả 1 — `best.pt` tự thổi phồng chính nó

Chọn `best.pt` = lấy max qua N lần rút thăm từ một phân phối nhiễu. Giá trị kỳ
vọng của max luôn cao hơn giá trị thật:

| chọn max qua | mAP50 báo cáo cao hơn thật |
|--------------|-----------------------------|
| 18 epoch     | +0,070 (+28% tương đối)     |
| 50 epoch     | +0,086 (+35%)               |
| 150 epoch    | **+0,102 (+41%)**           |

Với `epochs: 150`, con số val của `best.pt` bị thổi lên khoảng **+0,10 mAP50**
thuần tuý do cơ chế chọn. **Không bao giờ báo cáo số val của `best.pt`** —
chỉ báo số `test`, đúng một lần, ở cuối.

## 4. Hệ quả 2 — cổng `gate_recall = 0,70` là tung đồng xu

Với sd = 0,071 trên recall:

| recall THẬT | xác suất đo được ≥ 0,70 và qua cổng |
|-------------|--------------------------------------|
| 0,60        | 8%                                   |
| 0,65        | 24%                                  |
| 0,70        | 50%                                  |
| 0,75        | 76%                                  |

Một model recall thật 0,65 vẫn qua cổng 1 lần trong 4. Đã sửa: cổng giờ phán
trên **cận dưới** của khoảng bootstrap thay vì trên con số điểm — tức đòi hỏi
"kể cả khi 3 bo này là 3 bo may nhất, model vẫn hơn detector đang chạy".

---

## 5. Một khiếm khuyết riêng, cộng dồn vào cùng chỗ: `batch = 2`

Log cho 961 iteration/epoch trên 1.922 ảnh → **batch = 2**. AutoBatch không
hỏng, nó đúng; profile của chính nó trên T4:

| batch @1280 | VRAM     | kết quả                          |
|-------------|----------|----------------------------------|
| 1           | 3,87 G   |                                  |
| 2           | 8,78 G   | ← chọn, 9,04/14,56 G (62%)       |
| 4           | 16,97 G  | OOM (card chỉ 14,56 G)           |

**`imgsz = 1280` chặn cứng batch ở 2 trên T4.** YOLO dùng BatchNorm khắp nơi;
thống kê chuẩn hoá lấy từ 2 ảnh thì rất nhiễu. `nbs = 64` cộng dồn gradient bù
được phần gradient, nhưng **không bù được BN**.

Điều này quan trọng vì nó cộng vào đúng con số ở mục 2: một phần dao động giữa
các epoch là do model thật sự khác nhau giữa các epoch, chứ không chỉ do thước
đo ngắn. Từ log này **không tách được** hai nguồn — chỉ biết cả hai cùng đẩy
theo một hướng.

### Vì sao 1280 vốn không cần thiết

Mọi ảnh train/valid/test đều **1024×1024 hoặc nhỏ hơn** (trung vị cạnh dài =
1024). Nên 1280 là phóng to 1,25 lần: không thêm thông tin, chỉ đẩy box nhỏ
vượt ngưỡng ô lưới P3 (stride 8).

| imgsz | cạnh ngắn box trung vị | <8px | batch tối đa T4 |
|-------|------------------------|------|-----------------|
| 1024  | 16,2 px                | 8,0% | **4**           |
| 1280  | 20,2 px                | 2,8% | 2               |
| 1536  | 24,2 px                | 1,0% | 1               |

**Đã đổi sang `imgsz: 1024`, `batch: 4`** (đặt tường minh — để `-1` thì
AutoBatch nhắm 60% VRAM và vẫn chọn 2–3). Đánh đổi: thêm 5 điểm phần trăm box
dưới ngưỡng P3, đổi lấy BN gấp đôi số mẫu và epoch nhanh gần gấp đôi
(2:50 → ~1:50). 8,0% vẫn dưới ngưỡng cảnh báo 10% mà notebook tự đặt.

Đây là một đánh đổi thật, không phải một lỗi được sửa. Muốn quay lại: đặt
`imgsz` 1280 và `batch` 2.

---

## 6. Đã sửa gì trong notebook

1. `imgsz` 1280 → **1024**, `batch` -1 → **4**.
2. Mục **4b** mới: báo recall/mAP50 **theo từng bo**, cộng khoảng tin cậy
   bootstrap lấy mẫu lại ở mức bo. Recall cộng gộp được chính xác
   (= ΣTP/ΣGT) nên bootstrap này đúng về số học; mAP không cộng gộp tuyến tính
   nên chỉ báo biên độ giữa các bo.
3. Cổng phán quyết phán trên **cận dưới**, và in cảnh báo riêng cho trường hợp
   "con số điểm hơn incumbent nhưng cận dưới thì không".
4. Cảnh báo tự động khi khoảng tin cậy rộng hơn 0,10.

Sau đây mỗi lần train, con số tự khai báo độ chính xác của nó.

---

## 7. Việc chia lại tập KHÔNG sửa được vấn đề này

PCB-DSLR có 30 bo, đã dùng 28. Chuyển bo từ train sang test chỉ làm train tệ
đi mà test vẫn ít. Thêm tile từ chính các bo cũ hầu như không thêm thông tin
độc lập.

Cách duy nhất lấy được thước đo chặt hơn là **k-fold theo bo**: chia 28 bo
thành k nhóm, train k lần, mỗi bo được chấm đúng một lần. Chỉ số cuối cùng
tính trên cả 28 bo thay vì 3, thu hẹp khoảng tin cậy khoảng **√(28/3) ≈ 3 lần**.

Chi phí, ước từ nhịp đo được (~1:50/epoch ở cấu hình mới):

| phương án                       | GPU-giờ | ghi chú                             |
|---------------------------------|---------|-------------------------------------|
| 1 lần chạy, 150 epoch (hiện tại)| ~4,5    | khoảng tin cậy ±0,075               |
| 3-fold, 60 epoch (dò A/B)       | ~5,5    | đủ để xếp hạng cấu hình             |
| 3-fold, 150 epoch               | ~13,5   | khoảng tin cậy ~±0,025              |
| 5-fold, 150 epoch               | ~22,5   | vượt quota Kaggle 30h/tuần khá sát   |

Đề xuất: **3-fold ở 60 epoch để quyết các câu A/B** (1024 vs 1280, oversample
6 vs 1, model s vs m), rồi **một lần chạy đầy đủ 150 epoch** với cấu hình
thắng. Chưa làm — cần quyết định về ngân sách GPU.

---

## 8. Điều KHÔNG nên kết luận từ lần train đầu

mAP50 0,13–0,29 nhìn thì tệ, nhưng đó là **19/150 epoch, đo trên 3 bo**. Phần
lớn dao động là thước đo ngắn cộng batch = 2, không phải kiến trúc sai. Đừng
đổi model hay learning rate dựa trên nó.
