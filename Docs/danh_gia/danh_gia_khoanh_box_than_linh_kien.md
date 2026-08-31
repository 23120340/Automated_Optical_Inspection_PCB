# Đánh giá cách khoanh box thân linh kiện

> Đo 2026-08-31 trên checkpoint `joint_boxes (3).json`: **16 tile đã duyệt,
> 1.595 box**. Chạy lại bằng
> `python scripts/audit_component_boxes.py datasets/labelling/component_bodies --boxes "~/Downloads/joint_boxes (N).json"`.

## Kết luận nhanh

**Không quá bé, cũng không quá bự. Khoanh tốt hơn dataset công khai đo bằng
cùng một thước.** Cụ thể:

| | bám đúng mép (±1px) | quá to | quá bé | trung vị lệch |
|---|---:|---:|---:|---:|
| **Box của bạn** | **45,6%** | 30,9% | 23,5% | −1,0 px |
| Winnies v3 (công khai, gán nhãn kiểu vỏ) | 33,8% | 34,6% | 31,6% | 0,0 px |

Và về mặt hình học thì **sạch**: 0 box trùng nhau, 0 box bé dưới mức dùng được,
0 box suy biến.

Nên câu trả lời ngắn cho câu hỏi của bạn: **cứ khoanh tiếp như đang khoanh.**
Ba điều chỉnh nhỏ ở §4, không có điều chỉnh lớn nào.

---

## 1. Cách đo, và vì sao phải có mốc so sánh

Mép thân linh kiện là chỗ ảnh đổi đột ngột. Nên: phình hộp ra rồi thu vào từng
pixel, mỗi lần đo độ lớn gradient trung bình dọc theo viền. Viền nào gradient
mạnh nhất chính là mép thật.

```
lệch  0  → box bám đúng mép
lệch < 0 → mép thật nằm TRONG box   ⇒ box đang quá to
lệch > 0 → mép thật nằm NGOÀI box   ⇒ box đang quá bé
```

Riêng con số "45,6% bám đúng" **không tự đọc được** — không ai biết 45% là giỏi
hay kém. Vì vậy công cụ chạy đúng thước đo ấy lên **Winnies v3**, một dataset
công khai gán nhãn kiểu vỏ, và đó mới là thứ cho phép kết luận.

Hai chỗ tự kiềm chế trong phép đo, để nó không tạo ra kết luận giả:

- **Tầm quét co theo cỡ linh kiện** (±35% cạnh ngắn, tối đa ±8 px). Quét ±8 px
  quanh một con trở 14 px là quét quá nửa linh kiện, và khi đó phép đo trả về
  *hàng xóm* chứ không phải mép.
- **Đỉnh phải nhô hơn nền ít nhất 15%**, không thì ghi "không kết luận được"
  thay vì ép ra một con số. 215/1.595 box rơi vào nhóm này và **bị loại khỏi
  thống kê**, chứ không bị tính là đạt hay không đạt.

---

## 2. Cạm bẫy lớn nhất: cột "quá bé" phần lớn là PAD, không phải lỗi

Đây là chỗ dễ đọc sai nhất, nên nói kỹ.

Thứ nằm ngay bên ngoài thân linh kiện là **pad và chân hàn** — cạnh sắc nét nhất
trong cả vùng ảnh, sắc hơn cả mép thân. Thước đo bị chúng hút. Nghĩa là **một
box ôm thân ĐÚNG quy ước vẫn bị chấm là "quá bé"**.

Có bằng chứng đo được cho việc này, không phải suy đoán:

| Cỡ | trong nhóm "quá bé", bao nhiêu % có đỉnh **chạm trần tầm quét** |
|---|---:|
| nhỏ <20px | **31%** |
| vừa 20–60px | 5% |
| lớn ≥60px | 8% |

"Chạm trần" nghĩa là đỉnh gradient còn ở xa hơn cả tầm quét — mép thân không thể
nằm xa thế, nên đó là pad. Với linh kiện nhỏ (chip 2 chân), pad nằm sát ngay bên
cạnh thân nên gần một phần ba số ca "quá bé" thực chất là thước đo bắt vào pad.

⇒ **Con số 23,5% "quá bé" là chặn trên, không phải số lỗi thật.** Số lỗi thật
thấp hơn đáng kể.

---

## 3. Kiểm bằng mắt: quy ước đang được áp dụng đúng

Số liệu không thay được việc nhìn. Đã dựng bảng mẫu ngẫu nhiên (seed 17) theo ba
dải cỡ và soi từng cái:

**Linh kiện lớn (IC).** Đây là nhóm quan trọng nhất vì chân nhiều nhất, và cũng
là nhóm được khoanh **chuẩn nhất**: box ôm sát gói đen, toàn bộ chân cánh chim
nằm ngoài. Kiểm trên SOIC (`74HC...`, `P89AB 74VHC245`), TSSOP
(`74FCT 16543CTPV`), PQFP (`MICRONAS VPC 3230D`), PLCC (`SIEMENS SAB 80C535-N`,
`L3030-C9 AF`, `PEB 3065 N`) — **không cái nào bao chân**.

**Linh kiện vừa.** Điện trở có vòng màu, diode SOD, tụ tantalum có vạch cực: box
ôm thân, mũ kim loại hai đầu nằm ngoài. Đúng.

**Linh kiện nhỏ.** Chip 2 chân: box ôm thân, pad trắng hai bên nằm ngoài. Đúng.

**Một điểm đáng khen riêng:** ở vài ca, **khung lụa (silkscreen) trắng in trên
board bị bỏ ra ngoài box** và chỉ thân thật được khoanh — ví dụ tụ tantalum có
dấu `+` in cạnh nó, và thạch anh `SCG-004...` nằm trong khung lụa trắng. Đây
đúng là cách phải làm, và là chỗ người gán nhãn thiếu kinh nghiệm hay khoanh nhầm.

> Nghi ngờ ban đầu của tôi về tụ hoá tròn bị khoanh hụt **không đứng vững**: soi
> 144 box gần vuông cạnh ngắn ≥28px thì tất cả đều là PLCC/QFP và đều đúng. Ghi
> lại để bạn biết chuyện đó đã được kiểm chứ không bị bỏ qua.

---

## 4. Ba điều chỉnh nhỏ khi khoanh tiếp

**a) Lệch trung vị −1 px: box hơi rộng ra một pixel.** Cực nhỏ, và thật ra
**nên giữ nguyên**. Box rộng dư 1 px thì bước 5.5 vẫn đặt dải chân đúng chỗ; box
hụt 1 px thì dải chân bị đẩy vào trong thân. Sai về phía dư an toàn hơn sai về
phía thiếu. Không cần sửa gì.

**b) 19 box vượt biên tile.** Hợp lệ, **đừng cố kéo chúng vào trong**. Linh kiện
bị mép tile cắt ngang thì box phải giữ nguyên phần thật của nó; việc cắt về
trong khung là việc của lúc xuất nhãn YOLO, không phải việc của người vẽ. Kéo
vào là tạo ra một cái box mô tả sai kích thước linh kiện.

**c) 57 box tỉ lệ cạnh >3.** Đã soi: **phần lớn là connector / pin header thật**
— chúng dài thật, không phải khoanh nhầm dãy chân. Nhưng đây cũng đúng là hình
dạng mà một cái box vẽ trùm lên dãy chân sẽ tạo ra, nên khi gặp một linh kiện
rất dài, tự hỏi một câu: *"đây là một thân dài, hay là mình đang trùm lên hàng
chân của một thân ngắn?"* Ca cực đoan nhất trong bộ hiện tại là 56×1018 px
(tỉ lệ 18,2) và nó là connector rìa board — đúng.

### Nhắc lại quy ước, không đổi

- Khoanh **thân / gói / vỏ**: gói đen của IC, thân gốm, vỏ can.
- **Không** bao chân, pad, thiếc.
- **Không** bao khung lụa in trên board.
- Linh kiện bị mép tile cắt: khoanh phần nhìn thấy, để nó vượt biên.

---

## 5. Một quan sát về hiệu suất, không phải về chất lượng

Tile cắt chồng nhau 256 px, nên **cùng một linh kiện xuất hiện ở nhiều tile** và
bạn đang vẽ lại nó mỗi lần. Trong bộ mẫu, `L3030-C9 AF` xuất hiện 4 lần và
`SIEMENS PEB 3065 N` 3 lần.

Việc này **không sai** và không làm hỏng dữ liệu — packer chia tập theo **bo vật
lý** nên hai bản của cùng một linh kiện không thể rơi vào hai split khác nhau.
Nhưng nó là công sức lặp. Nếu muốn tiết kiệm, ưu tiên duyệt các tile thuộc **bo
chưa có tile nào được duyệt** trước, vì đó cũng đúng thứ đang chặn packer: cần
thêm 2 bo (`pcb_dslr:017`, `pcb_dslr:030`) là mở khoá được bộ train.

---

## 6. Số liệu đầy đủ

| Chỉ số | Đo được |
|---|---|
| Tile đã duyệt / box | 16 / 1.595 |
| Box mỗi tile | ít nhất 30, trung vị 87,5, nhiều nhất 229 |
| Cạnh ngắn | p05 12 · **trung vị 17** · p90 76 · max 240 px |
| Cạnh dài | p05 14 · trung vị 26 · p90 140 · max 1.018 px |
| Tỉ lệ cạnh | trung vị 1,38 · p99 7,63 |
| Box trùng nhau (IoU>0,5) | **0** |
| Box cạnh ngắn <6 px | **0** |
| Box vượt biên tile | 19 (hợp lệ) |
| Box tỉ lệ >3 | 57 (phần lớn là connector thật) |
| Bám đúng mép (±1 px) | **45,6%** — so với 33,8% của Winnies |

Ghi chú về `bám đúng 34,8%` ở nhóm lớn ≥60px: nhóm này gần như toàn IC nhiều
chân, tức là nhóm mà hàng chân ngay sát mép thân hút thước đo mạnh nhất. Kiểm
bằng mắt ở §3 cho thấy chính nhóm này lại là nhóm khoanh chuẩn nhất. **Đừng
dùng con số đó để kết luận nhóm IC bị khoanh hụt.**

---

## 7. Khoanh sát thân có ảnh hưởng classifier 6.1 không?

**Có, đo được — nhưng nó KHÔNG phải lỗi của cách bạn khoanh.** Đây là lệch *hợp
đồng*, và nó chỉ phát tác vào lúc detector lượt 1 được train lại trên nhãn tay.

Chạy lại bằng `python scripts/audit_crop_contract.py datasets/labelling/component_bodies --boxes "~/Downloads/joint_boxes (N).json"`.

### 7.1. Vì sao có lệch

Classifier 6.1 được train trên crop cắt từ box của **Consolidated**
(`model_manifest.json` → `dataset.slug`) — **đúng bộ đã train detector đang
chạy**. Nên detector hiện tại chính là bản sao sống của quy ước box mà classifier
mong đợi, và `CropConfig.padding_ratio = 0.15` là phần còn lại của hợp đồng đó.

Đo trên 861 cặp ghép được giữa detection và box tay:

> **Detector khoanh rộng hơn tay +22% mỗi cạnh** (trung vị; tỉ lệ diện tích 1,46).

Nghĩa là box tay của bạn *chặt hơn đáng kể* so với thứ classifier từng thấy.

### 7.2. Hậu quả, đo bằng chính classifier

Cắt crop hai kiểu trên **cùng một linh kiện** rồi hỏi classifier:

| | kết quả |
|---|---|
| Đổi nhãn khi cắt theo box tay | **192/861 = 22,3%** |
| Trong đó vẫn vượt ngưỡng accept 0,817 | **92 ca** |
| Cặp đổi nhiều nhất | `capacitor → led` **91 ca** (47% số ca đổi) |
| Kế tiếp | `resistor → capacitor` 33 · `ic → resistor` 12 |

Điều khó chịu nhất không phải con số 22,3%, mà là **92 ca đổi nhãn vẫn tự tin
vượt ngưỡng** — chúng sẽ được chấp nhận im lặng với nhãn mới, không rơi vào hàng
chờ xem tay. Ngưỡng review không bảo vệ được trước dạng lệch này.

### 7.3. Do hẹp hơn, hay do detector khoanh lệch chỗ?

Hai nguyên nhân này dễ lẫn, nên tách ra đo:

| Chênh lệch độ ôm | n | đổi nhãn |
|---|---:|---:|
| gần bằng nhau (<10%) | 288 | **10,1%** |
| rộng hơn 10–35% | 334 | 22,8% |
| rộng hơn >35% | 239 | **36,4%** |

Tăng đơn điệu theo mức chênh độ ôm ⇒ **đúng là do độ ôm**, không phải do lệch vị
trí. Mức 10,1% ở nhóm gần trùng nhau là nền nhiễu của chính classifier.

### 7.4. Nâng pad có cứu được không? — Không hẳn

Công thức giữ nguyên vùng nhìn cho pad `0,15 → 0,29`. Đo thật:

| | số ca đổi nhãn |
|---|---:|
| box tay + pad 0,15 | 192 |
| box tay + pad **0,29** | **154** |

Chỉ cứu được 38 ca (22,3% → 17,9%). **Không phải cách sửa.** Lý do: pad phục hồi
*diện tích* nhìn thấy chứ không phục hồi *tỉ lệ thân/nền* trong khung — mà đó mới
là thứ classifier đã học.

### 7.5. Vậy classifier hiện tại đã ổn chưa? Có cần đổi model không?

**Số của nó tốt, trên miền của chính nó:**

| | |
|---|---|
| accuracy (test) | 0,958 |
| macro F1 | 0,890 |
| accepted precision / coverage | 0,980 / 0,949 |

**Nhưng manifest của nó tự khai đúng cái điểm yếu vừa đo được:**

> `"unknown_policy_limit": "Confidence reject only; OOD behavior is not validated by this dataset."`

Thí nghiệm ở §7.2 chính là một phép thử OOD, và kết quả khớp lời cảnh báo đó: nó
đổi ý và **vẫn tự tin**.

**Khuyến nghị: KHÔNG đổi kiến trúc.**

1. Đây không phải vấn đề backbone. Tỉ lệ đổi nhãn tăng theo *chênh lệch độ ôm*
   (§7.3) — đổi sang ConvNeXt hay ViT cũng gặp đúng chuyện đó, vì nguyên nhân
   nằm ở dữ liệu train chứ không ở sức mạnh model.
2. Bản ConvNeXt-Base trong `models/library/` **không dùng thay được**: manifest
   của nó có `metrics: {}` — **chưa có số đo test nào cả**. Con số 0,9369 từng
   nhắc tới là macro recall trên model-val, không phải test, và ba cell cuối của
   notebook (calibration/test/export) chưa từng chạy. Không có số thì không
   nghiệm thu được, bất kể kiến trúc.

**Việc cần làm, theo thứ tự:**

1. **Bây giờ: không đụng gì.** Detector đang chạy vẫn sinh ra đúng loại crop mà
   classifier được fit. Hôm nay **không có gì hỏng**.
2. **Khi detector lượt 1 được train lại trên nhãn tay: train lại classifier trên
   crop cắt cùng kiểu.** Đây là *train lại*, không phải *đổi model* — và rẻ, vì
   crop sinh ra từ chính bộ nhãn đó, không cần gán nhãn thêm.
3. **Đừng dựa vào chỉnh pad** như cách sửa: đo được chỉ 4 điểm phần trăm (§7.4).
4. **Cổng nghiệm thu cho classifier mới:** chạy lại `audit_crop_contract.py`,
   yêu cầu tỉ lệ đổi nhãn giữa crop-cũ và crop-mới **≤ 10%** — tức về bằng nền
   nhiễu ở §7.3.
5. Ghi `padding_ratio` thực dùng vào manifest của classifier mới, để lần sau
   không ai phải đi đo ngược lại hợp đồng này.

### 7.6. Một con số phụ đáng chú ý

**734/1.595 box tay không có detection nào khớp** (IoU ≥ 0,4) — detector hiện tại
bỏ sót **46%** số linh kiện bạn đã khoanh trên chính các tile này. Con số đó
không liên quan tới cách khoanh của bạn, nhưng nó là lý do rõ ràng nhất cho việc
train lại detector lượt 1, và là thước đo để so sau khi train.
