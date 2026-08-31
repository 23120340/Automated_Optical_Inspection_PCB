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
