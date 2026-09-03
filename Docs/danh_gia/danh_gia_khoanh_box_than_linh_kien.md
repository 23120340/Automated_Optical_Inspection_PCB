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

### 7.7. Nới BOX lúc chạy để khỏi phải train lại classifier — có được không?

Ý tưởng: sau khi detector lượt 1 đã train xong trên nhãn tay và cho ra box chặt,
**nới box đó ra lúc suy luận** cho khớp quy ước cũ, rồi mới cắt crop. Như vậy
classifier giữ nguyên, không phải train lại.

Đây là ý khác với §7.4. Nới **lề** cộng một biên *bằng nhau cho bốn phía*, tính
theo cạnh dài (`pad = 0.15 × max(w,h)`). Nới **box** thì mỗi cạnh giãn theo chính
độ dài của nó — với linh kiện dài (trung vị tỉ lệ cạnh 1,38) hai cách cho ra hai
khung hình khác nhau. Nới box đúng về nguyên tắc hơn: nó dựng lại *cái box mà
detector cũ sẽ cho*, chứ không chỉ nới rộng khung.

Đo trên 861 cặp, so với A = box detector + pad 0,15 (thứ classifier được fit):

| Cách cắt | đổi nhãn | % |
|---|---:|---:|
| B — không bù gì | 192 | 22,3% |
| C — nới **lề** lên 0,29 | 154 | 17,9% |
| D — nới **box** ×1,13/1,15 (một hệ số chung) | 155 | **18,0%** |
| **F — nới box theo DẢI CỠ** (nhỏ ×1,25 · vừa ×1,13 · lớn ×1,05) | **124** | **14,4%** |
| E — nới box bằng hệ số **ORACLE** của từng linh kiện | 48 | **5,6%** |

**Đọc kết quả:**

- **Ý tưởng đúng, nhưng một hệ số chung thì vô ích.** D (18,0%) không hơn gì C
  (17,9%). Lý do đo được: hệ số nới **phụ thuộc cỡ linh kiện** — tương quan với
  cạnh ngắn `r = −0,45`. Linh kiện nhỏ cần ×1,25 còn linh kiện lớn chỉ ×1,05, mà
  D áp một con số cho tất cả nên nới hụt đúng ở nhóm đông nhất.
- **Chia theo cỡ thì ăn tiền thật:** F kéo từ 22,3% xuống **14,4%** — cắt được
  một phần ba thiệt hại, đổi lại khoảng mười dòng code và **không train lại gì**.
- **Nhưng vẫn chưa đủ.** Nền nhiễu của chính classifier là **10,1%** (§7.3), nên
  14,4% vẫn còn dư khoảng 4 điểm phần trăm so với mức không thể tránh.
- **Trần là 5,6%, và không cài đặt được.** E dùng đúng tỉ lệ thật của *từng*
  linh kiện — con số mà lúc chạy không ai biết, vì nó chính là box mà detector
  cũ *sẽ* cho ra. Khoảng cách F → E là **nhiễu per-box của detector** (độ lệch
  chuẩn 0,12–0,17 ngay trong từng dải cỡ), không quy tắc nào lấy lại được.

**Kết luận thực dụng:**

| Nếu bạn muốn | thì |
|---|---|
| Không train lại classifier, chấp nhận sai số | dùng **F** — nới box theo dải cỡ. 22,3% → 14,4% |
| Về đúng mức nhiễu nền (10%) | **phải train lại classifier** trên crop cắt cùng kiểu |

Nới box là **biện pháp giảm nhẹ tốt nhất trong các cách không train lại**, và
đáng làm ngay cả khi đã định train lại — nó che khoảng thời gian giữa lúc
detector mới lên và lúc classifier mới xong.

> **Một điều phải nói rõ để không đọc sai bảng trên.** Phép đo này lấy A làm
> chuẩn, tức coi *câu trả lời trên box của detector cũ* là đúng. Nó đo **độ lệch
> phân bố**, không đo **độ sai**. Nếu detector mới khoanh chính xác hơn detector
> cũ — mà nhiều khả năng là thế, vì nó học từ 1.595 box người duyệt — thì một
> phần trong 124 ca "đổi nhãn" của F có thể là classifier đang *đúng hơn*, chứ
> không phải sai đi. Muốn biết chắc thì phải có nhãn họ linh kiện thật cho các
> crop này, mà hiện chưa có.

---

## 8. Duyệt lượt hai trang gán nhãn package (2026-09-02)

Lượt duyệt độc lập trên `label_packages.html`, sau lượt của một agent khác.
Kiểm bằng cách **chạy**, không chỉ đọc code.

### 8.1. Luật gán nhãn sẵn đang NGƯỢC — đã gỡ

`scripts/prepare_package_labelling.py` gán `hai_chan` khi
`aspect >= 3.2 and area_fraction <= 0.0015`, với lý lẽ "thân rất nhỏ và thuôn
dài gần như chắc chắn là linh kiện hai chân".

Lý lẽ đó ngược. Chip hai chân thật (0402/0603/0805) có tỉ lệ thân khoảng
**1,5–2,5**; trên chính hàng đợi này **trung vị tỉ lệ là 1,38** (§6). Còn box
**tỉ lệ >3 phần lớn là connector / pin header** — đã kiểm bằng ảnh ở §4c. Ngưỡng
`>= 3.2` vì thế chọn đúng nhóm *ít khả năng là hai chân nhất*.

Soi ảnh cả 8 box mà luật bắn trúng, tỉ lệ cạnh
**4,7 · 17,9 · 3,3 · 8,3 · 9,6 · 3,4 · 4,5 · 3,5**:

| Box | Nhìn thấy gì |
|---|---|
| 8×143 (ar 17,9) | dải đen dọc theo **hàng ~15 pad** — thân connector ❌ |
| 100×12 (ar 8,3) | dải mỏng dọc **mép trên một IC lớn** (`PI … 8E10`), trên là hàng chân ❌ |
| 77×8 (ar 9,6) | dải mỏng ngay dưới **hàng 8 chân** ❌ |
| 5 cái còn lại | mơ hồ, không cái nào rõ là chip hai chân |

Luật chỉ bắn **8/3.855 box (0,2%)** nên không tiết kiệm được công đáng kể, trong
khi mỗi lần bắn sai lại tạo ra một box *trông như đã xong* — thứ người duyệt dễ
bấm qua theo phản xạ. **Đã gỡ hẳn**; 3.855/3.855 box về `unknown`.

Kiểm sau khi gỡ: **0 box lệch hình học, đúng 8 box đổi nhãn**, `source_geometry_sha256`
không đổi. Muốn điền sẵn lại thì đo trước — gán tay vài trăm box, tính tỉ lệ đúng
của quy tắc ứng viên, rồi mới bật.

### 8.2. Những chỗ nghi ngờ nhưng kiểm ra là ổn

**Một lần tôi nghi sai, ghi lại cho minh bạch.** Dòng xuất JSON
`cls: CLASSES[b.cls].name` không có guard tại chỗ, nên tôi tưởng nó nổ khi gặp
box `unknown`. Thực tế có **hai chốt chặn ngay trên vòng lặp**, và chạy thử cho
ra đúng thông báo *"Không xuất: 1 ảnh đã duyệt vẫn còn box unknown"*. Không im
lặng, không mất dữ liệu.

| Kiểm | Kết quả |
|---|---|
| Năm chốt chống `verified` + `unknown` (`mark`, import, `load_seed`, migration, export ×2) | tất cả kín |
| Migration mang việc đã duyệt sang trang package | **50 tile / 5.281 box sang đủ**; 6.574 box tổng, tất cả `unknown`, **0** tile bị đánh dấu duyệt nhầm |
| Phím `8`/`9` trên trang chỉ có 7 lớp | `setClass` chặn chỉ số ngoài dải — no-op |
| Smoke test `package_label_app_smoke.mjs` trên trang thật | pass |

### 8.3. Hai điều người duyệt phải biết trước

- **Phải mở trang package trong CÙNG trình duyệt** nơi đã duyệt trang thân.
  Migration đọc `localStorage` của trang thân; mở ở máy hoặc trình duyệt khác thì
  chỉ có 3.855 box từ seed, **mất 5.281 box** vừa duyệt.
- **Mọi box sẽ ở trạng thái `unknown`, kể cả trên tile đã duyệt thân.** Đó là chủ
  ý (`preserve_geometry_reset_box_classes_to_unknown`): hình học giữ nguyên, nhãn
  lớp chọn lại từ đầu. Thấy "0 tile đã duyệt" là đúng, không phải mất việc.

---

## 9. Soát toàn bộ trước khi train (2026-09-03)

Gán nhãn thân linh kiện **đã xong**: `joint_boxes (11).json` — 95 tile duyệt,
25 unusable, **9.493 box**. Lượt soát này chạy trên toàn bộ, **không dùng
detector có sẵn**: nó bỏ sót 46% box tay trên chính các tile này, nên không đủ
tư cách phán ai đúng ai sai.

### 9.1. Cách lọc: máy khoanh vùng nghi vấn, người quyết

Sáu nhóm nghi vấn được tính bằng hình học thuần tuý — **1.108/9.493 box
(11,7%)** — rồi soi bằng ảnh phóng to. Kết quả: **chỉ 7 box thật sự sai**.

| Nhóm | Số box | Phán quyết sau khi NHÌN |
|---|---:|---|
| Tí hon `<6px` | 257 | **GIỮ** — soi ra là chip 2 chân thật (thân đen giữa hai pad bạc) |
| Rất nhỏ `6–8px` | 573 | **GIỮ** — cùng lý do |
| Phủ gần hết tile | 9 | **GIỮ 6 / LOẠI 3** — 6 cái là socket RAM/DIMM và connector rìa board, dài 960–1030px là kích thước thật |
| Thuôn dài `ar>6` | 90 | **GIỮ** — khe ISA/PCI, pin header, cuộn cảm dài |
| Diện tích `>25%` tile | 5 | **GIỮ** — Socket 7, vỏ nhựa che, hộp chắn có tem serial |
| Lồng trong box khác | 205 | **GIỮ** — socket CPU có tụ lắp *bên trong* khoang; cả hai đều là linh kiện thật |
| Trùng nhau `IoU>0.5` | 8 | **LOẠI 4** — 4 cặp khoanh hai lần cùng một linh kiện |

Bảy box bị loại, kèm lý do từng cái, nằm ở
`datasets/labelling/component_bodies_round2_20260830/box_exclusions.json`.
Áp dụng bằng `scripts/apply_box_exclusions.py`, script này **từ chối chạy** nếu
checkpoint không phải bản đã soi — box được chỉ theo chỉ số, mà chỉ số chỉ đúng
với đúng file đó.

Ba box mép board là dạng nguy hiểm nhất trong cả bộ: chúng ôm **vát cạnh tối
trơn** và **nền đen ngoài board**, tức dạy model gọi nền là linh kiện.

### 9.2. Hai chỗ tôi định làm và đã dừng lại

**Không xoá 830 box nhỏ.** Nhìn ảnh thì chúng là linh kiện thật, không phải
nhiễu. Xoá là dạy model rằng linh kiện nhỏ là nền — một sai lầm tệ hơn nhiều so
với việc train trên vài trăm box khó. Cách xử lý đúng là **tăng `imgsz`**.

**Không xoá box lồng nhau.** Cái bao nhiều nhất bao 55 box khác, và đó thật sự
là supervision mâu thuẫn. Nhưng soi ảnh thì là socket CPU có tụ lắp trong
khoang — cả socket lẫn tụ đều tồn tại. Xoá là xoá linh kiện thật dựa trên phán
đoán của tôi. Ghi lại thành **ứng viên số một cần xem lại** nếu train ra kết quả
kém ở vùng socket.

### 9.3. Dataset đã xuất

`datasets/train/component_detect_v1`, gói bằng
`pack_component_detection_dataset.py` từ checkpoint đã dọn:

| | ảnh | box |
|---|---:|---:|
| train | 318 | 51.314 |
| valid | 10 | 1.300 |
| test | 11 | 640 |

**28 bo vật lý** (cổng cần 10), split khoá theo bo, 0 ảnh trùng giữa các split,
0 lỗi định dạng nhãn, một lớp `component`. Đối chiếu IC chính thức của PCB-DSLR:
446/448 khớp.

### 9.4. Một con số làm tôi phải sửa lại notebook

Viết notebook xong tôi mới đo cỡ box trên **chính gói đã xuất**, và nó bác bỏ lý
lẽ tôi vừa viết:

| imgsz | p05 | trung vị | **box < 8px** |
|---|---:|---:|---:|
| 1024 | 3,3 | 9,2 | **42,8%** |
| 1280 | 4,1 | 11,5 | 29,7% |
| **1536** | 4,9 | 13,8 | **20,9%** |
| 1792 | 5,7 | 16,1 | 14,3% |

Tôi đã viết "p05 là 7px trên tile 1024, nâng lên 1536 là thành 10,5px". Con số
đó đúng cho **tile của dự án**, nhưng gói này có **94% là ảnh công khai** cỡ lớn
(RF100 rộng 504–5985px) bị letterbox về 1536 — linh kiện nhỏ của chúng co lại
rất nhiều. Ở 1536 vẫn còn **21% box dưới 8px**, tức dưới một ô lưới P3.

Đã sửa notebook cho khớp số đo, và ghi rõ đòn bẩy đúng: **cắt tile ảnh công
khai**, không phải nâng tiếp imgsz (1792 chỉ bớt 6 điểm phần trăm mà tốn thêm
36% compute).

### 9.5. Notebook train

`training/kaggle/pcb_component_detector_v3_kaggle.py` (+ `.ipynb`, 16 cell).
Khác v2 ở ba chỗ: một lớp, split đã khoá (notebook **cấm** chia lại), và cổng
phán quyết so với detector đang chạy — `recall test > 0,54`, vì đó là recall của
model hiện tại trên chính các box này.
