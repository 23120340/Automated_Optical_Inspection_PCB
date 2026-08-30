# Kế hoạch kiểm tra lỗi toàn mạch (PCB defect)

> Soạn 2026-08-31. **Bản để bạn duyệt, chưa code gì cả.** Mọi con số ghi rõ đo
> từ đâu. Chỗ nào là suy luận hoặc trích từ bài báo thì nói thẳng.

## 1. Kết luận nhanh

- **Có một lỗ hổng thật, và nó lớn hơn tôi tưởng lúc bắt đầu đọc.** Golden
  Compare (bước 3.5) so **từng ô linh kiện** với ảnh chuẩn; 6.2 chấm **từng ROI
  mối hàn**. **Không có gì nhìn vào khoảng TRỐNG giữa các linh kiện** — chính là
  nơi vết xước, giọt thiếc bắn và sợi tóc nằm. Đây là lý do chính đáng để có
  tầng toàn mạch, và nó khác hẳn lý do của tầng đã bị gỡ ngày 28/08.
- **Hai model là đúng, nhưng ranh giới không phải như bạn nhớ.** Chia đúng
  không phải "lỗi biết trước / lỗi lạ" một cách trừu tượng, mà theo **có nhãn
  hay không có nhãn**: M1 = detector có giám sát cho vài lớp lỗi *đếm được và
  gọi tên được*; M2 = anomaly detection train **chỉ trên ảnh tốt** cho mọi thứ
  còn lại. M2 quan trọng hơn M1 và **nên làm trước**.
- **Dataset có sẵn — và tốt hơn dự kiến.** **VisA** có 4 bộ PCB (`pcb1`–`pcb4`),
  mỗi bộ ~1.000 ảnh tốt + 100 ảnh lỗi, board **đã gắn linh kiện**, và lớp lỗi
  gồm đúng **scratch / dent / colour spot / crack** cộng misplacement/missing.
  Repo **đã có sẵn `scripts/fetch_visa_reference_set.py`** kéo từ
  `BrachioLab/visa` — chỉ cần mở cho nó lấy ảnh lỗi (hiện nó cố tình chặn).
- **Cạm bẫy phải tránh:** DeepPCB, HRIPCB/PKU-Market-PCB, PCB-Defect (2025) đều
  là **board TRẦN chưa gắn linh kiện** — missing hole, mouse bite, open, short,
  spur, spurious copper. Chúng **không** chứa vết xước hay lem thiếc trên board
  đã lắp. Train trên chúng là train một bài toán khác.
- **Số quyết định số phận của cả kế hoạch là false call, không phải accuracy.**
  Một board 5144² cắt thành **36 tile**; board bị gọi lỗi nếu **bất kỳ** tile
  nào kêu. Ở FPR 1%/tile ⇒ **30,4% board tốt bị gọi lỗi**. Muốn xuống dưới 4%
  thì FPR mỗi tile phải **≤ 0,1%**. Bảng đầy đủ ở §5.

---

## 2. Ranh giới với Golden Compare và 6.2 — cái gì đã có người lo

Đọc từ mã nguồn, không phải từ tài liệu:

`aoi_pipeline/golden/compare.py` so sánh theo **`SlotRecipe`** — mỗi ô có
`expected_bbox_xyxy`, `fixed_roi_xyxy`, `template_path`, `compare_mask_path`,
`ignore_mask_path`. Tức **chỉ nhìn vào nơi có một ô đã đăng ký**. Grep toàn repo
`whole_board|full_board|board_level` trong `aoi_pipeline/` và `app/`: **không có
phép so sánh nào ở mức toàn board.**

| Lỗi | Ai bắt hôm nay | Còn thiếu gì |
|---|---|---|
| Thiếu linh kiện | Golden 3.5 (per-slot) + đối chiếu BOM | — |
| Lệch / xoay / tombstone | Golden 3.5 (`PositionTolerance`) | — |
| Sai linh kiện | 6.1 + Golden | — |
| Thiếu/thừa thiếc, cầu chì, hàn nguội | 6.2 (per-ROI) | model 6.2 chưa dùng để quyết được |
| **Xước mặt nạ hàn giữa các linh kiện** | **không ai** | ⬅ lỗ hổng |
| **Giọt thiếc bắn trên vùng trống** | **không ai** | ⬅ lỗ hổng |
| **Sợi tóc / xơ / vật lạ nằm vắt qua** | **không ai** | ⬅ lỗ hổng |
| **Cháy, ố, đổi màu mặt nạ** | **không ai** | ⬅ lỗ hổng |
| **Hỏng silkscreen, hở đồng** | **không ai** | ⬅ lỗ hổng |
| Cầu thiếc **giữa hai linh kiện khác nhau** | không ai¹ | ⬅ lỗ hổng |

¹ 6.2 chấm từng ROI của **một** linh kiện. Vệt thiếc nối pad của R12 sang pad của
C7 nằm giữa hai ROI, không thuộc ROI nào.

**Vì sao lần này khác lần bị gỡ ngày 28/08.** Commit `1447ed5` gỡ một tầng
"detect lỗi hàn toàn board" vì nó là **đường thứ ba trùng việc**: nó chấm *mối
hàn*, mà mối hàn đã có 5.5 + 6.2 lo, bằng một model train trên camera khác và
manifest của chính nó ghi `diagnostic_only`, không được quyết PASS/NG. Tầng đề
xuất ở đây **không đụng vào mối hàn**: nó nhìn phần mặt board mà không stage nào
nhìn. Nếu sau này nó bắt đầu phán về mối hàn thì nó đã lặp lại đúng sai lầm đó.

---

## 3. Danh sách lỗi và độ đọc được ở 46 µm/px

| Lỗi | Kích thước nhỏ nhất đáng bắt | px ở 46 µm | Đọc được? | Loại |
|---|---|---:|---|---|
| Xước mặt nạ hàn | rộng ~0,1 mm, dài 2–20 mm | 2 px rộng × 43–435 dài | ⚠️ thấy được nhờ **dài**, không nhờ rộng | anomaly |
| Giọt thiếc bắn | ⌀ 0,2–0,5 mm | 4–11 px | ⚠️ sát mép | anomaly + có giám sát |
| Vật lạ (tóc, xơ) | rộng 0,05–0,1 mm, dài | 1–2 px × dài | ⚠️ chỉ thấy vật dài | anomaly |
| Cháy / ố / đổi màu | vệt ≥ 1 mm | ≥ 22 px | ✅ | anomaly |
| Cầu thiếc giữa 2 linh kiện | dài 0,3–1 mm | 7–22 px | ✅ | có giám sát |
| Hở đồng / tróc mặt nạ | ≥ 0,5 mm | ≥ 11 px | ✅ | anomaly |
| Hỏng silkscreen | ≥ 0,5 mm | ≥ 11 px | ✅ | anomaly |
| Cong vênh board | mm | — | ❌ 2D không đo được | ngoài phạm vi |
| Bọt khí lớp phủ conformal | ⌀ ≥ 0,5 mm | ≥ 11 px | ⚠️ cần chiếu sáng UV | ngoài phạm vi |

**Đọc bảng cho đúng:** cột "px" là *bề rộng*. Một vết xước rộng 2 px vẫn nhìn
thấy được vì nó **dài hàng trăm px** — mắt và mô hình đều bắt được cấu trúc dài,
không cần bề rộng lớn. Đó là lý do xước khả thi còn giọt thiếc ⌀0,2 mm thì bấp
bênh: giọt thiếc *nhỏ theo cả hai chiều*.

---

## 4. Model 1 — lỗi có tên, có giám sát

- **Nhiệm vụ:** detect, 3–4 lớp: `solder_splash`, `bridge_between_parts`,
  `scratch`, `foreign_object`. Không nhiều hơn — mỗi lớp là một khoản nợ nhãn.
- **Kiến trúc:** YOLO (giống bước 4, tái dùng cả đường ONNX + `model_registry`).
- **Đầu vào:** ảnh board rất lớn (4096–5985 px) ⇒ **bắt buộc chia tile**. Dùng
  đúng `scripts/tile_test_images.py` đã có: `--tile 1024 --stride 768`
  (chồng 256 px). Vết xước dài hơn tile sẽ bị cắt qua nhiều tile — chấp nhận
  được, vì mỗi mảnh vẫn là một vệt bất thường; gộp lại ở mức board.
- **Dữ liệu:** phần lớn **phải tự gán nhãn**. VisA `pcb1`–`pcb4` cho ~400 ảnh
  lỗi tổng (100 mỗi bộ) nhưng nhãn ở mức **mask bất thường**, không phải bbox
  theo tên lỗi ⇒ dùng làm ảnh nguồn để gán, không dùng nhãn trực tiếp.
- **Cổng nghiệm thu:** recall theo từng lớp ≥ 0,80 trên test chia **theo board**,
  **và** false call ≤ ngưỡng ở §5. Không đạt ⇒ giữ tắt, đúng lệ
  `_NO_AUTO_ADOPT` của ô model lượt 2.

**Nói thẳng:** M1 là phần **yếu hơn** của kế hoạch. Nó cần nhãn mà chưa ai có, và
mỗi lớp lỗi cần vài trăm mẫu thật. Nếu phải chọn một, chọn M2.

---

## 5. Model 2 — anomaly detection, train chỉ trên ảnh tốt

Đây là phần đáng làm trước, vì nó **không cần ảnh lỗi để train** — mà ảnh lỗi
chính là thứ dự án không có.

**Phương pháp đề xuất: PatchCore** (hoặc EfficientAD nếu cần nhanh hơn), qua thư
viện `anomalib`.

Lý do chọn, gắn với ràng buộc của dự án:
- PatchCore **không train**: nó rút đặc trưng từ backbone đã pretrain và dựng một
  bộ nhớ patch từ ảnh tốt. Với dự án chỉ có **30 ảnh tốt cho một SKU**, "không
  train" là một ưu điểm quyết định — mọi phương pháp phải học phân phối đều đói
  dữ liệu hơn thế.
- Suy luận trên CPU chấp nhận được ở mức tile, và app đã chạy ONNX Runtime sẵn.
- EfficientAD nhanh hơn nhiều khi train, đổi lại cần nhiều ảnh hơn.

**Con số AUROC phải đọc cho cẩn thận.** Bảng công bố thường ghi PatchCore
~99,6% và EfficientAD ~99,9% **image AUROC trung bình trên toàn VisA (12 nhóm)**.
Nhưng phần lớn 12 nhóm đó là vật thể đơn giản (kẹo, hạt, ống). **4 nhóm PCB là
nhóm khó nhất trong VisA.** Đừng mang con số trung bình vào báo cáo như thể nó
là con số của PCB — **phải đo lại riêng trên `pcb1`–`pcb4`**, và đó là việc đầu
tiên của lộ trình vì nó rẻ và trả lời được ngay "hướng này có sống không".

**Điều kiện tiên quyết: căn ảnh.** PatchCore so patch theo **vị trí**. Lệch vài
px là mọi patch lệch bộ nhớ và bất thường nổi lên khắp nơi. Repo đã có bước 2
(ORB + ECC) và bước 3.5 dựng trên đó, nên hạ tầng có sẵn — nhưng **phải đo sai
số căn còn lại** trước khi tin bất kỳ con số anomaly nào. Một lỗi căn trông y hệt
một board đầy lỗi.

### Số học false call — phần quyết định

Board 5144², tile 1024, stride 768 ⇒ **6×6 = 36 tile/board**. Board bị gọi lỗi
nếu **bất kỳ** tile nào kêu, nên tỉ lệ báo động giả ở mức board là
`1 − (1 − p)³⁶`:

| FPR mỗi tile | Board tốt bị gọi lỗi | Board/ngày (100 board) | Phút/ngày @30 s |
|---:|---:|---:|---:|
| 5,00% | **84,2%** | 84,2 | 42,1 |
| 2,00% | 51,7% | 51,7 | 25,8 |
| 1,00% | **30,4%** | 30,4 | 15,2 |
| 0,50% | 16,5% | 16,5 | 8,3 |
| **0,10%** | **3,5%** | 3,5 | 1,8 |
| 0,05% | 1,8% | 1,8 | 0,9 |

**Đọc bảng này trước khi đọc bất cứ AUROC nào.** Một model "99% chính xác" ở mức
tile nghĩa là FPR 1% ⇒ **cứ 3 board tốt thì 1 bị chặn**. Dây chuyền sẽ tắt tính
năng trong tuần đầu. Ngưỡng phải chỉnh theo cột này, không theo AUROC.

Lưu ý cột "phút/ngày" trông nhỏ ngay cả ở mức tệ — vì 30 giây/lần xem là rẻ.
Cái đắt không phải thời gian mà là **lòng tin**: 84% board tốt bị gọi lỗi thì
công nhân bấm "bỏ qua" theo phản xạ, và lúc đó lỗi thật cũng bị bỏ qua.

---

## 6. Dataset: cái gì có sẵn, cái gì phải tự chụp

| Nguồn | Quy mô | Board trần hay đã lắp? | Có xước / lem thiếc? | Dùng được? |
|---|---|---|---|---|
| **VisA `pcb1`–`pcb4`** | ~1.000 tốt + 100 lỗi mỗi bộ | **đã lắp** (transistor, tụ, chip) | **có**: scratch, dent, colour spot, crack, misplacement, missing | ✅ **nguồn chính** |
| MPI-PCB (Zenodo 8213098) | 30 ảnh tốt **cùng một board** đã có trong repo; upstream có thêm `test/defect` | đã lắp | chưa kiểm | ✅ cho M2 một SKU |
| PCB-DSLR (CVL) | 175 ảnh, 165 board | đã lắp | không có nhãn lỗi | ảnh nền, không nhãn |
| RF100 printed-circuit-board | 177 cảnh | đã lắp | không | ảnh nền |
| DeepPCB | 1.500 cặp | **trần** | ❌ | ✗ khác bài toán |
| HRIPCB / PKU-Market-PCB | 1.386 → 10.886 ảnh | **trần** | ❌ (6 lỗi đường mạch) | ✗ |
| PCB-Defect (2025) | 230 ảnh, 1.704 lỗi | **trần, một lớp** | ❌ (missing pad, mouse bite, open, short, spur, spurious copper) | ✗ |
| SolDef_AI | 428 ảnh | đã lắp, **macro 1–3 µm/px** | mối hàn, không phải mặt board | ✗ lệch tỉ lệ ~20 lần |
| MVTec AD | 15 nhóm | — | **không có nhóm PCB nào** | ✗ |

**Phán quyết thẳng về câu hỏi của bạn ("dataset có sẵn không"):**
**Có — VisA.** Đây là bộ công khai duy nhất tôi tìm được vừa là board **đã gắn
linh kiện**, vừa có nhãn lỗi **bề mặt** (xước/lõm/đốm màu/nứt), vừa đủ ảnh tốt để
train anomaly. Và nó **đã nằm trong đường ống của repo**:
`scripts/fetch_visa_reference_set.py` kéo từ `BrachioLab/visa` với split
`pcb2.train`, hiện đã lấy 30 ảnh về `datasets/reference_sets/visa_pcb2_30/`.

Việc phải làm nhỏ hơn nhiều so với "đi tìm dataset": script hiện **cố tình từ
chối ảnh lỗi** — nó raise khi `label != 0`, vì viết cho mục đích enrollment
Golden. Mở rộng nó để lấy `pcb1..pcb4` cả `train` lẫn `test` là việc vài chục
dòng, không phải một cuộc khảo sát.

**Cái không dataset nào cho:** ảnh board **của chính dây chuyền bạn**. Repo hiện
có đúng **3 ảnh điện thoại** ở `real_pcb/`. 235 ảnh "toàn board" đang có là ảnh
công khai (CVL 175 + MPI 30 + PCB-DSLR 30). Và với anomaly detection thì điều này
nghiêm trọng hơn với các bước khác: **model anomaly là của riêng từng SKU** — bộ
nhớ patch dựng từ board A không nói được gì về board B. Hiện dự án có đúng **một
SKU với 30 khung ảnh tốt** (MPI gas pump).

---

## 7. Lộ trình

**Giai đoạn 0 — trả lời "hướng này có sống không" (làm được ngay, dữ liệu đã có)**
1. Mở `fetch_visa_reference_set.py` cho `pcb1..pcb4`, cả `train` và `test`.
2. Chạy PatchCore qua `anomalib` trên `pcb2`, **đo AUROC riêng cho nhóm PCB**,
   không dùng con số trung bình toàn VisA.
3. Dựng đường cong FPR/tile → false call mức board theo bảng §5, tìm ngưỡng cho
   FPR ≤ 0,1%, rồi xem recall còn lại bao nhiêu.
   **Đây là cổng đi/dừng.** Nếu ở FPR 0,1% mà recall dưới ~0,5 thì hướng anomaly
   toàn board chưa dùng được, và nên dừng lại nói thẳng thay vì đi tiếp.

**Giai đoạn 1 — đo trên board thật của dự án**
4. Đo sai số căn còn lại của bước 2 + 3.5 trên 30 khung MPI cùng board. Anomaly
   chỉ có nghĩa nếu sai số này nhỏ hơn kích thước lỗi cần bắt.
5. Dựng bộ nhớ PatchCore từ 30 ảnh tốt MPI, chạy trên `test/defect` upstream.

**Giai đoạn 2 — nối vào app (chỉ sau khi qua cổng ở bước 3)**
6. Ô model `models/active/board_anomaly/` + `model_manifest.json`, mục UI riêng.
7. Mặc định **TẮT**, không tự nạp.

**Giai đoạn 3 — M1 lỗi có tên**
8. Gán nhãn `solder_splash` / `bridge_between_parts` / `scratch` trên tile, tái
   dùng chính app gán nhãn HTML đã có.
9. Train, đo, giữ tắt cho tới khi vượt cổng.

**Chờ phần cứng (không code được):** ảnh từ chính dây chuyền; và với vết xước
rộng 0,1 mm thì camera 25 µm/px sẽ đổi hẳn cục diện (2 px → 4 px bề rộng).

---

## 8. Cách nối vào app

- **Ô model:** thêm `board_anomaly` vào `STAGE_FOLDERS`/`_MODEL_SLOTS` trong
  `app/streamlit_app.py` và `aoi_pipeline/model_registry.py`. Nhớ: registry chỉ
  liệt kê `.onnx` **có `model_manifest.json` bên cạnh**.
- **Bắt buộc vào `_NO_AUTO_ADOPT`**, như ô `lead_detector`. Một model bật sẵn mà
  chưa đo trên board thật là cách nhanh nhất để dây chuyền mất lòng tin.
- **Manifest phải khai thêm** (ngoài các trường chuẩn): `sku_id` — bộ nhớ patch
  thuộc về board nào; `reference_images` — dựng từ mấy ảnh tốt nào (sha256);
  `registration_requirement` — sai số căn tối đa còn hợp lệ; `fpr_per_tile` và
  `board_false_call_rate` tại ngưỡng đang đặt; `not_a_solder_grader: true` — ghi
  thẳng ranh giới đã làm chết tầng cũ.
- **Config:** section mới `board_anomaly` trong `_default_config()`. Nhớ bài học
  của `1447ed5`: `_use_model_entry` ghi vào key mà `_default_config` không tạo
  ⇒ app chết đúng lúc bấm chọn model. Có sẵn test canh việc này
  (`test_default_config_declares_every_section_the_sidebar_writes_into`).
- **Không có model = no-op tuyệt đối**, không phải lỗi — đúng lệ của mọi stage
  hiện có.
- **Cấm tuyệt đối:** tầng này **không** được ra phán quyết trên mối hàn. Nó chỉ
  báo "vùng này khác board chuẩn". Vượt ranh giới đó là lặp lại tầng đã gỡ.

---

## 9. Rủi ro

1. **Sai số căn giả dạng lỗi.** Rủi ro số một, và đã có tiền lệ ghi trong dự án:
   *"phép căn sai trông y hệt phép căn đúng nếu chỉ nhìn residual"*. Giảm thiểu:
   đo sai số căn **trước**, và đưa ngưỡng căn vào manifest như điều kiện chạy.
2. **False call giết tính năng.** Xem §5. Giảm thiểu: chỉnh ngưỡng theo cột
   board-level, không theo AUROC; báo cáo cả hai trong manifest.
3. **Một model cho một SKU.** Dự án có đúng một SKU đủ ảnh tốt. Mỗi board mới là
   một lần enrollment mới. Phải nói rõ với người vận hành, đừng để họ tưởng đây
   là model dùng chung.
4. **VisA khác miền với dây chuyền.** Giống hệt bài học của model lượt 2
   (`bootstrap_only`). Số của giai đoạn 0 là số **khả thi**, không phải số
   production.
5. **Chồng lấn ngầm với 6.2.** Anomaly sẽ kêu ở mối hàn xấu, vì mối hàn xấu đúng
   là bất thường. Nếu không chặn, một mối hàn lỗi bị đếm hai lần bởi hai tầng
   với hai phán quyết khác nhau. Giảm thiểu: **che vùng ROI mối hàn** khỏi bản đồ
   anomaly, để mỗi pixel chỉ có một chủ.

---

## 10. Câu hỏi cần bạn quyết

1. **Đồng ý làm M2 (anomaly) trước và M1 (lỗi có tên) sau chứ?** M2 không cần
   ảnh lỗi để train — mà ảnh lỗi đúng là thứ chúng ta không có.
2. **Đồng ý với cổng đi/dừng ở giai đoạn 0 chứ?** Cụ thể: nếu ở FPR 0,1%/tile mà
   recall trên VisA PCB dưới ~0,5 thì tôi **dừng** và báo lại, thay vì đi tiếp
   vào giai đoạn 1–3.
3. **Ngưỡng false call mức board nào là chấp nhận được với bạn?** Bảng §5 cho
   thấy 1%/tile ⇒ 30% board bị chặn. Tôi đề xuất trần **3,5% board** (tức
   0,1%/tile) — nhưng đây là quyết định vận hành, không phải kỹ thuật.
4. **Dây chuyền của bạn có bao nhiêu SKU?** Anomaly là một model một SKU. Nếu là
   1–3 SKU thì kế hoạch này chạy được; nếu là hàng chục thì chi phí enrollment
   đổi hẳn bài toán và ta nên bàn lại.
5. **Có chấp nhận việc tầng này KHÔNG phán quyết mối hàn không?** Nó sẽ im lặng ở
   vùng ROI mối hàn (bị che). Nếu bạn muốn nó nói cả về mối hàn thì ta đang dựng
   lại đúng tầng đã gỡ ngày 28/08, và tôi khuyên không.

---

Xem thêm: `Docs/ke_hoach_phan_nhom_package.md`,
`Docs/tien_do_detect_2_luot.md`, `Docs/danh_gia_model_6_2.md`,
`datasets/reference_sets/README.md`.
