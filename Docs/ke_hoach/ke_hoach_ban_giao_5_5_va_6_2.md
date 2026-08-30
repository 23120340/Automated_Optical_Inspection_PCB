# Kế hoạch bàn giao: phần còn lại của bước 5.5 / 6.2

> Soạn 2026-08-24. Viết cho **agent tiếp theo** nhận việc, không phải cho người
> đã ở trong ngữ cảnh. Mọi con số ở đây đều đo được, và chỗ nào chưa đo thì ghi
> rõ là chưa đo.
>
> Thay thế thứ tự P0→P5 trong `Docs/ke_hoach/phuong_an_sua_roi_moi_han_v2.md`. Tài liệu
> đó vẫn đáng đọc cho phần thiết kế; chỉ phần **thứ tự ưu tiên** là hết hạn.

## Bối cảnh

Kế hoạch v2 chia việc thành P0→P5. Đã làm xong P3 và một nửa P1/P2, nhưng **phép
đo đã lật lại tiền đề của nửa còn lại**:

- P1/P2 giả định "cho mọi thứ đọc ảnh chưa tăng cường". Đo ra thì **ngược nhau
  theo khâu**: khoanh ROI thì ảnh tăng cường tốt hơn (0 so với 3 ROI rơi trên
  chữ lụa), chấm điểm thì ảnh nguồn tốt hơn (2 so với 12 `bridge`). Đã sửa đúng
  chỗ nó gây hại (`65936b4`), nên phần refactor còn lại của P0/P1 **mất gần hết
  động cơ** — nó chỉ còn là dọn kiến trúc.
- Hệ quả chưa gỡ: mọi ngưỡng luật 6.2 là **tỉ lệ trên mặt nạ kim loại**, mà ảnh
  chưa tăng cường cho độ phủ thấp hơn ~19 điểm phần trăm. `insufficient` đi từ
  1 lên 6 — đúng chiều dự đoán, và **chưa có mốc thật** để biết đúng hay oan.
  Quyết định của chủ dự án: giữ nguyên, không thêm lớp đệm, hiệu chuẩn sau.
- Và có **một lỗi thật trong chính thay đổi đó**, phát hiện lúc soạn tài liệu
  này — xem Giai đoạn 0.

Phần việc của phiên song song (bước 6.2 tách vai trò, detector lỗi hàn, UI đổi
tên) đã commit ở `7515191`. Cây làm việc sạch, 736 test xanh.

## Cách làm việc — đọc trước khi bắt tay

Bốn điều rút ra từ phiên trước, mỗi điều đều đã có một lần trả giá:

1. **Đo trước khi kết luận, và đo end-to-end.** Bảng probe trong plan v2 nói
   `lead_outer_ratio = 0` cho 8 ROI; chạy qua đường ống thật thì U201 **mất một
   nửa số chân**. Probe cô lập (tắt refine, chưa fusion) không thay được phép
   chạy thật.
2. **Ba giả thuyết đã chết vì phép đo** (tách bằng màu, tách bằng kết cấu, chọn
   trục bằng vành ngoài). Khi một ý tưởng nghe hợp lý, hãy đo *trước* khi viết
   code cho nó.
3. **Bỏ sót nặng gấp 5 lần báo oan.** Bỏ sót là escape — board xấu ra khỏi
   chuyền; báo oan chỉ tốn công soát. Mọi phép chấm điểm phải dùng trọng số này.
4. **Luôn báo bán kính ảnh hưởng.** "Sửa được D201" là vô nghĩa nếu không kèm
   "và 38 linh kiện còn lại không đổi một pixel". Chỉnh 2 tham số trên 3 linh
   kiện của 1 ảnh là *ít* — nói rõ ra, đừng giấu.

Không tự ý làm: xoá ROI dựa trên mask khi chưa qua shadow mode; thêm OCR engine;
inpaint chữ lụa. Xem mục "Đã đo là ngõ cụt" ở cuối.

---

## Giai đoạn 0 — ĐÃ XONG (`d58c511`)

> Sửa xong 2026-08-24. Đo lại sau khi sửa: đường 1 không còn ghi đè; đường 2 cho
> tương quan **0,979–0,991** (trước là 0,176–0,269); ca có resize cũng chạy (hai
> khung đều ra 512×512). Ba mốc trên board chuẩn giữ nguyên. 744 test xanh.
>
> **Một ngõ cụt phát hiện lúc sửa, đã ghi thành test:** định thêm một vân tay
> nội dung để chặn cặp ảnh không khớp — **không được**. Cặp nhầm ghi điểm bằng
> đúng cặp đúng (0,985 ở mọi độ phân giải), còn fixture đúng lại thấp hơn cặp
> nhầm. Lý do là bản chất: `align` warp ảnh board **về hệ toạ độ của golden**,
> nên sai ở đây là *nhầm board* chứ không phải *lệch toạ độ*, mà golden thì
> trông giống hệt board đang kiểm. Chỗ này canh bằng **cấu trúc** — hai khung ra
> khỏi cùng một kết quả — không phải bằng phép kiểm nội dung.

Mô tả gốc của lỗi giữ lại bên dưới để đối chiếu.

## Giai đoạn 0 — Lỗi cần sửa ngay (chặn mọi thứ khác)

`radiometric_image` là **trạng thái đặt kèm theo tác dụng phụ của
`preprocess()`**, và cách đó sai theo hai đường. Cả hai đều là lỗi của thay đổi
`65936b4`, không phải lỗi có sẵn.

**Đường 1 — nạp ảnh golden là hỏng ngay, không cần alignment.**
`app/streamlit_app.py:1200` gọi `bridge.preprocess(reference)` để đưa ảnh golden
về cùng miền. Lời gọi đó chạy vào `AOIPipeline.preprocess`, và **ghi đè
`self.radiometric_image` bằng ảnh GOLDEN** thay vì ảnh board đang kiểm. Bước 6.2
sau đó chấm điểm trên pixel của một tấm ảnh khác. Cùng kích thước, nên phép kiểm
của `_radiometric_crops` không bắt được.

**Đường 2 — `align()` không warp theo.** `run()` chạy
`aligned = self.align(preprocessed.image, reference)` rồi chấm trên
`aligned.image`, trong khi `radiometric_image` giữ nguyên chưa căn. Lại cùng
kích thước, khác nội dung.

Chưa lộ ra vì mọi phép đo và test hiện có đều **không truyền ảnh tham chiếu**.
Nhưng app thì có — ngay khi người dùng nạp golden image.

**Cách sửa đúng: đừng để nó là trạng thái của pipeline.** Cho khung chưa tăng
cường đi *cùng* khung đã tăng cường, chứ không nằm rời trên `self`. Đây chính là
ý `FrameBundle` của plan v2 — hoá ra phần đó có động cơ thật, chỉ ở quy mô nhỏ
hơn nhiều: chỉ cần một cặp (ảnh phân tích, ảnh chưa tăng cường) đi kèm nhau qua
align → ROI → chấm điểm.

Tối thiểu phải có:

- `preprocess` trả về cặp ảnh thay vì đặt lên `self` (hoặc `align` nhận cả cặp
  và warp cả hai bằng `AlignmentResult.homography`, `aoi_pipeline/models.py:196`).
- `radiometric_image` **đặt về `None`** khi không warp được. Mất tính năng thì
  chấp nhận được; cắt nhầm pixel thì không.
- Hai test cho hai đường: (a) nạp golden rồi chấm — pixel phải thuộc ảnh board;
  (b) `run()` với golden **lệch vị trí** — ROI phải chấm đúng chỗ.

Trước khi sửa xong, coi như mọi kết luận 6.2 trên phiên có golden image là không
tin được.

---

## Giai đoạn 1 — Hiệu chuẩn lại ngưỡng 6.2

Hệ quả trực tiếp của `65936b4`, và nó **chặn việc tin bất kỳ đầu ra nào của
6.2**.

**Đã có sẵn công cụ, đừng viết lại:** `scripts/calibrate_solder_thresholds.py`
đo phân bố đặc trưng trên board **đã được chấp nhận** rồi đề xuất ngưỡng theo
phân vị. Nó gọi `pipeline.run()` nên **tự động phản ánh** nguồn pixel mới.

```bash
python scripts/calibrate_solder_thresholds.py <thu-muc-board-tot> \
    --model models/active/detector/best.onnx \
    --output config/solder_thresholds.json \
    --dump-features Docs/bench/solder_features_<ngay>.csv
```

- Cần **board đã chấp nhận**, không cần gán nhãn lỗi. Rẻ hơn nhiều so với gán
  nhãn từng mối hàn.
- Bảy ngưỡng cần đặt lại, tất cả ở `SolderGradingConfig`
  (`aoi_pipeline/config.py:697` trở đi): `missing_solder_ratio`,
  `insufficient_solder_ratio`, `excess_solder_ratio`, `insufficient_span_ratio`,
  `cold_specular_ratio`, `cold_contrast`, `bridge_edge_contact`.
- **Chạy hai lần**, một với `prefer_radiometric_image=True` và một với `False`,
  ghi cả hai bộ số vào tài liệu. Đó là bằng chứng cho việc đổi nguồn pixel, và
  là đường lui nếu bộ mới tệ hơn.
- Cổng: trên `00001__1024__1648___4120.png`, số `bridge` phải ≤ 2 (đã đo) **và**
  `insufficient` không được vượt quá số mối hàn thật sự thiếu thiếc khi soi tay.
  Chưa soi được thì ghi rõ là chưa có mốc, đừng tuyên bố đã tốt lên.

---

## Giai đoạn 2 — Nuisance mask chạy shadow (P4 của plan v2)

Đòn bẩy mạnh nhất còn lại, và đã có số đo ủng hộ.

Bộ phân loại 7 đặc trưng (H, S, V, L\*, a\*, b\*, độ lệch chuẩn cục bộ 5×5) tách
được ở **mức dải**:

| Dải | Có chân? | `segment_solder` hiện tại | Lọc 7 đặc trưng |
|---|---|---|---|
| SOIC-8 trên | có | 24 % | **45 %** |
| C239 cạnh trên | có | 40 % | **51 %** |
| SOIC-8 phải = chữ lụa | không | 21 % | **6 %** |
| D201 trái = viền lụa | không | 26 % | **2 %** |

Thước đo hiện tại **chồng lấn** (chữ lụa 26 % còn cao hơn dải thật 24 %); thước
đo 7 đặc trưng có khoảng trống thật. Ở mức pixel, kiểm trên vùng chưa từng thấy
đạt 78,8 %.

**Nhưng nó là một model, không phải một luật.** Đã thử tìm luật giải thích được
thay thế — ngưỡng trên S, trên `b*`, và kết hợp — cả ba không tách được. Phải
nhận nó đúng bản chất: cần dữ liệu gán nhãn, cần kiểm lại khi đổi board hoặc đổi
đèn.

- Module mới `aoi_pipeline/solder/nuisance.py`: trích 7 đặc trưng, huấn luyện từ
  vùng đã đánh dấu, chấm điểm cho một dải/ROI. Không tô đen, không inpaint.
- Nguồn nhãn: **mục "Đánh giá model" đã có sẵn**. Từ vựng bước `solder` trong
  `aoi_pipeline/modelops/model_feedback.py:89` đã đúng thứ cần: `roi_misplaced` (ROI sai
  chỗ) và `roi_missing` (thiếu ROI). Đọc bằng `load_feedback` /
  `entries_for_source`; cắt lại pixel bằng `evidence_bundle_for` — phần kiểm
  digest và từ chối khi ảnh đổi đã có sẵn và đã có test.
- **Shadow mode trước.** Ghi điểm và lý do vào metadata của ROI, **không xoá gì**.
  Chỉ bật chế độ loại sau khi có số trên vài chục dải của **nhiều board**.
- Cổng để bật loại: giảm ≥ 80 % ROI rơi trên chữ/viền, **và không mất một chân
  thật nào** trên tập kiểm đã khoá.

Điểm nối vào code: `_filter_bands_by_evenness` trong
`aoi_pipeline/solder/geometry.py` là đúng khuôn mẫu cần theo — nó đã là "một tín
hiệu phụ, im lặng khi không đủ căn cứ".

---

## Giai đoạn 3 — D201/D202

Còn **3 và 1 ROI trên viền lụa**. Chỉ số lược cố tình im lặng vì dải dưới 3 đốm,
mà cạnh 2 chân là chuyện thường (SOT-23) — đó là lựa chọn có chủ đích, không
phải thiếu sót.

Hai đường, làm cái nào có dữ liệu trước:

- **File pick-and-place thật** (đang chờ chủ dự án). Cột `footprint` cho biết
  gói SOT-23 có mấy chân, nên số đốm mong đợi thành *đã biết* thay vì phải đoán.
  Đường ống đã dùng `component.rotation` và đã có `axis_known` (`ad89694`); chỗ
  cần thêm là dùng `footprint` để chốt số chân.
- **Giai đoạn 2**, nếu nuisance mask đủ mạnh.

Lưu ý đã kiểm: `multi_pin` **không sai** với D201 — nó là SOT-23 **3 chân** thật,
và cả 3 pad đều đã khoanh đúng. Đừng đi sửa nhãn `ic` của detector với hy vọng
cứu được ca này; cả detector lẫn classifier đều gọi `ic` (0.69/0.50 và
0.85/0.84), và sửa nhãn cũng không bỏ được ROI thừa trên chữ lụa.

---

## Giai đoạn 4 — Provenance và contract profile (P0 thu gọn)

Làm **sau** ba giai đoạn trên. Phần `FrameBundle` đã bị Giai đoạn 0 lấy mất phần
lõi (cặp ảnh đi cùng nhau), nên chỗ này chỉ còn phần metadata.

Giữ lại hai phần có giá trị thật:

- `input_profile_id` + `preprocess_count` + nguồn gốc ảnh (`camera_raw`,
  `dataset_preprocessed`, `unknown`). Ảnh dataset hiện tại **đã qua tiền xử lý
  khi tạo dataset** — "ảnh gốc" trong mọi tài liệu phải hiểu là
  *source-as-received*, không phải RAW cảm biến.
- **Báo lỗi khi manifest của model đòi profile khác profile runtime.** Đây là
  phần đáng giá nhất của P0: nó biến một sai lệch im lặng thành một thông báo.

Bỏ khỏi phạm vi: `enhanced_analysis_bgr` như một nhánh riêng, và việc gỡ
letterbox kép ở classifier — cả hai chưa có số đo nào cho thấy đang gây hại.

---

## Giai đoạn 5 — Bộ định vị chân lượt 2

Theo đúng `Docs/ke_hoach/ke_hoach_fine_tune_cuc_bo.md`, không viết lại. Ba phép đo hỏng ở
mục "ngõ cụt" là lập luận mạnh nhất cho việc này: câu hỏi "đâu là chân" không
giải được bằng ngưỡng thủ công trên board này.

Board trong file docx (`00001__1024__1648___4120.png`) là **ứng viên gán nhãn
tốt nhất** hiện có: nó chứa cả ba ca khó — tụ hoá can tròn, SOT-23, và SOIC nằm
cạnh chữ lụa — trong một ảnh.

Việc còn thiếu: `scripts/feedback_to_yolo.py` (chuyển bản ghi đánh giá sang
dataset YOLO) và script train tại chỗ với `imgsz=256 batch=16 device=cpu
cache=False`.

---

## Giai đoạn 6 — Camera (P5) — đang bị chặn

Chờ phần cứng. Khi có: lưu **cả RAW/source lẫn ảnh dẫn xuất**, khoá ánh sáng,
thang đo và phơi sáng, ghi rõ profile xử lý. Chia tập theo **board/SKU**, không
chia ngẫu nhiên theo crop.

Không được dùng số đo trên ảnh dataset hiện tại để suy ngược chất lượng camera
tương lai.

---

## Đã đo là ngõ cụt — đừng làm lại

| Hướng | Vì sao hỏng |
|---|---|
| Tách chữ lụa khỏi mối hàn bằng **màu** | Cùng chữ ký: S = 47,8 so với 48,0. Ngưỡng đơn lẻ tốt nhất chỉ 65,2 % |
| Tách bằng **kết cấu** đơn thuần | Chữ lụa có Laplacian **16 004** so với chân thật 4 737 — lọc theo kết cấu giữ chữ, bỏ chân |
| Chọn trục tụ bằng **vành kim loại ngoài hộp** | Đúng 1/3: C232 và C231 có điện trở chip sát hai bên, mối hàn hàng xóm rơi vào vành đang đo |
| Đặt `lead_outer_ratio = 0` ("IC không mở box") | U201 mất một nửa số chân. Chân gull-wing nằm **ngoài** bao thân |
| Rào chắn tự phát hiện ca tách chân xấu | Ba cách (số đốm, độ đều pitch, tương phản biên dạng) đều không tách được — tương phản 60–67 ở ca hỏng, 45 và 78 ở ca chạy tốt |

**Điểm yếu đã biết, chưa giải:** phép tách chân lật kết quả **chỉ vì màu nền** —
(40,70,45) cho 12/12, (40,90,40) cho 10/12 với cùng độ sáng chân. Đã khoá bằng
test để nó hiện ra chứ không im lặng.

---

## Kiểm chứng

```bash
.venv/Scripts/python -m pytest -q                 # phải 736+ passed, 0 failed
PYTHONIOENCODING=utf-8 .venv/Scripts/python scripts/list_models.py
```

Board chuẩn để đối chiếu mọi thay đổi: `00001__1024__1648___4120.png`. Ba mốc
hiện tại, **không được tụt**:

| | |
|---|---|
| U201 (SOIC-8) | **8 ROI, 0 trên chữ lụa** |
| Toàn board | 39 linh kiện, 92 ROI |
| Chấm điểm | `bridge` 2, phải xem tay 10 |

Mỗi thay đổi phải kèm: số trước/sau, **bán kính ảnh hưởng** (bao nhiêu linh kiện
đổi), và số chân bị mất (phải là 0).

---

## Lưu ý vận hành

- Repo 404 MB, push đã hỏng vài lần với `HTTP 408`. Nếu lại hỏng: thử
  `git push --no-thin`, hoặc `git -c http.version=HTTP/1.1 push`. Kích thước
  không phải nguyên nhân — một commit 30 KB cũng từng hỏng.
- Model 105 MB đi qua **git-lfs** (`.gitattributes`). Đừng thêm trọng số lớn vào
  pack.
- Repo dùng chung với `bimhai11`. Một file chỉ nên có một người sửa tại một thời
  điểm — phiên trước đã phải dựng commit bằng `git update-index` để không cuốn
  theo phần việc chưa xong của phiên song song.
