# Kế hoạch: mối hàn KHÔNG thuộc linh kiện nào

Ghi 2026-09-06.

> **Toàn bộ đường dựng ROI hiện nay neo vào thân linh kiện.** Cái gì không nằm
> dưới một linh kiện thì không có ROI, không được chấm, và **không xuất hiện
> trong bất kỳ báo cáo nào** — kể cả báo cáo "không có lỗi".

---

## 1. Lỗ hổng, nói chính xác

`SolderJointCropper.derive()` lặp trên `detections`, tức box **thân linh kiện**.
Bước 2 (`detect_leads_in_components`) cũng chỉ chạy **bên trong một cửa sổ quanh
từng linh kiện**. Hệ quả:

- Không có linh kiện ⇒ không có cửa sổ ⇒ không có ROI ⇒ **không ai nhìn**.
- Và vì không có ROI, kết quả không nói "chỗ này chưa kiểm" — nó **im lặng**.

Những thứ rơi vào lỗ hổng này:

| loại | vì sao có mối hàn | thấy được từ trên xuống? |
|---|---|---|
| Test point | pad hàn hở, hoặc có chân cắm | có |
| Thermal pad / tản nhiệt hàn | diện tích hàn lớn dưới hoặc cạnh linh kiện | một phần |
| Chân vỏ chắn (shield can) | hàn xuống bo quanh chu vi | có |
| Lỗ xuyên (through-hole) không có linh kiện | jumper, dây nối, lỗ chờ | có |
| Pad chờ / vị trí không lắp (NP) | **không** được hàn — phải rỗng | có |
| Via | không phải mối hàn, **không** được tính là lỗi | có |

Hai dòng cuối quan trọng ngang những dòng trên: **thêm ROI vào chỗ không được
hàn là sinh ra báo giả**, và via thì nhiều vô kể.

## 2. Chưa đo được lỗ hổng này lớn bằng nào — và vì sao

Câu cần trả lời: *trên một bo thật, bao nhiêu phần trăm mối hàn không nằm dưới
một linh kiện?* **Chưa có số**, vì mọi công cụ đang có đều neo vào linh kiện:
lead detector chỉ chạy trong cửa sổ quanh linh kiện, nên **về nguyên tắc nó
không thể tìm thấy** mối hàn ở vùng trống. Đo bằng nó là đo cái thước tự vẽ.

> **Một số đo dễ bị đọc nhầm.** Trên 3 ảnh thật, ROI **suy từ hình học** chỉ phủ
> **666/2.170 = 31%** số mối hàn mà bước 2 tìm được; phần còn lại do bước 2 +
> fusion cứu. Con số đó nói bước 2 đang gánh phần lớn công việc — **nó không
> nói** gì về mối hàn ngoài linh kiện, vì cả 2.170 mối hàn đó đều nằm trong cửa
> sổ của một linh kiện nào đó.

**Cách đo đúng, chọn một trong hai:**

1. **Gán tay một tile**: khoanh *mọi* mối hàn nhìn thấy được, rồi đếm bao nhiêu
   cái không nằm trong hộp linh kiện nào. Rẻ, một tile là đã có tỉ lệ.
2. **Chạy detector mối hàn trên toàn ảnh** theo lưới trượt thay vì theo cửa sổ
   linh kiện, rồi so với tập ROI hiện tại. Cho số trên nhiều bo, nhưng recall
   của detector ở vùng trống thì chưa ai đo.

Nên làm (1) trước: nó trả lời được câu *"việc này có đáng làm không"* với chi
phí một buổi.

## 3. Ba nguồn có thể lấp, xếp theo độ tin

| nguồn | cho gì | có chưa |
|---|---|---|
| **IPC-D-356** | toạ độ **từng land một**, kể cả land không có linh kiện | ❌ chưa xin được; repo đã có bộ đọc |
| **CAD / Gerber** | vị trí pad, nhưng thường thiếu thermal pad và land cơ khí | ❌ chưa có |
| **Golden recipe** (người khoanh tay) | đúng những gì người duyệt khoanh | ✅ hạ tầng đã có (`SlotRecipe.fixed_roi_xyxy`) |

**Đường khả thi ngay là Golden recipe.** Nó không cần dữ liệu ngoài, và nó khớp
với việc dây chuyền kiểm **vài mẫu PCB cố định** — khoanh tay một lần cho mỗi
mẫu là trả đủ công.

## 4. Việc phải làm

| | việc | phụ thuộc |
|---|---|---|
| 1 | **Đo lỗ hổng** bằng cách gán tay một tile (§2 cách 1) | không |
| 2 | Cho `SlotRecipe` mang slot **không gắn linh kiện**: `expects_solder` = có / không / cấm | (1) nói rõ có đáng không |
| 3 | Bước 5.5 dựng ROI cho các slot đó **ngoài** vòng lặp theo linh kiện | (2) |
| 4 | Bước 6.2 chấm chúng bằng đúng bộ chấm hiện có; slot `cấm hàn` thì **có hàn là lỗi** | (3) |
| 5 | Đọc IPC-356 để tự sinh slot thay vì khoanh tay | có file IPC |

Bước 2 là chỗ dễ làm ẩu nhất. Một slot phải nói rõ nó thuộc **loại nào trong
ba**: *phải có hàn* (test point đã dùng), *có cũng được* (pad chờ), *không được
có* (vị trí NP). Gộp ba loại này làm một là hoặc bỏ sót lỗi, hoặc đẻ ra báo giả
— và báo giả thì làm người vận hành mất tin vào cả hệ thống.

## 5. Nguyên tắc giữ khi làm

- **Không suy "chỗ này không cần kiểm" từ việc không thấy linh kiện.** Đó đúng
  là loại suy luận trên sự vắng mặt mà kế hoạch package §8.1 đã cấm ở chỗ khác.
- **Via không phải mối hàn.** Đưa via vào tập ROI là sinh hàng nghìn báo giả
  mỗi bo.
- ROI thêm vào phải **đếm được và tách được** khỏi ROI của linh kiện, để đo
  riêng tỉ lệ báo giả của phần mới — trộn chung thì không biết phần nào hỏng.

## 6. Nghiệm thu

| mã | kịch bản | mong đợi |
|---|---|---|
| NL01 | Bo có test point đã hàn | Có ROI, được chấm, xuất hiện trong báo cáo |
| NL02 | Vị trí NP (không lắp) nhưng **có** vết hàn | Báo lỗi — đây là lỗi thật, hiện đang không ai thấy |
| NL03 | Bo đầy via | Không ROI nào sinh trên via |
| NL04 | Slot không gắn linh kiện, ảnh không nhìn thấy vùng đó | Báo "không kiểm được", **không** báo đạt |
| NL05 | So tỉ lệ báo giả trước/sau khi bật | Phần ROI mới đo được riêng, không trộn với ROI linh kiện |
