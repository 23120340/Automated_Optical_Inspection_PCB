# Ghi chú: Golden ghép từ nhiều ảnh — chưa có kế hoạch

> **Đây chưa phải kế hoạch.** Chỉ là ghi lại vấn đề, những gì đã tồn tại trong
> repo, và **một xung đột thiết kế** phải giải trước khi viết kế hoạch thật.
> Ghi 2026-09-05 theo yêu cầu.

---

## 1. Vấn đề

Một lần chụp chỉ có **một vùng nhìn chính diện** — vùng nằm dưới trục quang.
Càng ra xa tâm, linh kiện càng bị nhìn nghiêng: thân che mất pad phía xa, chân
bên khuất không thấy được, và toạ độ tâm lệch theo chiều cao linh kiện
(parallax).

Với AOI thì đó không phải chuyện thẩm mỹ. ROI mối hàn được đặt theo hình học
của thân linh kiện; thân bị nhìn nghiêng thì ROI lệch theo, và bước 6.2 chấm
một vùng không phải mối hàn.

Ý tưởng: chụp **nhiều ảnh**, mỗi ảnh chính diện một vùng khác nhau, rồi ghép
thành một ảnh trực giao duy nhất. Từ ảnh đó mới dựng sơ đồ dùng cho các lượt
kiểm sau.

---

## 2. Đã có gì trong repo

| | có | ghi chú |
|---|---|---|
| Nhận **nhiều** ảnh lúc enroll | ✅ | `golden/enrollment.py`, `min_images = 3` |
| **Chọn** ảnh tốt nhất trong số đó | ✅ | `select_reference()` chọn *medoid* sau khi qua cổng focus/clipping/exposure |
| **Ghép** nhiều ảnh thành một | ❌ | **cố ý không làm** — xem §3 |
| Ảnh → CAD/netlist/recipe | 📄 kế hoạch | `ke_hoach_so_hoa_mach_pcb_aoi.md` |
| Nắn phối cảnh toàn bo | ⚠️ một phần | `imaging/alignment.py` nắn theo bo, không nắn theo *độ cao linh kiện* |

Nói cách khác: repo đã biết **chọn** một ảnh tốt trong nhiều ảnh, và đã có kế
hoạch biến **một** ảnh tốt thành sơ đồ. Chỗ trống nằm đúng giữa hai thứ đó —
**không ảnh nào tốt ở mọi vùng**.

---

## 3. Xung đột thiết kế phải giải trước

`aoi_pipeline/golden/enrollment.py` mở đầu bằng đúng câu này:

> *"The selector **deliberately** returns one of the supplied files. It **never
> blends, stacks, or otherwise synthesises** a reference image: a Golden image
> must remain **traceable to a real acquisition**."*

Đây không phải thiếu sót, mà là ràng buộc có lý do: ảnh Golden là chuẩn để
phán một bo là PASS hay NG. Nếu nó là ảnh tổng hợp thì khi có tranh chấp,
không chỉ được vào một lần chụp thật nào cả.

**Ghép ảnh vi phạm ràng buộc đó.** Kế hoạch thật phải trả lời được: *lấy gì
thay cho tính truy xuất mà ràng buộc này đang bảo vệ?*

Hướng khả dĩ — chưa đo, chưa chọn:

1. **Giữ nguyên bản gốc + phép biến đổi.** Ảnh ghép là *dẫn xuất*, và lưu kèm
   danh sách ảnh nguồn (hash nội dung) + ma trận homography của từng ảnh. Khi
   tranh chấp thì dựng lại được, và chỉ được về đúng pixel gốc nào đã đóng góp
   cho vùng đang xét.
2. **Không ghép ảnh, chỉ ghép SƠ ĐỒ.** Mỗi ảnh chỉ đóng góp vùng chính diện
   của nó vào bản đồ linh kiện/pad; ảnh Golden vẫn là một file thật. Cách này
   giữ nguyên ràng buộc, nhưng phải định nghĩa "vùng chính diện" đo được.
3. **Đổi phần cứng thay vì đổi phần mềm.** Telecentric lens hoặc khoảng cách
   chụp xa hơn làm giảm hẳn phối cảnh. Đắt, nhưng không phải giải bài toán
   ghép.

---

## 4. Những gì phải đo trước khi viết kế hoạch

Chưa có số nào cho phần này, nên mọi lựa chọn ở §3 hiện là phỏng đoán.

1. **Sai lệch phối cảnh thật là bao nhiêu?** Đo dịch chuyển tâm của cùng một
   linh kiện giữa hai lần chụp lệch tâm quang — theo px và theo mm, tách theo
   khoảng cách tới tâm ảnh và theo **chiều cao linh kiện**.
2. **Nó có vượt dung sai ROI không?** Nếu lệch nhỏ hơn lề ROI hiện tại thì
   toàn bộ việc ghép không đáng làm. Đây là câu rẻ nhất và phải hỏi trước.
3. **Bao nhiêu ảnh là đủ?** Phụ thuộc góc mở ống kính và cỡ bo.
4. **Ghép xong có còn đo được không?** Đường nối giữa hai ảnh đi qua giữa một
   linh kiện thì ROI của nó nằm trên hai nguồn khác nhau, độ sáng khác nhau —
   6.2 chấm bằng ngưỡng ảnh nên chuyện đó ảnh hưởng trực tiếp.

---

## 5. Liên quan

- `Docs/ke_hoach/ke_hoach_so_hoa_mach_pcb_aoi.md` — ảnh → CAD/netlist/recipe,
  **giả định đã có một ảnh Golden tốt**. Ghi chú này là tiền đề còn thiếu của
  nó.
- `Docs/ke_hoach/ke_hoach_phan_nhom_package.md` §10.2 — chốt topology + ROI một
  lần lúc tạo golden recipe. Việc đó chỉ đúng nếu ảnh Golden đo được ở **mọi**
  vùng của bo.
- `Docs/thiet_ke/yeu_cau_phan_cung_camera.md` — nơi đặt phương án 3.
