# Báo cáo tiến độ — Tuần 3

**MSSV 23120340** · **31/08 – 06/09/2026** · Dự án AOI PCB

> Chỉ báo cáo phần **phát sinh trong tuần**. Nền tảng pipeline 0–6.2, hạ tầng
> train và các bước đã ổn định đã báo cáo ở kỳ trước
> ([bao_cao_tien_do.md](../bao_cao_tien_do.md), 31/08) nên không nhắc lại.

| | đầu tuần | cuối tuần |
|---|---:|---:|
| Bộ test | 1012 pass | **1157 pass** |
| Box thân linh kiện đã khoanh | dở dang | **9.486 box / 95 tile**, đã chốt |
| Tập kiểm gán nhãn tay | chưa có | **750 box, 667 có nhãn họ (88,9%)** |

---

## 1. Detector thân linh kiện — chuyển sang 1 lớp và đưa vào luồng

Bỏ hướng detect 22 lớp, chuyển sang **một lớp `component`** chỉ khoanh *thân*
linh kiện (loại chân/pad). Lý do: nhãn công khai không nhất quán ở mức lớp, còn
"thân linh kiện" thì gán tay được và ổn định.

- Hoàn tất và sao lưu **9.486 box** thân trên 95 tile, 34 bo vật lý.
- Sửa ba lỗi lộ ra ở lần train thật đầu tiên: ảnh công khai quá lớn phải **cắt
  tile** tại nguồn, `imgsz`/`batch` không khớp bộ nhớ T4, và `dataset_root` phải
  tự tìm thay vì sửa tay trên Kaggle.
- Promote **ver2**: nâng trần số detection **300 → 600**. Trần cũ nằm ngay trong
  đồ thị ONNX (YOLO26 không có NMS ngoài) nên phải khai lúc export, không chỉnh
  được lúc chạy. Kiểm chứng ở ngưỡng conf 0,02: bản cũ dừng đúng 300, bản mới ra
  384.

## 2. Bước 5.2 (phân nhóm package) — đổi từ train model sang **luật**

Đo trên 16.632 box có nhãn footprint: chip 2 chân chiếm **86,5%**, nên "luôn
đoán chip" đã đúng 86,5%; luật hình học toàn cục tốt nhất chỉ đạt **84,5%** —
*tệ hơn* baseline. Kết luận: không train model package, mà **nối luật sau
classifier 6.1** để biết trước họ, rồi chỉ chia nhỏ *bên trong* một họ.

Đã cài bộ luật, ba chốt an toàn, và một **cổng nghiệm thu riêng** (mất một pad
baseline là fail ngay, không cân nhắc đánh đổi).

**Kết quả đo được trong tuần:**

| việc | trước | sau |
|---|---:|---:|
| Luật bỏ qua (không kết luận được) | 15 / 39 | **3 / 39** |
| Chân rơi vào trong box thân | 18 / 60 | **0** |
| Tách họ `capacitor` (trụ đứng ↔ chip) | chưa có luật | **90,5%** vs baseline 68,2% |

Luật vẫn **mặc định TẮT** — chưa đủ bằng chứng để bật trên dây chuyền (xem §5).

## 3. Tập kiểm gán nhãn tay 750 box

Dựng tập phân tầng 750 box trên 34 bo, gán nhãn **bằng mắt** (không dùng 6.1,
vì tập này dùng để *đo* 6.1). Đã gán được **667/750 = 88,9%** nhãn họ.

Tập này mở khoá luôn phép đo tách tụ ở §2 — trước đó bế tắc vì chỉ có 21 mẫu.

> Ảnh crop của tập nằm ngoài git (`datasets/` bị `.gitignore` chặn: bộ ảnh
> gốc CVL PCB-DSLR giới hạn nghiên cứu phi thương mại, ràng buộc đi theo cả
> tile phái sinh). File nhãn là văn bản thuần nên đưa vào repo được nếu cần.

## 4. Năm lỗi tự tìm ra và đã sửa

Đây là phần đáng kể nhất của tuần, vì **ba trong năm lỗi nằm trong chính công cụ
đo**, tức chúng làm mọi con số trước đó mất giá trị.

1. **Cổng nghiệm thu bị mù.** Nó dựng lại ROI bằng đường riêng thay vì gọi hàm
   runtime, nên bỏ qua đúng thứ nó phải đo. "Trước" và "sau" bằng nhau *theo cấu
   trúc* ⇒ mọi kết quả PASS trước đó là **rỗng**. Trên mẫu kiểm chứng: cổng thấy
   4/4 → 4/4 trong khi runtime thật là 4/4 → **0/4**.
2. **ROI đặt sai cạnh.** Luật đo được chân nằm ở cạnh nào rồi **vứt đi**; bước
   5.5 luôn dựng ROI trên hai cạnh *dài* của thân. Hai thứ lệch nhau là ROI rơi
   trọn vào hai cạnh không có chân. Đã truyền cạnh đo được xuyên suốt xuống 5.5.
3. **Mất ROI khi phân loại ĐÚNG.** Với thân gần vuông, gói `tru_dung` sinh 2 ROI
   trong khi `two_terminal` sinh 4 — tức gán đúng lớp lại xoá mất hai vùng kiểm.
   Đo được 55% tụ trụ có hai cạnh lệch dưới 10%, nên "cạnh nào dài hơn" là nhiễu.
4. **Khung cắt sai câu hỏi.** Khung cắt của tập gán nhãn được chỉnh để thấy *pad*
   (trả lời câu **gói**), nên nó luôn cắt mất *silkscreen designator* (`R902`,
   `C450`) — thứ trả lời câu **họ**. Sửa thành **hai khung mỗi box**; riêng việc
   này gán thêm được **57/140 = 41%** số ảnh đang bế tắc.
5. **Ưu tiên nguồn bằng chứng.** Khi có dữ liệu CAD, hệ thống thay hình học
   nhưng không xoá cạnh chân do luật ảnh đo — lần dựng ROI sau sẽ trộn hai nguồn.

Mỗi lỗi đều kèm test canh; gỡ bản vá ra thì test đỏ (đã kiểm từng cái).

## 5. Đang chặn / tuần sau

- **Bo dây chuyền còn ở công ty.** Đã soạn xong quy trình chụp và script kiểm bộ
  ảnh; chờ mang bo về để chụp và đo.
- **Cổng mới đo tới bước dựng ROI, chưa đo đầu ra cuối.** Đo tay đầu-cuối trên
  bo thật: bật luật làm **82 ROI dịch chỗ**, nhưng độ phủ mối hàn **không đổi**
  (599/601 cả hai chế độ). Cần biến phép đo này thành cổng chạy lại được trước
  khi bật luật.
- **83 ô chưa kết luận được họ**, nằm ở vùng không có designator trong tầm nhìn.
  Bỏ qua được, nhưng phải ghi kèm **thiên lệch cỡ**: phần bỏ có trung vị cạnh dài
  17 px so với 82 px của phần giữ, nên số đo 6.1 trên phần còn lại sẽ lạc quan.
- **Chưa có tập nghiệm thu khoá** theo bo cho đường luật.

---

### Tài liệu chi tiết

| nội dung | file |
|---|---|
| Phân nhóm package (luật, chốt an toàn, cổng, các phép đo) | [ke_hoach_phan_nhom_package.md](../../plans/ke_hoach_phan_nhom_package.md) |
| Golden ghép sơ đồ + quy trình chụp bo | [ke_hoach_golden_ghep_so_do.md](../../plans/ke_hoach_golden_ghep_so_do.md) |
| Detect mối hàn 2 lượt (bảng công việc) | [tien_do_detect_2_luot.md](../tien_do_detect_2_luot.md) |
| Đánh giá classifier 6.1 | [danh_gia_classifier_6_1.md](../../evaluation/danh_gia_classifier_6_1.md) |
