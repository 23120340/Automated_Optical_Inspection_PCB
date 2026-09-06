# Trạng thái các kế hoạch — 2026-09-07

Trả lời hai câu: *cấu trúc còn ổn không*, và *cái gì chưa xong*.

---

## 1. Cấu trúc: ổn. Nhưng "chỉ cần fine-tune nữa" thì **không đúng**

Kiến trúc đúng và code đã nói rõ nó đúng: model **đề xuất**, recipe **quyết
định** (`golden/inspector.py`: *"this class is not an AI model"*). Đường ống
0 → 6.2 chạy được, 1.212 test xanh.

Nhưng những thứ dưới đây **không phải fine-tune** và cái nào cũng chặn (mục
gạch ngang đã xong 07/09):

| | vì sao fine-tune không giải quyết được |
|---|---|
| **Nhãn mối hàn có, nhưng SAI MIỀN** | Có **9.089 box** đã duyệt trên crop `fpic`/`winnies` — đó là dữ liệu train của `lead_detector`. Nhưng **không có box nào** trên tile PCB-DSLR hay bo dự án, nên không chấm được lượt 2 ở miền đang dùng thật. |
| **Golden recipe chưa dùng trong luồng chính** | `aoi_pipeline/pipeline.py` **không đọc** recipe; golden chạy qua `app/pipeline_bridge.py` — hai nhánh song song. Mà recipe mới là thứ ra quyết định. |
| ~~**Lớp lưu trữ chưa ai gọi**~~ | **Xong 07/09.** Trang kết quả bước 8 nay lưu được cả lần chạy: vị trí lỗi + ảnh cắt của từng lỗi, và nói luôn bo còn thiếu mặt nào. |
| **Luật 5.2 mặc định TẮT** | Chưa có tập nghiệm thu khoá theo bo nên chưa đủ căn cứ bật. |

Fine-tune đúng là **cần** — nhưng nó sửa được đúng một thứ: box lượt 1 sai trên
bo dự án (32% là pad tròn). Nó không tạo ra nhãn mối hàn và không nối recipe vào
luồng chính.

## 2. Đã làm, chưa xong hết

| kế hoạch | đã có | còn thiếu |
|---|---|---|
| [Phân nhóm package](ke_hoach_phan_nhom_package.md) | luật + 3 chốt an toàn, 2 cổng nghiệm thu, tách được họ tụ (90,5%), cạnh chân xuống tới 5.5 | **mặc định TẮT**; chưa có tập nghiệm thu khoá theo bo; nhánh `ic` mới chạy trên 1 board |
| [Số hoá dữ liệu / database](ke_hoach_so_hoa_du_lieu_va_truy_xuat_aoi.md) | schema v2 + migration, repository, phiên kiểm bo, cầu nối kết quả→bản ghi, **cắt ảnh lỗi + nối vào app** (giai đoạn 3) | chưa có hàng đợi/retry khi mất mạng; chưa có đối soát; **chưa có giao diện tra cứu** (giai đoạn 4) |
| [Golden ghép sơ đồ](ke_hoach_golden_ghep_so_do.md) | hướng đã chốt, phép đo chặn đã chạy, quy trình chụp + script kiểm bộ ảnh | **chờ bo dây chuyền**; chưa chụp, chưa đo |
| [Kiểm hai mặt PCB](ke_hoach_kiem_hai_mat_pcb.md) | bản ghi mang `board_id`/`side`, `scan_session`, quy tắc "một mặt đạt ≠ cả bo đạt", **app báo mặt còn thiếu sau khi lưu** | chưa nối recipe theo mặt; chưa có màn hình danh sách bo đang chờ mặt còn lại |
| [Pre-train 6.1](ke_hoach_pretrain_6_1_classification.md) | model chạy được, có manifest, đã đo lại sau khi đổi detector | chưa đo **độ đúng** thật trên miền dự án (mới đo độ tin) |
| [Detect mối hàn 2 lượt](ke_hoach_detect_moi_han_2_giai_doan.md) | giai đoạn A + B xong; model lượt 2 train trên **9.089 box đã duyệt**, mAP50 0,99 trên test khoá | nhãn chỉ có ở miền `fpic`/`winnies`; **chưa có box nào** trên tile PCB-DSLR hay bo dự án |
| [Mối hàn ngoài linh kiện](ke_hoach_moi_han_ngoai_linh_kien.md) | đã đo và xác nhận **chưa có gì** | toàn bộ; bước 1 là gán tay một tile để biết lỗ hổng lớn bằng nào |

## 3. Mới ở trên văn bản, **chưa code dòng nào**

| kế hoạch | nội dung | vì sao chưa làm |
|---|---|---|
| [Kiểm lỗi toàn mạch](ke_hoach_pcb_defect_toan_mach.md) | lỗi trên **đường mạch** (đứt, chập, mouse-bite), không phải mối hàn | tự ghi *"bản để bạn duyệt, chưa code gì cả"*; chờ quyết |
| [Fine-tune cục bộ](ke_hoach_fine_tune_cuc_bo.md) | train lại trên máy này bằng ảnh có feedback | **không có script nào**; chặn ở dữ liệu — đúng việc bạn đang hỏi |
| [Sửa ROI mối hàn v1](phuong_an_sua_roi_moi_han.md) / [v2](phuong_an_sua_roi_moi_han_v2.md) | chặn nhiễu chữ lụa/viền trắng ở chân IC | thiết kế để A/B, **A/B chưa chạy** |
| [Số hoá mạch PCB](ke_hoach_so_hoa_mach_pcb_aoi.md) | ảnh → CAD/netlist/recipe (digital twin) | R&D; **giả định đã có ảnh Golden tốt**, mà tiền đề đó chưa có |
| [RNN/LSTM cho AOI](ke_hoach_ung_dung_rnn_lstm_aoi_pcb.md) | chuỗi chân, đa góc, giám sát chuỗi thời gian | R&D thuần, xa nhất |
| [Bàn giao 5.5 / 6.2](ke_hoach_ban_giao_5_5_va_6_2.md) | tài liệu bàn giao cho người tiếp theo | là tài liệu, không phải việc |

## 4. Thứ tự tôi đề xuất

Xếp theo **cái gì mở khoá được nhiều thứ nhất**, không theo cái gì dễ nhất.

1. **Mở rộng nhãn mối hàn sang miền đang dùng thật.** Không phải khoanh lại từ
   đầu: đã có 9.089 box trên `fpic`/`winnies` và có sẵn app khoanh
   (`label_boxes.html`). Việc cần là thêm **2–3 tile PCB-DSLR** và **vài tile bo
   dự án**, để lần đầu tiên chấm được lượt 2 ở đúng miền đang chạy.
2. **Ảnh bo dự án + fine-tune**, kèm tập giữ riêng không bao giờ train. Sửa được
   32% box pad tròn.
3. **Golden recipe cho bo dây chuyền** khi bo về. Đây là thứ biến hệ thống từ
   "model đoán" thành "đo so chuẩn" — đổi nhiều nhất về chất lượng.
4. ~~**Nối kho lưu trữ vào app.**~~ **Xong 07/09** — xem §5.
5. Còn lại (toàn mạch, số hoá mạch, RNN) để sau khi ba việc trên xong.

Việc 1 và 2 làm được **ngay khi có ảnh**. Việc 3 chờ bo.

## 5. Đã làm 07/09: kho lưu trữ nay có người gọi

`aoi_pipeline/storage/capture.py` là mảnh còn thiếu giữa bước 3.5 và kho:
`defects_from_run` cho *vị trí*, `save_inspection` nhận *vị trí kèm ảnh*, nhưng
**chưa ai cắt ảnh** — nên mọi bản ghi đều không có ảnh, đúng thứ yêu cầu tối
thiểu của dây chuyền đòi.

Chỗ dễ sai nhất và đã xử lý: toạ độ trong kết quả kiểm nằm trong hệ
`golden_board_pixels`, tức hệ của ảnh **đã nắn**. Cắt từ ảnh test **gốc** bằng
đúng những toạ độ ấy vẫn ra một tấm ảnh hợp lệ, chỉ là ảnh của chỗ khác, và
không gì báo lỗi. Bài test chính cố tình cho ảnh test lệch 5 px so với Golden để
hai nguồn cắt cho ra hai kết quả khác nhau — dịch nguồn cắt đi 5 px là test đổ,
đã kiểm bằng cách phá tạm thời.

Hệ quy chiếu lạ thì **bỏ ảnh chứ không cắt bừa**: mất ảnh còn sửa được, ảnh sai
chỗ thì không ai biết mà sửa.

Trong app (bước 8 → Inspect Board) lưu là hành động **có chủ ý**, không tự động:
workbench còn dùng để thử nghiệm, tự động ghi mọi lần bấm Inspect thì lịch sử sản
xuất lẫn với ảnh thử và sau đó không tách ra được. Kho nằm ở `outputs/aoi_history`
— trong dự án nên sống qua khởi động lại, và bị `.gitignore` chặn nên ảnh bo thật
không lên repo.

Còn thiếu, đúng như giai đoạn 4 của kế hoạch: **màn hình tra cứu**. Hiện ghi được
và đọc lại được bằng code, nhưng chưa có chỗ tra theo serial, xem timeline, mở
ảnh lỗi.
