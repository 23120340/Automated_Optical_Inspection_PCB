# Trạng thái các kế hoạch — 2026-09-07

Trả lời hai câu: *cấu trúc còn ổn không*, và *cái gì chưa xong*.

---

## 1. Cấu trúc: ổn. Nhưng "chỉ cần fine-tune nữa" thì **không đúng**

Kiến trúc đúng và code đã nói rõ nó đúng: model **đề xuất**, recipe **quyết
định** (`golden/inspector.py`: *"this class is not an AI model"*). Đường ống
0 → 6.2 chạy được, 1.198 test xanh.

Nhưng bốn thứ dưới đây **không phải fine-tune** và cái nào cũng chặn:

| | vì sao fine-tune không giải quyết được |
|---|---|
| **Không có nhãn mối hàn** | 9.486 box đã khoanh đều là *thân*. Fixture pad đếm tay: **28 pad / 1 board**. Không đo được thì train xong cũng không biết tốt lên hay tệ đi. |
| **Golden recipe chưa dùng trong luồng chính** | `aoi_pipeline/pipeline.py` **không đọc** recipe; golden chạy qua `app/pipeline_bridge.py` — hai nhánh song song. Mà recipe mới là thứ ra quyết định. |
| **Lớp lưu trữ chưa ai gọi** | `aoi_pipeline/storage/` dựng xong, có test, nhưng **không nơi nào trong app gọi nó**. |
| **Luật 5.2 mặc định TẮT** | Chưa có tập nghiệm thu khoá theo bo nên chưa đủ căn cứ bật. |

Fine-tune đúng là **cần** — nhưng nó sửa được đúng một thứ: box lượt 1 sai trên
bo dự án (32% là pad tròn). Nó không tạo ra nhãn mối hàn, không nối recipe vào
luồng, không nối kho lưu trữ.

## 2. Đã làm, chưa xong hết

| kế hoạch | đã có | còn thiếu |
|---|---|---|
| [Phân nhóm package](ke_hoach_phan_nhom_package.md) | luật + 3 chốt an toàn, 2 cổng nghiệm thu, tách được họ tụ (90,5%), cạnh chân xuống tới 5.5 | **mặc định TẮT**; chưa có tập nghiệm thu khoá theo bo; nhánh `ic` mới chạy trên 1 board |
| [Số hoá dữ liệu / database](ke_hoach_so_hoa_du_lieu_va_truy_xuat_aoi.md) | schema v2 + migration, repository, phiên kiểm bo, cầu nối kết quả→bản ghi | **chưa nối vào app**; chưa có hàng đợi/retry khi mất mạng; chưa có đối soát; chưa có giao diện tra cứu |
| [Golden ghép sơ đồ](ke_hoach_golden_ghep_so_do.md) | hướng đã chốt, phép đo chặn đã chạy, quy trình chụp + script kiểm bộ ảnh | **chờ bo dây chuyền**; chưa chụp, chưa đo |
| [Kiểm hai mặt PCB](ke_hoach_kiem_hai_mat_pcb.md) | bản ghi mang `board_id`/`side`, `scan_session`, quy tắc "một mặt đạt ≠ cả bo đạt" | **giao diện chưa có** trạng thái "chờ mặt còn lại"; chưa nối recipe theo mặt |
| [Pre-train 6.1](ke_hoach_pretrain_6_1_classification.md) | model chạy được, có manifest, đã đo lại sau khi đổi detector | chưa đo **độ đúng** thật trên miền dự án (mới đo độ tin) |
| [Detect mối hàn 2 lượt](ke_hoach_detect_moi_han_2_giai_doan.md) | giai đoạn A + B xong, có model lượt 2 | chưa chấm được vì **không có nhãn mối hàn** |
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

1. **Khoanh tay mối hàn trên 2–3 tile.** Rẻ nhất, và nó mở khoá *mọi* câu hỏi về
   lượt 2 — hiện chưa câu nào trả lời bằng số được.
2. **Ảnh bo dự án + fine-tune**, kèm tập giữ riêng không bao giờ train. Sửa được
   32% box pad tròn.
3. **Golden recipe cho bo dây chuyền** khi bo về. Đây là thứ biến hệ thống từ
   "model đoán" thành "đo so chuẩn" — đổi nhiều nhất về chất lượng.
4. **Nối kho lưu trữ vào app.** Thư viện không ai gọi thì chưa biết nó có vừa dữ
   liệu thật không.
5. Còn lại (toàn mạch, số hoá mạch, RNN) để sau khi ba việc trên xong.

Việc 1 và 2 làm được **ngay khi có ảnh**. Việc 3 chờ bo. Việc 4 làm được bất cứ
lúc nào.
