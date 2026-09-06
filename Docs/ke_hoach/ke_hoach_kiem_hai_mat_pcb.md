# Kế hoạch: kiểm cả hai mặt của một tấm PCB

Ghi 2026-09-06, sau khi xác nhận dây chuyền **kiểm cả hai mặt**.

> **Phần dữ liệu đã xong; phần điều phối thì chưa.** Đọc §1 để biết ranh giới
> đó nằm ở đâu trước khi lên lịch làm.

---

## 1. Đã có gì, chưa có gì

| | trạng thái |
|---|---|
| Recipe biết mặt nào | ✅ `InspectionRecipe.side`, validate `top`/`bottom` |
| Bản ghi kiểm tra mang theo mặt | ✅ `InspectionRun.board_id`/`side` (06/09) |
| "Một mặt đạt ≠ cả PCB đạt" | ✅ `missing_required_sides()` và `InspectionStore.sides_still_missing()` |
| Lưu được nhiều lần kiểm của cùng một bo | ✅ `inspection` khoá theo `board_id` |
| **Ghép hai lần chụp thành một tấm bo** | ✅ `scan_session` (06/09) — xem §3 bước 1 |
| **Giữ danh tính bo qua thao tác lật** | ⚠️ dữ liệu đã chặn được; **giao diện thì chưa** — §3 bước 2 |
| **Đối chiếu vị trí giữa hai mặt** | ❌ chưa có, và §4 nói vì sao nó khó hơn vẻ ngoài |

Nói gọn: hệ thống **lưu** đúng hai mặt rồi, nhưng chưa có gì **dẫn** người vận
hành đi qua hai mặt và chưa có gì bảo đảm hai lần chụp đó là cùng một tấm bo.

## 2. Chỗ dễ hỏng nhất: lật bo là lúc mất danh tính

Chưa có serial (kế hoạch số hoá §4.2b), nên danh tính bo hiện là **ID nội bộ tự
sinh**. ID đó sống trong một phiên làm việc. Người vận hành lật bo rồi bấm nhầm
"bo mới" là hai mặt của cùng một tấm biến thành hai tấm — và **không có cách nào
phát hiện sau đó**.

Đây không phải lỗi phần mềm, nó là lỗi quy trình mà phần mềm phải chặn:

- Sau khi kiểm mặt thứ nhất, màn hình **phải** ở trạng thái *"đang chờ mặt còn
  lại của bo này"*, không quay về màn hình trống.
- Bắt đầu một bo mới trong khi bo cũ còn thiếu mặt thì phải **hỏi lại** và ghi
  lý do bỏ dở, chứ không im lặng đóng phiên.
- Có serial rồi thì rủi ro này biến mất: quét lại là ghép đúng bo. Nên **ưu tiên
  trả lời câu serial hơn là xây thêm cơ chế bù**.

## 3. Việc phải làm, theo thứ tự

| | việc | phụ thuộc |
|---|---|---|
| 1 | ✅ **Phiên kiểm bo** — `scan_session`, migration 2 (06/09) | lớp lưu trữ |
| 2 | **Trạng thái "chờ mặt còn lại"** trên giao diện, kèm cảnh báo khi bỏ dở | (1) |
| 3 | **Recipe theo từng mặt**: chọn đúng recipe `top`/`bottom` theo mặt đang kiểm | đã có `side` trong recipe |
| 4 | **Quyết định ở mức bo** dùng `sides_still_missing()` thay vì trạng thái của lần chạy cuối | đã có |
| 5 | Đối chiếu vị trí giữa hai mặt (§4) | chỉ khi thật sự cần |

> **Bước 1 đã xong 06/09.** `InspectionStore.open_session/close_session/`
> `cancel_session/session_sides_missing`. Bốn ràng buộc được ép ở tầng dữ liệu,
> không ở tầng ứng dụng:
>
> - **mỗi trạm một phiên đang mở** — chỉ mục một phần trong lược đồ, nên hai
>   tiến trình cùng mở thì cơ sở dữ liệu chặn, không phụ thuộc thứ tự chạy;
> - **đóng phiên khi còn thiếu mặt thì phải nêu lý do** (HM03);
> - **`required_sides` ghi vào phiên**, không lấy mặc định lúc đọc — bo một mặt
>   phải khai ra chứ không suy;
> - **gán inspection sang phiên của bo khác bị từ chối** — đó là tráo danh tính.
>
> Kết quả về sau khi phiên đã đóng/huỷ vẫn được ghi vào **đúng phiên sinh ra
> nó**, đánh dấu `arrived_after_close`, và **không** tính là đã kiểm mặt đó.

Bước 1–4 là đủ để chạy pilot. Bước 5 là việc riêng, đắt, và **chưa chắc cần**.

## 4. Đối chiếu vị trí giữa hai mặt — đọc kỹ trước khi hứa

Lật bo quanh trục dọc thì hệ toạ độ **soi gương**: một điểm ở `x` trên mặt TOP
nằm ở `W − x` trên mặt BOTTOM. Nên:

- Toạ độ lỗi của hai mặt **không so trực tiếp được**. Muốn nói *"lỗi mặt dưới
  nằm ngay dưới linh kiện U12 ở mặt trên"* thì phải có phép biến đổi gương, và
  phải biết bo được lật quanh **trục nào** — điều đó phụ thuộc thao tác tay của
  người vận hành, không suy ra được từ ảnh.
- Chốt được nó cần một mốc chung: lỗ định vị, mép bo, hoặc fiducial nhìn thấy
  được từ **cả hai** mặt.

Vì vậy: **bản đầu không hứa đối chiếu vị trí hai mặt.** Lưu riêng từng mặt, hiển
thị riêng, và quyết định ở mức bo dựa trên "đủ mặt bắt buộc" chứ không dựa trên
quan hệ hình học giữa hai mặt.

## 5. Nghiệm thu

| mã | kịch bản | mong đợi |
|---|---|---|
| HM01 | Kiểm TOP đạt, chưa kiểm BOTTOM | Bo **chưa** đủ điều kiện; giao diện nói rõ còn thiếu mặt nào |
| HM02 | Kiểm TOP, lật, kiểm BOTTOM | Hai lần kiểm gắn vào **cùng một** `board_id` |
| HM03 | Kiểm TOP rồi bắt đầu bo khác | Hệ thống hỏi lại; nếu bỏ dở thì ghi lý do, không im lặng |
| HM04 | TOP đạt, BOTTOM lỗi | Bo không đạt; lịch sử giữ **cả hai** bản ghi |
| HM05 | TOP đạt ở chế độ chạy thử, BOTTOM đạt thật | Bo **chưa** đủ điều kiện — lần chạy thử không làm căn cứ (§9.3) |
| HM06 | Dùng nhầm recipe mặt TOP để kiểm mặt BOTTOM | Phát hiện và từ chối, không cho ra kết quả |

HM05 và HM06 là hai ca dễ lọt nhất, vì cả hai đều cho ra một kết quả *trông
bình thường*.
