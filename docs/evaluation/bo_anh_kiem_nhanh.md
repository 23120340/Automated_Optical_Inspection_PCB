# Hai bộ ảnh kiểm — dùng ảnh nào để thử

Dựng 2026-09-06, sau khi xác định `00001__1024__1648___4120` là ảnh **ngoài
miền** ([loi_pad_tron_bo_du_an.md](loi_pad_tron_bo_du_an.md)).

```bash
python scripts/build_check_sets.py --mode good   # -> datasets/test_images/quick_check_set/
python scripts/build_check_sets.py --mode hard   # -> datasets/test_images/hard_set/
```

---

## Chọn thế nào

**Bằng số đo, không bằng mắt.** Mỗi tile trong 95 tile đã duyệt được chấm bằng
chính nhãn tay của dự án: ghép nhãn với box detect theo IoU ≥ 0,4, rồi lấy
**recall** và **tỉ lệ box thừa**. Sau đó trải **mỗi bo một tile** — một bộ toàn
tile của cùng một bo thì chạy tốt cũng không nói được gì về bo khác.

| bộ | điều kiện vào bộ *(chỉ tính lượt 1)* | kết quả |
|---|---|---|
| `quick_check_set` | recall ≥ 95% **và** thừa ≤ 15% | **15 tile / 15 bo**, recall 96–100% |
| `hard_set` | recall < 85% **hoặc** thừa > 25% | **17 tile / 17 bo**, recall 30–90%, thừa 4–69% |

Điều kiện của `hard_set` dùng **hoặc**, không dùng **và**: bỏ sót linh kiện và
đẻ box thừa là hai kiểu hỏng khác nhau, nối bằng "và" thì mất hẳn một kiểu.

Số kỳ vọng của từng ảnh nằm trong `manifest.json` và `README.md` của mỗi thư
mục, nên chạy lại là so được ngay.

## ⚠️ Cả hai bộ chỉ chấm LƯỢT 1, không chấm mối hàn

Sửa 2026-09-07, sau khi người dùng thử `pcb_dslr_011__rec1__1024__768___768` và
thấy ROI mối hàn sai trong khi tile đó nằm trong `quick_check_set`.

**Đó là thiếu sót của bộ, không phải mâu thuẫn.** Cả hai bộ được chọn bằng
**recall và box thừa của lượt 1** — tức chỉ đo *tìm thân linh kiện*. Không có
tiêu chí nào về ROI mối hàn, nên một tile lượt 1 hoàn hảo vẫn có thể có ROI mối
hàn tệ, và bộ vẫn nhận nó.

Ví dụ chính tile đó:

| | |
|---|---:|
| nhãn tay (**thân linh kiện**) | 77 |
| lượt 1 detect | 78 (recall 100%, thừa 1) |
| ROI mối hàn suy ra | 204 |
| lượt 2 detect chân | 175 |

**Vì sao chưa chấm được lượt 2 TRÊN TILE:**

> ⚠️ **Đính chính 2026-09-07.** Bản trước viết *"dự án không có nhãn tay cho mối
> hàn"*. **Sai.** Dự án có **9.089 box mối hàn đã duyệt** — 4.595 trên
> `fpic_components` (1.044 crop) và 4.494 trên `winnies_components` (987 crop) —
> và chính chúng là dữ liệu train của `models/active/lead_detector` (mAP50
> 0,9912; recall 0,9768 trên 25 cảnh test khoá). Xem
> `datasets/labelling/*/joint_boxes.reviewed.json`.

Điều đúng, hẹp hơn nhiều: nhãn mối hàn nằm trên **crop từng linh kiện** của hai
bộ công khai `fpic`/`winnies`, **không** nằm trên tile PCB-DSLR và **không** nằm
trên bo dự án. Nên chấm được lượt 2 *trong miền của nó*, mà không chấm được trên
`pcb_dslr_011` hay bo dây chuyền.

Ngoài ra 9.486 box khoanh trên tile đều là **thân linh kiện**, và fixture duy
nhất có pad đếm tay trên một board thật là `board_smd_00001` — **28 pad**.

Muốn chấm được lượt 2 thì phải khoanh tay mối hàn trên vài tile. Đó là việc chưa
ai làm, và nó chặn mọi câu hỏi dạng *"ROI mối hàn tốt tới đâu"*.

## Hai bộ trả lời hai câu khác nhau

**`quick_check_set` — "thay đổi vừa rồi có làm hỏng gì không?"**
Nó chỉ chứa ca detector **đang làm tốt**, nên số ở đó chỉ **tụt** khi có hồi
quy. Nó **không** đo được tiến bộ: sửa đúng hay sửa sai, số cũng gần như đứng
yên.

**`hard_set` — "sửa vừa rồi có tốt lên thật không?"**
Nó là nửa còn lại: tile detector **đang sai**. Đây mới là chỗ một thay đổi tốt
làm số **tăng**.

Có đúng một bộ là tự lừa. Chỉ nhìn bộ dễ thì mọi thay đổi trông đều vô hại; chỉ
nhìn bộ khó thì một thay đổi phá hỏng phần đang chạy tốt cũng không ai thấy.

## Cả hai đều KHÔNG chứng minh hệ thống dùng được

Chỗ dễ tự lừa nhất, nên nói thẳng:

- Mọi tile trong **cả hai** bộ đều **trong miền** của tập huấn luyện.
- Tile trong `hard_set` **đã được dùng để train**, nên số đo trên chúng là *lạc
  quan*: model đã nhìn thấy chúng rồi.
- Lỗi nặng nhất đang gặp là lỗi **ngoài miền**: trên bo của chính dự án, **32%**
  box lượt 1 là pad tròn chứ không phải linh kiện.

| dùng để hỏi | được |
|---|---|
| "thay đổi vừa rồi có làm hỏng gì không?" | ✅ `quick_check_set` |
| "sửa vừa rồi có tốt lên không?" | ✅ `hard_set`, nhưng số lạc quan |
| "hệ thống dùng được trên dây chuyền chưa?" | ❌ **cả hai đều không** |

## Còn thiếu: tập GIỮ RIÊNG trên bo dự án

Đây là thứ duy nhất trả lời được câu cuối, và nó **chưa tồn tại**. Khi có 10–20
tile bo dự án đã khoanh ([loi_pad_tron_bo_du_an.md](loi_pad_tron_bo_du_an.md) §6),
phải tách ngay làm hai: phần đưa vào train, và phần **không bao giờ train**. Đo
số box pad tròn trên phần giữ riêng, trước và sau — đó mới là con số nói được đã
sửa được hay chỉ làm model thuộc lòng thêm vài tile.

## Giấy phép

Ảnh phái sinh từ **CVL PCB-DSLR** (nghiên cứu phi thương mại), nên cả hai thư
mục nằm dưới `datasets/` và bị `.gitignore` chặn. Script và tài liệu thì có
trong git, nên dựng lại được bất cứ lúc nào mà không đưa ảnh lên repo.
