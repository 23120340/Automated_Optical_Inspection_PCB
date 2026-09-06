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

| bộ | điều kiện vào bộ | kết quả |
|---|---|---|
| `quick_check_set` | recall ≥ 95% **và** thừa ≤ 15% | **15 tile / 15 bo**, recall 96–100% |
| `hard_set` | recall < 85% **hoặc** thừa > 25% | **17 tile / 17 bo**, recall 30–90%, thừa 4–69% |

Điều kiện của `hard_set` dùng **hoặc**, không dùng **và**: bỏ sót linh kiện và
đẻ box thừa là hai kiểu hỏng khác nhau, nối bằng "và" thì mất hẳn một kiểu.

Số kỳ vọng của từng ảnh nằm trong `manifest.json` và `README.md` của mỗi thư
mục, nên chạy lại là so được ngay.

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
