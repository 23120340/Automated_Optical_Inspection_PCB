"""Bước 6.2: chấm chất lượng mối hàn trên các ROI do bước 5.5 sinh ra.

Ba tầng, dùng cùng nhau chứ không thay nhau:

* ``features`` đo đặc trưng vật lý của ROI, không cần train.
* ``rules`` biến số đo thành phán quyết kèm lý do, chạy được từ ngày đầu.
* ``classifier`` là model ONNX đã train (tuỳ chọn), cùng quy ước manifest với 6.1.
* ``inspector`` hợp nhất luật và model, với chốt chặn thiên về không bỏ lọt lỗi.

Chưa có model thì stage vẫn chạy bằng luật và ghi rõ ``source="rules"``.
"""
