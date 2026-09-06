# khao_sat — Ngoài kia có dữ liệu/model gì dùng được

- [dataset_lead_detection.md](dataset_lead_detection.md)
- [khao_sat_model_huggingface.md](khao_sat_model_huggingface.md)
- [pcb_aoi_component_datasets.md](pcb_aoi_component_datasets.md)
- [package_taxonomy_theo_lop.csv](package_taxonomy_theo_lop.csv) — 47 lớp của
  hai bộ công khai, kèm phân vị kích thước và đặc trưng contour
- [package_taxonomy_theo_cum.csv](package_taxonomy_theo_cum.csv) — 141 cụm
  KMeans trong từng lớp; cụm chỉ đánh số, tên package do người duyệt gán

Ảnh crop và contact sheet không vào git (336 MB). Dựng lại:
`python scripts/survey_package_taxonomy.py --out datasets/survey/<tên>`
