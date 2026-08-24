# Reference sets

## Bộ cùng board cho Golden/consensus/PnP và pilot một layout

`mpi_pcb_gas_pump_same_board_30/` là bộ bootstrap ưu tiên cho enrollment:

- 30 ảnh `train/good` của cùng một PCB máy bơm xăng, cùng layout và cùng mặt
  linh kiện;
- ảnh DSLR JPEG 4096×2816 đã được MPI-PCB đăng ký upstream về cùng canvas;
- lựa chọn theo 30 khoảng của chuỗi chụp, ưu tiên ảnh nguồn giàu chi tiết trong
  từng khoảng, không trộn `test/good` hoặc `test/defect`;
- nguồn Zenodo DOI `10.5281/zenodo.8213098`, giấy phép CC BY 4.0;
- bắt đầu review bằng
  `mpi_pcb_gas_pump_same_board_30/NHUNG_VIEC_BAN_CAN_LAM.md`.

`train/good` là nhãn upstream, không phải xác nhận OK của người vận hành. Căn
chỉnh upstream/identity chỉ hợp lệ cho enrollment trên dataset này; nó không thay
thế registration đã đo trên camera, lens và fixture production.

Golden và PnP hiện chỉ là bootstrap trong `golden_board_pixels`. Không tạo tọa
độ mm hoặc recipe production trước khi đo board/fiducial thật.

Bundle review hiện tại nằm tại
`../../golden_recipes/mpi_pcb_gas_pump/top/bootstrap_pixel_v1/`. Bắt đầu với
`pnp_pixels_NEEDS_REVIEW.csv` và `overlay_pnp_NEEDS_REVIEW.png`; giữ
`consensus_components.json/csv` làm detector audit đầy đủ, không coi các ID
`*_AUTO_*` là RefDes thật.

Tập này cũng có thể dùng làm pilot gắn nhãn Phase 1–5 cho **một layout**, nhưng
không cung cấp độ đa dạng giữa nhiều SKU/layout và không phải ảnh solder-side.

## Bộ nhiều layout cho pilot độ đa dạng Phase 1–5

`pcb_dslr_30_diverse/` là bộ pilot hiện tại:

- 30 ảnh, 30 PCB/layout khác nhau;
- ảnh DSLR 4928×3280 kèm board mask và upstream IC annotation;
- mỗi board có Golden candidate và PnP pixel `NEEDS_REVIEW` riêng trong
  `bootstrap/references/<board_id>/`;
- bắt đầu bằng `pcb_dslr_30_diverse/NHUNG_VIEC_BAN_CAN_LAM.md`.

Không tạo consensus hoặc một recipe chung giữa 30 board này, và không trộn chúng
vào Golden/PnP của bộ MPI cùng layout ở trên.

## Bộ repeated-OK cũ

`visa_pcb2_30/` gồm nhiều frame gần giống của cùng một layout. Bộ này **không
phải** tập 30 board đa dạng để gắn nhãn Phase 1–5. Chỉ giữ nó để thử enrollment,
alignment/repeatability và Golden Compare trên các lần chụp lặp của cùng SKU.
