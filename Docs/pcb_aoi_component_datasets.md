# Khảo sát dataset nhận dạng linh kiện PCB cho AOI

> Cập nhật: 2026-08-29. Mục tiêu là **phát hiện/phân loại linh kiện đã gắn trên PCB từ ảnh quang học**, không phải phát hiện lỗi đường mạch trên bare PCB.

## Kết luận nhanh

Không có dataset công khai nào bao phủ đủ danh sách 30 nhóm ở mức chi tiết yêu cầu. Dataset công khai chủ yếu nhận dạng **hình dạng/họ hoặc package** (`resistor`, `capacitor`, `IC`, `connector`...), trong khi các nhãn như `op-amp`, `ADC`, `MCU`, `PMIC`, `RS485 transceiver`, `NAND Flash` thường **không thể xác định chắc chắn chỉ từ ngoại hình**. Nhiều IC khác chức năng dùng cùng package; muốn phân biệt phải kết hợp OCR top-marking, BOM/CAD/centroid và cơ sở dữ liệu part number.

Khuyến nghị thực tế:

1. Dùng **PCB Component Detection Consolidated v1** làm bootstrap detector nhiều lớp. Kaggle API xác nhận gói ~2,87 GB và YAML dùng trực tiếp là `components_data_uncropped/data.yaml`; notebook đã có preset cho nguồn này.
2. Bổ sung **FPIC/FICS-PCB** chỉ sau khi được cấp quyền và xác minh license; dùng **WACV 2019** cho breadth, nhưng kiểm tra trùng nguồn trước khi ghép với Consolidated/RF100.
3. Dùng **PCB-SAID** cho trạng thái lắp ráp và lỗi SMD; không coi đây là nguồn chính để nhận dạng chức năng điện tử.
4. Tự chụp ảnh đúng camera, lens, ánh sáng, góc, PCB và dây chuyền đích; join annotation với BOM/centroid. Đây mới là tập quyết định độ chính xác AOI sản xuất.

## Cách đọc đánh giá

- **Độ phù hợp AOI** là đánh giá của người khảo sát về dữ liệu/nhãn/cách chụp, **không phải accuracy được tác giả công bố**.
- **Cao**: ảnh PCB thật, annotation trực tiếp, nguồn học thuật/tài liệu gốc rõ ràng.
- **Trung bình**: hữu ích nhưng ít board, khác miền AOI, nhãn thô, mất cân bằng hoặc nguồn tổng hợp/community.
- **Thấp**: ảnh sản phẩm rời/web, classification crop, license/annotation chưa rõ; chỉ nên dùng pretraining hoặc mining.
- Ký hiệu coverage: **D** = nhãn trực tiếp; **G** = chỉ nhãn họ chung; **P** = proxy/package gần đúng; **—** = không có nhãn đáng tin cậy.

## Dataset nên lấy

| # | Dataset / tải về | Quy mô & annotation | Linh kiện nhận diện trực tiếp | Phù hợp AOI | Giấy phép / lưu ý |
|---|---|---|---|---|---|
| 1 | [PCB Component Detection Consolidated (Kaggle)](https://www.kaggle.com/datasets/aryanstein/pcb-component-detection-consolidated-dataset/data) · [pipeline/mapping](https://github.com/aryan-programmer/pcb-fault-detection) | ~2.87 GB; hợp nhất WACV, FICS-PCB, PCB-Vision, CompDetect và oriented PCB; YOLO | `battery`, `button`, `buzzer`, `capacitor`, `clock`, `connector`, `diode`, `display`, `emi_filter`, `ferrite_bead`, `fuse`, `heatsink`, `ic`, `inductor`, `jumper`, `led`, `mosfet`, `potentiometer`, `resistor`, `transformer`, `transistor` (kiểm tra file YAML/version trước train) | **Cao để bootstrap**, nhưng **trung bình cho validation** vì nguồn trộn, taxonomy được remap, nguy cơ trùng ảnh/source leakage | Trang Kaggle ghi Apache-2.0; vẫn cần kiểm tra điều khoản của từng nguồn thành phần trước dùng thương mại. Chia train/val/test theo **board/source**, không random theo tile. |
| 2 | [FPIC – FICS PCB Image Collection](https://physicaldb.ece.ufl.edu/index.php/fics-pcb-image-collection-fpic/) · [paper](https://doi.org/10.1145/3588032) | 261 ảnh độ phân giải cao của 93 PCB; >71.000 instance gồm text và mounted components; polygon CSV, OCR, metadata | SMD theo họ: resistor, capacitor, inductor, transistor, diode, LED, IC và các lớp SMD/refdes khác; xem taxonomy thực tế sau khi được cấp quyền | **Cao** cho localization/segmentation và OCR; đa board/camera. **Không đủ** cho subtype chức năng | Cần yêu cầu access code. License không được trình bày rõ trên trang tổng hợp; phải xác nhận với chủ dữ liệu trước dùng thương mại. |
| 3 | [FICS-PCB multimodal](https://eprint.iacr.org/2020/366) | Ảnh DSLR độ phân giải cao + digital microscope với biến thiên chiếu sáng/scale; bbox CSV, marking/logo | **6 lớp:** IC, capacitor, resistor, inductor, transistor, diode | **Cao** cho 6 họ linh kiện; nhãn rộng, không phân subtype. Số board tương đối nhỏ nên dễ domain shift | Dataset qua Trust-Hub/PhysicalDB; kiểm tra quyền truy cập và license tại thời điểm tải. Không nhầm FICS-PCB với FPIC dù cùng nhóm nghiên cứu. |
| 4 | [WACV 2019 PCB Component Detection](https://sites.google.com/view/chiawen-kuo/home/pcb-component-detection) · [paper](https://arxiv.org/abs/1811.06994) | Trang tác giả: **47 ảnh độ phân giải cao, 31 loại linh kiện, khoảng 62.000 instance**, download ~287 MB; các bản downstream có thể đã chuẩn hóa còn 20 lớp/Pascal VOC | Các họ cơ bản như battery, button, buzzer, capacitor, clock, connector, diode, display, EMI filter, ferrite bead, fuse, heatsink, IC, inductor, jumper, LED, potentiometer, resistor, transformer, transistor; đọc taxonomy trong gói tải thay vì suy từ bản remap | **Trung bình–cao** cho breadth/ít-shot; **trung bình–thấp** cho generalization vì chỉ 47 ảnh, nhãn mất cân bằng và ảnh không hoàn toàn theo cell AOI | Trang tác giả cho tải trực tiếp nhưng không nêu license thương mại rõ ràng—cần xin phép nếu dùng sản phẩm. Có khả năng ảnh đã đi vào Consolidated/RF100; kiểm hash/board identity trước khi ghép. |
| 5 | [PCB-SAID](https://doi.org/10.1109/ICCVW69036.2025.00145) · [paper open access](https://www.openaccess.thecvf.com/content/ICCV2025W/VISION%2725/html/Mineo_PCB-SAID_A_Low-Cost_Camera-Based_Dataset_for_Few-Shot_SMD_Assembly_Inspection_ICCVW_2025_paper.html) | 175 RGB ảnh; bbox + polygon; 66 fine-grained state classes, 10 loại SMD, 22 package; good/missing/misaligned/lifted/rotated/short | Các loại/package SMD và **trạng thái lắp**; taxonomy chi tiết cần lấy từ release kèm paper | **Cao** cho bài toán defect/assembly few-shot; **trung bình** cho component taxonomy vì ảnh web-sourced và tập nhỏ | “Open dataset” theo paper; xác minh license cụ thể trong gói phát hành trước sử dụng thương mại. |
| 6 | [PCB-Vision](https://github.com/hifexplo/PCBVision) · [data record](https://doi.org/10.14278/rodare.2704) | 53 PCB; 53 RGB + 53 hyperspectral cubes 224 bands; pixel masks General/Monoseg | **IC, capacitor (chủ yếu electrolytic), connector** | **Cao** cho segmentation ba lớp và nghiên cứu RGB/HSI; **trung bình–thấp** để train detector tổng quát do chỉ 53 board và IC chiếm ưu thế lớn | Dữ liệu/codes công khai; xem license trong record/repository. Camera conveyor/recycling khác miền AOI lắp ráp. |
| 7 | [PCB DSLR Dataset](https://cvl.tuwien.ac.at/research/cvl-databases/pcb-dslr-dataset/) | 748 ảnh, 165 PCB, 9.313 bbox IC; 2.048 IC duy nhất; segmentation PCB | **IC** (và label text cho một phần IC) | **Cao** cho pretrain detector IC và OCR; **trung bình** cho AOI vì PCB phế liệu, ~220 ppi, điều kiện recycling | Chỉ miễn phí cho **nghiên cứu phi thương mại**. Không có nhãn các linh kiện khác. |
| 8 | [Printed Circuit Board / RF100 (Roboflow)](https://universe.roboflow.com/roboflow-100/printed-circuit-board) · [RF100 repo](https://github.com/roboflow/roboflow-100-benchmark) | 672 file object detection, **199 stem cảnh** trong export v4 local; nhiều cảnh có 2–4 bản crop/rectify/resize. YAML v4 local có 23 lớp; trang Universe hiện hiển thị 34 | `Button`, `Capacitor`, `Clock`, `Connector`, `Diode`, `IC`, `Inductor`, `LED`, `Resistor`, `Switch`, `Transistor` và các nhãn pad/pin/test-point cần loại khi gom về `component` | **Trung bình** để bootstrap body detector; **thấp cho validation** vì duplicate, nhãn community và chồng nguồn. Đây **không phải FPIC**; ảnh `pcbNrecM` khớp PCB DSLR | Trang Roboflow ghi CC BY 4.0, nhưng PCB DSLR upstream giới hạn nghiên cứu phi thương mại. Giữ điều kiện chặt hơn cho đến khi provenance được làm rõ. Khử trùng theo board, không theo file. |
| 9 | [PCB Component Detection – 9 classes](https://datasetninja.com/pcb-component-detection) | 1.410 ảnh, 11.119 object; bbox | `MOSFET`, `transformer`, `MOV`, `resistor` và 4 nhóm capacitor (`cap1`–`cap4`), cộng một nhãn resistor viết sai | **Trung bình–thấp**: có lớp hiếm hữu ích nhưng tên `cap1..4` không mang ngữ nghĩa nếu không đọc data card/source; cần relabel thủ công | Trang chỉ mục dẫn nguồn Kaggle/Roboflow; kiểm tra license gốc. Không merge mù vì typo/unknown labels. |
| 10 | [Electronic Component Recognizer (Kaggle)](https://www.kaggle.com/datasets/mdfaisalahmed025/electronic-component-recognizer) | 3.661 ảnh đã làm sạch, 20 lớp; chủ yếu classification/crop linh kiện | 20 lớp linh kiện điện tử theo data card (cần tải và xác minh `class_indices`) | **Thấp cho AOI**, **trung bình cho pretraining crop classifier**: background/scale thường khác linh kiện đã gắn trên PCB | Community dataset; xác minh nguồn ảnh, license và leakage. Không dùng làm test AOI. |

### Dataset không giải quyết bài toán nhận dạng linh kiện

- **DeepPCB**, PKU PCB defect và các bản sao: 1.500 cặp ảnh, nhãn 6 lỗi bare-board (`open`, `short`, `mouse bite`, `spur`, `missing hole`, `spurious copper`), không có component taxonomy.
- **PCBA-DET**: 4.000 ảnh nhưng nhãn chủ yếu lỗi vít/quạt/dây/scratch trên motherboard, không phải resistor/capacitor/IC.
- **PCB X-ray CT**: phù hợp kiểm tra inter-layer/X-ray, không phải detector linh kiện RGB top-view.
- Các dataset SPI/AOI dạng bảng: có thể hữu ích cho dự đoán lỗi hàn, nhưng thường không phát hành ảnh và bbox linh kiện để huấn luyện vision detector.

## Coverage theo danh sách yêu cầu

### Có thể train trực tiếp hoặc gom về nhãn họ

| Nhãn mục tiêu | Dataset có thể dùng | Mức thực tế | Ghi chú |
|---|---|---|---|
| Điện trở SMD/THT | Consolidated, FPIC/FICS, WACV, RF100 | **D/G** | Nguồn thường chỉ ghi `resistor`; phải tự tách SMD/THT. Resistor network thường bị nhầm IC/SIP. |
| Tụ gốm/điện phân/tantalum/film/mica/biến dung | Consolidated, FPIC/FICS, WACV, PCB-Vision, 9-class | **G**; điện phân **D/P** trong PCB-Vision | Phần lớn chỉ `capacitor`; subtype cần relabel theo package/marking. |
| Cuộn cảm | Consolidated, FPIC/FICS, WACV | **D/G** | `inductor` có nhãn trực tiếp ở cấp họ. |
| Ferrite bead | Consolidated, WACV | **D** | Dễ lẫn SMD inductor/0-ohm resistor; cần audit ảnh. |
| Common-mode choke / RF coil | WACV (`EMI filter`, `inductor`) | **P/G** | Không có nhãn subtype đáng tin cậy. |
| Transformer | Consolidated, WACV, 9-class | **D** | Transformer cách ly không được tách riêng. |
| Diode thường | Consolidated, FPIC/FICS, WACV | **D/G** | Thường chỉ `diode`. |
| Schottky/Zener/TVS/bridge/photodiode | Các tập có `diode`; 9-class có MOV, Consolidated có thể giữ nhãn nguồn | **G/P** | Không suy ra subtype chỉ từ package; cần OCR/BOM. TVS xuất hiện ở hai nhóm của yêu cầu nhưng nên là một class chức năng. |
| LED SMD/THT/công suất/RGB/IR | Consolidated, FPIC, WACV | **G** | Chỉ `LED`; 7-segment/matrix nên thuộc display riêng. |
| BJT/MOSFET/JFET/IGBT/Darlington/phototransistor | Consolidated, FPIC/FICS, WACV; 9-class có MOSFET | transistor **G**, MOSFET **D** ở nguồn hẹp | Package SOT-23/TO-220 không đủ phân biệt; power MOSFET cũng cần BOM/marking. |
| IC (chung) | Consolidated, FPIC/FICS, WACV, PCB-Vision, PCB-DSLR | **D** | Đây là lớp được hỗ trợ tốt nhất; không đồng nghĩa phân biệt chức năng IC. |
| Crystal/oscillator/resonator/clock | Consolidated/WACV (`clock`), RF100 | **G/P** | `clock` là taxonomy không đủ rõ để tách crystal/oscillator/RTC/generator. |
| Fuse | Consolidated, WACV | **D** | PTC/resettable fuse chưa tách. |
| MOV | 9-class | **D** | Cần kiểm tra chất lượng/nguồn và bổ sung ảnh dây chuyền. |
| Battery / coin cell / holder | Consolidated, WACV | **G/P** | Thường `battery`, không đảm bảo tách cell và holder. |
| Connector (chung) | Consolidated, WACV, PCB-Vision | **D** | Cần relabel subtype theo geometry/BOM. |
| Button/switch/potentiometer/jumper | Consolidated, WACV, RF100 | **D/G** | WACV có button, potentiometer, jumper; RF100 có button/switch. Trimmer thường gom potentiometer. |
| Buzzer | Consolidated, WACV | **D** | Microphone/speaker không được bảo đảm. |
| Display (chung) | Consolidated, WACV | **D/G** | Không tách LCD/OLED/TFT/e-paper/7-segment/matrix. |
| Heat sink | Consolidated, WACV | **D** | Fan/thermal pad/spreader không có nhãn tốt. |

### Không có coverage công khai đủ tin cậy ở mức subtype

Các mục dưới đây phải tạo nhãn từ **BOM/CAD + OCR part marking + ảnh nội bộ**; dùng detector `IC`, `connector`, `module` hoặc `transistor` làm tầng 1:

| Nhóm | Các lớp chưa được hỗ trợ trực tiếp đáng tin cậy |
|---|---|
| Điện trở/tụ đặc biệt | mạng điện trở, biến trở/trimmer tách biệt chắc chắn; tụ tantalum/film/mica/biến dung |
| Diode/quang | Schottky, Zener, TVS, bridge, photodiode tách chức năng |
| Transistor | BJT, JFET, IGBT, Darlington, phototransistor và power MOSFET tách chức năng |
| IC analog/digital | analog IC, digital IC, logic, op-amp, comparator, mux/demux, ADC, DAC |
| Xử lý & bộ nhớ | MCU, MPU, CPU, SoC, FPGA, CPLD, DSP; EEPROM, Flash, SRAM, DRAM, NAND/NOR |
| Nguồn/driver | regulator, LDO, PMIC, buck/boost/buck-boost, DC-DC, AC-DC, power module; motor/gate/LED/display/relay/MOSFET driver |
| Clock | crystal, oscillator, resonator, RTC, clock generator tách riêng |
| Bảo vệ/cách ly | resettable fuse/PTC, NTC/PTC, ESD, surge protector, SSR, optocoupler, digital isolator, isolation transformer |
| Sensor | toàn bộ temperature/humidity/pressure/light/Hall/current/voltage/IMU/proximity/gas/touch sensor |
| Interface | USB/UART/RS232/RS485/CAN/Ethernet PHY/I2C/SPI/LIN IC |
| RF/wireless | Wi-Fi, Bluetooth, Zigbee, LoRa, NFC/RFID, RF transceiver, GNSS, GSM/LTE/5G, RF amplifier |
| Connector subtype | pin/female header, terminal block, board-to-board, wire-to-board, FFC/FPC, JST, Molex |
| Physical port | USB-A/B/C/Micro/Mini, HDMI, DisplayPort, RJ45, audio/DC jack, VGA, DVI |
| Socket | IC/CPU/SIM/SD/MicroSD/battery socket |
| Mechanical controls | tactile/slide/toggle/DIP/rotary switch, rotary encoder tách riêng |
| Audio/display/camera | microphone, speaker, audio amplifier/codec; LCD/OLED/TFT/7-segment/matrix/e-paper; camera/image sensor/IR/optical sensor |
| Antenna/RF | PCB/chip/external antenna, RF/SMA/U.FL connector, RF shield |
| Thermal/mechanical | fan, thermal pad, heat spreader; shield can, spacer, standoff, mounting hole, screw, clip |
| Test/config | test point, jumper cap, programming/debug/JTAG/SWD header tách riêng |
| Module | Wi-Fi/Bluetooth/GPS/cellular/camera/power/sensor/MCU/RF module tách chức năng |

## Lớp phổ biến nên ưu tiên

Không có một ranking phổ quát cho mọi PCB. Danh sách dưới là **ưu tiên kỹ thuật cho một detector AOI tổng quát**, không phải thống kê thị trường; cần thay bằng tần suất từ BOM của nhà máy.

| Ưu tiên | Lớp đề xuất | Lý do |
|---|---|---|
| **P0 – gần như mọi PCBA** | resistor SMD, capacitor SMD (ceramic), IC, diode, transistor/MOSFET, connector, LED, inductor/ferrite bead, crystal/oscillator, test point | Tần suất cao hoặc quan trọng cho kiểm tra presence/absence/polarity; có dữ liệu bootstrap tương đối tốt. |
| **P1 – rất thường gặp theo sản phẩm** | electrolytic/tantalum capacitor, regulator/power IC, fuse/PTC, button/switch, pin header/FFC/FPC/JST, USB/RJ45/DC jack, optocoupler, transformer, relay, heatsink, buzzer, display/module | Có mặt nhiều trên nguồn, công nghiệp, consumer/IoT nhưng không phải mọi board. |
| **P2 – domain-specific** | RF modules/connectors/antenna, sensors, camera, cellular/GNSS, FPGA/CPU/socket, large display, fan, isolation/power modules | Cần dữ liệu theo dòng sản phẩm; khó học đúng từ dataset PCB chung. |
| **Không nên là class thị giác tầng 1** | op-amp, comparator, ADC/DAC, MCU/MPU/SoC, memory type, protocol transceiver, buck/boost, driver type | Đây là **chức năng/part identity**, không phải ngoại hình ổn định. Detect package → OCR → lookup BOM/part database. |

### Taxonomy MVP đề xuất

Để có model đầu tiên khả dụng, bắt đầu với 20–25 lớp hình thái: `resistor_smd`, `resistor_tht`, `capacitor_chip`, `capacitor_electrolytic`, `inductor`, `ferrite_bead`, `transformer`, `diode`, `led`, `transistor_small`, `power_semiconductor`, `ic`, `crystal_oscillator`, `fuse_ptc`, `relay`, `optocoupler`, `connector`, `physical_port`, `switch_button`, `potentiometer`, `display`, `buzzer`, `battery_holder`, `heatsink`, `module`, cộng `unknown_component`.

Sau đó tạo tầng identity: `bbox/crop → OCR marking → normalize → candidate lookup từ BOM → geometry/package/pin-count verification`. Với AOI theo một SKU, cách đáng tin cậy hơn là so từng vị trí với CAD/centroid và golden board thay vì yêu cầu model đoán mọi chức năng từ ảnh.

## Trạng thái pack detector một lớp `component`

Audit ngày 2026-08-30 trên checkpoint đã duyệt mới nhất hiện có **8/10 board
vật lý mục tiêu** tối thiểu. Phép chia ổn định theo hash của board cho 7 nhóm
`train`, 0 nhóm `valid` và 1 nhóm `test`, nên readiness đang là **false — thiếu
`valid`**. Packer chỉ audit và từ chối ghi dataset ở trạng thái này; cần duyệt
thêm các board vật lý độc lập rồi chạy audit lại, không hạ điều kiện bằng cách
chia các tile của cùng board sang nhiều split.

RF100 và Winnies chỉ được đưa vào **train**. Validation/test phải đến từ ảnh
local đã `verified` trong miền camera đích; nếu dùng ảnh public để chấm điểm,
augmentation và các board PCB-DSLR chồng nguồn sẽ làm metric lạc quan giả. Mọi
ảnh public trùng board với local holdout bị loại khỏi pack. Annotation IC chính
thức của PCB-DSLR chỉ dùng audit độ đầy đủ, không tự nhập làm ground truth cho
detector một lớp. Lệnh audit/pack được ghi trong `datasets/public/README.md`.

## Checklist trước khi gộp và train

1. **Tải và inventory:** lưu URL, version/hash, license, số board, số instance/class, resolution, camera và annotation format.
2. **Audit 100–300 instance mỗi nguồn:** sai bbox, class ambiguity, linh kiện không được gán, ảnh duplicate, bbox cực nhỏ, orientation/polarity.
3. **Chuẩn hóa taxonomy có provenance:** giữ `source_label`, `source_dataset`, `board_id`, `component_refdes`; không xóa nhãn gốc.
4. **Chuẩn hoá board identity trước khi chia:** ví dụ `pcb7rec1`, `pcb7__rec1` và `pcb_dslr_007__rec1` đều phải về cùng board `pcb_dslr:007`. Tất cả rec/crop/tile/biến thể của board đó ở cùng split; dùng cả exact hash và mapping tên, không chỉ stem file.
5. **Tách vai trò nguồn:** dữ liệu public/community chủ yếu vào train. Validation/test mục tiêu nên là board đã duyệt từ camera/domain đích. Nếu RF100/WACV/Consolidated trùng một board local giữ lại, loại bản public khỏi train hoặc buộc nó theo cùng split; không để một biến thể của board test lọt vào train.
6. **Khử augment trước split:** Winnies có 173 file từ 73 ảnh nguồn qua flip/xoay; RF100 có nhiều bản crop/rectify/resize của cùng cảnh. Nhóm theo nguồn trước rồi mới chia. Không tin split do export community cung cấp khi chưa audit.
7. **Không tin accuracy ngẫu nhiên theo ảnh:** nhiều dataset chụp cùng board 3–5 lần; random split gây leakage và điểm ảo.
8. **Xử lý vật thể nhỏ:** train bằng tile/SAHI, giữ độ phân giải đủ lớn, cân bằng rare class; augmentation phải giống biến thiên thật của cell AOI.
9. **Đánh giá theo class và kích thước:** mAP50-95, precision/recall, miss rate, false call per board, confusion matrix; riêng polarity/presence/offset phải có metric riêng.
10. **Calibrate confidence trên dữ liệu nhà máy:** “độ tin cậy dataset” trong tài liệu này không thay thế confidence threshold của model.

## Khoảng trống dữ liệu cần tự thu thập

Cho mỗi SKU/board side, nên có golden images và defect injection có kiểm soát: missing, wrong component, offset X/Y, rotation, polarity reverse, tombstone/lifted lead, bridge/insufficient/excess solder. Thu thập trên đầy đủ lot, supplier/package variant, solder mask, silkscreen, illumination drift và camera recalibration. Với lớp hiếm, ưu tiên chụp fixture/coupon bằng đúng hệ quang học thay vì lấy ảnh sản phẩm rời trên Internet.

## Nguồn tham khảo chính

- FPIC: 261 ảnh/93 PCB/>71k instance trên [PhysicalDB](https://physicaldb.ece.ufl.edu/index.php/fics-pcb-image-collection-fpic/) và bài báo [ACM JETC](https://doi.org/10.1145/3588032).
- FICS-PCB: taxonomy sáu lớp và mô tả multimodal trong [paper gốc](https://eprint.iacr.org/2020/366).
- WACV PCB component detection: [trang tác giả và download](https://sites.google.com/view/chiawen-kuo/home/pcb-component-detection), [paper](https://arxiv.org/abs/1811.06994).
- RF100 `printed-circuit-board`: [trang dataset](https://universe.roboflow.com/roboflow-100/printed-circuit-board), [benchmark repo](https://github.com/roboflow/roboflow-100-benchmark). Việc khử 672 file thành 194 PCB và phân biệt với FPIC được mô tả trong [PCB-Detection](https://github.com/SanderGi/PCB-Detection).
- PCB-Vision: dữ liệu/codes ở [GitHub](https://github.com/hifexplo/PCBVision), [paper](https://arxiv.org/abs/2401.06528), [DOI dataset](https://doi.org/10.14278/rodare.2704).
- PCB DSLR: quy mô, annotation và điều khoản nghiên cứu phi thương mại trên [TU Wien CVL](https://cvl.tuwien.ac.at/research/cvl-databases/pcb-dslr-dataset/).
- PCB-SAID: 175 ảnh, 66 state classes, bbox/mask theo [IEEE](https://doi.org/10.1109/ICCVW69036.2025.00145).
- Consolidated dataset: nguồn thành phần, format YOLO và license khai báo trên [Kaggle](https://www.kaggle.com/datasets/aryanstein/pcb-component-detection-consolidated-dataset/data).
