# Quy trình làm việc

Ba quy trình: **(A)** người dùng thu thập ảnh cho một loại dataset, **(B)** cho từng loại dataset, **(C)** người phát
triển sửa code. Kiến trúc chi tiết: [architecture.md](architecture.md).

---

## A. Thu thập ảnh cho một loại dataset

```
 ① check-setup ► ② search TYPE ► ③ run TYPE ► ④ stats TYPE ► ⑤ review + clean TYPE ──► ảnh sẵn sàng
  kiểm tra máy    xem trước       thu thập     kiểm tra        dạy bộ lọc nhiễu          để gán nhãn
  + datasets.yaml (lưu link)      + lọc        số lượng        (lặp lại khi cần)         (ngoài công cụ)
                                     ▲              │
                                     └── thiếu ảnh? thêm câu tìm kiếm vào datasets.yaml, chạy ③ lại
```

`TYPE` là một tên trong [`datasets.yaml`](../datasets.yaml): `birds`, `food_factory_workers`, `vehicle`, hoặc loại bạn thêm.

| Bước | Lệnh | Kết quả |
|---|---|---|
| ① | `mbfs-crawler check-setup` | ✓/✗ cho Python, thư viện, ffmpeg, GPU, model, `datasets.yaml`, quyền ghi, Internet |
| ② | `mbfs-crawler search birds` | các link tìm được lưu vào `links/birds/found.txt` (sửa tay được; chạy lại chỉ thêm link mới); chưa tải video / ảnh nào |
| ③ | `mbfs-crawler run birds` (dùng `datasets.yaml`, gồm cả `found.txt`) hoặc `mbfs-crawler run birds -i links/birds/found.txt` | `../dataset/birds/images/` + báo cáo cuối lần chạy |
| ④ | `mbfs-crawler stats birds` | số ảnh theo nguồn / license / lớp phát hiện, số bị loại và lý do |
| ⑤ | `mbfs-crawler review birds --noise "images/xxx*.jpg"` rồi `mbfs-crawler clean birds` | ảnh giống ví dụ nhiễu chuyển vào `rejected/noise/` |

**Chạy lại an toàn:** mọi lệnh có thể chạy lại. Nguồn đã xong được bỏ qua; nguồn bị dừng giữa chừng (Ctrl+C, mất SSH)
được làm lại từ đầu, sạch sẽ. Không cần dọn gì bằng tay.

**Sau bước ⑤:** ảnh trong `../dataset/<TYPE>/images/` là đầu vào cho việc gán nhãn và train. Hai việc đó không thuộc
công cụ này. `crops.csv` ghi nguồn (video / link) và license của từng ảnh, dùng để chia train/val theo nguồn và kiểm
tra bản quyền.

---

## B. Theo từng loại dataset

```
                       ┌──────────── datasets.yaml ────────────┐
                       │ processor "crop"          processor "frame"
                       ▼                           ▼
  food_factory_workers  người → ảnh từng người     birds   khung hình đầy đủ, giữ khung có chim
  vehicle               xe → ảnh từng xe                   (yolov8n chỉ để CHỌN khung có chim)
                                                           + một phần khung trời trống (ảnh nền)
```

| Loại | Model phát hiện | Xử lý | Ghi chú |
|---|---|---|---|
| `food_factory_workers` | `mbfsn_human_det_v6.onnx` (person) | crop | link tìm được nằm ở `links/food_factory_workers/found.txt` |
| `vehicle` | `mbfsn_vehicle_det_v6.onnx` (car, motorbike, truck, bus) | crop | `crops.csv` ghi loại xe (`det_class`) |
| `birds` | `yolov8n.pt` (COCO bird), chia ô | frame | yolov8n chỉ dùng để chọn khung có chim; khi có model riêng → đổi `detector.model` |
| loại mới | lớp COCO: `yolov8n.pt`; không có: không detector | frame (chỉ lọc theo chuyển động) → gán nhãn → train → đổi sang model mới | chỉ cần thêm một khối vào `datasets.yaml` |

**Đổi cái gì ở đâu** (tất cả trong `datasets.yaml`):

| Muốn | Sửa |
|---|---|
| tìm thêm / tìm khác | `search:` |
| dùng thêm danh sách link | `links:` (từng file `.txt`, đường dẫn tính từ thư mục dự án; mỗi loại một thư mục `links/<TYPE>/`) |
| bỏ kết quả kiểu hoạt hình, game... | `exclude_words:` |
| nhiều / ít kết quả hơn, nguồn khác | `sources:` (`limit`) |
| model khác, lớp khác | `detector:` (`model`, `classes`) |
| nhiều / ít ảnh mỗi người, mỗi xe | `processor:` (`track_gap`, `max_per_track`) |
| ảnh nét hơn / to hơn | `quality:` (`min_sharpness`, `min_height`, `min_conf`) |
| loại trùng mạnh hơn | `filters:` → `duplicate: {max_similarity: ...}` (thấp hơn = loại nhiều hơn) |

Sửa xong: `mbfs-crawler types` hoặc `check-setup` báo ngay nếu có thiết lập sai (kèm gợi ý tên đúng).

---

## C. Phát triển (sửa code / thêm tính năng)

```
 sửa code ──► viết / sửa test ──► ruff check ──► pytest ──► pytest -m system ──► (trước khi phát hành)
                                   (lint)        (≈15 s,      (≈1 phút, model     pytest -m network
                                                 offline)     thật, offline)      benchmark + checklist
```

1. Đọc [architecture.md](architecture.md) để biết sửa ở module nào; quy tắc code trong `CLAUDE.md`.
2. Thêm nguồn / bộ lọc / processor: một lớp + decorator trong module tương ứng (bảng "Mở rộng"). Tuỳ chọn của nó
   dùng được ngay trong `datasets.yaml` (được kiểm tra theo tham số của hàm khởi tạo).
3. Mỗi thay đổi đi kèm test (xem [test-plan.md](test-plan.md)); `ruff check mbfs_crawler tests` và `pytest` phải xanh.
4. Lệnh mới / tuỳ chọn mới: cập nhật README.md (tiếng Việt, cả bash và PowerShell).
5. Trước khi gửi đồng nghiệp: làm theo "Checklist phát hành" trong [test-plan.md](test-plan.md).
