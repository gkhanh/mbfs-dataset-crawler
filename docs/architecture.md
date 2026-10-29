# Kiến trúc mbfs-crawler

Tài liệu này để **nắm toàn bộ dự án trong một trang**: có những gì, nằm ở đâu, nối với nhau thế nào.
Quy trình sử dụng xem [workflow.md](workflow.md); kế hoạch kiểm thử xem [test-plan.md](test-plan.md).

---

## 1. Bức tranh tổng thể

Một công cụ duy nhất để gom ảnh cho **mọi** loại model phát hiện. Mỗi loại dataset là **một khối trong
`datasets.yaml`**, không phải một dự án mới. Công cụ **chỉ thu thập ảnh**: gán nhãn và train là bước riêng, sau đó.

```
        ĐẦU VÀO                         mbfs-crawler                                 KẾT QUẢ
 ┌───────────────────────┐   ┌──────────────────────────────────────┐   ┌──────────────────────────────┐
 │ Tìm kiếm Internet     │   │  1. thu thập  → MediaItem             │   │ dataset/<TYPE>/              │
 │  youtube, wikimedia,  │──►│  2. bỏ qua cái đã làm / trùng / dài   │──►│   images/   ảnh để gán nhãn  │
 │  openverse, archive…  │   │  3. tải về (cache dùng chung)         │   │   crops.csv nguồn + license  │
 │ Link (≈1800 trang)    │   │  4. YOLO: cắt vật thể | giữ khung hình│   │   rejected/ ảnh bị loại      │
 │ File / thư mục máy    │   │  5. lọc: chất lượng, trùng, nhiễu(RAG)│   └──────────────┬───────────────┘
 └───────────────────────┘   │  6. lưu + đánh dấu đã xong            │                  │
                             └──────────────────▲───────────────────┘                  ▼
                                                │                         gán nhãn, train (ngoài công cụ)
                datasets.yaml: birds · food_factory_workers · vehicle · …
                (tìm gì, nguồn nào, model nào, ngưỡng lọc) → `mbfs-crawler run TYPE`
```

**Model dùng:** YOLO để phát hiện (`models/mbfsn_human_det_v6`, `models/mbfsn_vehicle_det_v6`, `yolov8n.pt` cho các lớp
COCO như chim) và **DINOv2-small** chỉ để so sánh ảnh (bỏ trùng + lọc nhiễu RAG). Không có model nào khác.

---

## 2. Cấu trúc thư mục dự án

```
mbfs-dataset-crawler/                ← DỰ ÁN (git repo)
├── datasets.yaml                    ← CÁC LOẠI DATASET (file duy nhất người dùng cần sửa)
├── links/<TYPE>/found.txt           link do lệnh search lưu (mỗi loại một thư mục); datasets.yaml `links:` đọc lại
├── mbfs_crawler/                    ← MÃ NGUỒN (một thư mục, không thư mục con)
│   ├── cli.py          các lệnh: check-setup, types, search, run, review, clean, stats, …
│   ├── pipeline.py     điều phối một lần chạy: thu thập → tải → xử lý → lọc → lưu; và lệnh clean
│   ├── sources.py      nguồn: youtube, bilibili, niconico, wikimedia, openverse, internet_archive,
│   │                   bing_images, google_images, url_list (link), local (file); license
│   ├── downloads.py    tải ảnh (song song) và video (yt-dlp), cache trong ../downloads/
│   ├── processors.py   đọc khung hình (thích ứng), luật chất lượng, xử lý "crop" và "frame"
│   ├── filters.py      bộ lọc: duplicate (trùng), reference (RAG: ví dụ tốt/nhiễu)
│   ├── models.py       YOLO (UltralyticsDetector), DINOv2 (DinoV2Encoder), VectorIndex
│   ├── store.py        thư mục dataset: lưu ảnh, crops.csv, rejected/, báo cáo stats
│   ├── datasets.py     đọc và kiểm tra datasets.yaml (DatasetType)
│   ├── config.py       đường dẫn, registry plugin, RunContext (trạng thái một lần chạy), RunStats
│   ├── utils.py        log, tên file an toàn, đọc/ghi ảnh, độ nét, thiết bị (GPU/CPU)
│   └── setup_check.py  kiểm tra máy (lệnh check-setup)
├── models/                          model phát hiện (.pt / .onnx)
├── tests/                           kiểm thử (xem test-plan.md)
├── docs/                            tài liệu: kiến trúc, quy trình, kế hoạch kiểm thử
├── pyproject.toml                   cài đặt + lệnh mbfs-crawler
├── README.md                        hướng dẫn dùng (tiếng Việt)
└── CLAUDE.md                        quy tắc cho người / AI sửa code

Python tự tạo (không đưa vào git, xoá được): .venv/  __pycache__/

NGOÀI dự án, ngang hàng với nó (vị trí do lớp Paths trong config.py quyết định; đổi bằng --data-dir / --downloads-dir):
mbfs-projects/
├── mbfs-dataset-crawler/            ← dự án: chỉ có mã nguồn và cấu hình
├── dataset/                         ← những gì bạn giữ lại
│   ├── <TYPE>/                      KẾT QUẢ (xem mục 6)
│   ├── references/<TYPE>/           good|noise: ví dụ cho bộ lọc RAG
│   └── logs/                        log của --debug, báo cáo lỗi crash-*.log
└── downloads/                       ← cache video / ảnh tải về, dùng chung mọi loại
```

`mbfs_crawler/` phải tách khỏi thư mục dự án vì Python cần một "package" để `pip install` và lệnh `mbfs-crawler`
hoạt động, và để các tên chung như `config`, `utils`, `models` không đụng thư viện khác.

---

## 3. Các module phụ thuộc nhau thế nào

Mũi tên = "dùng". Tầng dưới không bao giờ dùng tầng trên.

```
                        ┌─────────┐        ┌───────────┐
       tầng lệnh        │ cli.py  │        │ setup_check │
                        └────┬────┘        └───────────┘
                             ▼
       tầng điều phối   ┌─────────────┐
                        │ pipeline.py │
                        └──────┬──────┘
          ┌───────────┬────────┼──────────────┬─────────────┐
          ▼           ▼        ▼              ▼             ▼
   ┌────────────┐┌────────────┐┌──────────────┐┌──────────┐┌──────────┐
   │ sources.py ││downloads.py││processors.py ││filters.py││ store.py │   tầng công việc
   └────────────┘└────────────┘└──────┬───────┘└──────────┘└──────────┘
          │            │              │             │           │
          ▼            ▼              ▼             ▼           ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │   models.py (YOLO, DINOv2, VectorIndex)     datasets.py (YAML)   │   tầng nền
   │   config.py (Paths, registry, RunContext)               utils.py │
   └──────────────────────────────────────────────────────────────────┘
```

Thư viện nặng (torch, ultralytics, yt-dlp, selenium) chỉ được import **bên trong hàm khi cần**, nên các lệnh nhẹ
(`types`, `stats`, `search`, `review`) khởi động trong khoảng 1 giây.

---

## 4. Các lớp chính

| Module | Lớp / hàm chính | Vai trò |
|---|---|---|
| `config.py` | `Paths`, `Registry` (`SOURCES`, `PROCESSORS`, `FILTERS`, `DETECTORS`), `RunContext`, `RunStats` | đường dẫn, đăng ký plugin, trạng thái + bộ đếm của một lần chạy, tạo model khi cần |
| `datasets.py` | `DatasetType`, `load_types`, `load_type` | đọc `datasets.yaml`, kiểm tra mọi tuỳ chọn theo plugin, báo lỗi rõ ràng (kèm gợi ý tên đúng) |
| `sources.py` | `MediaItem`, `Source` + 10 nguồn, `resolve_inputs`, `normalize_license` | biến truy vấn / link / file thành `MediaItem` (link, key, license, tiêu đề, thời lượng) |
| `downloads.py` | `ImageDownloader`, `VideoDownloader` | tải về `../downloads/`, không bao giờ tải lại |
| `processors.py` | `Sample`, `QualityRules`, `FrameSampler`, `ObjectCropProcessor` ("crop"), `FrameProcessor` ("frame") | biến một video / ảnh thành các `Sample` (ảnh ứng viên) |
| `filters.py` | `FilterChain`, `DuplicateFilter`, `ReferenceFilter`, `ReferenceSet` | giữ hoặc loại từng `Sample`, bộ lọc rẻ chạy trước |
| `models.py` | `UltralyticsDetector`, `DinoV2Encoder`, `VectorIndex` | YOLO (detect + track + chia ô), vector ảnh, tìm ảnh giống |
| `store.py` | `DatasetStore`, `RejectedStore`, `dataset_report` | ghi dataset an toàn khi bị dừng giữa chừng, tiếp tục được |
| `pipeline.py` | `Pipeline`, `collect`, `prefetch`, `clean_dataset` | ghép tất cả lại |

---

## 5. Bên trong một lần `run`

```
 cli.cmd_run ──► load_type(TYPE, datasets.yaml) ──► Pipeline.run( collect(search, links | -i, sources) )

 ┌─ luồng tải (chạy song song, đi trước 1 video) ──────────────────────────────┐
 │  MediaItem ─► _wanted?  ✗ đã xong (processed.txt) / đã gặp trong lần chạy   │
 │                         ✗ tiêu đề có exclude_words / quá dài (max_duration) │
 │       │ ảnh → để riêng, xử lý theo lô ở cuối                                │
 │       ▼ video                                                               │
 │  VideoDownloader.fetch (cache ../downloads/videos/<key>.mp4)                   │
 └───────┬─────────────────────────────────────────────────────────────────────┘
         ▼  luồng chính
  Processor.process_video
    FrameSampler: 1 khung / `every` giây; không thấy gì 2 s → giãn dần tới `every_max`
    "crop":  YOLO track → cắt từng vật thể → QualityRules (conf, cao, vỡ, mờ)
             → giữ ảnh NÉT NHẤT mỗi `track_gap` giây, tối đa `max_per_track` / vật thể
    "frame": bỏ khung không đổi (motion) → YOLO (chia ô) → giữ khung có vật thể
             (+ một phần khung trống làm ảnh nền)
         │  danh sách Sample
         ▼
  Pipeline.submit:  DINOv2 mã hoá CẢ LÔ một lần → FilterChain (theo thứ tự trong datasets.yaml)
         ├─ giữ  → DatasetStore.add  (images/, crops.csv, embeddings.f32)
         └─ loại → RejectedStore     (rejected.csv; lưu ảnh nếu lý do = noise)
         │  `stop_after` lần loại liên tiếp → dừng video sớm (không còn gì mới)
         ▼
  DatasetStore.mark_done(key)  → processed.txt  (Ctrl+C trước bước này: lần sau làm lại sạch)
```

---

## 6. Thư mục kết quả `../dataset/<TYPE>/`

```
../dataset/<TYPE>/
├── images/                  ảnh (crop hoặc khung hình) để gán nhãn; công cụ không gán nhãn
├── crops.csv                mỗi ảnh: nguồn, link gốc, khung, thời điểm, độ tin cậy, độ nét, license, lớp phát hiện
├── processed.txt            các nguồn đã xử lý xong (lần sau bỏ qua)
├── embeddings.f32           vector DINOv2 của từng ảnh (dòng i = dòng i của crops.csv), chỉ ghi thêm
└── rejected/                rejected.csv + noise/ (ảnh bị bộ lọc RAG loại, để xem lại)
```

Định dạng giống `worker-dataset` (cùng tên file ảnh, cùng các cột đầu của `crops.csv`, `processed.txt`): một thư mục
dataset cũ chép vào `../dataset/<TYPE>/` sẽ được chạy tiếp.

---

## 7. Mở rộng

| Muốn thêm | Làm gì | Ở đâu |
|---|---|---|
| Loại dataset mới | chép một khối, đổi tên và câu tìm kiếm | `datasets.yaml` |
| Nguồn tìm kiếm mới | một lớp `Source` + `@SOURCES.register("tên")`, viết `search(query)`; rồi thêm tên vào `sources:` của loại cần dùng | `sources.py` |
| Bộ lọc mới | một lớp `Filter` + `@FILTERS.register("tên")`, viết `check(sample)` | `filters.py` |
| Cách xử lý mới | một lớp `Processor` + `@PROCESSORS.register("tên")` | `processors.py` |
| Model mới | file `.pt`/`.onnx` vào `models/`, trỏ `detector.model` trong `datasets.yaml` | `models/` |

Mỗi phần mới cần kiểm thử (xem [test-plan.md](test-plan.md), mục 7).
