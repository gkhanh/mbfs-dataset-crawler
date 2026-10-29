# mbfs-crawler: thu thập ảnh cho dataset phát hiện đối tượng

Một công cụ duy nhất, chạy hoàn toàn bằng **dòng lệnh** (không cửa sổ, chạy được qua SSH), để gom ảnh cho các model YOLO:
đàn chim ở sân bay, công nhân nhà máy thực phẩm (găng tay, lưới tóc), xe cộ, và các loại sau này.

- Tìm video và ảnh trên Internet (không cần API key), hoặc dùng link / file / thư mục có sẵn.
- Dùng model YOLO trong `models/` để **cắt từng người / xe** hoặc **giữ cả khung hình** (chim).
- Tự bỏ ảnh mờ, ảnh trùng, và ảnh "nhiễu" giống các ví dụ xấu bạn đã đánh dấu (lọc RAG).
- Kết quả: thư mục `../dataset/<loại>/images/` (nằm **ngoài** thư mục dự án), sẵn sàng để gán nhãn.
- **Chỉ thu thập ảnh**: công cụ không gán nhãn, không tạo dataset train. Gán nhãn và train là bước riêng, sau khi có ảnh.
- **Thu thập cái gì** nằm trong **một file duy nhất: [`datasets.yaml`](datasets.yaml)**. Muốn đổi câu tìm kiếm, model, ngưỡng
  lọc, hay thêm loại mới: chỉ sửa file này, không đụng vào code.

📚 Tài liệu trong [`docs/`](docs/): [kiến trúc + sơ đồ](docs/architecture.md) ·
[quy trình làm việc](docs/workflow.md) · [kế hoạch kiểm thử](docs/test-plan.md)

---

## 1. Yêu cầu

- Python **3.10 trở lên**, git. Windows 10/11, macOS hoặc Linux.
- Khuyến nghị GPU NVIDIA (hoặc chip Apple M). Không có GPU vẫn chạy, chậm hơn.
- Không cần cài ffmpeg (đi kèm gói `imageio-ffmpeg`).

## 2. Cài đặt

**macOS / Linux (bash, zsh)**

```bash
cd mbfs-dataset-crawler
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[google]"
```

**Windows (PowerShell)**

```powershell
cd mbfs-dataset-crawler
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
# Chỉ khi có GPU NVIDIA (cài torch bản CUDA trước):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[google]"
```

- PowerShell báo không chạy được `Activate.ps1`: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` rồi thử lại.
- `.[google]` = cài thêm phần tuỳ chọn `google` (nguồn Google Images, cần Google Chrome). Không dùng thì: `pip install -e .`
- Mỗi lần mở terminal mới: `source .venv/bin/activate` (macOS/Linux) hoặc `.venv\Scripts\Activate.ps1` (Windows).
- Luôn chạy lệnh **trong thư mục dự án** (nơi có `datasets.yaml` và `models/`), hoặc đặt biến `MBFS_HOME` trỏ tới đó.

## 3. Kiểm tra cài đặt

```bash
mbfs-crawler check-setup   # ✓ là ổn, ✗ là lỗi phải sửa (có gợi ý →), ! là tuỳ chọn. Kiểm tra cả datasets.yaml
mbfs-crawler types         # các loại dataset trong datasets.yaml
mbfs-crawler sources       # các nguồn tìm kiếm
mbfs-crawler models        # model trong models/ và các lớp của chúng
mbfs-crawler --help        # mọi lệnh; mbfs-crawler run --help: mọi tuỳ chọn của run
```

## 4. Các loại dataset có sẵn

| Loại (`TYPE`) | Thu thập gì | Lưu gì | Model |
|---|---|---|---|
| `birds` | đàn chim trên trời, gần sân bay / bãi trực thăng | **cả khung hình** có chim (+ ít khung không có chim làm mẫu âm) | `yolov8n.pt` (tự tải lần đầu), chia ô để thấy chim nhỏ |
| `food_factory_workers` | công nhân nhà máy thực phẩm, bếp nhà hàng (găng tay, lưới tóc) | **mỗi người một ảnh** | `mbfsn_human_det_v6.onnx` |
| `vehicle` | ô tô, xe máy, xe tải, xe buýt trong giao thông | **mỗi xe một ảnh** | `mbfsn_vehicle_det_v6.onnx` |

Cấu hình đầy đủ của từng loại (câu tìm kiếm, nguồn, ngưỡng...) nằm trong [`datasets.yaml`](datasets.yaml), có chú thích
tiếng Việt cho mọi thiết lập.

## 5. Thu thập dữ liệu: từng bước

```
 check-setup → search TYPE → run TYPE → stats TYPE → review / clean TYPE → (gán nhãn, train: ngoài công cụ)
               xem trước     thu thập    kiểm tra     dạy bộ lọc nhiễu
```

Mọi lệnh có dạng **`mbfs-crawler <lệnh> <TYPE>`**, với `TYPE` là một tên trong `datasets.yaml`.
Chi tiết từng bước và sơ đồ: [docs/workflow.md](docs/workflow.md).

### Bước 1: tìm link (không tải video / ảnh)

```bash
mbfs-crawler search birds                     # chạy các câu tìm kiếm của birds, in kết quả, LƯU link vào links/birds/found.txt
mbfs-crawler search birds --limit 5           # ít kết quả hơn, nhanh hơn
mbfs-crawler search birds -o links/birds/tuan1.txt   # lưu vào file khác (nhớ khai báo nó trong `links:` nếu muốn `run` dùng)
```

Link của mỗi loại nằm trong **thư mục riêng ngay trong dự án**: `links/birds/`, `links/food_factory_workers/`,
`links/vehicle/` (không nằm ở thư mục gốc). Mở `found.txt`, xoá dòng không muốn hoặc thêm link của bạn. Chạy `search` lại
thì chỉ **thêm link mới vào cuối** file, không ghi đè phần bạn đã sửa. Mỗi dòng ghi kèm license và nguồn.

### Bước 2: thu thập

```bash
# Cách 1: dùng câu tìm kiếm + danh sách link (`links:`, gồm links/<TYPE>/found.txt) trong datasets.yaml
mbfs-crawler run birds
mbfs-crawler run food_factory_workers --cookies-from-browser firefox   # YouTube chặn thì dùng đăng nhập của trình duyệt
mbfs-crawler run vehicle --limit 3                                      # thử nhanh: 3 kết quả mỗi nguồn mỗi câu

# Cách 2: CHỈ dùng link / file của bạn (bỏ qua tìm kiếm)
mbfs-crawler run birds -i links/birds/found.txt                         # danh sách vừa sửa ở bước 1 (license được giữ)
mbfs-crawler run vehicle -i "https://www.youtube.com/watch?v=XXXX"
mbfs-crawler run vehicle -i ~/Videos/traffic/                           # macOS / Linux
mbfs-crawler run vehicle -i C:\Videos\traffic\                          # Windows
```

- `-i` nhận: link (YouTube và ≈1800 trang khác), link ảnh, file video/ảnh, thư mục, file `.txt` (mỗi dòng một link).
  Lặp lại được: `-i a.txt -i b.mp4`.
- Kết quả ở **`../dataset/<TYPE>/images/`** (ngang hàng với thư mục dự án). Cuối mỗi lần chạy có báo cáo: số khung đã
  đọc / bỏ qua, số lần gọi model, số ảnh lưu / loại (và lý do), dung lượng tải, thời gian.
- **Chạy lại cùng lệnh là an toàn**: nguồn đã xong được bỏ qua, nguồn đang làm dở (Ctrl+C, mất mạng) được làm lại sạch.
  Muốn thêm ảnh: chạy lại, hoặc thêm câu tìm kiếm vào `datasets.yaml` rồi chạy lại.

### Bước 3: kiểm tra và lọc nhiễu

```bash
mbfs-crawler stats food_factory_workers                                    # số ảnh theo nguồn, license, lớp, lý do bị loại
mbfs-crawler review food_factory_workers --noise "images/*_id7.jpg"        # đánh dấu ảnh nhiễu (poster, TV, ma-nơ-canh...)
mbfs-crawler review food_factory_workers --good images/tot1.jpg images/tot2.jpg
mbfs-crawler clean food_factory_workers --dry-run                          # xem trước những ảnh sẽ bị loại
mbfs-crawler clean food_factory_workers                                    # chuyển chúng vào rejected/noise/
```

Ví dụ được lưu ở `../dataset/references/<TYPE>/good|noise/`. Từ đó mỗi lần `run` tự loại ảnh giống ví dụ nhiễu: càng
đánh dấu nhiều, dataset càng sạch. Đặt pattern trong dấu ngoặc kép (PowerShell không tự mở rộng `*`, công cụ tự làm).

### Sau đó

Ảnh trong `../dataset/<TYPE>/images/` là đầu vào cho bước gán nhãn và train. Hai bước đó **không** thuộc công cụ này.
`crops.csv` cho biết mỗi ảnh lấy từ video / link nào và license của nó (để chia train/val theo nguồn và kiểm tra bản quyền).

## 6. Sửa hoặc thêm loại dataset (`datasets.yaml`)

Mở [`datasets.yaml`](datasets.yaml) ở thư mục dự án. Phần đầu file giải thích từng thiết lập bằng tiếng Việt.
Sửa xong, chạy `mbfs-crawler check-setup` hoặc `mbfs-crawler types`: thiết lập sai được báo ngay, kèm gợi ý.

**Đổi câu tìm kiếm** (việc hay làm nhất): sửa danh sách `search:` của loại đó.

```yaml
food_factory_workers:
  search:
    - food factory workers gloves hairnet
    - seafood processing workers gloves      # thêm một dòng
```

**Thêm loại mới**: chép một khối có sẵn, đổi tên (chữ, số, `_`), sửa `description`, `search`, `detector`:

```yaml
drones:
  description: Drone bay gần sân bay; cả khung hình
  search: [drone flying sky far away, drone near airport camera]
  sources:
    youtube: {limit: 10}
    wikimedia: {limit: 30, media: [image, video]}
  processor: {type: frame, every: 1.0, min_change: 0.00003}   # chưa có model: giữ khung hình có chuyển động
  filters:
    duplicate: {max_similarity: 0.92}
```

Rồi: `mbfs-crawler run drones`. Khi đã train được model: chép vào `models/`, thêm
`detector: {model: models/drone_v1.pt, classes: [drone]}`. Đối tượng là một trong 80 lớp COCO (person, car, bird, dog...):
dùng ngay `models/yolov8n.pt`.

Không đặt tên loại là `references` hoặc `logs` (trùng thư mục của công cụ).

## 7. Nguồn dữ liệu (không cần API key)

| Nguồn | Loại | License |
|---|---|---|
| `youtube`, `bilibili`, `niconico` | video | không rõ (rủi ro bản quyền) |
| `wikimedia` | ảnh + video | CC / public domain, ghi theo từng file |
| `openverse` | ảnh | Creative Commons (đôi khi chậm) |
| `internet_archive` | video | theo từng mục |
| `google_images` | ảnh (cần Chrome) | không rõ |
| `bing_images` | ảnh | không rõ; **không bật sẵn** (hay trả kết quả không liên quan, xem trước bằng `search`) |
| link (`-i URL`), file, thư mục | mọi loại | theo link / `local` |

Nguồn của mỗi loại và số kết quả mỗi câu tìm kiếm: mục `sources:` trong `datasets.yaml`.

## 8. Tiết kiệm tài nguyên

Mặc định đã tối ưu. Điều chỉnh trong `datasets.yaml` (theo từng loại) hoặc bằng tuỳ chọn dòng lệnh:

| Ở đâu | Thiết lập | Tác dụng |
|---|---|---|
| `processor:` | `every: 0.2` / `every_max: 1.0` | phân tích 1 khung mỗi 0.2 s; không thấy gì thì giãn dần tới 1 khung / 1 s |
| `processor:` | `stop_after: 50` | dừng video sau 50 ảnh bị loại liên tiếp (camera tĩnh, không còn gì mới); `0` để tắt |
| `download:` | `max_res: 720` | độ phân giải tải về (model chạy ở 640 px, 720p là đủ) |
| `download:` | `max_minutes: 5` | chỉ tải 5 phút đầu của mỗi video |
| `download:` | `max_duration: 1800` | bỏ qua video dài hơn 30 phút |
| dòng lệnh | `--delete-videos` | xoá video sau khi xử lý xong |
| dòng lệnh | `--batch 16` | số ảnh xử lý mỗi lô |
| dòng lệnh | `--threads 4` | giới hạn số luồng CPU (máy dùng chung) |
| dòng lệnh | `--device cpu\|mps\|0` | chọn thiết bị (mặc định tự chọn: GPU NVIDIA → Apple → CPU) |

Gợi ý laptop: `mbfs-crawler run vehicle --delete-videos --threads 4`. Server GPU: để mặc định.
Video tải về nằm ở `../downloads/` và được **dùng lại** cho mọi loại: không bao giờ tải hai lần.

## 9. Chạy lâu trên server qua SSH

**Linux / macOS**

```bash
tmux new -s crawl                                   # thoát ra: Ctrl+B rồi D;  vào lại: tmux attach -t crawl
mbfs-crawler run food_factory_workers --log-file crawl.log
# hoặc chạy nền:
nohup mbfs-crawler run food_factory_workers --log-file crawl.log --no-progress > /dev/null 2>&1 &
tail -f crawl.log
```

**Windows (PowerShell)**

```powershell
Start-Process mbfs-crawler -ArgumentList "run food_factory_workers --log-file crawl.log --no-progress" -NoNewWindow
Get-Content crawl.log -Wait
```

Mất kết nối? Chạy lại đúng lệnh đó: công cụ tiếp tục từ chỗ dừng. Mã thoát: `0` xong, `1` lỗi, `2` sai tham số hoặc
`datasets.yaml`, `3` không có gì để làm, `130` bị dừng (Ctrl+C).

## 10. Tham số dòng lệnh

**`mbfs-crawler run TYPE`**

| Tham số | Ý nghĩa |
|---|---|
| `TYPE` | loại dataset trong `datasets.yaml` (`mbfs-crawler types` để xem) |
| `-i PATH\|URL\|LIST.txt` (lặp lại được) | chỉ thu thập từ những đầu vào này, bỏ qua tìm kiếm |
| `--limit N` | số kết quả mỗi nguồn mỗi câu tìm kiếm (mặc định: theo `datasets.yaml`) |
| `--force` | xử lý lại cả nguồn đã làm |
| `--delete-videos` | xoá video sau khi xử lý |
| `--cookies-from-browser firefox` | dùng đăng nhập YouTube của trình duyệt khi bị chặn |

**Dùng được với mọi lệnh**

| Tham số | Ý nghĩa |
|---|---|
| `--home DIR` | thư mục dự án (chứa `datasets.yaml`, `models/`); mặc định thư mục hiện tại hoặc biến `MBFS_HOME` |
| `--data-dir DIR` | nơi đặt kết quả, ví dụ, log (mặc định `../dataset`; hoặc biến `MBFS_DATA`) |
| `--downloads-dir DIR` | nơi lưu video / ảnh tải về (mặc định `../downloads`; hoặc biến `MBFS_DOWNLOADS`) |
| `--device`, `--batch`, `--threads`, `--no-half` | thiết bị và tài nguyên (mục 8) |
| `--debug` | chế độ gỡ lỗi (mục 13): ghi mọi chi tiết vào `<data-dir>/logs/` |
| `--log-file`, `--no-progress`, `--quiet` | đầu ra |

Các lệnh khác: `search TYPE [--limit N] [-o links.txt]`, `stats TYPE`, `review TYPE --good ... --noise ...`,
`clean TYPE [--dry-run]`, `types`, `sources`, `models`, `check-setup [--offline]`.

## 11. Cấu trúc thư mục

```
mbfs-dataset-crawler/
├── datasets.yaml        ← CÁC LOẠI DATASET: file duy nhất cần sửa
├── links/               link do lệnh `search` lưu, mỗi loại một thư mục
│   ├── birds/found.txt
│   ├── food_factory_workers/found.txt
│   └── vehicle/found.txt
├── models/              model phát hiện
├── mbfs_crawler/        mã nguồn
├── tests/               kiểm thử
└── docs/                kiến trúc, quy trình, kế hoạch kiểm thử
```

**Dự án chỉ chứa mã nguồn và cấu hình.** Mọi thứ công cụ tạo ra nằm ở hai thư mục **ngang hàng** với thư mục dự án:

```
mbfs-projects/
├── mbfs-dataset-crawler/      ← dự án
├── dataset/                   ← những gì bạn giữ lại (đổi bằng --data-dir)
│   ├── food_factory_workers/  KẾT QUẢ của một loại: images/ crops.csv processed.txt rejected/ ...
│   ├── birds/
│   ├── references/<TYPE>/     ví dụ good|noise cho bộ lọc nhiễu
│   └── logs/                  log của --debug và báo cáo lỗi (crash-*.log)
└── downloads/                 ← cache video / ảnh tải về, dùng chung mọi loại (đổi bằng --downloads-dir)
```

Muốn đặt chỗ khác (ổ đĩa lớn, ổ mạng): `mbfs-crawler run birds --data-dir D:\data --downloads-dir E:\cache`, hoặc đặt
biến `MBFS_DATA`, `MBFS_DOWNLOADS`. Sơ đồ module và luồng dữ liệu: [docs/architecture.md](docs/architecture.md).

Thư mục Python tự tạo, không đưa vào git, xoá lúc nào cũng được: `__pycache__/` (bản biên dịch khi chạy), `.venv/`
(môi trường Python, chứa cả thông tin cài đặt của `pip install -e .`).

## 12. Bản quyền

- Mỗi ảnh có cột `license` trong `crops.csv` (`cc` = Creative Commons cho phép thương mại; `cc-nc` = phi thương mại /
  không phái sinh; `unknown` = YouTube, Bing, Google; `local` = file của bạn). `stats TYPE` đếm ảnh theo license.
- **MPII Cooking 2** (tên `generic_s..`) chỉ dùng cho nghiên cứu.
- Video YouTube có rủi ro bản quyền: hỏi công ty trước khi dùng thương mại.
- Model Ultralytics theo giấy phép AGPL-3.0.

## 13. Xử lý sự cố

| Triệu chứng | Cách xử lý |
|---|---|
| `datasets.yaml not found` | chạy lệnh trong thư mục dự án, hoặc thêm `--home <thư mục dự án>` |
| `unknown dataset type 'x'` | `mbfs-crawler types` để xem tên đúng (lỗi có gợi ý "Did you mean ...") |
| `datasets.yaml, type 'x': unknown setting ...` | gõ sai tên thiết lập trong `datasets.yaml`; lỗi chỉ ra chỗ sai và gợi ý tên đúng |
| YouTube báo 403 / "Sign in to confirm" | `--cookies-from-browser firefox` (trên Windows dùng firefox: Chrome mã hoá cookie) |
| `libGL.so.1: cannot open shared object` (Linux server) | `sudo apt install libgl1` |
| Không dùng GPU | `python -c "import torch; print(torch.cuda.is_available())"`; trên Windows cài torch bản CUDA (mục 2) |
| Openverse báo "no answer" | nguồn đó chậm; bỏ qua hoặc giảm `--limit` |
| Google Images rỗng | cần Google Chrome; Google hay thay đổi trang |
| Windows: lỗi đường dẫn quá dài | đặt dự án gần gốc ổ đĩa, ví dụ `C:\mbfs\` |
| Một video / link báo `! cannot download ...` | bình thường: link đó bị bỏ qua, các link còn lại vẫn chạy tiếp; cuối lần chạy có dòng `warnings: N` |
| Wikimedia báo "too many requests" (429) | công cụ tự chờ và thử lại; nếu vẫn bị, giảm `--limit` rồi chạy lại (cái đã tải được giữ lại) |

### Chế độ gỡ lỗi `--debug`

Thêm `--debug` vào **bất kỳ lệnh nào** để xem và ghi lại mọi chi tiết:

```bash
mbfs-crawler run birds -i links/birds/found.txt --debug
```

- Terminal hiện thêm từng yêu cầu mạng và câu trả lời (`GET ...: HTTP 404`), từng lần thử tải, thông điệp của yt-dlp.
- Dòng đầu cho biết file log: `<data-dir>/logs/run-<ngày>-<giờ>.log` (mặc định `../dataset/logs/`). File này có giờ, tên
  luồng (`[download]` là luồng tải trước video kế tiếp) và **toàn bộ traceback** của từng lỗi, kể cả lỗi đã được bỏ qua.
- Không cần `--debug` từ trước: nếu chương trình dừng vì lỗi bất ngờ, nó **luôn** ghi `logs/crash-<giờ>.log` (traceback,
  lệnh đã chạy, phiên bản, hệ điều hành) và in đường dẫn file đó. Gửi file này khi báo lỗi.
- Muốn log ở chỗ khác: `--log-file duongdan.log`.
- Bình thường (không `--debug`) terminal chỉ hiện các dòng ngắn: mỗi lỗi một dòng bắt đầu bằng `!`, không có thanh tiến
  độ bị chen ngang, không có traceback.

## 14. Kiểm thử

```bash
pip install -e ".[dev]"
pytest                      # unit + integration: offline, ≈ 15 giây
pytest -m system            # chạy lệnh thật với model thật, ≈ 1 phút
pytest -m network -rs       # nguồn Internet thật (chậm)
pytest --cov=mbfs_crawler   # độ phủ
ruff check mbfs_crawler tests
```

Kế hoạch đầy đủ, kết quả và checklist phát hành: [docs/test-plan.md](docs/test-plan.md).

## 15. Báo lỗi

Gửi kèm: (1) kết quả `mbfs-crawler check-setup`, (2) đúng lệnh đã chạy, (3) file log: chạy lại đúng lệnh đó với
`--debug`, rồi gửi file `logs/run-<giờ>.log` (đường dẫn in ngay dòng đầu). Nếu chương trình báo lỗi bất ngờ, nó tự ghi
`logs/crash-<giờ>.log`: gửi file này.
