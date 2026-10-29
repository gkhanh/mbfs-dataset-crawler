# Kế hoạch kiểm thử

Tài liệu này trả lời bốn câu hỏi: **kiểm thử cái gì**, **vì sao chọn cách kiểm thử như vậy**, **file test nào bảo vệ phần nào của
code (và khỏi lỗi gì)**, và **kết quả hiện tại**. Kiến trúc xem [architecture.md](architecture.md); quy trình xem
[workflow.md](workflow.md).

---

## 1. Mục tiêu và nguyên tắc

Công cụ sắp được gửi cho đồng nghiệp, mỗi người một máy khác nhau. Vì vậy phải chứng minh ba điều:

| Mục tiêu | Vì sao quan trọng | Phần kiểm thử chính |
|---|---|---|
| **Chạy đúng**: ra đúng ảnh, đúng thông tin nguồn và license, đúng cấu hình trong `datasets.yaml` | Dataset sai âm thầm rất tai hại: model train xong kém mà không ai biết nguyên nhân | `test_processors`, `test_storage`, benchmark |
| **Chạy được ở mọi nơi**: Windows, macOS, Linux, SSH không màn hình, đường dẫn có dấu tiếng Việt | Người dùng không ngồi cạnh tôi để sửa; lỗi chỉ có ở Windows sẽ chỉ lộ ra khi đã gửi | `test_utils`, `test_cli`, `test_system`, thử tay trên Windows |
| **Không bỏ cuộc giữa chừng và tiết kiệm**: một link hỏng không làm mất cả lần chạy, không tải / tính hai lần | Lần chạy dài hàng chục phút; mất giữa chừng là tốn công và băng thông | `test_cli` (robustness), `test_models` (downloads), `test_integration` |

Bốn nguyên tắc dẫn đến cách tổ chức phía dưới:

1. **Mặc định phải nhanh và chạy offline** (khoảng 10 giây, không mạng, không tải model). Test chậm thì người ta ngại chạy;
   test cần mạng thì lúc đỏ lúc xanh. Vì vậy `pytest` thường chỉ chạy loại nhanh; loại chậm có cờ riêng (`-m system`, `-m network`).
2. **Dùng đồ giả cho thứ chậm hoặc không ổn định, dùng đồ thật cho thứ không thể giả.** Model YOLO / DINOv2, Internet, Chrome
   là đồ chậm hoặc thất thường, nên test logic dùng bản giả (xem mục 3). Nhưng đồ giả có thể che lỗi ghép nối (ví dụ thiếu gói
   `onnx` chỉ lộ ra với model thật), nên vẫn có tầng `system` chạy model thật.
3. **Mỗi lỗi thật đã gặp biến thành một test** để không quay lại (mục 5).
4. **Test hành vi, không test cài đặt bên trong.** Test hỏi "kết quả có đúng không", không hỏi "hàm gọi hàm nào", để sau này
   được phép tối ưu / viết lại code mà test vẫn đúng.

---

## 2. Các tầng kiểm thử

```
                      ▲ ít, chậm, sát thực tế
                     ╱ ╲
   network  (10)    ╱   ╲   Internet thật: nguồn còn trả đúng định dạng không?
   system   ( 5)   ╱─────╲  lệnh thật + model thật + không màn hình + gói cài đặt
   integration(12)╱───────╲ cả chương trình trong một tiến trình, model giả, "Internet" giả
   unit (237)    ╱─────────╲ từng module riêng, mọi thứ giả, mỗi test vài mili-giây
                ▼ nhiều, nhanh, dễ chỉ ra lỗi ở đâu
```

| Tầng | Lệnh | Thời gian | Cần gì | Khi nào chạy | Vì sao có tầng này |
|---|---|---|---|---|---|
| Lint | `ruff check mbfs_crawler tests` | < 1 s | — | mỗi lần sửa code | bắt lỗi chính tả biến, import thừa trước khi chạy test |
| Unit | `pytest` | ≈ 10 s | không mạng, không model | mỗi lần sửa code | chỉ ngay module nào hỏng; đủ nhanh để chạy liên tục |
| Integration | `pytest` (cùng lệnh) | gộp trong ≈ 10 s | như trên | mỗi lần sửa code | các module đúng riêng lẻ chưa chắc ghép đúng với nhau |
| Độ phủ | `pytest --cov=mbfs_crawler` | ≈ 15 s | như trên | trước khi merge (≥ 85%) | phát hiện code chưa có test nào chạm tới |
| System | `pytest -m system` | ≈ 25 s | model trong `models/`, DINOv2 (tải 1 lần) | trước khi merge | đồ giả có thể che lỗi ghép nối với model / hệ điều hành thật |
| Network | `pytest -m network -rs` | 5–15 phút | Internet | trước khi phát hành | trang web đổi định dạng thì code vẫn pass test cũ mà hỏng thật; chỉ Internet thật mới lộ |
| Benchmark | `python tests/benchmark.py VIDEO` | vài phút | một video | trước khi phát hành | đo tăng tốc tiết kiệm được bao nhiêu và đổi lấy bao nhiêu ảnh |

**Vì sao không chỉ giữ một tầng?** Chỉ có unit thì không biết các phần ghép có chạy; chỉ có system thì chậm và khi đỏ không biết
lỗi ở đâu. Tháp test cho nhiều test nhanh ở dưới (tìm lỗi chính xác), ít test chậm ở trên (xác nhận ghép nối).

**Vì sao network tách riêng và được phép "skip"?** Openverse đôi khi chậm, Bilibili / Niconico / Google có thể bị chặn theo mạng.
Nếu gộp vào test thường, một trang web chập chờn sẽ làm đỏ mọi lần merge dù code không sai. Skip kèm lý do là chấp nhận được;
test fail vì sai dữ liệu thì không.

**Không có CI tự động** (repo chưa có remote; file workflow đã được gỡ chủ ý). Vì vậy checklist phát hành (mục 8) yêu cầu
chạy các tầng trên bằng tay, **kể cả trên một máy Windows**. **Vì sao phải thử Windows?** Có lỗi chỉ xảy ra trên Windows
(tên file `CON`, `NUL`, đường dẫn có dấu, mã hoá console cp1252); chạy trên máy macOS sẽ không bao giờ thấy. Code đã phòng
các lỗi đó (và có test unit cho chúng), nhưng chỉ máy Windows thật mới xác nhận được.

---

## 3. Đồ hỗ trợ kiểm thử và lý do

| Đồ hỗ trợ | Là gì | Vì sao dùng |
|---|---|---|
| `FakeDetector` (`helpers.py`) | "Model" tìm các khối sáng (pixel > 200) trong ảnh, đánh số track từ trái sang phải | Test **logic cắt ảnh** (chọn ảnh nét nhất, giới hạn số ảnh, dừng sớm) chứ không phải độ chính xác của YOLO. Kết quả xác định, chạy trong mili-giây, không cần file model |
| `FakeEncoder` (`helpers.py`) | Vector từ vùng giữa của ảnh thu nhỏ: ảnh giống nhau → vector giống nhau | Test bộ lọc trùng / RAG mà không tải DINOv2 (~90 MB). Hai ảnh cùng nội dung phải ra độ giống 1.0, khác nội dung thì thấp |
| Ảnh / video tổng hợp (`helpers.py`) | Nền nhiễu + khối sáng chuyển động, tạo bằng OpenCV lúc chạy test | Không phải lưu video nhị phân lớn trong git; kiểm soát được chính xác "có người lúc nào, rời đi lúc nào" |
| `web` (`conftest.py`) | Web server cục bộ (`127.0.0.1`) đóng vai Internet | Chạy **yt-dlp thật, requests thật, code tải thật** mà không cần mạng, không bị giới hạn tốc độ, kết quả xác định |
| `ctx_factory` (`conftest.py`) | Tạo `RunContext` có sẵn detector / encoder giả, thư mục tạm | Mỗi test một "môi trường" riêng, không dùng chung trạng thái |
| `log_lines`, `restore_logging` (`conftest.py`) | Thu lại các dòng log của test; trả logger về như cũ sau mỗi test | `cli.main()` cấu hình lại logger toàn cục; không khôi phục thì test này làm sai test kia |
| `tests/fixtures/*.json` | **Phản hồi thật** đã lưu của Wikimedia, Openverse, Internet Archive | Ghim lại định dạng thật mà code đang phân tích. Nếu trang đổi định dạng, test network đỏ → cập nhật fixture và code cùng lúc |
| Ảnh mẫu của Ultralytics (`bus.jpg`, `zidane.jpg`) | Ảnh có sẵn trong thư viện, có người và xe buýt thật | Test `system` dùng model thật trên ảnh thật mà không cần tải thêm gì |

---

## 4. File test ↔ module: kiểm tra gì và vì sao

### `test_utils.py` (74 test) — nền tảng (`utils.py`, `config.py`, `datasets.py`)

| Nhóm test | Kiểm tra | Vì sao cần |
|---|---|---|
| Tên file (`slug`, `local_key`, `url_key`, `video_key`) | không còn ký tự cấm; tên Windows dành riêng (`CON`, `NUL`, `COM1`…) được đổi; khoá ổn định giữa các lần chạy | Tên file sai làm lỗi ghi file trên Windows; khoá không ổn định làm "việc đã xong" bị coi là mới và làm lại |
| Ảnh (`pad_box`, `sharpness`, `is_broken`, `changed_fraction`) | khung không vượt ngoài ảnh; độ nét so sánh được giữa ảnh to và nhỏ; ảnh phẳng / viền đen / dải bị cắt bị loại; phát hiện con chim nhỏ chuyển động nhưng bỏ qua nhiễu camera | Các ngưỡng này quyết định ảnh nào vào dataset; sai là dataset lẫn ảnh rác hoặc mất ảnh tốt |
| Đọc / ghi ảnh đường dẫn tiếng Việt | `read_image` / `write_jpeg` chạy với thư mục `ảnh thử Khánh` | `cv2.imread` thất bại âm thầm với đường dẫn có dấu trên Windows |
| Chọn thiết bị | NVIDIA → Apple → CPU; lựa chọn tường minh được giữ | Chọn nhầm thiết bị làm chậm hoặc lỗi |
| Registry | đăng ký / tạo theo tên; trùng tên bị chặn; tên sai được gợi ý ("Did you mean 'youtube'?") | Thêm plugin chỉ bằng một lớp + decorator phải đáng tin |
| `datasets.yaml` | file của dự án có đủ 3 loại và đều hợp lệ; tên loại gõ sai được gợi ý; thiếu file / YAML hỏng / thiết lập sai kiểu **báo đúng loại và đúng chỗ sai** (kèm gợi ý tên đúng); tuỳ chọn được kiểm theo plugin; `frame` không detector hợp lệ; `links:` tính từ thư mục chứa file; giữ thứ tự loại như trong file | Đây là file duy nhất người dùng sửa, bằng tay; lỗi mơ hồ thì họ bỏ cuộc |
| `Paths` | mặc định `../dataset` và `../downloads` nằm **ngoài** thư mục dự án; đổi bằng tuỳ chọn / biến môi trường; tên loại `references`, `logs` bị cấm | Yêu cầu "dự án chỉ chứa mã nguồn"; tránh loại trùng tên thư mục của công cụ |
| Log (`setup`, `--debug`, `short_error`, `YtDlpLogger`) | terminal ngắn, file log đầy đủ có giờ + tên luồng + traceback; đếm cảnh báo; thông điệp lỗi một dòng dễ đọc; yt-dlp không in đè thanh tiến độ | Yêu cầu "log terminal gọn" và "debug mode khi chạy hỏng" |

### `test_sources.py` (53 test) — nguồn dữ liệu (`sources.py`)

| Nhóm test | Kiểm tra | Vì sao cần |
|---|---|---|
| License | chuẩn hoá `CC BY-SA 4.0`, URL Creative Commons, CC0, public domain; phân loại `pd / cc / cc-nc / local / unknown` | cột `license` của `crops.csv` và `stats` dựa vào đây để biết ảnh nào được dùng thương mại |
| Wikimedia, Openverse, Internet Archive | phân tích **phản hồi thật đã lưu**: số lượng theo `limit`, trang kế tiếp, license, ảnh thu nhỏ 1920 px; dịch vụ không trả lời thì dừng êm | Phân tích sai định dạng = không có dữ liệu. Phản hồi thật bảo đảm test không chỉ khớp với tưởng tượng của tôi |
| Bing, Google | trang HTML mẫu: bỏ qua mục JSON hỏng, bỏ trùng, bỏ ảnh thu nhỏ của chính Google, kiểm tra bộ lọc hợp lệ, thiếu Selenium chỉ cảnh báo | Hai nguồn này phân tích HTML nên dễ vỡ nhất; test chỉ rõ chỗ cần sửa khi trang đổi |
| Link và yt-dlp (giả) | playlist → từng video; link ảnh không cần hỏi yt-dlp; **link video trực tiếp (`.webm`, `.mp4`) không hỏi yt-dlp**; trang yt-dlp không đọc được (lỗi `ValueError`) chỉ là một cảnh báo | Chính là lỗi `play-av` và 429 của Wikimedia đã gặp |
| File / thư mục / `.txt` | quét đệ quy, bỏ file ẩn, bỏ thư mục kết quả của chính công cụ, ghi chú `# license=…`, `C:\…` không bị nhầm là link | Đầu vào cục bộ phải đúng trên cả Windows lẫn Unix; không quét lại kết quả của mình |
| `http_get` | chờ khi bị giới hạn tốc độ (429 / 5xx), `Retry-After` là ngày thì dùng giá trị mặc định, hết thời gian thì bỏ cuộc có cảnh báo, mỗi luồng một session | Một trang chậm / đang bận không được làm sập cả lần chạy |

### `test_models.py` (29 test) — YOLO và tải về (`models.py`, `downloads.py`)

| Nhóm test | Kiểm tra | Vì sao cần |
|---|---|---|
| Detector (dùng "Ultralytics giả") | lọc lớp ngay trong model; tên lớp lạ báo rõ model có những lớp nào; **FP16 chỉ trên NVIDIA + `.pt`**; detection chưa xác nhận có `track_id = -1` | FP16 trên CPU / Apple làm sai hoặc lỗi; track chưa xác nhận từng làm mất ảnh |
| Chia ô (`tile_boxes`, `merge_boxes`) | các ô phủ kín ảnh, không ô nào vượt cỡ; chồng nhau thì giữ khung tin cậy nhất | Chim bé xíu ở xa chỉ thấy khi chia ô; chia thiếu là bỏ sót, không gộp là trùng khung |
| Tải ảnh (web server cục bộ) | tải một lần rồi dùng cache; trang web / file giả không phải ảnh bị từ chối; giới hạn dung lượng; tải song song giữ thứ tự | Không tải hai lần, không lưu rác |
| Tải video (yt-dlp giả) | giới hạn độ phân giải / số phút / bỏ âm thanh; thử lại bằng HLS khi lần đầu lỗi; cookie trình duyệt | YouTube hay trả 403 cho đường tải trực tiếp |
| **Video trực tiếp và lỗi** | file video tải bằng HTTP thường (không cần yt-dlp), file `.part` không lọt vào cache; Wikimedia dùng User-Agent rõ ràng và tự thử lại khi 429; vẫn bị giới hạn thì một cảnh báo; file quá lớn bị bỏ; **lỗi bất ngờ của yt-dlp hay bất kỳ ngoại lệ nào chỉ là một cảnh báo** | Tái hiện đúng sự cố người dùng gặp (429 và `opening play-av tag not found`) |
| Vị trí cache | `../downloads/videos`, `../downloads/images` | Yêu cầu cache nằm ngoài dự án |

### `test_processors.py` (22 test) — biến video / ảnh thành mẫu (`processors.py`)

| Nhóm test | Kiểm tra | Vì sao cần |
|---|---|---|
| `FrameSampler` | đúng khung mỗi `every` giây; lấy mẫu thích ứng chỉ giãn **sau 2 giây không thấy gì** và quay lại tốc độ cũ khi thấy lại; đếm khung bỏ qua; video hỏng báo lỗi | Giãn quá sớm làm mất ảnh (lỗi đã gặp ở bản đầu); số đếm để người dùng thấy công cụ tiết kiệm được gì |
| `cut_object` | luật chất lượng: độ tin cậy, chiều cao, ảnh vỡ, mờ; khung có lề | Ảnh xấu vào dataset làm hỏng nhãn |
| **Vòng lặp theo dõi giống hệt worker-dataset** | cho cùng chuỗi phát hiện ngẫu nhiên (5 kịch bản), phiên bản mới (chỉ xét track đang xuất hiện) cho **đúng cùng kết quả** với vòng lặp gốc (xét mọi track từng thấy) | Tôi đã tối ưu cách quản lý track; phải chứng minh tối ưu không đổi kết quả |
| `ObjectCropProcessor` | giữ ảnh nét nhất mỗi `track_gap` giây, tối đa `max_per_track`; dừng sớm khi liên tiếp bị loại; không cắt detection chưa xác nhận nhưng vẫn giữ tốc độ lấy mẫu; ảnh tĩnh xử lý theo lô | Chính là lõi lấy từ worker-dataset |
| `FrameProcessor` | bỏ khung tĩnh **trước khi** gọi detector; cổng `min_count`; giữ một phần khung trống làm ảnh nền; khung giữ nguyên kích thước; dừng sớm; **không tự gán nhãn, không tạo thư mục `labels/`** | Camera trời cố định phần lớn là khung giống nhau; yêu cầu "chỉ thu thập ảnh, không gán nhãn" |

### `test_storage.py` (23 test) — lưu trữ, bộ lọc (`models.py` `VectorIndex`, `filters.py`, `store.py`)

| Nhóm test | Kiểm tra | Vì sao cần |
|---|---|---|
| `VectorIndex` | tìm ảnh giống nhất, k láng giềng; tự nới dung lượng; **chỉ ghi thêm phần mới** (kích thước file tăng đúng bằng số dòng mới); cắt về dòng cũ khi bị dừng; bỏ dòng ghi dở; đổi `.npy` cũ sang định dạng mới | Bản gốc ghi lại cả file sau mỗi nguồn (chậm dần theo bình phương); và file không được hỏng khi bị `kill` |
| Bộ lọc | thứ tự chuỗi lọc, **dừng ở bộ lọc đầu loại** (bộ lọc đắt không bị gọi khi không cần); ngưỡng trùng; bỏ phiếu good / noise; **không có ví dụ thì không nạp model** | Bộ lọc rẻ chạy trước để tiết kiệm; RAG không được tốn gì khi chưa ai dạy |
| `DatasetStore` | lưu ảnh + dòng `crops.csv` + vector + `processed.txt` (chỉ những thứ đó, không có nhãn); **nguồn dừng giữa chừng bị dọn sạch ở lần sau**; nâng cấp thư mục cũ của worker-dataset; `remove` giữ csv và vector khớp dòng; điền vector còn thiếu | Cho phép Ctrl+C / mất SSH mà không cần dọn tay; tương thích với dữ liệu cũ |
| `RejectedStore`, `dataset_report` | chỉ lưu ảnh với lý do đáng xem lại (noise), không lưu bản trùng; báo cáo `stats` đúng | Không tốn đĩa cho hàng nghìn bản trùng |

### `test_cli.py` (36 test) — điều phối, dòng lệnh, kiểm tra máy (`pipeline.py`, `cli.py`, `setup_check.py`)

| Nhóm test | Kiểm tra | Vì sao cần |
|---|---|---|
| Tải trước (`prefetch`) | giữ thứ tự, chỉ đi trước tối đa hai mục, lỗi được ném lại ở luồng chính, luồng là daemon | Luồng phụ không được giữ chương trình khi bấm Ctrl+C và không được nuốt lỗi |
| Sàng lọc trước khi tải | bỏ mục đã làm, đã gặp, tiêu đề có từ cấm, video quá dài (trừ khi chỉ tải `max_minutes` phút đầu) | Việc rẻ nhất là việc không phải làm |
| `submit` | mã hoá **cả lô một lần**, không tính vector nếu không bộ lọc nào cần | Mỗi ảnh một lần gọi model thì chậm |
| `collect`, `make_sources` | đầu vào trước, truy vấn sau; một nguồn hỏng không dừng nguồn khác; tuỳ chọn sai báo ngay từ đầu | Lỗi cấu hình phải lộ ra trước khi mất công làm |
| `clean` | dùng giá trị và vector đã lưu, **không đọc lại ảnh, không chạy model** | `clean` phải nhanh và rẻ để chạy lặp |
| Dòng lệnh, lưu link | `plan_work` (có `-i` thì chỉ dùng `-i`, không thì dùng `search` + `links` của loại), mở rộng `*` (PowerShell không tự mở), mã thoát 0 / 1 / 2 / 3 / 130, ghi file log; tên loại gõ sai được gợi ý, thiếu `datasets.yaml` báo rõ cách sửa; `save_links` chỉ **thêm** link mới vào cuối file, không ghi đè dòng bạn đã sửa / xoá tay, không lặp tiêu đề | Script và cron dựa vào mã thoát; người dùng cần biết sửa gì |
| **Không màn hình** | lệnh nhẹ không nạp torch (khởi động nhanh), không bao giờ đọc stdin | Chạy qua SSH / cron không có người gõ |
| Chế độ `--debug` và báo cáo lỗi | log vào `../dataset/logs/`; lỗi bất ngờ luôn ghi `crash-*.log` một dòng thông báo; ghi báo cáo lỗi thất bại cũng không làm chương trình sập thêm | Yêu cầu "khi chạy hỏng vẫn có log để xem" |
| Một mục hỏng không dừng cả lần chạy | video xử lý lỗi → bỏ qua, video kế tiếp vẫn chạy, nguồn lỗi không được đánh dấu xong; lô ảnh lỗi → bỏ qua | Chính là yêu cầu "chạy mượt, không crash" |
| `setup_check` | báo ✓ / ✗ kèm cách sửa; chỉ lỗi bắt buộc làm lệnh thoát khác 0; kiểm tra mạng là tuỳ chọn | Đồng nghiệp chạy một lệnh là biết máy mình thiếu gì |

### `test_integration.py` (12 test) — cả chương trình, trong một tiến trình (model giả, web server cục bộ)

| Test | Chứng minh | Vì sao cần |
|---|---|---|
| `run_collects_from_links_lists_and_folders_then_resumes` | link video + danh sách `.txt` + thư mục cục bộ đều thu thập được, license đúng; chạy lại không tải lại, không làm lại; **thư mục dự án không có thêm gì** | Đường đi phổ biến nhất của người dùng |
| `interrupted_run_resumes_to_the_same_result` | bị ngắt giữa chừng rồi chạy lại ra **đúng cùng** tập ảnh như chạy một mạch | Đảm bảo "an toàn khi Ctrl+C" |
| `search_writes_a_link_list…`, `search_saves_to_links_folder_of_the_type_by_default`, `run_without_inputs_uses_the_searches…` | `search TYPE -o` rồi `run TYPE -i`; `search TYPE` không `-o` lưu vào `links/<TYPE>/found.txt` (thư mục dự án gốc không có file lạ), chạy lại chỉ thêm link mới; `run TYPE` không `-i` dùng câu tìm kiếm + nguồn trong `datasets.yaml` | Hai cách thu thập chính của người dùng |
| `review_then_clean…` | đánh dấu ảnh nhiễu → `clean` đưa ảnh giống nó vào `rejected/noise/` | Chức năng lọc nhiễu RAG từ đầu đến cuối |
| `every_type_in_the_project_datasets_yaml_runs` (mỗi loại một test) | `birds`, `food_factory_workers`, `vehicle` trong **chính file `datasets.yaml` của dự án** đều chạy được; loại mới thêm vào file tự có test | Loại có sẵn hỏng thì người dùng gặp ngay |
| `options_move_results_and_downloads_anywhere` | `--data-dir` / `--downloads-dir` đưa kết quả đi nơi khác; mặc định không đụng thư mục dự án | Yêu cầu vị trí đầu ra |
| `bad_links_never_stop_the_run…` | trong danh sách có link chết, trang web giả ảnh, file `.mp4` hỏng → vẫn thu được ảnh tốt, mỗi lỗi một dòng `!`, không traceback, có dòng `warnings: N` | Tái hiện đúng tình huống người dùng báo |
| `debug_run_records_the_whole_run…` | `--debug` ghi từng yêu cầu mạng và mã trả về vào file log | Yêu cầu chế độ debug |

### `test_system.py` (5 system + 10 network) — như người dùng thật (`-m system`) và Internet thật (`-m network`)

| Test | Chứng minh | Vì sao cần |
|---|---|---|
| `people_end_to_end…` (system) | lệnh thật + **model người thật + DINOv2 thật**, loại `food_factory_workers`: chạy → `kill` giữa chừng → chạy lại (giống hệt chạy sạch) → `stats` → không có thư mục `labels/` → `review` + `clean` | Đồ giả có thể che lỗi ghép nối với model thật |
| `vehicle_type_crops_the_bus` (system) | model xe ONNX nhận ra xe buýt, `det_class = bus` | Kiểm tra đường ONNX (cần gói `onnx` — thiếu gói này đã từng gây lỗi) |
| `frame_type_skips_static_frames…` (system) | video trời tĩnh: bỏ khung không đổi nhưng vẫn giữ khung có chim, ảnh giữ nguyên 1280×720 | Cách xử lý `frame` của loại `birds` |
| `every_command_runs_headless` (system) | mọi lệnh chạy với stdin đóng, không `DISPLAY`, log không có ký tự thanh tiến độ, mã thoát đúng | Chạy qua SSH |
| `installed_wheel_runs_with_a_datasets_yaml` (system) | build gói, cài, chạy `types` từ gói với `datasets.yaml` của dự án; có lệnh `mbfs-crawler` | Đồng nghiệp có thể cài từ gói, không từ thư mục nguồn; thiếu file trong gói là lỗi chỉ lộ khi cài |
| `live_sources_return_results` (network) | YouTube, Wikimedia, Internet Archive, Openverse trả kết quả thật | Phát hiện sớm khi trang đổi |
| `optional_live_sources_never_crash` (network) | Bilibili, Niconico, Bing, Google: có thể rỗng nhưng không được văng lỗi | Nguồn tuỳ chọn không được làm hỏng lần chạy |
| `live_video_download_takes_only_the_first_seconds` (network) | `max_minutes` thật sự chỉ tải phần đầu | Tiết kiệm băng thông |
| `birds_type_keeps_real_bird_photos…` (network) | `run birds` với Wikimedia: `yolov8n` tìm thấy chim trong ảnh thật, không có `labels/` | Đường tự tải `yolov8n.pt` và cổng chim |

### `benchmark.py` — không phải test, là script đo
So mbfs-crawler tắt tăng tốc và mbfs-crawler mặc định trên cùng một video. **Vì sao script riêng?** Thời gian chạy
phụ thuộc máy nên không đưa vào test đỏ / xanh; nhưng cần chạy lại trước mỗi lần phát hành để biết tối ưu còn hiệu quả.

---

## 5. Lỗi thật đã được test phát hiện (và nay có test canh giữ)

Danh sách này là bằng chứng cho việc kế hoạch kiểm thử có tác dụng, và là lý do nguyên tắc 3 (mục 1) tồn tại.

| Lỗi | Phát hiện bằng | Hậu quả nếu không có test | Test canh giữ |
|---|---|---|---|
| Lấy mẫu thích ứng giãn ngay khi khung đầu không thấy ai | benchmark | còn 64 / 80 ảnh, mất ảnh âm thầm | `test_adaptive_sampling_backs_off_only_after_a_quiet_period_and_resets` |
| Kiểm tra chuyển động dùng độ chênh trung bình nên bỏ sót chim nhỏ | test với chim 12×10 px | dataset chim thiếu đúng loại khó nhất | `test_changed_fraction_notices_a_small_moving_object_but_not_noise` |
| `rejected.csv` mất tiêu đề sau khi bị `kill`, làm `stats` lỗi | test system (kill rồi chạy lại) | lệnh `stats` sập sau khi tiếp tục | `test_rejected_store_keeps_images_only_for_reviewable_reasons` + `test_people_end_to_end…` |
| Detection chưa được tracker xác nhận bị bỏ, làm lấy mẫu thích ứng tưởng "không có ai" | benchmark | giãn khung sai, mất ảnh | `test_unconfirmed_detections_are_not_cropped_but_keep_full_rate` |
| Thiếu gói `onnx`: Ultralytics tự `pip install` rồi thất bại | người dùng chạy `check-setup` | cảnh báo đỏ, có thể hỏng ở máy khác | kiểm tra gói trong `setup_check` + `test_vehicle_type_crops_the_bus` |
| `pip install -e .` tạo thư mục `mbfs_crawler.egg-info` trong dự án | người dùng hỏi | dự án lẫn file không phải mã nguồn | đổi công cụ build sang hatchling (ghi thông tin cài đặt vào `.venv/`); `test_installed_wheel_runs_with_a_datasets_yaml` build gói từ bản sao |
| yt-dlp ném `ValueError` (`opening play-av tag not found`) với trang Archive.org không có trình phát | người dùng chạy `run` | **cả lần chạy dừng**, lần sau lại dừng ở đúng link đó | `test_an_unexpected_ytdlp_error_is_one_warning_and_the_run_goes_on`, `test_a_page_yt_dlp_cannot_parse_is_one_warning_not_a_crash` |
| Wikimedia trả 429 vì yt-dlp dò từng link `.webm` | người dùng chạy `run` | 5 video mất, chỉ còn cảnh báo khó hiểu | `test_direct_video_files_download_without_ytdlp`, `test_wikimedia_files_are_requested_politely_and_retried_on_429` |
| `Retry-After` là một ngày giờ chứ không phải số | rà soát code `http_get` | `float()` văng lỗi giữa lúc thử lại | `test_retry_after_accepts_numbers_and_ignores_dates` |
| Thông báo lỗi dài, lặp lại link, chen vào thanh tiến độ | đọc đầu ra thật | terminal khó đọc | `test_short_error_gives_one_readable_line`, `test_ytdlp_messages_go_to_the_debug_log_not_the_terminal` |
| Ffmpeg in thẳng ra terminal khi dùng `--max-minutes` | chạy thật không `--debug` | terminal rối | `test_video_options_limit_resolution_audio_and_minutes` (đòi `-loglevel error`) |

---

## 6. Kết quả hiện tại (28/10/2026, macOS, Apple M-series)

| Tầng | Kết quả |
|---|---|
| Lint | sạch |
| Unit + integration | **249 passed** (≈ 10 s), độ phủ **94%** |
| System | **5 passed** (≈ 16 s) |
| Network | **8 passed**, 2 skipped: Niconico và Google Images không trả kết quả từ mạng này (nguồn tuỳ chọn) |
| Chạy thật, 7 nguồn (5 video Wikimedia, Internet Archive, YouTube) + link chết | exit 0, 63 ảnh, không 429, mỗi lỗi một dòng `!`, kết thúc có `warnings: N` |

**Vì sao độ phủ là 94% chứ không phải 100%?** 143 dòng chưa được test thường chạy tới. Chúng thuộc các loại mà dựng bản giả chỉ là
kiểm tra cái giả, nên không đáng viết: nạp DINOv2 thật (~90 MB; được test system chạy nhưng không tính vào độ phủ của `pytest`
thường), mã điều khiển Chrome cho Google Images, các kiểm tra phụ thuộc máy trong `setup_check` (GPU, mạng thật), nhánh lỗi hiếm
(ổ đĩa đầy, tuỳ chọn sai hiếm gặp), và vài dòng chỉ chạy trên Windows (đường dẫn ngắn 8.3). Mức tối thiểu **85%** nhằm chặn
code mới không có test, chứ không đuổi theo 100%. Phần lõi (cắt ảnh, lọc, lưu, đọc `datasets.yaml`) phủ 98–99%.

**Benchmark:** video nhà máy 8 phút, loại `food_factory_workers` (model `.onnx` như trong `datasets.yaml`),
Mac Apple M-series. Lệnh: `python tests/benchmark.py <video>`.

| Chạy | Thời gian | Lần gọi detector | Ảnh lưu |
|---|---|---|---|
| tắt tăng tốc | 65.7 s | 2470 | 82 |
| mặc định (`every_max 1.0`) | 52.2 s | 1826 (−26%) | 80 (97.5%) |

Mặc định gọi detector ít hơn 26% và nhanh hơn 20%, đổi lại mất 2 / 82 ảnh. Lợi ích lớn hơn của các tối ưu khác (vector chỉ ghi thêm, quản lý track chỉ trên người đang xuất hiện, cache tải về
dùng chung) chỉ hiện ra khi dataset lớn và chạy lâu, nên không thấy rõ trong video 8 phút.

---

## 7. Vấn đề đã biết

| Vấn đề | Ảnh hưởng | Cách xử lý | Vì sao không sửa hẳn |
|---|---|---|---|
| Bing trả ảnh không liên quan cho yêu cầu tự động | phí lượt tải | không bật trong `datasets.yaml`; muốn dùng thì thêm `bing_images` vào `sources:` sau khi xem `search` | là hành vi phía Bing; không nên lách chặn bot |
| Openverse đôi khi rất chậm | nguồn bị bỏ qua lần đó | tự thử lại một lần rồi bỏ qua, có cảnh báo | phía dịch vụ; test network cho phép skip |
| Google Images cần Chrome, trang hay thay đổi | có thể trả về rỗng | nguồn tuỳ chọn (`.[google]`) | đổi trang là chuyện thường; đã cô lập trong một lớp |
| `yolov8n` yếu với chim rất nhỏ ở xa | bỏ sót một số khung có chim | train model chim riêng rồi đổi `detector.model` | giới hạn của model COCO, không phải của công cụ |
| Tăng tốc thích ứng mất một ít ảnh | ≈ 2–3% ảnh ít hơn (benchmark) | trong `datasets.yaml` đặt `every_max` bằng `every` để tắt | đánh đổi chủ ý lấy tốc độ; người dùng chọn được |
| Chưa chạy trên máy Windows thật | lỗi riêng Windows chưa lộ | checklist yêu cầu thử thủ công | cần máy thật; repo hiện không có CI |
| Model người `.onnx` chạy bằng CPU trên Mac | chậm hơn chạy trên GPU (ONNX Runtime trên Mac không dùng được MPS) | dùng GPU NVIDIA, hoặc chép `.pt` vào `models/` và đổi `detector.model` trong `datasets.yaml` | model `.pt` đã được gỡ chủ ý |

---

## 8. Checklist phát hành (trước khi gửi đồng nghiệp)

- [ ] `ruff check`, `pytest`, `pytest -m system` và `pytest -m network -rs` đạt trên máy macOS / Linux và một máy Windows (skip có lý do là chấp nhận được).
      *Vì sao:* mỗi tầng bắt một loại lỗi khác nhau (mục 2).
- [ ] `python tests/benchmark.py <video>`, ghi số liệu vào mục 6.
      *Vì sao:* tối ưu có thể hỏng âm thầm khi sửa code; số liệu mới là bằng chứng.
- [ ] Một người chưa từng dùng làm theo README.md **từng lệnh một** trên máy Windows sạch và máy macOS sạch, từ `check-setup` đến `stats`.
      *Vì sao:* người viết quá quen nên bỏ qua bước thiếu; chỉ người lạ mới thấy README thiếu gì.
- [ ] Đặt version trong `pyproject.toml` (bắt đầu `0.1.0`), tạo git tag.
      *Vì sao:* khi đồng nghiệp báo lỗi phải biết họ đang dùng bản nào (`check-setup` và log đều ghi phiên bản).

---

## 9. Thêm test cho phần mới

Quy tắc chung: **không có test thì chưa xong**. Chọn test theo câu hỏi "phần mới này có thể hỏng theo cách nào, và tầng nào nhìn thấy?".

| Thêm | Test tối thiểu | Vì sao |
|---|---|---|
| Nguồn mới | lưu một phản hồi thật vào `tests/fixtures/`, test parser trong `test_sources.py` (giới hạn, trang tiếp, license, phản hồi rỗng / hỏng); thêm một dòng vào test network | phản hồi thật ghim định dạng; test network báo khi trang đổi |
| Bộ lọc mới | `test_storage.py`: giữ / loại / thứ tự trong chuỗi; dùng `FakeEncoder` | thứ tự quyết định bộ lọc đắt có bị gọi thừa không |
| Processor mới | `test_processors.py` với video tổng hợp và `FakeDetector` | kết quả xác định, không cần model thật |
| Loại dataset mới | thêm vào `datasets.yaml` là tự được kiểm bởi `test_the_shipped_datasets_yaml…` (hợp lệ) và `test_every_type_in_the_project_datasets_yaml_runs` (chạy được) | loại hỏng lộ ngay khi chạy test |
| Lệnh mới | `test_cli.py` (mã thoát) + `test_integration.py` (chạy thật) + README | mã thoát là hợp đồng với script; README là hợp đồng với người dùng |
| Sửa một lỗi | thêm test tái hiện lỗi **trước**, rồi sửa, rồi thêm một dòng vào mục 5 | để lỗi không quay lại và người sau hiểu vì sao có test đó |
