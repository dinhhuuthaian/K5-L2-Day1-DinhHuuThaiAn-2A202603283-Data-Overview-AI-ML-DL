# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy: 11/6/2026**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: 
  {
    "sample_id": "traffic",
    "coco_image_id": 210273,
    "image_width": 640,
    "image_height": 428,
    "task": "image_classification",
    "taxonomy_name": "ImageNet-1K",
    "model_file": "yolo11n-cls.pt",
    "model_sha256": "c62d41bf9625777760018bf914d2e6cd472420ccd01706d97a61cb6c82502bd7",
    "ultralytics_version": "8.4.145",
    "rank": 1,
    "class_id": 468,
    "class_name": "cab",
    "score": 0.510915
  }
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
 + Record này mô tả toàn bộ bức ảnh chứ không phải một vật thể riêng lẻ. 
 Dựa vào việc trường task có giá trị image_classification, đồng thời output chỉ chứa class_id, class_name, rank, score mà không có thông tin vị trí (bounding box). 
 Ngoài ra notebook cũng nêu rõ mô hình classification gán một nhãn cho toàn bộ ảnh thay vì cho từng đối tượng.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
 + Class list đến từ taxonomy ImageNet-1K gắn với checkpoint yolo11n-cls.pt.
 Model chỉ chọn một nhãn trong tập lớp đã định nghĩa sẵn, không tự tạo tên lớp mới.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
 + class_name để đọc, class_id để định danh chính xác, taxonomy_name để biết nhãn thuộc bộ taxonomy nào.
 Ba trường này giúp tránh mơ hồ và hỗ trợ truy vết kết quả mô hình.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
 + Guideline cần định nghĩa thế nào là "đối tượng/chủ đề chính" và quy tắc ưu tiên khi có nhiều đối tượng xuất hiện trong cùng ảnh.
 Điều này giúp các annotator gán cùng một nhãn cho các trường hợp tương tự.
6. Vì sao không được lấy `score` của model làm ground truth?
 + score là confidence của model, còn ground truth là sự thật do con người hoặc dữ liệu chuẩn xác nhận. Score cao không đồng nghĩa với dự đoán đúng.
## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=… ; `score`=… ; `bbox_xyxy`=[…, …, …, …] ; `bbox_width`=… ; `bbox_height`=…
   {
    "class_name": "person",
    "score": 0.912625,
    "coordinate_unit": "pixel",
    "bbox_format": "xyxy",
    "bbox_xyxy": [
      385.33,
      69.24,
      498.92,
      348.92
    ],
    "bbox_width": 113.58,
    "bbox_height": 279.68
  },
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
 Góc trên-trái: (385.33, 69.24).
 Góc dưới-phải: (498.92, 348.92).
 Box rộng 113.58 pixel và cao 279.68 pixel.
 Box bao quanh một người đứng ở khu vực bên phải của ảnh.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
 Threshold 0.20: 17 prediction.
 Threshold 0.60: 6 prediction.
 Khi tăng threshold, số prediction giảm vì các box có score thấp không được giữ lại.
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
 + Khi hạ ngưỡng, số lượng prediction tăng lên nên reviewer phải kiểm tra thêm nhiều bounding box có score thấp, loại bỏ các box phát hiện nhầm (false positive), sửa nhãn nếu cần và xác nhận các box còn lại là hợp lệ.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
 + Bounding box phải ôm sát đối tượng nhất có thể, bao phủ toàn bộ phần nhìn thấy của đối tượng và hạn chế chứa nền không cần thiết.
 Nếu đối tượng bị che khuất một phần, vẫn gán nhãn và vẽ box theo phần nhìn thấy được của đối tượng.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
Trong ảnh kitchen, đối tượng cup trên bàn bị che khuất một phần bởi các vật thể khác.
 + Tôi sẽ đưa tình huống này vào guideline thay vì escalation vì đây là trường hợp phổ biến.
 Quy tắc có thể là: "Đối tượng bị che khuất một phần nhưng vẫn nhận diện được thì vẫn gán nhãn và vẽ box theo phần nhìn thấy.
 " Escalation chỉ cần thiết khi không thể xác định chắc chắn đối tượng.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=… ; `class_name`=… ; `score`=… ; số điểm polygon=`polygon_point_count`=… ; 3–5 điểm đầu của `polygon_xy`=…3
{
	"instance_id": "kitchen-001",
	"class_name": "person",
	"score": 0.899318,
	"polygon_point_count": 348,
	"polygon_xy": [
      	[
        	446.0,
        	70.0
      	],
      	[
        	445.0,
        	71.0
      	],
      	[
        	444.0,
        	71.0
      	],
      	[
        	443.0,
        	72.0
      	],
   ]
}
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
 + Bounding box chỉ cho biết vị trí và kích thước của đối tượng thông qua một hình chữ nhật.
 Polygon cho biết đường viền thực tế của đối tượng bằng nhiều điểm biên, nên mô tả hình dạng chính xác hơn và giảm phần nền không thuộc đối tượng.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
 + instance_id dùng để nhận diện một đối tượng cụ thể trong dữ liệu annotation.
 Nó khác class_id vì class_id chỉ cho biết loại đối tượng (person, bowl, oven...), và khác tracking ID vì không theo dõi đối tượng qua nhiều frame video.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
 + Mask phải bám sát biên thật của đối tượng, chỉ bao gồm các pixel thuộc đối tượng và không lấy nền xung quanh.
 Nếu đối tượng bị che khuất một phần thì chỉ vẽ phần nhìn thấy được.
 Tất cả annotator phải áp dụng cùng nguyên tắc để đảm bảo tính nhất quán.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
 + Trong ảnh kitchen, các bowl ở góc dưới bên trái nằm sát nhau và một phần biên bị chồng lấp.
 Tôi chọn guideline thay vì escalation vì đây là trường hợp phổ biến.
 Quy tắc là: nếu vẫn xác định được ranh giới giữa các vật thể thì vẽ mask riêng cho từng instance theo phần nhìn thấy được.
 Chỉ escalation khi không thể xác định được ranh giới hoặc nhãn của đối tượng.
## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | 1 ảnh → 1 nhãn lớp (ví dụ: cab, street, kitchen...) | Ảnh có nhiều đối tượng nên khó xác định chủ đề chính của ảnh | Chọn nhãn phù hợp nhất cho toàn bộ ảnh theo guideline | Kiểm tra nhãn đã đúng chủ đề chính của ảnh chưa |
| Phát hiện vật thể | Mỗi đối tượng → 1 class + 1 bounding box | Threshold thấp tạo nhiều box hơn, có thể xuất hiện false positive hoặc box chồng lấp | Gán class và vẽ bounding box ôm sát đối tượng | Kiểm tra class đúng chưa, box có ôm sát đối tượng không, có bỏ sót hoặc trùng lặp không |
| Instance segmentation | Mỗi đối tượng → 1 class + 1 polygon/mask riêng | Vật thể bị che khuất, biên mờ hoặc nhiều vật chạm nhau (ví dụ các bowl trên bàn) | Vẽ mask bám sát biên thật của từng đối tượng | Kiểm tra mask có bám đúng biên, tách đúng từng instance và không ăn vào nền |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
 + Không chia sẻ, tải lên hoặc phát tán dữ liệu của bài lab cho người không có thẩm quyền và chỉ sử dụng dữ liệu đúng mục đích học tập được cho phép.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
 + Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng công việc và báo ngay cho Lab Coach hoặc người quản lý dự án để được hướng dẫn xử lý.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
