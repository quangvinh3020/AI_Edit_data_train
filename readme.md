# YOLO + SAM Semi‑Automatic Labeling Pipeline — Gán nhãn bán tự động cho ảnh (VN/EN)

## 🇻🇳 Tổng quan | 🇬🇧 Overview
Công cụ gán nhãn bán tự động kết hợp **Ultralytics YOLOv8** (phát hiện đối tượng) và **Segment Anything (SAM)** (tinh chỉnh mask/bounding box) để tạo **dataset chuẩn YOLO** trực tiếp từ thư mục ảnh. Tối ưu cho bài toán biển số xe (heuristic ưu tiên box có tỷ lệ w/h lớn & diện tích nhỏ), nhưng vẫn linh hoạt cho nhiều lớp khác.  
A semi‑automatic labeling tool that pairs **Ultralytics YOLOv8** (object detection) with **Segment Anything (SAM)** (mask/bbox refinement) to export a **YOLO‑format dataset** straight from your images folder. Optimized for license plate scenarios with heuristics, but generic enough for other classes.

---

## ✨ Tính năng chính | Key Features
- YOLO phát hiện → SAM tinh chỉnh box từ mask (box‑prompt).  
  YOLO detection → SAM refines boxes from masks (box prompt).
- Heuristic ưu tiên “biển số”: **w/h > 2.5** và **area < 2% ảnh** được gợi ý/annotate trước.  
  License‑plate heuristic: prioritize boxes with high aspect ratio & small area.
- Hỗ trợ **nhập nhãn thủ công** (rename/override) theo từng box hoặc **vẽ ROI** để thêm biển số.  
  Manual labeling per box or add plates via ROI drawing.
- Xuất **dataset YOLO chuẩn**: `dataset/images/{train,val}` & `dataset/labels/{train,val}` + `classes.txt`.  
  Exports YOLO dataset structure with consistent `classes.txt` mapping.
- (Tuỳ chọn) Lưu **preview** ảnh đã vẽ box để review.  
  Optional preview images for visual QA.

---

## 📁 Cấu trúc thư mục | Project Structure
```
.
├─ images/                # ảnh đầu vào (input images)
├─ dataset/
│  ├─ images/
│  │  ├─ train/
│  │  └─ val/
│  └─ labels/
│     ├─ train/
│     └─ val/
├─ labeled_images/        # preview có vẽ box (optional)
├─ labels/                # legacy labels (optional)
├─ classes.txt            # mapping class -> id
└─ script.py              # (file mã nguồn này)
```

---

## 🧩 Yêu cầu hệ thống | Requirements
- Python 3.9–3.11
- GPU NVIDIA + CUDA (khuyến nghị) hoặc CPU (chậm hơn)  
  NVIDIA GPU with CUDA recommended; CPU supported but slower
- Thư viện chính | Core libs:
  - torch (PyTorch)
  - ultralytics (YOLOv8)
  - opencv-python
  - numpy
  - segment-anything (SAM, predictor)

---

## 🛠️ Cài đặt môi trường | Environment Setup

### Cách 1 — Conda (khuyến nghị) | Recommended
```bash
# Tạo môi trường
conda create -n yolo-sam python=3.10 -y
conda activate yolo-sam

# Cài PyTorch (chọn lệnh đúng với phiên bản CUDA của bạn)
# Tham khảo pytorch.org để lấy lệnh phù hợp CUDA 11.x/12.x hoặc CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Thư viện còn lại
pip install ultralytics opencv-python numpy
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### Cách 2 — Virtualenv (pip)
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip

# PyTorch theo CUDA/CPU của bạn
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Thư viện còn lại
pip install ultralytics opencv-python numpy
pip install git+https://github.com/facebookresearch/segment-anything.git
```

> 📥 **Weights cần chuẩn bị | Required weights**
> - Tải `yolov8l.pt` (Ultralytics) và đặt cạnh script hoặc cung cấp đường dẫn trong biến `YOLO_WEIGHTS`.
> - Tải checkpoint SAM `sam_vit_h_4b8939.pth` (SAM ViT‑H) và cập nhật `SAM_CHECKPOINT`.
> - Nếu dùng model YOLO/SAM khác, chỉnh `YOLO_WEIGHTS`, `SAM_ARCH`, `SAM_CHECKPOINT` tương ứng.

---

## ⚙️ Cấu hình nhanh | Quick Config
Trong mã nguồn có sẵn các tham số chính có thể chỉnh:
```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

YOLO_WEIGHTS = "yolov8l.pt"
YOLO_IMGSZ   = 1000
YOLO_CONF    = 0.05
YOLO_IOU     = 0.4
YOLO_MAXDET  = 1000

SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
SAM_ARCH = "vit_h"

SUS_RATIO_THRESH = 2.5
SUS_AREA_FRAC    = 0.02

IMAGES_DIR   = "images"
PREVIEW_DIR  = "labeled_images"
LABELS_DIR   = "labels"
CLASSES_FILE = "classes.txt"

DATASET_DIR  = "dataset"
VAL_RATIO    = 0.2
```

---

## ▶️ Cách chạy | How to Run
1. **Chuẩn bị ảnh** vào thư mục `images/` (`.jpg/.jpeg/.png`).  
   Put your input images into `images/`.
2. Đảm bảo **weights YOLO & SAM** có sẵn và biến đường dẫn đúng.  
   Ensure YOLO & SAM weights are present and paths are correct.
3. Chạy script:  
   ```bash
   python script.py
   ```
4. Quy trình annotate:
   - Script tự chạy YOLO → hiển thị crop từng box để bạn **nhập nhãn**:
     - Enter để chấp nhận nhãn gợi ý (nếu có), nhập tên khác để override, `q` để bỏ qua box.
   - Có thể **vẽ ROI** để thêm biển số thủ công (nhiều lần): nhập `r` khi được hỏi.
   - Khi xong, dữ liệu sẽ được lưu vào `dataset/…` theo split `train/val` (tự chia theo `VAL_RATIO`).  
   The tool will iterate over detections, ask for labels per crop, allow manual plate ROIs, and finally save YOLO dataset into `dataset/` with train/val splits.

5. Sau khi chạy, script tự cập nhật `classes.txt` và in gợi ý `dataset.yaml`:
   ```yaml
   path: /absolute/path/to/dataset
   train: images/train
   val: images/val
   names:
     0: car
     1: license_plate
   ```

---

## ⌨️ Phím tắt & Tương tác | Shortcuts & Interaction
- **CROP window**: nhập nhãn → Enter (nhận) | `q` (bỏ qua).  
- **ROI (biển số)**: chọn vùng → Enter (nhận) | `c` (hủy).  
- **PREVIEW/REVIEW**: nhấn Enter ở terminal để xác nhận lưu.  
> Lưu ý: Cần môi trường có GUI (OpenCV HighGUI). Server/headless cần cấu hình X11/Virtual Display.

---

## 🧪 Kiểm thử nhanh | Quick Test
- Dùng một ảnh đơn giản trong `images/`, chạy script, gán nhãn 1–2 box, kiểm tra file:
  - `dataset/images/train/*.jpg`
  - `dataset/labels/train/*.txt`
  - `classes.txt`
  - `labeled_images/*.jpg` (preview, nếu bật)

---

## 🩺 Troubleshooting
- **Không thấy cửa sổ ảnh**: Kiểm tra môi trường GUI/X11. WSL/remote cần `export DISPLAY=:0` hoặc xserver.  
- **CUDA OOM / SAM lỗi**: Script tự fallback bbox YOLO nếu SAM lỗi. Giảm `YOLO_IMGSZ` hoặc dùng model nhỏ (`yolov8n.pt`).  
- **Hiệu năng thấp**: Dùng GPU, giảm kích thước ảnh hoặc số lượng ảnh/lượt.  
- **Mapping nhãn sai**: Xoá `classes.txt` để reset mapping (script tạo lại theo thứ tự xuất hiện).

---

## 📜 Bản quyền & Ghi công | License & Credits
- Dựa trên **Ultralytics YOLOv8** và **Segment Anything (Meta)** — tuân thủ giấy phép tương ứng.  
  Built on **Ultralytics YOLOv8** and **Meta’s Segment Anything**; respect their licenses.
- Tác giả/Author: **Trần Quang Vinh**.

---

## 🙌 Góp ý & Mở rộng | Contributions & Extensions
- Hỗ trợ nhiều class hơn: thêm logic gợi ý tên theo `results.names`.  
- Lưu mask (COCO) bên cạnh bbox nếu muốn đa định dạng.  
- Thêm tham số CLI (argparse) để chạy không cần sửa code.

—  
**Made by / Người làm:** Trần Quang Vinh
