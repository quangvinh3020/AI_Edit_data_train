import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from pathlib import Path
import random

# ====================
# CONFIG
# ====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# YOLO model (detection)
YOLO_WEIGHTS = "yolov8l.pt"        # bạn có thể đổi sang model đã fine-tune
YOLO_IMGSZ   = 1000                # tăng nếu cần bắt vật thể nhỏ (1200-1600)
YOLO_CONF    = 0.05
YOLO_IOU     = 0.4
YOLO_MAXDET  = 1000

# SAM model
SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
SAM_ARCH = "vit_h"

# Heuristic ưu tiên "biển số"
SUS_RATIO_THRESH = 2.5    # w/h > ngưỡng
SUS_AREA_FRAC    = 0.02   # diện tích < 2% ảnh => nghi là biển số

# IO
IMAGES_DIR   = "images"           # thư mục ảnh input (nguồn)
LABELS_DIR   = "labels"           # NẾU bạn vẫn muốn lưu nhãn rời như cũ (tuỳ chọn)
PREVIEW_DIR  = "labeled_images"   # ảnh có vẽ box để review (tuỳ chọn)
CLASSES_FILE = "classes.txt"      # lưu mapping class -> id

# Dataset YOLO để train
DATASET_DIR  = "dataset"          # đích lưu dataset chuẩn YOLO
VAL_RATIO    = 0.2                # tỉ lệ val
# ====================

# ====================
# INIT MODELS
# ====================
yolo_model = YOLO(YOLO_WEIGHTS).to(DEVICE)

sam = sam_model_registry[SAM_ARCH](checkpoint=SAM_CHECKPOINT).to(DEVICE)
predictor = SamPredictor(sam)

# ====================
# LOAD / SAVE CLASSES
# ====================
label_map = {}
next_id = 0

if os.path.exists(CLASSES_FILE):
    with open(CLASSES_FILE, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f.readlines() if line.strip() != ""]
        label_map = {name: i for i, name in enumerate(classes)}
        next_id = len(classes)
    print(f"📂 Loaded {len(label_map)} classes từ {CLASSES_FILE}")

# ====================
# UTILS
# ====================
def yolo_txt_line(cls_id, xmin, ymin, xmax, ymax, img_w, img_h):
    # YOLO format (normalized)
    x_center = ((xmin + xmax) / 2) / img_w
    y_center = ((ymin + ymax) / 2) / img_h
    bw = (xmax - xmin) / img_w
    bh = (ymax - ymin) / img_h
    return f"{cls_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}"

def ensure_dirs():
    os.makedirs(LABELS_DIR, exist_ok=True)
    os.makedirs(PREVIEW_DIR, exist_ok=True)

def ensure_dataset_dirs():
    for split in ["train", "val"]:
        os.makedirs(os.path.join(DATASET_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(DATASET_DIR, "labels", split), exist_ok=True)

def add_label_line(cls_name, xmin, ymin, xmax, ymax, label_lines, w, h):
    """Đảm bảo class -> id nhất quán + thêm dòng nhãn YOLO normalized."""
    global next_id, label_map
    if cls_name not in label_map:
        label_map[cls_name] = next_id
        next_id += 1
    cls_id = label_map[cls_name]
    label_lines.append(yolo_txt_line(cls_id, xmin, ymin, xmax, ymax, w, h))
    return True

def add_plate_by_roi(img, canvas, label_lines, w, h):
    win = "ADD_LICENSE_PLATE (Enter=nhận, c=hủy)"
    roi = cv2.selectROI(win, img, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(win)

    x, y, bw, bh = map(int, roi)
    if bw == 0 or bh == 0:
        return False

    xmin, ymin, xmax, ymax = x, y, x + bw, y + bh

    # Cho phép đặt tên (Enter = 'license_plate')
    label_name = input("Nhãn cho ROI (Enter = 'license_plate'): ").strip()
    if label_name == "":
        label_name = "license_plate"

    # Thêm dòng nhãn YOLO
    global next_id, label_map
    if label_name not in label_map:
        label_map[label_name] = next_id
        next_id += 1
    cls_id = label_map[label_name]
    label_lines.append(yolo_txt_line(cls_id, xmin, ymin, xmax, ymax, w, h))

    # Vẽ và hiện lại ngay
    cv2.rectangle(canvas, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
    cv2.putText(canvas, label_name, (xmin, max(0, ymin - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow("PREVIEW", canvas)
    cv2.waitKey(1)

    return True


# ====================
# CORE
# ====================
def process_image(img_path, split_for_dataset="train", also_save_preview=True, also_save_legacy_labels=False):
    """
    - Annotate như code cũ (YOLO + SAM refine + nhập nhãn thủ công cho mỗi box)
    - Thêm bước: cho phép thêm nhiều biển số bằng ROI
    - Lưu trực tiếp thành dataset YOLO: dataset/images/{split}/, dataset/labels/{split}/
    - (tuỳ chọn) vẫn lưu preview & labels cũ nếu muốn
    """
    global next_id

    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Không đọc được ảnh: {img_path}")
        return

    h, w = img.shape[:2]

    # --- YOLO predict ---
    results = yolo_model.predict(
        img,
        conf=YOLO_CONF,
        iou=YOLO_IOU,
        imgsz=YOLO_IMGSZ,
        max_det=YOLO_MAXDET,
        verbose=False
    )[0]

    predictor.set_image(img)

    # Lấy boxes + thuộc tính
    if results.boxes is None or results.boxes.xyxy is None or len(results.boxes.xyxy) == 0:
        print("⚠️ Không có box từ YOLO.")
        # vẫn cho bạn thêm biển số thủ công nếu muốn
        label_lines = []
        while True:
            choice = input("Không có box nào. Thêm biển số thủ công? (r = ROI, Enter = bỏ): ").strip().lower()
            if choice == "r":
                added = add_plate_by_roi(img, label_lines, w, h)
                print("✅ Đã thêm license_plate." if added else "❌ Không thêm.")
            else:
                break
        if not label_lines:
            return
        # lưu dataset
        ensure_dataset_dirs()
        stem = Path(img_path).stem
        out_img = os.path.join(DATASET_DIR, "images", split_for_dataset, stem + ".jpg")
        out_lbl = os.path.join(DATASET_DIR, "labels", split_for_dataset, stem + ".txt")
        cv2.imwrite(out_img, img)
        with open(out_lbl, "w", encoding="utf-8") as f:
            f.write("\n".join(label_lines))
        print(f"✅ Saved (no-yolo) {out_img}, {out_lbl}")
        return

    boxes = results.boxes.xyxy.cpu().numpy()                       # (N, 4) xyxy
    confs = results.boxes.conf.cpu().numpy() if results.boxes.conf is not None else None
    clses = results.boxes.cls.cpu().numpy() if results.boxes.cls is not None else None
    names = getattr(results, "names", None)  # dict id->name nếu có

    # --- ƯU TIÊN BIỂN SỐ + SẮP XẾP NHỎ -> LỚN ---
    H, W = h, w
    img_area = W * H
    sus, oth = [], []

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        ratio = bw / bh
        area  = bw * bh
        if ratio > SUS_RATIO_THRESH and area < SUS_AREA_FRAC * img_area:
            sus.append(i)   # nghi là biển số
        else:
            oth.append(i)

    def sort_by_area(idxs):
        if not idxs:
            return []
        areas = ((boxes[idxs, 2] - boxes[idxs, 0]) * (boxes[idxs, 3] - boxes[idxs, 1]))
        return [idx for _, idx in sorted(zip(areas, idxs), key=lambda x: x[0])]

    sus_sorted = sort_by_area(sus)  # biển số nhỏ -> lớn
    oth_sorted = sort_by_area(oth)  # các box khác nhỏ -> lớn
    order = sus_sorted + oth_sorted

    boxes = boxes[order]
    if confs is not None: confs = confs[order]
    if clses is not None: clses = clses[order]

    # --- Vòng annotate ---
    label_lines = []
    canvas = img.copy()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in box]

        # SAM bằng box prompt (ổn định hơn point prompt)
        box_prompt = np.array([x1, y1, x2, y2])[None, :]
        try:
            masks, scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box_prompt,
                multimask_output=True
            )
            if masks is None or len(masks) == 0:
                # fallback: dùng bbox gốc
                xmin, ymin, xmax, ymax = x1, y1, x2, y2
            else:
                mask = masks[np.argmax(scores)]
                ys, xs = np.where(mask)
                if len(xs) == 0:
                    xmin, ymin, xmax, ymax = x1, y1, x2, y2
                else:
                    xmin, xmax = int(xs.min()), int(xs.max())
                    ymin, ymax = int(ys.min()), int(ys.max())
        except Exception:
            # nếu SAM lỗi (thiếu VRAM...), dùng bbox YOLO
            xmin, ymin, xmax, ymax = x1, y1, x2, y2

        # crop review
        xmin_c, ymin_c = max(0, xmin), max(0, ymin)
        xmax_c, ymax_c = min(w-1, xmax), min(h-1, ymax)
        if xmax_c <= xmin_c or ymax_c <= ymin_c:
            continue

        crop = img[ymin_c:ymax_c, xmin_c:xmax_c]
        if crop.size == 0:
            continue

        # Gợi ý nhãn mặc định từ YOLO (nếu có)
        suggested = None
        conf_txt = ""
        if clses is not None and names is not None:
            try:
                pred_id = int(clses[i])
                suggested = names.get(pred_id, None) if isinstance(names, dict) else None
            except Exception:
                suggested = None
        if confs is not None:
            conf_txt = f" ({confs[i]:.2f})"

        # Hiển thị crop và hỏi người dùng
        cv2.imshow("CROP", crop)
        cv2.waitKey(1)
        if suggested:
            prompt = f"→ Nhập tên/số cho box này [Enter = '{suggested}'{conf_txt}, q = bỏ]: "
        else:
            prompt = "→ Nhập tên/số cho box này (q = bỏ qua): "

        user_input = input(prompt).strip()
        cv2.destroyWindow("CROP")

        if user_input.lower() == "q":
            continue
        label_name = suggested if (user_input == "" and suggested is not None) else user_input

        if label_name == "" or label_name is None:
            # nếu vẫn rỗng, bỏ qua
            continue

        # ánh xạ label -> id
        if label_name not in label_map:
            label_map[label_name] = next_id
            next_id += 1
        cls_id = label_map[label_name]

        # lưu dòng nhãn YOLO
        label_lines.append(yolo_txt_line(cls_id, xmin, ymin, xmax, ymax, w, h))

        # vẽ lên canvas (preview)
        cv2.rectangle(canvas, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(canvas, f"{label_name}", (xmin, max(0, ymin - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # === Thêm BIỂN SỐ thủ công bằng ROI (có thể thêm nhiều cái) ===
    while True:
        choice = input("Thêm biển số thủ công? (r = vẽ ROI, Enter = xong): ").strip().lower()
        if choice == "r":
            added = add_plate_by_roi(img, canvas, label_lines, w, h)
            print("✅ Đã thêm license_plate." if added else "❌ Không thêm.")
        else:
            break

    # Xác nhận & LƯU DATASET YOLO
    if label_lines:
        # Cho xem preview (không bắt buộc)
        if also_save_preview:
            cv2.imshow("REVIEW", canvas)
            cv2.waitKey(1)
            _ = input("Nhấn Enter để lưu…")
            cv2.destroyWindow("REVIEW")

        # Lưu **dataset chuẩn YOLO**
        ensure_dataset_dirs()
        stem = Path(img_path).stem
        out_img = os.path.join(DATASET_DIR, "images", split_for_dataset, stem + ".jpg")
        out_lbl = os.path.join(DATASET_DIR, "labels", split_for_dataset, stem + ".txt")

        # ảnh gốc dùng để train (không phải ảnh vẽ box)
        cv2.imwrite(out_img, img)
        with open(out_lbl, "w", encoding="utf-8") as f:
            f.write("\n".join(label_lines))
        print(f"✅ Saved YOLO dataset: {out_img}, {out_lbl}")

        # (tuỳ chọn) vẫn lưu “labels/” và “labeled_images/” kiểu cũ để bạn xem
                # (luôn) Lưu ảnh preview để xem lại
        os.makedirs(PREVIEW_DIR, exist_ok=True)
        preview_img = os.path.join(PREVIEW_DIR, stem + ".jpg")
        cv2.imwrite(preview_img, canvas)
        print(f"🖼️  Saved preview: {preview_img}")

        # (tuỳ chọn) vẫn lưu nhãn “kiểu cũ” vào labels/
        if also_save_legacy_labels:
            os.makedirs(LABELS_DIR, exist_ok=True)
            legacy_label = os.path.join(LABELS_DIR, stem + ".txt")
            with open(legacy_label, "w", encoding="utf-8") as f:
                f.write("\n".join(label_lines))
            print(f"ℹ️  Also saved legacy label: {legacy_label}")


# ====================
# RUN
# ====================
if __name__ == "__main__":
    # Duyệt thư mục ảnh input
    if not os.path.exists(IMAGES_DIR):
        print(f"❌ Không tìm thấy thư mục ảnh: {IMAGES_DIR}")
    else:
        files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        files.sort()

        # Chia train/val cố định theo shuffle 1 lần (80/20)
        random.seed(42)
        random.shuffle(files)
        split_idx = int(len(files) * (1 - VAL_RATIO))
        train_files = files[:split_idx]
        val_files   = files[split_idx:]
        print(f"📊 Số ảnh: {len(files)} → Train: {len(train_files)}, Val: {len(val_files)}")

        # Annotate & lưu thẳng vào dataset/
        for img_file in train_files:
            process_image(os.path.join(IMAGES_DIR, img_file), split_for_dataset="train",
                          also_save_preview=True, also_save_legacy_labels=False)
        for img_file in val_files:
            process_image(os.path.join(IMAGES_DIR, img_file), split_for_dataset="val",
                          also_save_preview=True, also_save_legacy_labels=False)

    # Cập nhật classes.txt cuối cùng (giữ thứ tự ID)
    classes_sorted = sorted(label_map.items(), key=lambda x: x[1])
    with open(CLASSES_FILE, "w", encoding="utf-8") as f:
        for name, _ in classes_sorted:
            f.write(name + "\n")
    print(f"📄 Updated {CLASSES_FILE} với {len(classes_sorted)} classes.")

    # Gợi ý tạo dataset.yaml (in ra màn hình)
    yaml_path = os.path.abspath(DATASET_DIR)
    print("\n👉 Tạo file dataset.yaml với nội dung (sửa path cho đúng):")
    print(f"""\
path: {yaml_path}
train: images/train
val: images/val
names:
  0: car
  1: license_plate
""")
