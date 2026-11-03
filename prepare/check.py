import os
import random
import cv2
import numpy as np

base_dir = os.environ.get(
    "DATASET_YOLO_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset_yolo")
)
img_dir_train = os.path.join(base_dir, "images", "train")
img_dir_val = os.path.join(base_dir, "images", "val")
lbl_dir_train = os.path.join(base_dir, "labels", "train")
lbl_dir_val = os.path.join(base_dir, "labels", "val")

def fix_label_dir(label_dir):
    for fname in os.listdir(label_dir):
        if not fname.lower().endswith(".txt"):
            continue
        fpath = os.path.join(label_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        new_lines = []
        changed = False

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 7:
                continue

            cls_id = parts[0]
            nums = list(map(float, parts[1:]))
            if len(nums) < 6:
                continue

            poly = nums[4:]
            if len(poly) < 6 or len(poly) % 2 != 0:
                continue

            xs_raw = poly[0::2]
            ys_raw = poly[1::2]

            # 将多边形坐标裁剪到 [0,1]，与训练时约定一致
            xs = [min(max(x, 0.0), 1.0) for x in xs_raw]
            ys = [min(max(y, 0.0), 1.0) for y in ys_raw]

            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            bbox_w = x_max - x_min
            bbox_h = y_max - y_min
            if bbox_w <= 0 or bbox_h <= 0:
                continue
            bbox_xc = x_min + bbox_w / 2
            bbox_yc = y_min + bbox_h / 2

            # 再次确保 bbox 在 [0,1] 内
            bbox_xc = min(max(bbox_xc, 0.0), 1.0)
            bbox_yc = min(max(bbox_yc, 0.0), 1.0)
            bbox_w = min(max(bbox_w, 0.0), 1.0)
            bbox_h = min(max(bbox_h, 0.0), 1.0)

            nums_fixed = [bbox_xc, bbox_yc, bbox_w, bbox_h] + poly
            line_fixed = cls_id + " " + " ".join(f"{v:.6f}" for v in nums_fixed) + "\n"

            if line_fixed != line:
                changed = True
            new_lines.append(line_fixed)

        if changed:
            with open(fpath, "w", encoding="utf-8") as f:
                f.writelines(new_lines)


def visualize_one(img_path, lbl_dir):
    name = os.path.splitext(os.path.basename(img_path))[0]
    lbl_path = os.path.join(lbl_dir, name + ".txt")
    if not os.path.exists(lbl_path):
        print("no label:", name)
        return

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]

    with open(lbl_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            cls = int(parts[0])
            # bbox 4 个 + 后面多边形
            nums = list(map(float, parts[1:]))
            poly = nums[4:]  # 去掉 bbox，只要多边形点
            if len(poly) < 6:
                continue

            pts = []
            for i in range(0, len(poly), 2):
                xn = min(max(poly[i], 0.0), 1.0)
                yn = min(max(poly[i + 1], 0.0), 1.0)
                x = xn * w
                y = yn * h
                pts.append([x, y])
            pts = np.array(pts, dtype=np.int32)

            color = (0, 0, 255)  # 红色
            cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)

    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    fix_label_dir(lbl_dir_train)
    fix_label_dir(lbl_dir_val)

    all_imgs_train = [os.path.join(img_dir_train, f) for f in os.listdir(img_dir_train) if f.lower().endswith(".tif")]
    for p in random.sample(all_imgs_train, min(10, len(all_imgs_train))):
        print("check train:", p)
        visualize_one(p, lbl_dir_train)

    if os.path.isdir(img_dir_val):
        all_imgs_val = [os.path.join(img_dir_val, f) for f in os.listdir(img_dir_val) if f.lower().endswith(".tif")]
        if all_imgs_val:
            for p in random.sample(all_imgs_val, min(5, len(all_imgs_val))):
                print("check val:", p)
                visualize_one(p, lbl_dir_val)
