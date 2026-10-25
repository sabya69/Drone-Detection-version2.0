"""
build_dataset.py - Generates a complete, balanced YOLO training dataset:
1. Hard Negatives: Extracted from user's false-positive screenshots (faces, eyeglasses, backgrounds).
   Heavily augmented with flips, rotations, brightness, blur, and scale variations, paired with 0-byte empty labels.
2. Positive Markers: Generates ArUco markers 41, 32, 43, 44 placed realistically on indoor scenes
   with perspective warps, lighting variations, and exact YOLO bounding box coordinates.
3. Creates dataset.yaml and directory hierarchy ready for immediate training.
"""
import glob
import os
import random
import cv2
import numpy as np

# Reproducibility
random.seed(42)
np.random.seed(42)

DATASET_DIR = os.path.abspath("dataset")
TRAIN_IMG = os.path.join(DATASET_DIR, "images", "train")
VAL_IMG = os.path.join(DATASET_DIR, "images", "val")
TRAIN_LBL = os.path.join(DATASET_DIR, "labels", "train")
VAL_LBL = os.path.join(DATASET_DIR, "labels", "val")

for d in [TRAIN_IMG, VAL_IMG, TRAIN_LBL, VAL_LBL]:
    os.makedirs(d, exist_ok=True)

CLASS_MAP = {
    41: 0,
    32: 1,
    43: 2,
    44: 3
}


# ---------------- Overlay Cleaning ----------------
def clean_overlay(img):
    """Inpaints green/yellow/red overlay boxes and HUD text."""
    h, w = img.shape[:2]
    # Trim top title banner if present
    start_y = 30 if h > 100 else 0
    img_trimmed = img[start_y:, :].copy()
    th, tw = img_trimmed.shape[:2]

    # Convert to HSV to detect overlay elements reliably
    hsv = cv2.cvtColor(img_trimmed, cv2.COLOR_BGR2HSV)

    # Yellow bounding box & text
    m_yellow = cv2.inRange(hsv, (18, 90, 90), (38, 255, 255))
    # Green HUD text and lines
    m_green = cv2.inRange(hsv, (36, 70, 70), (85, 255, 255))
    # Red target dots
    m_red1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
    m_red2 = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))

    overlay_mask = m_yellow | m_green | m_red1 | m_red2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    overlay_mask = cv2.dilate(overlay_mask, kernel, iterations=2)

    # Text in center / top-left regions
    cx1, cy1, cx2, cy2 = int(tw * 0.35), int(th * 0.20), int(tw * 0.75), int(th * 0.45)
    center_roi = img_trimmed[cy1:cy2, cx1:cx2]
    m_txt = (center_roi < 45).all(axis=2) | (center_roi > 215).all(axis=2)
    overlay_mask[cy1:cy2, cx1:cx2] |= (m_txt.astype(np.uint8) * 255)

    overlay_mask = cv2.dilate(overlay_mask, kernel, iterations=1)
    cleaned = cv2.inpaint(img_trimmed, overlay_mask, 5, cv2.INPAINT_TELEA)

    return cleaned


# ---------------- Image Augmentations ----------------
def augment_negative(img):
    """Generates a randomized augmentation of a negative image."""
    h, w = img.shape[:2]
    out = img.copy()

    # Random horizontal flip
    if random.random() > 0.5:
        out = cv2.flip(out, 1)

    # Random brightness / contrast
    alpha = random.uniform(0.75, 1.25)  # contrast
    beta = random.randint(-20, 20)      # brightness
    out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

    # Random slight rotation (-10 to +10 deg)
    angle = random.uniform(-10.0, 10.0)
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    out = cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Random blur
    if random.random() > 0.65:
        ksize = random.choice([3, 5])
        out = cv2.GaussianBlur(out, (ksize, ksize), 0)

    # Standardize output size
    out = cv2.resize(out, (640, 480))
    return out


# ---------------- Positive Marker Generator ----------------
def generate_positive_sample(bg_img, marker_id, class_idx):
    """Pastes an ArUco marker realistically onto a background with perspective & lighting."""
    bg = cv2.resize(bg_img, (640, 480)).copy()
    bg_h, bg_w = bg.shape[:2]

    # Generate ArUco marker from DICT_4X4_50
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    m_size = random.randint(75, 210)
    marker = cv2.aruco.generateImageMarker(d, marker_id, m_size)
    marker = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)

    # Add white border like real printed marker on paper/cube
    border = int(m_size * random.uniform(0.12, 0.25))
    marker_bordered = cv2.copyMakeBorder(
        marker, border, border, border, border,
        cv2.BORDER_CONSTANT, value=(245, 245, 245)
    )
    mb_h, mb_w = marker_bordered.shape[:2]

    # Random perspective warp (simulating 3D cube tilt)
    src_pts = np.float32([[0, 0], [mb_w, 0], [mb_w, mb_h], [0, mb_h]])
    warp_amt = random.uniform(0.04, 0.18)
    dst_pts = np.float32([
        [random.uniform(0, mb_w * warp_amt), random.uniform(0, mb_h * warp_amt)],
        [mb_w - random.uniform(0, mb_w * warp_amt), random.uniform(0, mb_h * warp_amt)],
        [mb_w - random.uniform(0, mb_w * warp_amt), mb_h - random.uniform(0, mb_h * warp_amt)],
        [random.uniform(0, mb_w * warp_amt), mb_h - random.uniform(0, mb_h * warp_amt)]
    ])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(marker_bordered, M, (mb_w, mb_h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    # Create mask of warped marker
    gray_w = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, mask_w = cv2.threshold(gray_w, 1, 255, cv2.THRESH_BINARY)

    # Random position on background
    max_x = bg_w - mb_w - 10
    max_y = bg_h - mb_h - 10
    if max_x <= 10 or max_y <= 10:
        return None, None

    px = random.randint(10, max_x)
    py = random.randint(10, max_y)

    roi = bg[py:py + mb_h, px:px + mb_w]
    mask_inv = cv2.bitwise_not(mask_w)
    bg_sub = cv2.bitwise_and(roi, roi, mask=mask_inv)
    fg_sub = cv2.bitwise_and(warped, warped, mask=mask_w)
    bg[py:py + mb_h, px:px + mb_w] = cv2.add(bg_sub, fg_sub)

    # Compute bounding box from mask
    contours, _ = cv2.findContours(mask_w, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    x, y, w, h = cv2.boundingRect(contours[0])
    bx1 = px + x
    by1 = py + y
    bx2 = bx1 + w
    by2 = by1 + h

    # YOLO normalized coordinates: class x_center y_center width height
    x_c = ((bx1 + bx2) / 2.0) / bg_w
    y_c = ((by1 + by2) / 2.0) / bg_h
    bb_w = (bx2 - bx1) / bg_w
    bb_h = (by2 - by1) / bg_h

    label_line = f"{class_idx} {x_c:.6f} {y_c:.6f} {bb_w:.6f} {bb_h:.6f}"
    return bg, label_line


def main():
    print("=" * 60)
    print("Building balanced YOLO training dataset with hard negatives...")
    print("=" * 60)

    false_files = sorted(glob.glob(r"false\*.png"))
    if not false_files:
        print("Error: No images found in false/ directory!")
        return

    print(f"Found {len(false_files)} false-positive reference images in false/.\n")

    # 1. Clean and extract base negative frames & crops
    base_negatives = []
    for f in false_files:
        raw = cv2.imread(f)
        if raw is None:
            continue
        cleaned = clean_overlay(raw)
        base_negatives.append(cleaned)

        h, w = cleaned.shape[:2]
        if h > 200 and w > 200:
            # Face, glasses & torso crop (upper/center)
            fy1, fy2 = int(h * 0.10), int(h * 0.90)
            fx1, fx2 = int(w * 0.15), int(w * 0.85)
            torso_crop = cleaned[fy1:fy2, fx1:fx2]
            base_negatives.append(torso_crop)

            # Left side crop (clothesline, background room, doorway)
            bg_left = cleaned[:, :int(w * 0.50)]
            base_negatives.append(bg_left)

            # Right side crop (bed, wall, room background)
            bg_right = cleaned[:, int(w * 0.50):]
            base_negatives.append(bg_right)

    print(f"Extracted {len(base_negatives)} base negative image patches.")

    # 2. Generate Augmented Negative Samples (Hard Negatives)
    num_neg_train = 160
    num_neg_val = 40

    neg_idx = 0
    for split, count in [("train", num_neg_train), ("val", num_neg_val)]:
        img_dir = TRAIN_IMG if split == "train" else VAL_IMG
        lbl_dir = TRAIN_LBL if split == "train" else VAL_LBL
        for i in range(count):
            base = random.choice(base_negatives)
            neg_img = augment_negative(base)
            fname = f"neg_scene_{neg_idx:04d}"
            img_path = os.path.join(img_dir, f"{fname}.jpg")
            lbl_path = os.path.join(lbl_dir, f"{fname}.txt")

            cv2.imwrite(img_path, neg_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            # Empty 0-byte label file signals negative sample in YOLO!
            with open(lbl_path, "w") as lf:
                pass
            neg_idx += 1

    print(f"Created {num_neg_train} train and {num_neg_val} val negative samples (empty 0-byte labels).")

    # 3. Generate Positive Marker Samples (ArUco 41, 32, 43, 44)
    num_pos_train_per_class = 50
    num_pos_val_per_class = 12

    pos_idx = 0
    for marker_id, class_idx in CLASS_MAP.items():
        for split, count in [("train", num_pos_train_per_class), ("val", num_pos_val_per_class)]:
            img_dir = TRAIN_IMG if split == "train" else VAL_IMG
            lbl_dir = TRAIN_LBL if split == "train" else VAL_LBL

            created = 0
            while created < count:
                bg_candidate = random.choice(base_negatives)
                pos_img, label_line = generate_positive_sample(bg_candidate, marker_id, class_idx)
                if pos_img is None:
                    continue

                fname = f"pos_marker_{marker_id}_{pos_idx:04d}"
                img_path = os.path.join(img_dir, f"{fname}.jpg")
                lbl_path = os.path.join(lbl_dir, f"{fname}.txt")

                cv2.imwrite(img_path, pos_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                with open(lbl_path, "w") as lf:
                    lf.write(label_line + "\n")

                pos_idx += 1
                created += 1

    total_pos_train = num_pos_train_per_class * len(CLASS_MAP)
    total_pos_val = num_pos_val_per_class * len(CLASS_MAP)
    print(f"Created {total_pos_train} train and {total_pos_val} val positive marker samples.")

    # 4. Generate dataset.yaml
    yaml_content = f"""# YOLO Training Configuration - Drone Target Detection
path: {DATASET_DIR.replace(chr(92), '/')}
train: images/train
val: images/val

names:
  0: '41'
  1: '32'
  2: '43'
  3: '44'
"""
    yaml_path = os.path.join(DATASET_DIR, "dataset.yaml")
    with open(yaml_path, "w") as yf:
        yf.write(yaml_content)

    print(f"\nSaved dataset config to: {yaml_path}")
    print("\nDataset Summary:")
    print(f"  Training:   {total_pos_train} positives + {num_neg_train} hard negatives = {total_pos_train + num_neg_train} images")
    print(f"  Validation: {total_pos_val} positives + {num_neg_val} hard negatives = {total_pos_val + num_neg_val} images")
    print(f"  Total Images: {total_pos_train + num_neg_train + total_pos_val + num_neg_val}")
    print("=" * 60)


if __name__ == "__main__":
    main()
