# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 16:44:31 2025

Author: czheng23
Description:
    Convert segmentation masks into YOLO-style bounding box annotations.
    Detects red regions in segmentation masks and exports YOLO format .txt files.
"""

import os
import cv2
import numpy as np

# ==== Manual configuration ====
masks_dir = "C:/Users/czheng23/Downloads/Tomato-IMG/masks"      # Path to segmentation masks
images_dir = "C:/Users/czheng23/Downloads/Tomato-IMG/images"    # Path to original images (optional)
out_labels = "C:/Users/czheng23/Downloads/Tomato-IMG/labels"    # Output folder for YOLO .txt annotations
class_id = 0                                                   # YOLO class ID (e.g., 0 = tomato)
min_area = 20                                                  # Minimum area of connected components
save_preview = True                                            # Save preview images with bounding boxes
preview_dir = "C:/Users/czheng23/Downloads/Tomato-IMG/preview"  # Folder to save preview images
os.makedirs(out_labels, exist_ok=True)
if save_preview:
    os.makedirs(preview_dir, exist_ok=True)

def extract_red_mask(mask_bgr):
    """
    Extract red regions from a BGR segmentation mask using HSV color thresholding.
    Returns a binary mask where red regions are 255 and others are 0.
    """
    hsv = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 80, 80])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([170, 80, 80])
    upper2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    return cv2.bitwise_or(mask1, mask2)

def binary_to_bboxes(binary, min_area=50):
    """
    Convert a binary mask into bounding boxes.
    Returns a list of bounding boxes in [x_min, y_min, x_max, y_max] format.
    """
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        bboxes.append([x, y, x+w, y+h])
    return bboxes

def to_yolo_bbox(xmin, ymin, xmax, ymax, img_w, img_h):
    """
    Convert bounding box from (x_min, y_min, x_max, y_max) format
    to YOLO format: (x_center, y_center, width, height), normalized to [0,1].
    """
    w = xmax - xmin
    h = ymax - ymin
    xc = xmin + w / 2
    yc = ymin + h / 2
    return xc / img_w, yc / img_h, w / img_w, h / img_h

# ==== Main process ====
for fname in os.listdir(masks_dir):
    if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
        continue

    mask_path = os.path.join(masks_dir, fname)
    mask = cv2.imread(mask_path)
    if mask is None:
        print(f"[WARN] Failed to read: {mask_path}")
        continue

    # Match the original image if available
    base = os.path.splitext(fname)[0]
    img_path = os.path.join(images_dir, fname)
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]
    else:
        img = mask.copy()
        img_h, img_w = mask.shape[:2]

    # Extract red regions as binary mask
    binary = extract_red_mask(mask)

    # Generate bounding boxes
    bboxes = binary_to_bboxes(binary, min_area=min_area)

    # Write YOLO annotation file
    out_txt = os.path.join(out_labels, base + ".txt")
    lines = []
    for (x1, y1, x2, y2) in bboxes:
        xc, yc, w, h = to_yolo_bbox(x1, y1, x2, y2, img_w, img_h)
        lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    with open(out_txt, "w") as f:
        f.write("\n".join(lines))
    print(f"[OK] {fname}: {len(bboxes)} boxes -> {out_txt}")

    # Save preview images with bounding boxes
    if save_preview:
        for (x1, y1, x2, y2) in bboxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        out_img = os.path.join(preview_dir, base + "_preview.jpg")
        cv2.imwrite(out_img, img)
