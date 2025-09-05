🍅 Tomato Segmentation → YOLO Annotation Converter

This repository provides a simple Python script to convert segmentation masks (where tomatoes are highlighted in red) into YOLO-style bounding box annotations.
It is designed to help you quickly prepare datasets compatible with Roboflow, YOLOv5/v8, Detectron2, or other object detection frameworks.

✨ Features

🟥 Red-region detection — Extracts red regions from segmentation masks using HSV color thresholding.

📦 Bounding box generation — Automatically converts red regions into bounding boxes.

📝 YOLO annotation export — Generates .txt files in YOLO format:

class x_center y_center width height


🖼️ Preview visualization — Optionally saves images with bounding boxes drawn.

⚡ Roboflow compatible — Output format works seamlessly with Roboflow.
