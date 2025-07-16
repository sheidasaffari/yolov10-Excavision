#!/usr/bin/env python3
"""
convert_output_to_yolo.py

Usage
-----
python convert_output_to_yolo.py \
    --src /path/to/output \
    --dst /path/to/yolo_dataset_root

Resulting structure
-------------------
/path/to/yolo_dataset_root/
├── images/
│   └── 2025-05-26_21-31-59-06-06-0003.png
└── labels/
    └── 2025-05-26_21-31-59-06-06-0003.txt     # YOLO-format bboxes
"""

import argparse, json, re, shutil
from pathlib import Path

# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────
IMG_RE = re.compile(
    r"Excavator_(\d+)_cam(\d+)_frame(\d+)\.(png|jpg|jpeg)$", re.IGNORECASE
)

def coco_bbox_to_yolo(bbox, img_w, img_h):
    """Convert COCO [x, y, w, h] → YOLO [x_c, y_c, w, h] (all normalized)."""
    x_min, y_min, w, h = bbox
    x_c = (x_min + w / 2) / img_w
    y_c = (y_min + h / 2) / img_h
    return x_c, y_c, w / img_w, h / img_h


# ────────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────────
def main(src_root: Path, dst_root: Path):
    images_dir  = dst_root / "images"
    labels_dir  = dst_root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for json_path in src_root.rglob("*_2d.json"):
        # The matching image has same base name minus '_2d.json'
        img_path = json_path.with_name(json_path.stem.replace("_2d", "") + json_path.suffix.replace("json", "png"))
        if not img_path.exists():   # fallback for jpg / jpeg
            possibles = list(json_path.parent.glob(json_path.stem.replace("_2d", "") + ".*"))
            img_candidates = [p for p in possibles if p.suffix.lower() in {'.png','.jpg','.jpeg'}]
            if not img_candidates:
                print(f"[WARN] No image for {json_path.name}")
                continue
            img_path = img_candidates[0]

        m = IMG_RE.match(img_path.name)
        if not m:
            print(f"[WARN] Image filename pattern not recognized: {img_path}")
            continue

        excav_id, cam_id, frame_no, _ = m.groups()
        parent_stamp = json_path.parent.name          # e.g. 2025-05-26_21-31-59
        new_stem = f"{parent_stamp}-{int(excav_id):02d}-{int(cam_id):02d}-{frame_no}"
        dst_img  = images_dir / f"{new_stem}.png"
        dst_lbl  = labels_dir / f"{new_stem}.txt"

        # ── Load JSON and extract bboxes ──────────────────────────────────
        with open(json_path, "r") as f:
            data = json.load(f)

        img_meta = data["images"][0]
        W, H     = img_meta["width"], img_meta["height"]

        yolo_lines = []
        for ann in data["annotations"]:
            bbox = ann["bbox"]                        # [x_min, y_min, w, h]
            x_c, y_c, w_n, h_n = coco_bbox_to_yolo(bbox, W, H)
            yolo_lines.append(f"0 {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}")

        if not yolo_lines:
            print(f"[WARN] No annotations in {json_path}")
            continue

        # ── Copy/rename image and write label file ───────────────────────
        shutil.copy2(img_path, dst_img)
        dst_lbl.write_text("\n".join(yolo_lines))
        print(f"[OK] {dst_img.name}   ({len(yolo_lines)} boxes)")

    print("\n✅ Conversion complete!")
    print(f"Images saved to: {images_dir}")
    print(f"Labels saved to: {labels_dir}")

# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert custom excavator data → YOLO format")
    parser.add_argument("--src", required=True, type=Path, help="Path to /output folder")
    parser.add_argument(
        "--dst",
        required=True,
        type=Path,
        help="Destination root (images/ and labels/ will be created here)",
    )
    args = parser.parse_args()
    main(args.src.expanduser().resolve(), args.dst.expanduser().resolve())




##Use case
#python format-conversion.py --src /home/sheida/Documents/Research/Blender/output --dst /home/sheida/yolov10-Excavision/data
