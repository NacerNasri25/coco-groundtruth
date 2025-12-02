#!/usr/bin/env python
"""
Task 3 – COCO Panoptic Ground Truth (val2017)

For each image, we generate a list of dictionaries:

    {
        "instance_id": int,          # segment ID from panoptic segmentation
        "class_id": int,             # remapped category index [0..N-1]
        "mask": np.ndarray(H, W)     # binary mask for this segment
    }

The script uses:
- panoptic_val2017.json
- panoptic_val2017/*.png  (RGB-encoded panoptic masks)
"""

import os
import json
import numpy as np
from PIL import Image


def rgb_to_id(color):
    """
    Convert RGB color to segment id as defined in COCO Panoptic:

        id = R + 256 * G + 256^2 * B

    This works for:
    - a full RGB image (H, W, 3) → (H, W) ids
    """
    color = color.astype(np.int64)
    return color[:, :, 0] + 256 * color[:, :, 1] + (256**2) * color[:, :, 2]


def build_category_id_mapping(panoptic_data):
    """
    Build mapping from COCO category_id → continuous class index [0..N-1].

    Panoptic categories include both 'things' and 'stuff', so
    N will typically be > 80 (e.g., 133 categories).
    """
    cats = sorted(panoptic_data["categories"], key=lambda c: c["id"])
    mapping = {cat["id"]: idx for idx, cat in enumerate(cats)}
    return mapping


def index_panoptic_structures(panoptic_data):
    """
    Build quick lookup dictionaries:
    - image_id → image_info
    - image_id → panoptic_annotation (file_name + segments_info)
    """
    img_id_to_img = {img["id"]: img for img in panoptic_data["images"]}
    img_id_to_ann = {ann["image_id"]: ann for ann in panoptic_data["annotations"]}
    return img_id_to_img, img_id_to_ann


def get_panoptic_gt_for_image(
    img_id,
    img_id_to_img,
    img_id_to_ann,
    catid_to_classid,
    panoptic_masks_dir,
):
    """
    Ground-truth for ONE image in panoptic format.

    Parameters
    ----------
    img_id : int
        COCO image id
    img_id_to_img : dict[int, dict]
    img_id_to_ann : dict[int, dict]
    catid_to_classid : dict[int, int]
    panoptic_masks_dir : str
        Directory containing panoptic_val2017 PNGs.

    Returns
    -------
    list[dict]:
        [
          {
            "instance_id": int,
            "class_id": int,
            "mask": np.ndarray(H, W)
          },
          ...
        ]
    """
    img_info = img_id_to_img[img_id]
    ann = img_id_to_ann[img_id]

    file_name = ann["file_name"]  # e.g. "000000397133.png"
    mask_path = os.path.join(panoptic_masks_dir, file_name)

    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Panoptic mask not found: {mask_path}")

    # Load RGB panoptic mask and convert to segment-id map
    panoptic_rgb = np.array(Image.open(mask_path))
    seg_id_map = rgb_to_id(panoptic_rgb)  # shape (H, W)

    H, W = seg_id_map.shape

    panoptic_instances = []

    for seg in ann["segments_info"]:
        seg_id = seg["id"]           # integer segment ID
        cat_id = seg["category_id"]  # panoptic category id

        # Remap to continuous class index
        class_id = catid_to_classid[cat_id]

        # Binary mask for this segment
        mask = (seg_id_map == seg_id).astype(np.uint8)

        panoptic_instances.append(
            {
                "instance_id": seg_id,
                "class_id": class_id,
                "mask": mask,
            }
        )

    return panoptic_instances, (H, W)


def main():
    # Base directory of the project (one level above this file)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(base_dir)

    panoptic_json = os.path.join(base_dir, "data", "panoptic_val2017.json")
    panoptic_masks_dir = os.path.join(base_dir, "data", "panoptic_val2017")

    if not os.path.exists(panoptic_json):
        raise FileNotFoundError(f"Panoptic JSON not found: {panoptic_json}")
    if not os.path.isdir(panoptic_masks_dir):
        raise FileNotFoundError(f"Panoptic masks directory not found: {panoptic_masks_dir}")

    print("Loading COCO Panoptic annotations from:")
    print(f"  {panoptic_json}")

    with open(panoptic_json, "r") as f:
        panoptic_data = json.load(f)

    # Category mapping
    catid_to_classid = build_category_id_mapping(panoptic_data)
    print(f"Number of panoptic categories (mapped): {len(catid_to_classid)}")

    # Index images and annotations by image_id
    img_id_to_img, img_id_to_ann = index_panoptic_structures(panoptic_data)
    img_ids = sorted(img_id_to_img.keys())
    print(f"Total images in panoptic val2017: {len(img_ids)}")

    # Small subset for testing (first 5 images)
    test_img_ids = img_ids[:5]

    for img_id in test_img_ids:
        instances, (H, W) = get_panoptic_gt_for_image(
            img_id,
            img_id_to_img,
            img_id_to_ann,
            catid_to_classid,
            panoptic_masks_dir,
        )

        print(f"\nImage ID: {img_id}")
        print(f"  Image size: {H} x {W}")
        print(f"  Number of panoptic instances: {len(instances)}")

        if instances:
            first = instances[0]
            print(
                f"  First entry → instance_id: {first['instance_id']}, "
                f"class_id: {first['class_id']}, "
                f"mask shape: {first['mask'].shape}, "
                f"mask dtype: {first['mask'].dtype}"
            )


if __name__ == "__main__":
    main()
