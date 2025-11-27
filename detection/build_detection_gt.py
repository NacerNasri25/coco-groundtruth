#!/usr/bin/env python
# for using in pipline we should use -> from detection.build_detection_gt import get_detection_gt_for_image
"""
Task 3 – Part 1: Object Detection Ground Truth (COCO 2017 val)

For each image, we generate a list of dictionaries like:

    {"class_id": <int in [0..79]>, "mask": np.ndarray(H, W)}

- class_id: continuous class index 0..79 (remapped from COCO category_id)
- mask: rectangular binary mask derived from the COCO bounding box
"""

import os
import numpy as np
from pycocotools.coco import COCO


def bbox_to_mask(bbox, height, width):
    """
    Convert a COCO bounding box [x, y, w, h] into a binary mask
    of shape (height, width) with values in {0, 1}.
    """
    x, y, w, h = bbox

    mask = np.zeros((height, width), dtype=np.uint8)

    x0 = int(round(x))
    y0 = int(round(y))
    x1 = int(round(x + w))
    y1 = int(round(y + h))

    # clip to image boundaries
    x0 = max(0, min(x0, width))
    x1 = max(0, min(x1, width))
    y0 = max(0, min(y0, height))
    y1 = max(0, min(y1, height))

    mask[y0:y1, x0:x1] = 1
    return mask


def build_category_id_mapping(coco):
    """
    Build mapping from COCO category_id → continuous class index [0..79].
    """
    cat_ids = coco.getCatIds()
    cats = coco.loadCats(cat_ids)
    cats_sorted = sorted(cats, key=lambda c: c["id"])

    mapping = {cat["id"]: idx for idx, cat in enumerate(cats_sorted)}
    return mapping


def get_detection_gt_for_image(coco, image_id, catid_to_classid):
    """
    Ground truth for ONE image (detection view).

    Returns:
        list of {"class_id": int, "mask": np.ndarray(H, W)}
    """
    img_info = coco.loadImgs([image_id])[0]
    width = img_info["width"]
    height = img_info["height"]

    ann_ids = coco.getAnnIds(imgIds=[image_id], iscrowd=None)
    anns = coco.loadAnns(ann_ids)

    detections = []

    for ann in anns:
        coco_cat_id = ann["category_id"]
        class_id = catid_to_classid[coco_cat_id]

        bbox = ann["bbox"]
        mask = bbox_to_mask(bbox, height, width)

        detections.append(
            {
                "class_id": class_id,
                "mask": mask,
            }
        )

    return detections


def main():
    # Find JSON: coco_task3/data/instances_val2017.json
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ann_file = os.path.join(base_dir, "data", "instances_val2017.json")

    if not os.path.exists(ann_file):
        raise FileNotFoundError(f"Annotation file not found: {ann_file}")

    print(f"Loading COCO annotations from:\n  {ann_file}")
    coco = COCO(ann_file)

    catid_to_classid = build_category_id_mapping(coco)
    print(f"Number of classes (mapped): {len(catid_to_classid)}")

    img_ids = coco.getImgIds()
    print(f"Total images in val2017: {len(img_ids)}")

    # small subset for testing
    test_img_ids = img_ids[:5]

    for image_id in test_img_ids:
        detections = get_detection_gt_for_image(coco, image_id, catid_to_classid)
        print(f"\nImage ID: {image_id}")
        print(f"  Number of detections: {len(detections)}")
        if detections:
            first = detections[0]
            print(f"  First entry → class_id: {first['class_id']}, "
                  f"mask shape: {first['mask'].shape}, "
                  f"mask dtype: {first['mask'].dtype}")


if __name__ == "__main__":
    main()
