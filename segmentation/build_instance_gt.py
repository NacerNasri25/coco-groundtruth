#!/usr/bin/env python
"""
Task 3 – Part 2: Instance Segmentation Ground Truth (COCO 2017 val)

For each image, we generate a list of dictionaries like:

    {"instance_id": int,
     "class_id": <int in [0..79]>,
     "mask": np.ndarray(H, W)}

- instance_id: COCO annotation id (ann["id"])
- class_id: continuous class index 0..79 (remapped from COCO category_id)
- mask: binary segmentation mask from COCO (using coco.annToMask)
"""

import os
import numpy as np
from pycocotools.coco import COCO


def build_category_id_mapping(coco):
    """
    Build mapping from COCO category_id → continuous class index [0..79].
    """
    cat_ids = coco.getCatIds()
    cats = coco.loadCats(cat_ids)
    cats_sorted = sorted(cats, key=lambda c: c["id"])

    mapping = {cat["id"]: idx for idx, cat in enumerate(cats_sorted)}
    return mapping


def get_instance_gt_for_image(coco, image_id, catid_to_classid):
    """
    Ground truth for ONE image (instance segmentation view).

    Parameters
    ----------
    coco : COCO
        COCO API object
    image_id : int
        COCO image id
    catid_to_classid : dict[int, int]
        Mapping from COCO category_id → class index [0..79]

    Returns
    -------
    list[dict]
        Each entry:
            {
                "instance_id": int,
                "class_id": int,
                "mask": np.ndarray(H, W)
            }
    """
    img_info = coco.loadImgs([image_id])[0]
    width = img_info["width"]
    height = img_info["height"]

    ann_ids = coco.getAnnIds(imgIds=[image_id], iscrowd=None)
    anns = coco.loadAnns(ann_ids)

    instances = []

    for ann in anns:
        coco_cat_id = ann["category_id"]
        class_id = catid_to_classid[coco_cat_id]

        # COCO API gives us the segmentation mask directly
        mask = coco.annToMask(ann)  # shape: (H, W), dtype=uint8

        # Sanity check: shapes should match
        if mask.shape != (height, width):
            # If something is off, we skip this annotation for now.
            # (Can be refined later.)
            continue

        instance_id = ann["id"]

        instances.append(
            {
                "instance_id": instance_id,
                "class_id": class_id,
                "mask": mask,
            }
        )

    return instances


def main():
    # Path to instances_val2017.json (same as in detection script)
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

    # Small subset for testing
    test_img_ids = img_ids[:5]

    for image_id in test_img_ids:
        instances = get_instance_gt_for_image(coco, image_id, catid_to_classid)
        print(f"\nImage ID: {image_id}")
        print(f"  Number of instances: {len(instances)}")
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
