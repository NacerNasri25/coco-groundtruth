# COCO 2017 Ground-Truth Extraction (Detection & Instance Segmentation)

This mini-project implements **Task 3** from a research assignment:
extracting ground-truth annotations from the **COCO 2017 validation set**
for:

- **Object detection** (bounding boxes as masks + class)
- **Instance segmentation** (segmentation masks + class + instance ID)

The code is designed to be integrated into a larger research pipeline.

---

## 📂 Project Structure

```text
coco_task3/
│
├── data/
│   ├── instances_val2017.json   # COCO 2017 val annotations (NOT in repo)
│   └── val2017/                 # COCO 2017 val images (NOT in repo)
│
├── detection/
│   ├── __init__.py
│   └── build_detection_gt.py    # Object detection GT extraction
│
└── segmentation/
    ├── __init__.py
    └── build_instance_gt.py     # Instance segmentation GT extraction
