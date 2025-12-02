# COCO 2017 Ground-Truth Extraction  
(Detection • Instance Segmentation • Panoptic Segmentation)

This mini-project implements **Task 3** from a research assignment:  
extracting ground-truth annotations from the **COCO 2017 validation set**  
for three separate tasks:

- **Object detection**  
  Convert bounding boxes → binary masks + continuous class IDs (0–79)

- **Instance segmentation**  
  Convert polygon/RLE segmentation → binary masks + class ID + instance ID

- **Panoptic segmentation**  
  Decode RGB panoptic masks → binary masks + class ID + segment ID  
  (covering both “things” and “stuff”; 133 categories)

The code is modular and designed to be integrated into a larger research pipeline.

---

## 📥 Download COCO 2017 Data

All data must be downloaded manually (NOT included in this repo).  
Here are the official download links:

### **🔸 COCO 2017 Images**
- **Validation images (5K):**  
  https://images.cocodataset.org/zips/val2017.zip

### **🔸 COCO 2017 Instance Segmentation Annotations**
- **instances_val2017.json:**  
  https://images.cocodataset.org/annotations/annotations_trainval2017.zip  
  (file path inside: `annotations/instances_val2017.json`)

### **🔸 COCO 2017 Panoptic Segmentation**
- **Panoptic annotations:**  
  https://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip  
  Contains:
  - `panoptic_val2017.json`
  - Folder `panoptic_val2017/` with RGB mask PNG files

---

## 📂 Project Structure

```text
coco_task3/
│
├── data/
│   ├── instances_val2017.json          # COCO 2017 val annotations (NOT in repo)
│   ├── val2017/                        # COCO 2017 val images (NOT in repo)
│   ├── panoptic_val2017.json           # COCO 2017 panoptic annotations (NOT in repo)
│   └── panoptic_val2017/               # COCO 2017 panoptic RGB masks (NOT in repo)
│
├── detection/
│   ├── __init__.py
│   └── build_detection_gt.py           # Object detection GT extraction
│
├── segmentation/
│   ├── __init__.py
│   └── build_instance_gt.py            # Instance segmentation GT extraction
│
└── panoptic/
    ├── __init__.py
    └── build_panoptic_gt.py            # Panoptic segmentation GT extraction
