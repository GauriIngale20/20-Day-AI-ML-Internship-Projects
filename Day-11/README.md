# Day 11: Computer Vision Foundations & Image Annotation Analysis

## Project Overview
This project explores computer vision fundamentals using the COCO8 image dataset. It covers image properties, YOLO object-detection annotations, bounding box visualization, and annotation quality validation.

## Objectives
- Inspect image dimensions, channels, and formats.
- Understand RGB, BGR, and grayscale images.
- Explore YOLO annotation format.
- Visualize object bounding boxes.
- Validate annotation quality.
- Generate image inventory and annotation reports.

## Technologies Used
- Python
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Pillow
- Jupyter Notebook
- Pytest

## Dataset
- Training images: 4
- Validation images: 4
- Total images: 8
- Total object annotations: 30
- Valid image-label files: 8
- Invalid image-label files: 0

## Project Structure
```text
Day-11/
├── data/
│   └── coco8/
│       ├── images/
│       │   ├── train/
│       │   └── val/
│       ├── labels/
│       │   ├── train/
│       │   └── val/
│       ├── coco8.yaml
│       ├── LICENSE
│       └── README.md
├── notebooks/
│   ├── day11_computer_vision_foundations.ipynb
│   └── day11_backup.ipynb
├── reports/
│   ├── figures/
│   ├── image_inventory.csv
│   ├── annotation_quality.csv
│   └── annotation_quality_notes.md
├── src/
│   ├── image_loader.py
│   ├── image_inspector.py
│   ├── annotation_visualizer.py
│   └── inventory.py
├── tests/
│   ├── test_image_inspector.py
│   └── test_yolo_labels.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md

```

## Key Results
- Inspected 8 images.
- Analyzed image dimensions, formats, and color channels.
- Visualized YOLO bounding boxes.
- Validated 8 image-label files containing 30 object annotations.
- Found 0 invalid and 0 missing label files.
- Generated 5 visualization figures.
- Generated image inventory and annotation quality reports.

## Run the Project

Install dependencies:
```bash
pip install -r requirements.txt
```

Run:
```bash
python main.py
```

Run tests:
```bash
python -m pytest -v
```

## Testing
All 5 automated tests passed successfully.

## Conclusion
This project provided practical experience with image inspection, YOLO object-detection annotations, bounding box visualization, and annotation quality validation using Python and OpenCV.