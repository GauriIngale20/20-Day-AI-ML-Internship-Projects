# Day 12: YOLO Inference and Object Detection

## Project Overview
This project demonstrates object detection using a pretrained YOLOv8 Nano model. It performs single-image and batch inference on sample images and analyzes the detection results.

## Technologies Used
- Python
- YOLOv8 (Ultralytics)
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Jupyter Notebook

## Project Structure
```text
Day-12/
├── data/
│   └── input_images/
├── models/
│   └── yolov8n.pt
├── notebooks/
│   └── yolo_inference_analysis.ipynb
├── src/
│   ├── config.py
│   ├── model_loader.py
│   ├── inference.py
│   ├── batch_inference.py
│   └── result_analyzer.py
├── outputs/
│   ├── annotated_images/
│   ├── plots/
│   ├── detection_results.csv
│   └── batch_summary.csv
├── reports/
│   └── inference_observations.md
├── tests/
├── main.py
├── requirements.txt
└── README.md
```

## Tasks Completed
1. Loaded pretrained YOLOv8 Nano model.
2. Performed single-image object detection.
3. Implemented batch inference.
4. Extracted bounding boxes, class IDs and confidence scores.
5. Saved annotated output images.
6. Generated detection summary CSV files.
7. Created object distribution and confidence visualizations.
8. Prepared inference observation report.

## Dataset Summary
- Total Images: 2
- Total Objects Detected: 9
- Unique Classes: 4
- Most Detected Class: Person

## Deliverables
- Inference script
- Batch inference script
- Annotated images
- Detection result CSV
- Batch summary CSV
- Visualization plots
- Jupyter Notebook
- Observation report

## Conclusion
The project successfully demonstrates object detection using YOLOv8 Nano. The model identifies objects, generates bounding boxes and confidence scores, and saves the inference results for further analysis.