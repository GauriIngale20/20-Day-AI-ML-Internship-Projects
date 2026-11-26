
# YOLO Inference Analysis - Day 12

## Project Overview
Pretrained YOLOv8 Nano model was used for object detection.

## Dataset Summary
- Total Images Processed: 2
- Total Objects Detected: 9
- Unique Object Classes: 4

## Class-wise Detection
Class_Name
person       6
bus          1
stop sign    1
tie          1

## Confidence Analysis
- Average Confidence: 0.6533
- Confidence Threshold: 0.25

## Image-wise Detection
     Image  Total_Objects
   bus.jpg              6
zidane.jpg              3

## Key Observations
1. YOLO successfully detected objects in both sample images.
2. Person was the most frequently detected class.
3. Different objects received different confidence scores.
4. Annotated images and detection results were saved successfully.

## Conclusion
YOLOv8 Nano performed object detection on sample images
and generated bounding boxes, class labels, and confidence scores.
