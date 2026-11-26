import cv2
import numpy as np
import pytest

from src.annotation_visualizer import draw_yolo_annotations


def test_draw_yolo_annotations(tmp_path):
    image_path = tmp_path / "test.jpg"
    label_path = tmp_path / "test.txt"

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(image_path), image)

    label_path.write_text("0 0.5 0.5 0.4 0.4")

    result = draw_yolo_annotations(image_path, label_path)

    assert result.shape == (100, 100, 3)
    assert result.dtype == np.uint8


def test_missing_label_file(tmp_path):
    image_path = tmp_path / "test.jpg"

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(image_path), image)

    with pytest.raises(FileNotFoundError):
        draw_yolo_annotations(
            image_path,
            tmp_path / "missing.txt"
        )


def test_invalid_yolo_annotation(tmp_path):
    image_path = tmp_path / "test.jpg"
    label_path = tmp_path / "test.txt"

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(image_path), image)

    label_path.write_text("invalid annotation")

    with pytest.raises(ValueError):
        draw_yolo_annotations(image_path, label_path)