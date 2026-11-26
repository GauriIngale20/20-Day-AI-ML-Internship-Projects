import cv2
import numpy as np
import pytest

from src.image_inspector import inspect_image


def test_inspect_image(tmp_path):
    image_path = tmp_path / "test_image.jpg"

    test_image = np.zeros((100, 200, 3), dtype=np.uint8)
    cv2.imwrite(str(image_path), test_image)

    result = inspect_image(str(image_path))

    assert result["filename"] == "test_image.jpg"
    assert result["width"] == 200
    assert result["height"] == 100
    assert result["channels"] == 3
    assert result["format"] == "JPEG"
    assert result["color_mode"] == "BGR"


def test_missing_image():
    with pytest.raises(FileNotFoundError):
        inspect_image("missing_image.jpg")