from pathlib import Path
from src import inference


def test_missing_image_raises_error():
    missing_image = Path("missing_test_image.jpg")

    try:
        inference.run_inference(missing_image)
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        assert True


def test_inference_saves_annotated_image(tmp_path, monkeypatch):

    class FakeResult:
        boxes = []

        def save(self, filename):
            Path(filename).write_text("test image")

    class FakeModel:
        names = {}

        def __init__(self, model_path):
            pass

        def predict(self, **kwargs):
            return [FakeResult()]

    monkeypatch.setattr(inference, "YOLO", FakeModel)
    monkeypatch.setattr(inference, "ANNOTATED_DIR", tmp_path)

    image_path = tmp_path / "sample.jpg"
    image_path.write_text("sample image")

    result = inference.run_inference(image_path)

    output_path = tmp_path / "sample_detected.jpg"

    assert result is not None
    assert output_path.exists()
