from pathlib import Path
import cv2


def draw_yolo_annotations(image_path, label_path):
    """
    Draw YOLO bounding boxes on an image.

    Returns:
        Annotated image in BGR format.
    """

    image_path = Path(image_path)
    label_path = Path(label_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not label_path.exists():
        raise FileNotFoundError(f"Label not found: {label_path}")

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Unable to load image: {image_path}")

    height, width = image.shape[:2]

    lines = label_path.read_text(encoding="utf-8").strip().splitlines()

    for line_number, line in enumerate(lines, start=1):

        values = line.split()

        if len(values) != 5:
            raise ValueError(
                f"Invalid annotation at line {line_number}"
            )

        try:
            class_id = int(values[0])
            x_center, y_center, box_width, box_height = map(
                float, values[1:]
            )
        except ValueError as error:
            raise ValueError(
                f"Invalid values at line {line_number}"
            ) from error

        x_center *= width
        y_center *= height
        box_width *= width
        box_height *= height

        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)
        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            image,
            f"Class {class_id}",
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    return image