from pathlib import Path
import pandas as pd

from src.image_inspector import inspect_image


def generate_image_inventory(image_dirs, label_dirs):
    """
    Generate an inventory of images and their YOLO annotations.

    Args:
        image_dirs: Dictionary containing train and validation image folders.
        label_dirs: Dictionary containing train and validation label folders.

    Returns:
        Pandas DataFrame containing image inventory.
    """

    records = []

    for split in image_dirs:

        image_dir = Path(image_dirs[split])
        label_dir = Path(label_dirs[split])

        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        if not label_dir.exists():
            raise FileNotFoundError(f"Label directory not found: {label_dir}")

        for image_path in sorted(image_dir.iterdir()):

            if image_path.suffix.lower() not in {
                ".jpg", ".jpeg", ".png", ".bmp"
            }:
                continue

            details = inspect_image(str(image_path))

            label_path = label_dir / f"{image_path.stem}.txt"

            if label_path.exists():
                content = label_path.read_text(encoding="utf-8").strip()
                object_count = len(content.splitlines()) if content else 0
                annotation_available = True
            else:
                object_count = 0
                annotation_available = False

            records.append({
                "filename": image_path.name,
                "split": split,
                "width": details["width"],
                "height": details["height"],
                "channels": details["channels"],
                "format": details["format"],
                "file_size_kb": details["file_size_kb"],
                "annotation_available": annotation_available,
                "object_count": object_count
            })

    return pd.DataFrame(records)


def save_inventory_report(inventory_df, output_path):
    """Save image inventory as a CSV report."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    inventory_df.to_csv(output_path, index=False)

    return output_path