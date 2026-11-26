from src.image_loader import load_image, get_image_files
from src.image_inspector import inspect_image


def main():
    image_folder = "data/coco8/images/train"

    images = get_image_files(image_folder)

    print("=" * 50)
    print("DAY 11 - COMPUTER VISION FOUNDATIONS")
    print("=" * 50)

    print(f"\nTotal training images: {len(images)}")

    for image_path in images:
        image = load_image(str(image_path))
        details = inspect_image(str(image_path))

        print("\nImage:", details["filename"])
        print("Dimensions:", details["width"], "x", details["height"])
        print("Channels:", details["channels"])
        print("Color Mode:", details["color_mode"])
        print("PIL Mode:", details["pil_mode"])
        print("Format:", details["format"])
        print("File Size:", details["file_size_kb"], "KB")


if __name__ == "__main__":
    main()
