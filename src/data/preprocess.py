import os
import shutil
from pathlib import Path

import cv2

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR


def apply_clahe(img_path, save_path):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"cannot read: {img_path}")
        return

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)

    img_bgr = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2BGR)

    cv2.imwrite(str(save_path), img_bgr)


def process_dataset(input_dir: Path, output_dir: Path):
    categories = ["benign", "malignant", "normal"]

    for category in categories:
        cat_in_dir = input_dir / category
        cat_out_dir = output_dir / category

        if not cat_in_dir.exists():
            continue

        cat_out_dir.mkdir(parents=True, exist_ok=True)
        files = os.listdir(cat_in_dir)
        print(f"category '{category}' ({len(files)} files)")

        for file_name in files:
            in_file = cat_in_dir / file_name
            out_file = cat_out_dir / file_name

            if "_mask" in file_name:
                shutil.copy2(in_file, out_file)
            else:
                apply_clahe(in_file, out_file)


if __name__ == "__main__":
    process_dataset(RAW_DATA_DIR, PROCESSED_DATA_DIR)
    print(f"done, data directory: {PROCESSED_DATA_DIR}")
