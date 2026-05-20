"""
Code to download or generate data
"""

import glob
import os
import shutil
from pathlib import Path

import kagglehub
from PIL import Image
from torch.utils.data import Dataset

from src.config import RAW_DATA_DIR


class BUSIDataset(Dataset):
    def __init__(self, base_dir, filenames, transform=None):
        self.base_dir = Path(base_dir)
        self.filenames = filenames
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.base_dir / self.filenames[idx]

        mask_path = Path(str(img_path).replace(".png", "_mask.png"))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        mask = (mask > 0.5).float()

        return image, mask


def download_data():

    download_path = kagglehub.dataset_download(
        "aryashah2k/breast-ultrasound-images-dataset"
    )
    target_base_dir = RAW_DATA_DIR
    categories = ["benign", "malignant", "normal"]

    for category in categories:
        source_dirs = glob.glob(
            os.path.join(download_path, "**", category), recursive=True
        )

        if not source_dirs:
            print(f"no images for category: {category}")
            continue

        source_dir = source_dirs[0]
        target_dir = os.path.join(target_base_dir, category)

        os.makedirs(target_dir, exist_ok=True)

        for file_name in os.listdir(source_dir):
            source_file = os.path.join(source_dir, file_name)
            target_file = os.path.join(target_dir, file_name)

            if os.path.isfile(source_file):
                shutil.copy2(source_file, target_file)


def main():
    download_data()


if __name__ == "__main__":
    main()
