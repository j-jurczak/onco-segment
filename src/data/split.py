import os

from sklearn.model_selection import train_test_split


def get_data_splits(base_dir, val_ratio=0.15, test_ratio=0.15, seed=42):
    all_files = []

    for category in ["benign", "malignant"]:
        cat_dir = base_dir / category
        if cat_dir.exists():
            images = [
                f"{category}/{f}" for f in os.listdir(cat_dir)
                if f.endswith(".png") and "_mask" not in f
            ]
            all_files.extend(images)

    train_val_files, test_files = train_test_split(
        all_files, test_size=test_ratio, random_state=seed, shuffle=True
    )

    val_size_adjusted = val_ratio / (1.0 - test_ratio)

    train_files, val_files = train_test_split(
        train_val_files, test_size=val_size_adjusted, random_state=seed, shuffle=True
    )

    return train_files, val_files, test_files
