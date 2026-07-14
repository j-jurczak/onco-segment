from pathlib import Path

import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.data.split import get_data_splits
from src.data.transforms import get_train_transforms, get_val_transforms
from src.dataset import BUSIDataset
from src.models.factory import create_segmentation_model


def compute_dice_score(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()

    if union == 0:
        return 1.0

    return (2.0 * intersection / union).item()


def train_model(
    architecture: str = "unet",
    encoder: str = "resnet34",
    data_path: Path = PROCESSED_DATA_DIR,
):
    EPOCHS = 10
    BATCH_SIZE = 8
    LR = 1e-4

    model_name = f"{architecture}_{encoder}_best.pth"
    model_save_path = MODELS_DIR / model_name

    train_files, val_files, test_files = get_data_splits(data_path)
    print(f"set divided: {len(train_files)} Train | {len(val_files)} Val | {len(test_files)} Test")

    train_ds = BUSIDataset(data_path, train_files, transform=get_train_transforms())
    val_ds = BUSIDataset(data_path, val_files, transform=get_val_transforms())

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    print("init model")
    model = create_segmentation_model(architecture, encoder)

    device = torch.device("cuda" if torch.cuda.is_available() 
                          else "mps" if torch.mps.is_available()
                          else "cpu")

    model.to(device)

    criterion = smp.losses.DiceLoss(mode="binary", from_logits=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_dice = 0.0
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"starting training on {device}")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        val_dice = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)

                loss = criterion(outputs, masks)
                val_loss += loss.item()

                preds = torch.sigmoid(outputs)
                val_dice += compute_dice_score(preds, masks)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)

        print(
            f"Epoch {epoch+1:02d}/{EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Dice: {avg_val_dice:.4f}"
        )

        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            torch.save(model.state_dict(), model_save_path)
            print(f"   new best: {model_save_path}")


if __name__ == "__main__":
    train_model()
