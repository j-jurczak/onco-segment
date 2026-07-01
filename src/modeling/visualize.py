import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import FIGURES_DIR, MODELS_DIR, PROCESSED_DATA_DIR
from src.data.split import get_data_splits
from src.data.transforms import get_val_transforms
from src.dataset import BUSIDataset
from src.models.factory import create_segmentation_model


def visualize_predictions(architecture="unet", encoder="resnet34", num_images=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = f"{architecture}_{encoder}_best.pth"
    model_path = MODELS_DIR / model_name

    print(f"loading model: {model_path}")
    model = create_segmentation_model(architecture, encoder)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print("getting test data")
    _, _, test_files = get_data_splits(PROCESSED_DATA_DIR)
    test_ds = BUSIDataset(PROCESSED_DATA_DIR, test_files, transform=get_val_transforms())

    test_loader = DataLoader(test_ds, batch_size=num_images, shuffle=True)

    images, masks_true = next(iter(test_loader))
    images = images.to(device)

    print("starting prediction")
    with torch.no_grad():
        outputs = model(images)
        preds = torch.sigmoid(outputs)
        preds = (preds > 0.5).float()

    images = images.cpu().numpy()
    masks_true = masks_true.cpu().numpy()
    preds = preds.cpu().numpy()

    fig, axes = plt.subplots(num_images, 4, figsize=(16, 4 * num_images))

    cols = ['USG', "Doctor's Mask (Ground Truth)", 'AI Mask', 'Overlay (Yellow = Hit)']
    for ax, col in zip(axes[0], cols):
        ax.set_title(col, size=14, fontweight='bold')

    for i in range(num_images):
        img = images[i].transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)

        mask_t = masks_true[i, 0]
        mask_p = preds[i, 0]

        overlay = np.zeros_like(img)
        overlay[mask_t == 1] = [0, 1, 0]
        overlay[mask_p == 1] = [1, 0, 0]
        overlay[(mask_t == 1) & (mask_p == 1)] = [1, 1, 0]

        axes[i, 0].imshow(img)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(mask_t, cmap='gray')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(mask_p, cmap='gray')
        axes[i, 2].axis('off')

        axes[i, 3].imshow(img)
        axes[i, 3].imshow(overlay, alpha=0.4)
        axes[i, 3].axis('off')

    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / f"prediction_grid_{architecture}_{encoder}.png"

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"saved visualization to: {output_path}")
    plt.close()


if __name__ == "__main__":
    visualize_predictions()
