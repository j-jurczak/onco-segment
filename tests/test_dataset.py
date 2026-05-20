import pytest
import torch
import torchvision.transforms as T
from PIL import Image

from src.dataset import BUSIDataset

DEFAULT_TRANSFORM = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor()
])


def test_busi_dataset_loading_and_shapes(tmp_path):
    mock_dir = tmp_path / "benign"
    mock_dir.mkdir()

    Image.new("RGB", (100, 100), color="gray").save(mock_dir / "benign (1).png")
    Image.new("L", (100, 100), color="white").save(mock_dir / "benign (1)_mask.png")

    dataset = BUSIDataset(base_dir=tmp_path, filenames=["benign/benign (1).png"], transform=DEFAULT_TRANSFORM)
    img, mask = dataset[0]

    assert isinstance(img, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    assert img.shape == (3, 256, 256)
    assert mask.shape == (1, 256, 256)


def test_dataset_handles_grayscale_source_images(tmp_path):
    mock_dir = tmp_path / "benign"
    mock_dir.mkdir()

    Image.new("L", (100, 100), color=128).save(mock_dir / "benign_gray.png")
    Image.new("L", (100, 100), color=255).save(mock_dir / "benign_gray_mask.png")

    dataset = BUSIDataset(base_dir=tmp_path, filenames=["benign/benign_gray.png"], transform=DEFAULT_TRANSFORM)
    img, _ = dataset[0]

    assert img.shape[0] == 3


def test_dataset_correctly_binarizes_fuzzy_masks(tmp_path):
    mock_dir = tmp_path / "malignant"
    mock_dir.mkdir()

    Image.new("RGB", (100, 100), color="black").save(mock_dir / "tumor.png")

    fuzzy_mask = Image.new("L", (100, 100), color=130)
    fuzzy_mask.save(mock_dir / "tumor_mask.png")

    dataset = BUSIDataset(base_dir=tmp_path, filenames=["malignant/tumor.png"], transform=DEFAULT_TRANSFORM)
    _, mask = dataset[0]

    assert torch.all(mask == 1.0)


def test_dataset_raises_error_when_mask_is_missing(tmp_path):
    mock_dir = tmp_path / "benign"
    mock_dir.mkdir()

    Image.new("RGB", (100, 100)).save(mock_dir / "lost_mask.png")

    dataset = BUSIDataset(base_dir=tmp_path, filenames=["benign/lost_mask.png"], transform=DEFAULT_TRANSFORM)

    with pytest.raises(FileNotFoundError):
        _ = dataset[0]


def test_dataset_empty_behavior():
    dataset = BUSIDataset(base_dir="data/raw", filenames=[], transform=DEFAULT_TRANSFORM)

    assert len(dataset) == 0
    with pytest.raises(IndexError):
        _ = dataset[0]


def test_dataset_handles_completely_black_masks(tmp_path):
    mock_dir = tmp_path / "normal"
    mock_dir.mkdir()

    Image.new("RGB", (100, 100), color="gray").save(mock_dir / "normal (1).png")
    Image.new("L", (100, 100), color="black").save(mock_dir / "normal (1)_mask.png")

    dataset = BUSIDataset(base_dir=tmp_path, filenames=["normal/normal (1).png"], transform=DEFAULT_TRANSFORM)
    _, mask = dataset[0]

    assert torch.sum(mask) == 0.0
