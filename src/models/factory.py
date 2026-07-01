import segmentation_models_pytorch as smp
import torch.nn as nn


def create_segmentation_model(architecture_name: str, encoder_name: str) -> nn.Module:
    models_factory = {
        "unet": smp.Unet,
        "unetplusplus": smp.UnetPlusPlus,
        "deeplabv3plus": smp.DeepLabV3Plus,
    }

    arch_lower = architecture_name.lower().replace("+", "plus")

    if arch_lower in models_factory:
        model_class = models_factory[arch_lower]
        return model_class(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )
    else:
        raise ValueError(
            f"no '{architecture_name}' model found, check spelling. "
            f"available architectures are: {list(models_factory.keys())}"
        )
