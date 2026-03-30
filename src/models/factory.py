from torchvision import models
from torch import nn

model_mapping = {
    "densenet121": (
        models.densenet121,
        {"weights": models.DenseNet121_Weights.DEFAULT, "family": "densenet"},
    ),
    "densenet161": (
        models.densenet161,
        {"weights": models.DenseNet161_Weights.DEFAULT, "family": "densenet"},
    ),
    "densenet169": (
        models.densenet169,
        {"weights": models.DenseNet169_Weights.DEFAULT, "family": "densenet"},
    ),
    "densenet201": (
        models.densenet201,
        {"weights": models.DenseNet201_Weights.DEFAULT, "family": "densenet"},
    ),
    "resnet50": (
        models.resnet50,
        {"weights": models.ResNet50_Weights.IMAGENET1K_V2, "family": "resnet"},
    ),
    "resnet101": (
        models.resnet101,
        {"weights": models.ResNet101_Weights.IMAGENET1K_V2, "family": "resnet"},
    ),
    "resnet152": (
        models.resnet152,
        {"weights": models.ResNet152_Weights.IMAGENET1K_V2, "family": "resnet"},
    ),
    "vit-b-16": (
        models.vit_b_16,
        {"weights": models.ViT_B_16_Weights.DEFAULT, "family": "vit"},
    ),
    "vit-b-32": (
        models.vit_b_32,
        {"weights": models.ViT_B_32_Weights.DEFAULT, "family": "vit"},
    ),
    "swin-t": (
        models.swin_t,
        {"weights": models.Swin_T_Weights.DEFAULT, "family": "swin"},
    ),
    "swin-s": (
        models.swin_s,
        {"weights": models.Swin_S_Weights.DEFAULT, "family": "swin"},
    ),
    "swin-b": (
        models.swin_b,
        {"weights": models.Swin_B_Weights.DEFAULT, "family": "swin"},
    ),
    "swin-v2-b": (
        models.swin_v2_b,
        {"weights": models.Swin_V2_B_Weights.DEFAULT, "family": "swin"},
    ),
    "efficientnet-b0": (    # 224x224 input size
        models.efficientnet_b0,
        {"weights": models.EfficientNet_B0_Weights.DEFAULT, "family": "efficientnet"},
    ),
    "efficientnet-b1": (    # 240x240 input size
        models.efficientnet_b1,
        {"weights": models.EfficientNet_B1_Weights.DEFAULT, "family": "efficientnet"},
    ),
    "efficientnet-b2": (    # 260x260 input size
        models.efficientnet_b2,
        {"weights": models.EfficientNet_B2_Weights.DEFAULT, "family": "efficientnet"},
    ),
    "efficientnet-b3": (    # 300x300 input size
        models.efficientnet_b3,
        {"weights": models.EfficientNet_B3_Weights.DEFAULT, "family": "efficientnet"},
    ),
    "efficientnet-b4": (    # 380x380 input size
        models.efficientnet_b4,
        {"weights": models.EfficientNet_B4_Weights.DEFAULT, "family": "efficientnet"},
    ),
    "efficientnet-b5": (    # 456x456 input size
        models.efficientnet_b5,
        {"weights": models.EfficientNet_B5_Weights.DEFAULT, "family": "efficientnet"},
    ),
    "efficientnet-b6": (    # 528x528 input size
        models.efficientnet_b6,
        {"weights": models.EfficientNet_B6_Weights.DEFAULT, "family": "efficientnet"},
    ),
    "efficientnet-b7": (    # 600x600 input size
        models.efficientnet_b7,
        {"weights": models.EfficientNet_B7_Weights.DEFAULT, "family": "efficientnet"},
    ),
    "efficientnet-v2-s": (  # 384x384 input size
        models.efficientnet_v2_s,
        {"weights": models.EfficientNet_V2_S_Weights.DEFAULT, "family": "efficientnet"},
    ),
    "efficientnet-v2-m": (  # 480x480 input size
        models.efficientnet_v2_m,
        {"weights": models.EfficientNet_V2_M_Weights.DEFAULT, "family": "efficientnet"},
    ),
    "efficientnet-v2-l": (  # 480x480 input size
        models.efficientnet_v2_l,
        {"weights": models.EfficientNet_V2_L_Weights.DEFAULT, "family": "efficientnet"},
    ),
    # Add more models as needed with their respective configurations.
}


class Model(nn.Module):
    """Moodel definition."""

    def __init__(self, model_name: str, num_classes: int, freeze_backbone: bool = True):
        """
        Initialize Model instance.

        Args:
            model_name (str): Name of the model architecture.
            num_classes (int): Number of output classes.
            freeze_backbone (bool): If True, freeze pretrained weights. If False, train the whole model.
        """
        super(Model, self).__init__()

        model_class, model_config = model_mapping[model_name]
        self.model = model_class(weights=model_config["weights"])

        # Optionally freeze backbone parameters
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        in_features = self._get_in_features(model_config["family"])

        if model_config["family"] == "densenet":
            self.model.classifier = self._create_classifier(in_features, num_classes)
        elif model_config["family"] == "resnet":
            self.model.fc = self._create_classifier(in_features, num_classes)
        elif model_config["family"] == "vit":
            self.model.heads = self._create_classifier(in_features, num_classes)
        elif model_config["family"] == "swin":
            self.model.head = self._create_classifier(in_features, num_classes)
        elif model_config["family"] == "efficientnet":
            self.model.classifier = self._create_classifier(in_features, num_classes)

    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)

    def _get_in_features(self, family: str) -> int:
        """Return the number of input features for the classifier."""
        if family == "densenet":
            return self.model.classifier.in_features
        elif family == "resnet":
            return self.model.fc.in_features
        elif family == "vit":
            return self.model.heads.head.in_features
        elif family == "swin":
            return self.model.head.in_features
        elif family == "efficientnet":
            return self.model.classifier[1].in_features

    def _create_classifier(self, in_features: int, num_classes: int) -> nn.Sequential:
        """Create the classifier module."""
        return nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )


class ModelFactory:
    """
    Factory for creating different models based on their names.

    Args:
        name (str): The name of the model factory.
        num_classes (int): The number of output classes.

    Raises:
        ValueError: If the specified model factory is not implemented.
    """

    def __init__(self, name: str, num_classes: int, freeze_backbone: bool = True):
        """
        Initialize ModelFactory instance.

        Args:
            name (str): The name of the model.
            num_classes (int): The number of output classes.
            freeze_backbone (bool): Whether to freeze backbone weights.
        """
        self.name = name
        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone

    def __call__(self):
        """
        Create a model instance based on the provided name.

        Args:
            model_name (str): Name of the model architecture.
            num_classes (int): Number of output classes.

        Returns:
            Model: An instance of the selected model.
        """
        if self.name not in model_mapping:
            valid_options = ", ".join(model_mapping.keys())
            raise ValueError(
                f"Invalid model name: '{self.name}'. Available options: {valid_options}"
            )

        return Model(self.name, self.num_classes, self.freeze_backbone)


if __name__ == "__main__":
    model = ModelFactory("resnet50", 5)()