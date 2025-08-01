import torch
import torch.nn as nn

from models import *


class SimCLR(nn.Module):
    """
    SimCLR model for self-supervised contrastive learning on 1D signals.

    Architecture:
      - Backbone encoder (e.g., ResNet1D variants) without final classification head
      - MLP projection head mapping encoder features to contrastive space
      - Optional batch normalization in projection head
      - Downstream linear classifier for fine-tuning/evaluation

    Args:
        backbone: str
            Name of the backbone to use: 'resnet1d18', 'resnet1d34', or 'resnet1d50'.
        projection_dim: int
            Dimension of the projection (hidden and output) in the projection head.
        num_classes: int
            Number of classes for downstream classification.
        input_channels: int
            Number of input channels for the backbone encoder.
        use_batchnorm: bool, default=False
            Whether to include a BatchNorm1d layer after the first projection.
    """
    def __init__(
        self,
        backbone: str,
        projection_dim: int,
        num_classes: int,
        input_channels: int,
        use_batchnorm: bool = False
    ):
        super().__init__()
        # Map backbone string to constructor
        backbones = {
            'resnet1d18': resnet1d18,
            'resnet1d34': resnet1d34,
            'resnet1d50': resnet1d50,
        }
        if backbone not in backbones:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Initialize encoder
        self.encoder = backbones[backbone](num_classes=num_classes, input_channels=input_channels)

        # Save feature dimension
        feat_dim = self.encoder.fc.in_features

        # Remove original classifier
        self.encoder.fc = nn.Identity()

        # Projection head
        layers = [
            nn.Linear(feat_dim, projection_dim),
            nn.BatchNorm1d(projection_dim) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Linear(projection_dim, projection_dim),
        ]
        self.projection_head = nn.Sequential(*layers)

        # Temperature parameter for contrastive loss (learnable log scale)
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/0.07)))

        # Linear classifier for downstream tasks
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for contrastive embeddings.

        Args:
            x: Tensor of shape (batch_size, channels, length)
        Returns:
            z: Tensor of shape (batch_size, projection_dim)
        """
        h = self.encoder(x)
        z = self.projection_head(h)
        return z

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for supervised classification.

        Args:
            x: Tensor of shape (batch_size, channels, length)
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        h = self.encoder(x)
        logits = self.classifier(h)
        return logits


if __name__ == '__main__':
    # Example usage
    model = SimCLR(
        backbone='resnet1d18',
        projection_dim=128,
        num_classes=4,
        input_channels=1,
        use_batchnorm=True
    )
    print(model)
