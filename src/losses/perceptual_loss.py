from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG19 features.
    Computes the weighted L1 loss between VGG19 feature maps of the input and target images.
    """

    DEFAULT_WEIGHTS = (1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0)

    def __init__(self, weights: Optional[tuple[float, ...]] = None):
        """
        Initializes the PerceptualLoss module.
        Args:
            weights (Optional[tuple[float, ...]]): Weights for each VGG19 layer's loss contribution.
                If None, default weights are used.
        """
        super(PerceptualLoss, self).__init__()
        self.vgg = VGG19(requires_grad=False)
        self.vgg.eval()
        self.criterion = nn.L1Loss()
        self.weights = weights if weights is not None else self.DEFAULT_WEIGHTS
        if len(self.weights) != VGG19.VGG_SLICES:
            raise ValueError(
                f"Weights length must be {VGG19.VGG_SLICES}, got {len(self.weights)}"
            )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = sum(
            w * self.criterion(x_feat, y_feat.detach())
            for w, x_feat, y_feat in zip(self.weights, x_vgg, y_vgg)
        )
        return loss


class VGG19(torch.nn.Module):
    """
    Module with slices of pretrained VGG19 network.
    """

    VGG_SLICES = 5

    def __init__(self, requires_grad=False):
        super(VGG19, self).__init__()

        vgg_features = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features

        self.slice1 = torch.nn.Sequential(*vgg_features[0:2])
        self.slice2 = torch.nn.Sequential(*vgg_features[2:7])
        self.slice3 = torch.nn.Sequential(*vgg_features[7:12])
        self.slice4 = torch.nn.Sequential(*vgg_features[12:21])
        self.slice5 = torch.nn.Sequential(*vgg_features[21:30])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X: torch.Tensor) -> list[torch.Tensor]:
        """
        Forward pass through VGG19 slices.
        Args:
            X (torch.Tensor): Input tensor of shape (B, C, H, W) where:
                C (Channels) is supported as 3 for RGB or 1 for grayscale (converted to 3 channels for VGG19).
        Returns:
            list[torch.Tensor]: List of feature maps from each slice.
        """
        if X.shape[1] == 1:
            X = X.repeat(1, 3, 1, 1)

        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)

        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
