from typing import Literal

import torch
import torch.nn as nn


class AdversarialLoss(nn.Module):
    """
    Adversarial loss as described in
    https://arxiv.org/abs/1711.10337

    Generator is considered our Video Restoration model and Discriminator is a separate network
    that tries to distinguish between real ground truths frames and generated frames.
    """

    def __init__(
        self,
        loss_type: Literal["nsgan", "lsgan", "hinge"] = "nsgan",
        target_real_label: float = 1.0,
        target_fake_label: float = 0.0,
        generator_weight: float = 1.0,
        discriminator_weight: float = 1.0,
    ):
        """
        Initializes the AdversarialLoss module.
        Args:
            loss_type (str): Type of adversarial loss. One of 'nsgan', 'lsgan', 'hinge'.
            target_real_label (float): Label for real samples
            target_fake_label (float): Label for fake samples
            generator_weight (float): Weight for the generator loss component.
            discriminator_weight (float): Weight for the discriminator loss component.
        """
        super(AdversarialLoss, self).__init__()
        self.loss_type = loss_type
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))

        self.real_label: torch.Tensor
        self.fake_label: torch.Tensor

        criterion_mapping = {
            "nsgan": nn.BCEWithLogitsLoss,
            "lsgan": nn.MSELoss,
            "hinge": nn.ReLU,
        }
        criterion_cls = criterion_mapping.get(loss_type)
        if criterion_cls is None:
            raise ValueError(
                f"Unknown loss type: {loss_type} use one of {list(criterion_mapping.keys())}"
            )
        self.criterion = criterion_cls()
        self.generator_weight = generator_weight
        self.discriminator_weight = discriminator_weight

    def forward(
        self, outputs: torch.Tensor, is_real: bool, is_disc: bool
    ) -> torch.Tensor:
        """
        Calculates loss based on the discriminator outputs.
        Args:
            outputs (torch.Tensor): Discriminator outputs.
            is_real (bool): Whether the targets are real or fake.
            is_disc (bool): Whether the loss is for the discriminator or generator.
        """
        if self.loss_type == "hinge":
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean() * self.discriminator_weight
            else:
                return (-outputs).mean() * self.generator_weight
        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(
                outputs
            )
            loss = self.criterion(outputs, labels)
            if is_disc:
                loss = loss * self.discriminator_weight
            else:
                loss = loss * self.generator_weight
            return loss
