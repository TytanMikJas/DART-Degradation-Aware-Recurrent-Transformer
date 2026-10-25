import torch
import torch.nn as nn

from src.losses.adversarial_loss import AdversarialLoss

MultiLossResult = tuple[torch.Tensor, dict[str, torch.Tensor]]


class WeightedMultiLoss(nn.Module):
    """
    Combines multiple loss functions with specified weights.
    AdversarialLoss should not be included here. Use a dedicated model class as it requires
    special handling.
    """

    def __init__(self, loss_functions: list[nn.Module], weights: list[float]):
        """
        Initializes the WeightedMultiLoss module.
        Args:
            loss_functions (list[nn.Module]): List of loss function instances.
            weights (list[float]): Corresponding weights for each loss function.
        """
        super(WeightedMultiLoss, self).__init__()
        if len(loss_functions) != len(weights):
            raise ValueError("Loss functions and weights must have the same length.")

        for loss_fn in loss_functions:
            if not isinstance(loss_fn, nn.Module):
                raise ValueError("All loss functions must be instances of nn.Module.")
            if isinstance(loss_fn, AdversarialLoss):
                raise ValueError(
                    "AdversarialLoss should not be included in WeightedMultiLoss."
                    "Use a dedicated model class for adversarial training."
                )
        self.loss_functions = nn.ModuleList(loss_functions)
        self.fn_names = [type(loss_fn).__name__ for loss_fn in loss_functions]
        self.weights = weights

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> MultiLossResult:
        """
        Computes the weighted sum of the individual losses.
        Args:
            outputs (torch.Tensor): Model predictions.
            targets (torch.Tensor): Ground truth targets.
        Returns:
            total_loss (torch.Tensor): The combined weighted loss.
            loss_dict (dict[str, torch.Tensor]): Dictionary of unweighted individual loss values.
        """
        total_loss = 0.0
        loss_dict: dict[str, torch.Tensor] = {}

        for name, loss_fn, weight in zip(
            self.fn_names, self.loss_functions, self.weights
        ):
            loss_value = loss_fn(outputs, targets)
            total_loss += weight * loss_value
            loss_dict[name] = loss_value

        return total_loss, loss_dict  # type: ignore
