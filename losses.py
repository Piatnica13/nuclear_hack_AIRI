import torch
import torch.nn as nn

from config import (
    EPS,
    LOSS_MSE_WEIGHT,
    LOSS_MAE_WEIGHT,
)


class MSELoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, prediction, target):
        return self.loss(prediction, target)


class MAELoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, prediction, target):
        return self.loss(prediction, target)


class RMSELoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.mse = nn.MSELoss()

    def forward(self, prediction, target):
        return torch.sqrt(
            self.mse(prediction, target) + EPS
        )


class CombinedLoss(nn.Module):

    def __init__(
        self,
        mse_weight=LOSS_MSE_WEIGHT,
        mae_weight=LOSS_MAE_WEIGHT,
    ):
        super().__init__()

        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

        self.mse_weight = mse_weight
        self.mae_weight = mae_weight

    def forward(self, prediction, target):

        loss = 0.0

        if self.mse_weight > 0:
            loss += (
                self.mse_weight
                * self.mse(prediction, target)
            )

        if self.mae_weight > 0:
            loss += (
                self.mae_weight
                * self.mae(prediction, target)
            )

        return loss


def get_loss(name="mse", **kwargs):

    name = name.lower()

    if name == "mse":
        return MSELoss()

    if name == "mae":
        return MAELoss()

    if name == "rmse":
        return RMSELoss()

    if name == "combined":
        return CombinedLoss(**kwargs)

    raise ValueError(f"Unknown loss: {name}")