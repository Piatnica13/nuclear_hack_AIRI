import torch
import torch.nn.functional as F

from config import (
    IMAGE_HEIGHT,
    IN_CHANNELS,
    EPS,
    SURFACE_CHANNELS,
    PRESSURE_CHANNELS,
    CHANNEL_STDS,
)


# =====================================================
# Latitude weights
# =====================================================

latitudes = torch.linspace(-90.0, 90.0, IMAGE_HEIGHT)

LAT_WEIGHTS = torch.cos(torch.deg2rad(latitudes))
LAT_WEIGHTS = LAT_WEIGHTS.clamp(min=0.0)
LAT_WEIGHTS /= LAT_WEIGHTS.mean()


# =====================================================
# Basic metrics
# =====================================================

def mse(pred, target):
    return F.mse_loss(pred, target)


def rmse(pred, target):
    return torch.sqrt(mse(pred, target) + EPS)


def mae(pred, target):
    return F.l1_loss(pred, target)


def psnr(pred, target):
    value = mse(pred, target)
    return 20 * torch.log10(1.0 / torch.sqrt(value + EPS))


# =====================================================
# Helpers
# =====================================================

def _channel_std(device):
    return CHANNEL_STDS.to(device)


# =====================================================
# Latitude weighted NRMSE
# =====================================================

def channel_nrmse(pred, target):

    weights = LAT_WEIGHTS.to(pred.device).view(1, 1, -1, 1)

    error2 = (pred - target) ** 2

    weighted_error = (error2 * weights).sum(dim=(0, 2, 3))
    weighted_norm = (
        weights.sum()
        * pred.shape[0]
        * pred.shape[3]
    )

    weighted_rmse = torch.sqrt(
        weighted_error / weighted_norm + EPS
    )

    sigma = _channel_std(pred.device)

    return weighted_rmse / sigma


def surface_score(pred, target):
    return channel_nrmse(pred, target)[SURFACE_CHANNELS].mean()


def pressure_score(pred, target):
    return channel_nrmse(pred, target)[PRESSURE_CHANNELS].mean()


def overall_score(pred, target):

    return (
        0.5 * surface_score(pred, target)
        + 0.5 * pressure_score(pred, target)
    )


# =====================================================
# Full report
# =====================================================

def metrics(pred, target):

    nrmse = channel_nrmse(pred, target)

    surface = nrmse[SURFACE_CHANNELS].mean()
    pressure = nrmse[PRESSURE_CHANNELS].mean()

    result = {

        "mse": mse(pred, target).item(),
        "rmse": rmse(pred, target).item(),
        "mae": mae(pred, target).item(),
        "psnr": psnr(pred, target).item(),

        "surface_score": surface.item(),
        "pressure_score": pressure.item(),

        "overall_score": (
            0.5 * surface
            + 0.5 * pressure
        ).item(),

        "overall_nrmse": nrmse.mean().item(),
    }

    for i in range(IN_CHANNELS):
        result[f"nrmse_ch_{i:02d}"] = nrmse[i].item()

    return result


def compute_rmse(pred, target):
    return rmse(pred, target).item()


def compute_nrmse(pred, target):
    return channel_nrmse(pred, target).mean().item()


def compute_surface_score(pred, target):
    return surface_score(pred, target).item()


def compute_pressure_score(pred, target):
    return pressure_score(pred, target).item()


def compute_overall_score(pred, target):
    return overall_score(pred, target).item()