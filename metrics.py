import torch
import torch.nn.functional as F

from config import IMAGE_HEIGHT

EPS = 1e-8

SURFACE = slice(0, 8)
PRESSURE = slice(8, 28)

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

def _channel_std(target):
    """
    sigma_f,train (приближение).
    Вместо train-датасета используется target.
    """

    return target.std(dim=(0, 2, 3)).clamp(min=EPS)


# =====================================================
# Latitude weighted NRMSE
# =====================================================

def channel_nrmse(pred, target):

    weights = LAT_WEIGHTS.to(pred.device).view(1, 1, -1, 1)

    error2 = (pred - target) ** 2

    weighted_error = (error2 * weights).sum(dim=(0, 2, 3))
    weighted_norm = weights.sum() * pred.shape[0] * pred.shape[3]

    weighted_rmse = torch.sqrt(weighted_error / weighted_norm + EPS)

    sigma = _channel_std(target)

    return weighted_rmse / sigma


def surface_score(pred, target):
    return channel_nrmse(pred, target)[SURFACE].mean()


def pressure_score(pred, target):
    return channel_nrmse(pred, target)[PRESSURE].mean()


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

    result = {
        "mse": mse(pred, target).item(),
        "rmse": rmse(pred, target).item(),
        "mae": mae(pred, target).item(),
        "psnr": psnr(pred, target).item(),

        "surface_score": nrmse[SURFACE].mean().item(),
        "pressure_score": nrmse[PRESSURE].mean().item(),
        "overall_score": (
            0.5 * nrmse[SURFACE].mean()
            + 0.5 * nrmse[PRESSURE].mean()
        ).item(),

        "overall_nrmse": nrmse.mean().item(),
    }

    for i in range(28):
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