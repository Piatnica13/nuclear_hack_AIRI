from pathlib import Path
import torch

# ===========================
# PATHS
# ===========================

ROOT_DIR = Path(__file__).resolve().parent

DATASET_PATH = ROOT_DIR / "data" / "era5_28ch_0p25_6h.zarr"

MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"
RESULTS_DIR = ROOT_DIR / "results"
PREDICTIONS_DIR = ROOT_DIR / "predictions" / "predictions_512"

MODEL_PATH = MODELS_DIR / "autoencoder.pth"


# ===========================
# DATASET
# ===========================

LIMIT = None

PATCH_SIZE = 256

RANDOM_CROP = True
HORIZONTAL_FLIP = True

# ===========================
# DATA
# ===========================

IN_CHANNELS = 28

IMAGE_HEIGHT = 720
IMAGE_WIDTH = 1440

SURFACE_VARIABLES = [
    "2m_temperature",
    "mean_sea_level_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_precipitation_6hr",
    "sea_surface_temperature",
    "total_column_water_vapour",
    "total_cloud_cover",
]

PRESSURE_VARIABLES = [
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "geopotential",
    "specific_humidity",
]

PRESSURE_LEVELS = [1000, 925, 850, 700]

# ===========================
# TRAINING
# ===========================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 2
EPOCHS = 5

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

NUM_WORKERS = 8
PIN_MEMORY = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 2

RANDOM_SEED = 42

# ===========================
# MODEL
# ===========================

BASE_CHANNELS = 48
LATENT_CHANNELS = 128



CHANNEL_MEANS = torch.tensor([
    280.0,
    101325.0,
    0.0,
    0.0,
    0.0002,
    290.0,
    25.0,
    0.5,

    270.0, 265.0, 260.0, 250.0,

    0.0, 0.0, 0.0, 0.0,

    0.0, 0.0, 0.0, 0.0,

    40000.0, 42000.0, 45000.0, 48000.0,

    0.004, 0.003, 0.002, 0.001,
])

CHANNEL_STDS = torch.tensor([
    15.0,
    2500.0,
    15.0,
    15.0,
    0.002,
    15.0,
    15.0,
    0.3,

    18.0, 18.0, 18.0, 18.0,

    20.0, 20.0, 20.0, 20.0,

    20.0, 20.0, 20.0, 20.0,

    3000.0, 3000.0, 3000.0, 3000.0,

    0.003, 0.003, 0.003, 0.003,
])


NUM_RESIDUAL_BLOCKS = 5

# ==========================
# Losses
# ==========================

LOSS_MSE_WEIGHT = 1.0
LOSS_MAE_WEIGHT = 0.0

EPS = 1e-8


# ==========================
# Metrics
# ==========================

SURFACE_CHANNELS = slice(0, 8)
PRESSURE_CHANNELS = slice(8, IN_CHANNELS)


# ==========================
# Training
# ==========================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

LOSS_NAME = "mse"

WEIGHT_DECAY = 1e-5

MIN_LEARNING_RATE = LEARNING_RATE / 100

LOG_EVERY_STEPS = 50

EXPERIMENT_DATASET_SIZES = [
    50,
]


# ==========================
# Inference
# ==========================

MAX_INFERENCE_SAMPLES = 64


RESULTS_JSON = "results.json"

PER_SAMPLE_METRICS_CSV = "per_sample_metrics.csv"