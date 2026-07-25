from pathlib import Path
import torch

# ===========================
# PATHS
# ===========================

ROOT_DIR = Path(__file__).resolve().parent

DATASET_PATH = ROOT_DIR / "data"/ "era5_28ch_0p25_6h.zarr"

MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"
RESULTS_DIR = ROOT_DIR / "results"
PREDICTIONS_DIR = ROOT_DIR / "predictions"

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

IMAGE_HEIGHT = 721
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
EPOCHS = 20

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

BASE_CHANNELS = 32
LATENT_CHANNELS = 256