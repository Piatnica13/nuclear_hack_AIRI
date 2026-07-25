import os

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset, DataLoader

from config import (
    DATASET_PATH,
)



def create_dataloader(limit=None, batch_size=1):

    dataset = TestDataset()

    if limit is not None:
        dataset.length = min(limit, dataset.length)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return dataset, loader


class TestDataset(Dataset):

    def __init__(self, dataset_path=DATASET_PATH):

        print("Opening test dataset...")

        print("DATASET_PATH =", dataset_path)
        print("Exists =", os.path.exists(dataset_path))
        print("Contents =", os.listdir(dataset_path)[:10])

        self.ds = xr.open_zarr(
            dataset_path,
            consolidated=False,
        )

        self.surface = [
            "2m_temperature",
            "mean_sea_level_pressure",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "total_precipitation_6hr",
            "sea_surface_temperature",
            "total_column_water_vapour",
            "total_cloud_cover",
        ]

        self.pressure = [
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind",
            "geopotential",
            "specific_humidity",
        ]

        self.levels = [1000, 925, 850, 700]

        self.length = self.ds.time.size

        self.means = torch.tensor([
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
        ]).view(-1, 1, 1)

        self.stds = torch.tensor([
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
        ]).view(-1, 1, 1)

        print(f"Test samples: {self.length}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        channels = []

        for var in self.surface:

            data = (
                self.ds[var]
                .isel(time=idx)
                .values
            )

            channels.append(data)

        for var in self.pressure:

            data = (
                self.ds[var]
                .sel(level=self.levels)
                .isel(time=idx)
                .values
            )

            channels.extend(data)

        sample = np.stack(channels)

        sample = np.nan_to_num(
            sample,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        sample = torch.from_numpy(sample).float()

        sample = (sample - self.means) / self.stds

        sample = torch.nan_to_num(
            sample,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        sample = sample[:, :720, :]

        return sample, sample