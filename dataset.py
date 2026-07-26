import os

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset, DataLoader

from config import (
    DATASET_PATH,
    SURFACE_VARIABLES,
    PRESSURE_VARIABLES,
    PRESSURE_LEVELS,
    CHANNEL_MEANS,
    CHANNEL_STDS,
    IN_CHANNELS,
    BATCH_SIZE,
    PATCH_SIZE,
)


def create_dataloader(
    train=True,
    limit=None,
    batch_size=BATCH_SIZE,
):

    dataset = (
        TrainDataset()
        if train
        else TestDataset()
    )

    if limit is not None:
        dataset.length = min(limit, dataset.length)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),   # <-- изменение
    )

    return dataset, loader


class BaseDataset(Dataset):

    def __init__(
        self,
        dataset_path=DATASET_PATH,
    ):

        print(f"Opening dataset: {dataset_path}")

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(dataset_path)

        self.ds = xr.open_zarr(
            dataset_path,
            consolidated=False,
        )

        self.length = self.ds.time.size

        self.means = CHANNEL_MEANS.view(-1, 1, 1)
        self.stds = CHANNEL_STDS.view(-1, 1, 1)

        print(f"Samples: {self.length}")

    def __len__(self):
        return self.length

    def load_sample(self, idx):

        channels = []

        # --------------------------
        # Surface variables
        # --------------------------

        for var in SURFACE_VARIABLES:

            data = (
                self.ds[var]
                .isel(time=idx)
                .values
            )

            channels.append(data)

        # --------------------------
        # Pressure variables
        # --------------------------

        for var in PRESSURE_VARIABLES:

            data = (
                self.ds[var]
                .sel(level=PRESSURE_LEVELS)
                .isel(time=idx)
                .values
            )

            channels.extend(data)

        sample = np.stack(channels)

        assert sample.shape[0] == IN_CHANNELS, (
            f"Expected {IN_CHANNELS} channels, "
            f"got {sample.shape[0]}"
        )

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

        # -------------------------------------------------------
        # ERA5 0.25° содержит 721 строку.
        # Для соревнования используем общую сетку 720×1440.
        # -------------------------------------------------------
        sample = sample[:, :720, :]     # <-- оставить

        return sample.contiguous()      # <-- изменение


class TrainDataset(BaseDataset):

    def __getitem__(self, idx):

        sample = self.load_sample(idx)

        height, width = sample.shape[1:]

        top = torch.randint(
            0,
            height - PATCH_SIZE + 1,
            (1,),
        ).item()

        left = torch.randint(
            0,
            width - PATCH_SIZE + 1,
            (1,),
        ).item()

        sample = sample[
            :,
            top:top + PATCH_SIZE,
            left:left + PATCH_SIZE,
        ]

        return sample, sample


class TestDataset(BaseDataset):

    def __getitem__(self, idx):

        sample = self.load_sample(idx)

        sample = sample[:, :720, :].contiguous()

        return sample, sample