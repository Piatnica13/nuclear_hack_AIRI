import math
import os

import matplotlib.pyplot as plt
import torch

from config import DEVICE
from dataset import create_dataloader
from metrics import rmse, mae, psnr
from model import AutoEncoder

RESULT_DIR = "results"

os.makedirs(
    RESULT_DIR,
    exist_ok=True,
)


def visualize(model_path, limit, samples=2):

    dataset, _ = create_dataloader(limit)

    model = AutoEncoder().to(DEVICE)

    state_dict = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(state_dict)

    model.eval()

    indices = torch.randperm(
        len(dataset)
    )[:samples]

    with torch.no_grad():

        for idx in indices:

            sample = dataset[idx]

            image = sample[0]

            image_gpu = image.unsqueeze(0).to(
                DEVICE,
                non_blocking=True,
            )

            output_gpu = model(image_gpu)

            batch_rmse = rmse(
                output_gpu,
                image_gpu,
            ).item()

            batch_mae = mae(
                output_gpu,
                image_gpu,
            ).item()

            batch_psnr = psnr(
                output_gpu,
                image_gpu,
            ).item()

            image = image_gpu.squeeze().cpu()
            output = output_gpu.squeeze().cpu()

            error = torch.abs(
                image - output
            )

            channels = image.shape[0]

            channels_per_row = 4
            columns_per_channel = 3

            rows = math.ceil(
                channels / channels_per_row
            )

            cols = (
                channels_per_row
                * columns_per_channel
            )

            fig, axes = plt.subplots(
                rows,
                cols,
                figsize=(
                    cols * 2.5,
                    rows * 2.8,
                ),
            )

            if rows == 1:
                axes = axes.reshape(
                    1,
                    -1,
                )

            for ch in range(channels):

                row = ch // channels_per_row

                col = (
                    ch % channels_per_row
                ) * columns_per_channel

                original = image[ch]

                recon = output[ch]

                err = error[ch]

                vmin = min(
                    original.min().item(),
                    recon.min().item(),
                )

                vmax = max(
                    original.max().item(),
                    recon.max().item(),
                )

                axes[row, col].imshow(
                    original,
                    cmap="coolwarm",
                    vmin=vmin,
                    vmax=vmax,
                )
                axes[row, col].set_title(
                    f"Ch {ch}",
                    fontsize=9,
                )
                axes[row, col].axis("off")

                axes[row, col + 1].imshow(
                    recon,
                    cmap="coolwarm",
                    vmin=vmin,
                    vmax=vmax,
                )
                axes[row, col + 1].set_title(
                    "Recon",
                    fontsize=9,
                )
                axes[row, col + 1].axis("off")

                im = axes[row, col + 2].imshow(
                    err,
                    cmap="inferno",
                )
                axes[row, col + 2].set_title(
                    "Error",
                    fontsize=9,
                )
                axes[row, col + 2].axis("off")

                fig.colorbar(
                    im,
                    ax=axes[row, col + 2],
                    fraction=0.046,
                    pad=0.04,
                )

            fig.suptitle(
                f"Sample {idx.item()}\n"
                f"Patch size: {image.shape[1]} x {image.shape[2]}\n"
                f"RMSE = {batch_rmse:.5f}    "
                f"MAE = {batch_mae:.5f}    "
                f"PSNR = {batch_psnr:.2f} dB",
                fontsize=18,
            )

            plt.tight_layout()

            filename = os.path.join(
                RESULT_DIR,
                f"sample_{idx.item()}.png",
            )

            plt.savefig(
                filename,
                dpi=250,
                bbox_inches="tight",
            )

            plt.close(fig)

            print(
                f"Saved -> {filename}"
            )


if __name__ == "__main__":

    visualize(
        model_path="models/autoencoder_50.pth",
        limit=512,
        samples=2,
    )