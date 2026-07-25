import os

import matplotlib.pyplot as plt
import torch

from dataset import create_dataloader
from metrics import rmse, mae
from model import AutoEncoder, DEVICE

os.makedirs("results_images", exist_ok=True)


def visualize(model_path, limit, samples=2):

    dataset, _ = create_dataloader(limit)

    model = AutoEncoder().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        image, _ = dataset[0]
        model(image.unsqueeze(0).to(DEVICE))

    indices = torch.randperm(len(dataset))[:samples]

    with torch.no_grad():

        for idx in indices:

            image, _ = dataset[idx]

            image_gpu = image.unsqueeze(0).to(
                DEVICE,
                non_blocking=True,
            )

            output_gpu = model(image_gpu)

            batch_rmse = rmse(output_gpu, image_gpu).item()
            batch_mae = mae(output_gpu, image_gpu).item()

            image = image_gpu.squeeze().cpu()
            output = output_gpu.squeeze().cpu()
            error = torch.abs(image - output)

            fig, axes = plt.subplots(
                7,
                12,
                figsize=(28, 18),
            )

            for ch in range(28):

                row = ch // 4
                col = (ch % 4) * 3

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
                axes[row, col].set_title(f"Ch {ch}")
                axes[row, col].axis("off")

                axes[row, col + 1].imshow(
                    recon,
                    cmap="coolwarm",
                    vmin=vmin,
                    vmax=vmax,
                )
                axes[row, col + 1].set_title("Recon")
                axes[row, col + 1].axis("off")

                axes[row, col + 2].imshow(
                    err,
                    cmap="inferno",
                )
                axes[row, col + 2].set_title("Error")
                axes[row, col + 2].axis("off")

            fig.suptitle(
                f"Sample {idx.item()}\n"
                f"Compression: {model.compression_ratio:.2f}x\n"
                f"RMSE={batch_rmse:.4f}    MAE={batch_mae:.4f}",
                fontsize=20,
            )

            plt.tight_layout()

            filename = f"results/sample_{idx.item()}.png"

            plt.savefig(
                filename,
                dpi=250,
                bbox_inches="tight",
            )

            plt.close()

            print(f"Saved -> {filename}")


if __name__ == "__main__":

    visualize(
        model_path="models/autoencoder_512.pth",
        limit=512,
        samples=2,
    )