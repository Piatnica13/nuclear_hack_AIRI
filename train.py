import csv
import os
import time

import mlflow
import torch

from config import (
    LEARNING_RATE,
    EPOCHS,
    MODELS_DIR,
    LOGS_DIR,
    PATCH_SIZE,
)

from dataset import create_dataloader
from metrics import metrics
from model import AutoEncoder, DEVICE

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("ERA5-28ch-AutoEncoder-plus")


def train(limit):

    dataset, loader = create_dataloader(limit)

    model = AutoEncoder().to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-5,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=LEARNING_RATE / 100,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=False,
    )

    criterion = torch.nn.MSELoss()

    os.makedirs(
        LOGS_DIR,
        exist_ok=True,
    )

    csv_path = os.path.join(
        LOGS_DIR,
        f"training_log_{limit}.csv",
    )

    csv_file = open(
        csv_path,
        "w",
        newline="",
    )

    writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "global_step",
            "epoch",
            "batch_in_epoch",
            "train_loss",
            "val_loss_quick",
            "val_rmse_quick",
            "val_nrmse_quick",
        ],
    )

    writer.writeheader()

    quick_images, _ = next(iter(loader))
    quick_images = quick_images.to(
        DEVICE,
        non_blocking=True,
    )

    global_step = 0
    best_score = float("inf")

    with mlflow.start_run():

        mlflow.log_param("dataset_size", len(dataset))
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", loader.batch_size)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("patch_size", PATCH_SIZE)

        first = True

        for epoch in range(EPOCHS):

            start = time.time()

            model.train()

            epoch_loss = 0.0

            for batch_idx, (images, _) in enumerate(loader):

                images = images.to(
                    DEVICE,
                    non_blocking=True,
                )

                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(
                    device_type="cuda",
                    enabled=False,
                ):

                    output = model(images)

                    loss = criterion(
                        output,
                        images,
                    )

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()

                global_step += 1

                if global_step % 50 == 0:

                    model.eval()

                    with torch.no_grad():

                        quick_output = model(quick_images)

                        quick_loss = criterion(
                            quick_output,
                            quick_images,
                        ).item()

                        quick_metrics = metrics(
                            quick_output,
                            quick_images,
                        )

                    writer.writerow(
                        {
                            "global_step": global_step,
                            "epoch": epoch + 1,
                            "batch_in_epoch": batch_idx + 1,
                            "train_loss": loss.item(),
                            "val_loss_quick": quick_loss,
                            "val_rmse_quick": quick_metrics["rmse"],
                            "val_nrmse_quick": quick_metrics["overall_score"],
                        }
                    )

                    csv_file.flush()

                    mlflow.log_metric(
                        "train_loss_step",
                        loss.item(),
                        step=global_step,
                    )

                    mlflow.log_metric(
                        "val_loss_quick",
                        quick_loss,
                        step=global_step,
                    )

                    mlflow.log_metric(
                        "val_rmse_quick",
                        quick_metrics["rmse"],
                        step=global_step,
                    )

                    mlflow.log_metric(
                        "val_nrmse_quick",
                        quick_metrics["overall_score"],
                        step=global_step,
                    )

                    model.train()

            epoch_loss /= len(loader)

            model.eval()

            with torch.no_grad():

                output = model(quick_images)

                result = metrics(
                    output,
                    quick_images,
                )

            if first:

                mlflow.log_param(
                    "input_shape",
                    str(model.input_shape),
                )

                mlflow.log_param(
                    "latent_shape",
                    str(model.latent_shape),
                )

                mlflow.log_param(
                    "compression_ratio",
                    model.compression_ratio,
                )

                first = False

            mlflow.log_metric(
                "loss",
                epoch_loss,
                step=epoch,
            )

            for key, value in result.items():

                mlflow.log_metric(
                    key,
                    value,
                    step=epoch,
                )

            print(
                f"[{epoch + 1:03d}/{EPOCHS}] "
                f"Loss={epoch_loss:.6f} | "
                f"Overall={result['overall_score']:.5f} | "
                f"Surface={result['surface_score']:.5f} | "
                f"Pressure={result['pressure_score']:.5f} | "
                f"LR={scheduler.get_last_lr()[0]:.2e} | "
                f"RMSE={result['rmse']:.5f} | "
                f"PSNR={result['psnr']:.2f} | "
                f"{time.time() - start:.1f}s"
            )

            scheduler.step()

            if result["overall_score"] < best_score:

                best_score = result["overall_score"]

                torch.save(
                    model.state_dict(),
                    os.path.join(
                        MODELS_DIR,
                        "best_autoencoder.pth",
                    ),
                )

        csv_file.close()

        torch.save(
            model.state_dict(),
            os.path.join(
                MODELS_DIR,
                f"autoencoder_{limit}.pth",
            ),
        )


if __name__ == "__main__":

    EXPERIMENTS = [
        512,
    ]

    for limit in EXPERIMENTS:

        print("\n" + "=" * 70)
        print(f"DATASET SIZE: {limit}")
        print("=" * 70)

        train(limit)