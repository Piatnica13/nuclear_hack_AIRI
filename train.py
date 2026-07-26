import csv
import os
import time

import mlflow
import torch

from tqdm import tqdm

from config import (
    DEVICE,
    LEARNING_RATE,
    EPOCHS,
    MODELS_DIR,
    LOGS_DIR,
    PATCH_SIZE,
    WEIGHT_DECAY,
    MIN_LEARNING_RATE,
    LOG_EVERY_STEPS,
    LOSS_NAME,
    EXPERIMENT_DATASET_SIZES,
)

from dataset import create_dataloader
from losses import get_loss
from metrics import (
    compute_rmse,
    mae,
    psnr,
)
from model import AutoEncoder

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("ERA5-28ch-AutoEncoder-plus")


def train(limit):

    dataset, loader = create_dataloader(train=True, limit=limit)

    model = AutoEncoder().to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=MIN_LEARNING_RATE,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=DEVICE.type == "cuda",
    )

    criterion = get_loss(LOSS_NAME)

    os.makedirs(
        LOGS_DIR,
        exist_ok=True,
    )

    os.makedirs(
        MODELS_DIR,
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
            "val_mae_quick",
        ],
    )

    writer.writeheader()

    monitor_batch, _ = next(iter(loader))

    monitor_batch = monitor_batch.to(
        DEVICE,
        non_blocking=True,
    )

    global_step = 0
    best_score = float("inf")

    with mlflow.start_run():

        mlflow.log_param(
            "dataset_size",
            len(dataset),
        )

        mlflow.log_param(
            "epochs",
            EPOCHS,
        )

        mlflow.log_param(
            "batch_size",
            loader.batch_size,
        )

        mlflow.log_param(
            "learning_rate",
            LEARNING_RATE,
        )

        mlflow.log_param(
            "patch_size",
            PATCH_SIZE,
        )

        mlflow.log_param(
            "loss",
            LOSS_NAME,
        )

        for epoch in range(EPOCHS):

            start = time.time()

            model.train()

            epoch_loss = 0.0

            for batch_idx, (images, _) in enumerate(loader):

                images = images.to(
                    DEVICE,
                    non_blocking=True,
                )

                optimizer.zero_grad(
                    set_to_none=True,
                )

                with torch.autocast(
                    device_type=DEVICE.type,
                    enabled=DEVICE.type == "cuda",
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

                if global_step % LOG_EVERY_STEPS == 0:

                    model.eval()

                    with torch.no_grad():

                        quick_output = model(
                            monitor_batch
                        )

                        quick_loss = criterion(
                            quick_output,
                            monitor_batch,
                        ).item()

                        quick_rmse = compute_rmse(
                            quick_output,
                            monitor_batch,
                        )

                        quick_mae = mae(
                            quick_output,
                            monitor_batch,
                        ).item()

                    writer.writerow(
                        {
                            "global_step": global_step,
                            "epoch": epoch + 1,
                            "batch_in_epoch": batch_idx + 1,
                            "train_loss": loss.item(),
                            "val_loss_quick": quick_loss,
                            "val_rmse_quick": quick_rmse,
                            "val_mae_quick": quick_mae,
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
                        quick_rmse,
                        step=global_step,
                    )

                    mlflow.log_metric(
                        "val_mae_quick",
                        quick_mae,
                        step=global_step,
                    )

                    model.train()

            epoch_loss /= len(loader)

            model.eval()

            with torch.no_grad():

                output = model(
                    monitor_batch,
                )

                result = {
                    "rmse": compute_rmse(
                        output,
                        monitor_batch,
                    ),
                    "mae": mae(
                        output,
                        monitor_batch,
                    ).item(),
                    "psnr": psnr(
                        output,
                        monitor_batch,
                    ).item(),
                }

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
                f"RMSE={result['rmse']:.5f} | "
                f"MAE={result['mae']:.5f} | "
                f"PSNR={result['psnr']:.2f} | "
                f"LR={scheduler.get_last_lr()[0]:.2e} | "
                f"{time.time() - start:.1f}s"
            )

            scheduler.step()

        csv_file.close()

        torch.save(
            model.state_dict(),
            os.path.join(
                MODELS_DIR,
                f"autoencoder_{limit}.pth",
            ),
        )


if __name__ == "__main__":

    for limit in EXPERIMENT_DATASET_SIZES:

        print("\n" + "=" * 70)
        print(f"DATASET SIZE: {limit}")
        print("=" * 70)

        train(limit)