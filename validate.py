import json
import os
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import (
    DATASET_PATH,
    DEVICE,
    MAX_INFERENCE_SAMPLES,
    RESULTS_DIR,
)
from dataset import TestDataset
from metrics import metrics
from model import AutoEncoder


@torch.no_grad()
def validate(
    checkpoint_path,
    save_bitstream=False,
):
    os.makedirs(
        RESULTS_DIR,
        exist_ok=True,
    )

    dataset = TestDataset(DATASET_PATH)

    model = AutoEncoder().to(DEVICE)

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=DEVICE,
        )
    )

    model.eval()

    inference_times = []
    per_sample = []

    metric_sum = None

    if save_bitstream:
        bitstream_dir = os.path.join(
            RESULTS_DIR,
            "bitstreams",
        )

        os.makedirs(
            bitstream_dir,
            exist_ok=True,
        )

    num_samples = min(
        MAX_INFERENCE_SAMPLES,
        len(dataset),
    )

    for idx in tqdm(
        range(num_samples),
        desc="Validation",
    ):

        x, _ = dataset[idx]

        x = x.unsqueeze(0).to(
            DEVICE,
            non_blocking=True,
        )

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        prediction = model(x)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        inference_times.append(
            time.perf_counter() - start
        )

        pred_cpu = prediction.cpu()
        target_cpu = x.cpu()

        if save_bitstream:
            pred_cpu.squeeze(0).numpy().astype(
                np.float32
            ).tofile(
                os.path.join(
                    bitstream_dir, # type: ignore
                    f"{idx:03d}.bin",
                )
            )

        sample_metrics = metrics(
            pred_cpu,
            target_cpu,
        )

        if metric_sum is None:
            metric_sum = {
                k: 0.0
                for k in sample_metrics.keys()
            }

        for k in metric_sum:
            metric_sum[k] += float(sample_metrics[k])

        per_sample.append(
            {
                "sample": idx,
                "rmse": sample_metrics["rmse"],
                "nrmse": sample_metrics["overall_nrmse"],
                "surface_score": sample_metrics["surface_score"],
                "pressure_score": sample_metrics["pressure_score"],
            }
        )

        del prediction
        del pred_cpu
        del target_cpu
        del x

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    result = {
        k: v / num_samples
        for k, v in metric_sum.items()  # type: ignore
    }

    results = {
        "metrics": {
            "rmse": result["rmse"],
            "nrmse": result["overall_nrmse"],
            "surface_score": result["surface_score"],
            "pressure_score": result["pressure_score"],
            "overall_score": result["overall_score"],
        }
    }

    with open(
        os.path.join(
            RESULTS_DIR,
            "results.json",
        ),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=4,
        )

    pd.DataFrame(
        per_sample,
    ).to_csv(
        os.path.join(
            RESULTS_DIR,
            "per_sample_metrics.csv",
        ),
        index=False,
    )

    print("\n==============================")
    print("Validation finished")
    print("==============================")
    print(f"Samples           : {num_samples}")
    print(
        f"Average time      : "
        f"{np.mean(inference_times) * 1000:.2f} ms"
    )
    print(f"MSE               : {result['mse']:.8f}")
    print(f"RMSE              : {result['rmse']:.8f}")
    print(f"MAE               : {result['mae']:.8f}")
    print(f"PSNR              : {result['psnr']:.2f}")
    print(f"Surface Score     : {result['surface_score']:.8f}")
    print(f"Pressure Score    : {result['pressure_score']:.8f}")
    print(f"Overall Score     : {result['overall_score']:.8f}")
    print(f"Overall NRMSE     : {result['overall_nrmse']:.8f}")
    print("==============================")

    return result


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        help="Path to checkpoint",
    )

    parser.add_argument(
        "--save-bitstream",
        action="store_true",
        help="Save predictions as .bin files",
    )

    args = parser.parse_args()

    validate(
        checkpoint_path=args.model,
        save_bitstream=args.save_bitstream,
    )