import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import *
from dataset import TestDataset
from metrics import (
    compute_rmse,
    compute_nrmse,
    compute_surface_score,
    compute_pressure_score,
)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    dataset = TestDataset(DATASET_PATH)

    rmse_list = []
    nrmse_list = []
    surface_list = []
    pressure_list = []

    per_sample = []

    for idx in tqdm(range(len(dataset)), desc="Evaluation"):
        _, target = dataset[idx]

        prediction = np.fromfile(
            os.path.join(PREDICTIONS_DIR, f"{idx:03d}.bin"),
            dtype=np.float32,
        ).reshape(target.shape)

        target = target.numpy()

        rmse = compute_rmse(prediction, target)
        nrmse = compute_nrmse(prediction, target)
        surface = compute_surface_score(prediction, target)
        pressure = compute_pressure_score(prediction, target)

        rmse_list.append(rmse)
        nrmse_list.append(nrmse)
        surface_list.append(surface)
        pressure_list.append(pressure)

        per_sample.append(
            {
                "sample": idx,
                "rmse": rmse,
                "nrmse": nrmse,
                "surface_score": surface,
                "pressure_score": pressure,
            }
        )

    overall_rmse = float(np.mean(rmse_list))
    overall_nrmse = float(np.mean(nrmse_list))
    overall_surface = float(np.mean(surface_list))
    overall_pressure = float(np.mean(pressure_list))
    overall_score = 0.5 * overall_surface + 0.5 * overall_pressure

    results = {
        "metrics": {
            "rmse": overall_rmse,
            "nrmse": overall_nrmse,
            "surface_score": overall_surface,
            "pressure_score": overall_pressure,
            "overall_score": overall_score,
        }
    }

    with open(
        os.path.join(RESULTS_DIR, "results.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(results, f, indent=4)

    pd.DataFrame(per_sample).to_csv(
        os.path.join(RESULTS_DIR, "per_sample_metrics.csv"),
        index=False,
    )

    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    main()