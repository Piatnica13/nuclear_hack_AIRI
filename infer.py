import os
import time
import numpy as np
import torch
import argparse
from tqdm import tqdm

from config import *
from model import AutoEncoder
from dataset import TestDataset
from metrics import metrics


parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    required=True,
    help="Path to checkpoint"
)

parser.add_argument(
    "--output",
    required=True,
    help="Directory for predictions"
)

args = parser.parse_args()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output, exist_ok=True)

    dataset = TestDataset(DATASET_PATH)

    model = AutoEncoder().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    inference_times = []

    all_pred = []
    all_target = []

    with torch.no_grad():

        N = min(256, len(dataset))

        for idx in tqdm(range(N), desc="Inference"):
        
            x = dataset[idx]
    
            x = x.unsqueeze(0).to(device) # type: ignore
    
            if device.type == "cuda":
                torch.cuda.synchronize()
    
            start = time.perf_counter()
    
            prediction = model(x)
    
            if device.type == "cuda":
                torch.cuda.synchronize()
    
            inference_times.append(time.perf_counter() - start)
    
            prediction.squeeze(0).cpu().numpy().astype(np.float32).tofile(
                os.path.join(args.output, f"{idx:03d}.bin")
            )
    
            all_pred.append(prediction.cpu())
            all_target.append(x.cpu())

    all_pred = torch.cat(all_pred, dim=0)
    all_target = torch.cat(all_target, dim=0)

    result = metrics(all_pred, all_target)

    print("\n==============================")
    print("Inference finished")
    print("==============================")
    print(f"Samples           : {N}")
    print(f"Average time      : {np.mean(inference_times) * 1000:.2f} ms")
    print(f"MSE               : {result['mse']:.8f}")
    print(f"RMSE              : {result['rmse']:.8f}")
    print(f"MAE               : {result['mae']:.8f}")
    print(f"PSNR              : {result['psnr']:.2f}")
    print(f"Surface Score     : {result['surface_score']:.8f}")
    print(f"Pressure Score    : {result['pressure_score']:.8f}")
    print(f"Overall Score     : {result['overall_score']:.8f}")
    print(f"Overall NRMSE     : {result['overall_nrmse']:.8f}")
    print("==============================")


if __name__ == "__main__":
    main()