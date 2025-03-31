import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm  # For progress bars
import wandb    
import urllib.request

def evaluate_ood(model, distortion_name, severity, CONFIG):
    data_dir = CONFIG["ood_dir"]
    device = CONFIG["device"]

    # Load the OOD images
    images = np.load(os.path.join(data_dir, f"{distortion_name}.npy"))

    # Select the subset of images for the given severity
    start_index = (severity - 1) * 10000
    end_index = severity * 10000
    images = images[start_index:end_index]

    # Convert to PyTorch tensors and create DataLoader
    images = torch.from_numpy(images).float() / 255.0  # Normalize to [0, 1]
    images = images.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
    dataset = TensorDataset(images)
    dataloader = DataLoader(
        dataset, 
        batch_size=CONFIG["batch_size"], 
        shuffle=False, 
        num_workers=CONFIG["num_workers"], 
        pin_memory=True)

    # Normalize after converting to tensor
    normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    
    predictions = []  # Store predictions
    with torch.no_grad():
        for inputs in tqdm(dataloader, desc=f"Evaluating {distortion_name} (Severity {severity})", leave=False):
           inputs = inputs[0]
           inputs = normalize(inputs) # Apply normalization
           inputs = inputs.to(device)

           outputs = model(inputs)
           _, predicted = outputs.max(1)
           predictions.extend(predicted.cpu().numpy())
    return predictions

# Check if the files are already downloaded
def files_already_downloaded(directory, num_files):
    for i in range(num_files):
        file_name = f"distortion{i:02d}.npy"
        file_path = os.path.join(directory, file_name)
        if not os.path.exists(file_path):
            return False
    return True


def evaluate_ood_test(model, CONFIG):
    data_dir = CONFIG["ood_dir"]
    device = CONFIG["device"]

    num_files = 19  # Number of files to download

    # Only download if files aren't already downloaded
    if not files_already_downloaded(data_dir, num_files):
        # Create the directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)

        # Base URL for the files
        base_url = "https://github.com/DL4DS/ood-test-files/raw/refs/heads/main/ood-test/"

        # Download files distortion00.npy to distortion18.npy
        for i in range(num_files):
            file_name = f"distortion{i:02d}.npy"
            file_url = base_url + file_name
            file_path = os.path.join(data_dir, file_name)

            print(f"Downloading {file_name}...")
            urllib.request.urlretrieve(file_url, file_path)
            print(f"Downloaded {file_name} to {file_path}")

        print("All files downloaded successfully.")
    else:
        print("All files are already downloaded.")

    distortions = [f"distortion{str(i).zfill(2)}" for i in range(19)]

    all_predictions = []  # Store all predictions for the submission file

    model.eval()  # Ensure model is in evaluation mode
    for distortion in distortions:
        for severity in range(1, 6):
            predictions = evaluate_ood(model, distortion, severity, CONFIG)
            all_predictions.extend(predictions)  # Accumulate predictions
            print(f"{distortion} (Severity {severity})")

    return all_predictions

def create_ood_df(all_predictions):

    distortions = [f"distortion{str(i).zfill(2)}" for i in range(19)]
    
    # --- Create Submission File (OOD) ---
    # Create IDs for OOD (assuming the order is as evaluated)
    ids_ood = []
    for distortion in distortions:
        for severity in range(1, 6):
            for i in range(10000):
              ids_ood.append(f"{distortion}_{severity}_{i}")

    submission_df_ood = pd.DataFrame({'id': ids_ood, 'label': all_predictions})
    return submission_df_ood

