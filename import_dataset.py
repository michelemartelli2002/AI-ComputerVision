import os
import kaggle

# Requirement: Kaggle API key in ~/.kaggle/kaggle.json
#              pip kaggle package
dataset_id = "zippyz/cats-and-dogs-breeds-classification-oxford-dataset"
out_folder = ".data/oxford-cats-dogs"
os.makedirs(out_folder, exist_ok=True)

print(f"Downloading dataset...")
kaggle.api.dataset_download_files(dataset=dataset_id, path=out_folder, unzip=True, quiet=False)

print("Download and extraction complete.")