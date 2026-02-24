import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import time

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from unet import ConditionalUNet
from train import transform, denoise_image_with_condition

# -----------------------------
# DATASET
# -----------------------------
class CustomFolderDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir, transform=None):
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.transform = transform
        self.filenames = sorted(os.listdir(clean_dir))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        clean_path = os.path.join(self.clean_dir, fname)
        noisy_path = os.path.join(self.noisy_dir, fname)

        clean_img = Image.open(clean_path).convert("L")
        noisy_img = Image.open(noisy_path).convert("L")

        if self.transform:
            clean_img = self.transform(clean_img)
            noisy_img = self.transform(noisy_img)

        return clean_img, noisy_img, fname

# -----------------------------
# UTILITIES
# -----------------------------
def tensor_to_np(tensor):
    np_img = tensor.squeeze().cpu().numpy()
    if np_img.ndim == 3:
        np_img = np_img.transpose(1, 2, 0)
    return (np_img * 0.5 + 0.5).clip(0, 1)  # De-normalize to [0,1]

def save_np_image(np_img, path):
    img_uint8 = (np_img * 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(path)

def calculate_rmse(img1, img2):
    return np.sqrt(np.mean((img1 - img2) ** 2))

def evaluate_metrics(den_np, gt_np):
    """
    Compute SSIM, RMSE, PSNR using numpy images already in [0,1].
    Matches the test file exactly.
    """
    ssim_val = ssim(gt_np, den_np, data_range=1.0, channel_axis=None)
    rmse_val = calculate_rmse(gt_np, den_np)
    psnr_val = psnr(gt_np, den_np, data_range=1.0)
    return ssim_val, rmse_val, psnr_val

# -----------------------------
# MODEL LOADING
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ConditionalUNet()
model = torch.nn.DataParallel(model)
model = model.to(device)

checkpoint = torch.load("unet checkpoints/best_model.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
model.eval()

# -----------------------------
# PATHS
# -----------------------------
clean_folder = "test/clean"
noisy_folder = "test/noisy"
output_folder = "Outputs"
os.makedirs(output_folder, exist_ok=True)

dataset = CustomFolderDataset(clean_folder, noisy_folder, transform=transform)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# -----------------------------
# EVALUATION LOOP
# -----------------------------
total_ssim = 0
total_rmse = 0
total_psnr = 0
total_inference_time = 0.0 
count = 0

print("\nEvaluating and saving denoised outputs...\n")

for clean_img, noisy_img, fname in tqdm(loader):
    clean_img = clean_img.to(device)
    noisy_img = noisy_img.to(device)

    fname = fname[0]

    # -------------------------
    # Measure inference time
    # -------------------------
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()

    with torch.no_grad():
        den = denoise_image_with_condition(model, noisy_img, num_timesteps=8)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()

    inference_time = end_time - start_time
    total_inference_time += inference_time
    # -------------------------

    clean_np = tensor_to_np(clean_img)
    den_np   = tensor_to_np(den)

    # Save denoised image
    save_path = os.path.join(output_folder, fname)
    save_np_image(den_np, save_path)

    # Compute metrics
    ssim_val, rmse_val, psnr_val = evaluate_metrics(den_np, clean_np)

    total_ssim += ssim_val
    total_rmse += rmse_val
    total_psnr += psnr_val
    count += 1

# -----------------------------
# FINAL RESULTS
# -----------------------------
print("\n==================== RESULTS ====================\n")
print(f"Total Images Evaluated: {count}")
print(f"Average SSIM : {total_ssim / count:.4f}")
print(f"Average RMSE : {total_rmse / count:.4f}")
print(f"Average PSNR : {total_psnr / count:.2f} dB")

# -----------------------------
# INFERENCE TIME RESULTS
# -----------------------------
avg_inf_time = total_inference_time / count
print(f"\nTotal Inference Time : {total_inference_time:.4f} seconds")
print(f"Average Inference Time per Image : {avg_inf_time:.4f} seconds")

print(f"\nDenoised images saved to: {output_folder}")
print("\n=================================================\n")
