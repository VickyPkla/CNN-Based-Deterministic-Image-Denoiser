from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn as nn
import os
from PIL import Image
from unet import ConditionalUNet
import torch
import matplotlib.pyplot as plt
from loss import DenoisingLoss
import time

# -------------------------------
# Seed & Determinism
# -------------------------------
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -------------------------------
# Dataset
# -------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.clean_dir = os.path.join(root_dir, 'gt')
        self.noisy_dir = os.path.join(root_dir, 'Mixed')
        self.transform = transform
        self.image_filenames = sorted(os.listdir(self.clean_dir))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        filename = self.image_filenames[idx]
        clean_path = os.path.join(self.clean_dir, filename)
        noisy_path = os.path.join(self.noisy_dir, filename)

        clean_image = Image.open(clean_path).convert('L')
        noisy_image = Image.open(noisy_path).convert('L')

        if self.transform:
            clean_image = self.transform(clean_image)
            noisy_image = self.transform(noisy_image)

        return clean_image, noisy_image


# -------------------------------
# DataLoader
# -------------------------------
dataset = CustomDataset(root_dir='train', transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# -------------------------------
# Device setup
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device set: {device}')
losses = []


# -------------------------------
# Add Noise
# -------------------------------
def add_noise(clean_image, noisy_image, alpha):
    return (1 - alpha) * clean_image + alpha * noisy_image


# -------------------------------
# Training Function
# -------------------------------
def train_conditional_ddpm(model, dataloader, num_timesteps=200, num_epochs=1000, lr=2e-4):
    global losses

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    prev_epochs = 0
    checkpoint_dir = "unet checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_loss = float('inf')

    # Try to resume
    load_checkpoint_path = os.path.join("unet checkpoints", "best_model.pth")
    if os.path.exists(load_checkpoint_path):
        print(f"Found existing checkpoint at {load_checkpoint_path}. Loading...")
        checkpoint = torch.load(load_checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        prev_epochs = checkpoint['epoch']
        losses = checkpoint['losses']
        best_loss = min(losses)

        print(f"Resuming from epoch {prev_epochs} with best loss {best_loss:.6f}")
    else:
        print("No previous checkpoint found. Training from scratch.")

    model.train()
    num_epochs = prev_epochs + num_epochs

    for epoch in range(prev_epochs, num_epochs):
        tick = time.time()
        epoch_loss = 0.0
        ssim_loss = 0
        charb_loss = 0

        w_charb = 0.8
        w_ssim = 0.2

        loss_fn = DenoisingLoss(w_charb=w_charb, w_ssim=w_ssim)

        for clean_image, noisy_image in dataloader:
            clean_image = clean_image.to(device)
            noisy_image = noisy_image.to(device)

            t = torch.randint(num_timesteps//2, num_timesteps, (clean_image.shape[0],)).long().to(device)

            alpha = t.float() / num_timesteps
            alpha = alpha.view(-1, 1, 1, 1)

            intermediate_image = add_noise(clean_image, noisy_image, alpha)

            optimizer.zero_grad()

            predicted_clean = model(intermediate_image, t)
            loss = loss_fn(predicted_clean, clean_image)

            ssim_loss += loss_fn.ssim_ind
            charb_loss += loss_fn.charb_ind

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(dataloader)
        avg_ssim_loss = ssim_loss / len(dataloader)
        avg_charb_loss = charb_loss / len(dataloader)
        tock = time.time()

        print(f"Epoch {epoch + 1}/{num_epochs} | LR: {lr:.6e}")
        print(f"Avg Loss: {avg_epoch_loss:.6f} | Avg Charb: {avg_charb_loss:.6f} | Avg SSIM: {avg_ssim_loss:.6f}")


        losses.append(avg_epoch_loss)

        # Save every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'losses': losses
            }, checkpoint_path)
            print(f"Checkpoint saved at: {checkpoint_path} ({tock - tick:.2f}s)")

        # Best model update
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_model_path = os.path.join(checkpoint_dir, f"best_model_{epoch+1}.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'losses': losses
            }, best_model_path)
            print(f"#############   Best model updated with loss: {best_loss:.6f}  ############")


# -------------------------------
# Denoising Function
# -------------------------------
def denoise_image_with_condition(model, input_image, num_timesteps):
    noisy_image = input_image.clone()
    for t in reversed(range(1, num_timesteps+1)):
        alpha_t = t / num_timesteps
        alpha_t_1 = max((t - 1), 0) / num_timesteps
        with torch.no_grad():
            predicted_clean = model(input_image, torch.tensor([t], device=device))
            x_t = (1 - alpha_t) * predicted_clean + alpha_t * noisy_image
            x_t_1 = (1 - alpha_t_1) * predicted_clean + alpha_t_1 * noisy_image
            input_image = input_image - x_t + x_t_1
    return input_image



# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":

    n_gpus = torch.cuda.device_count()
    device_ids = list(range(n_gpus))
    model = ConditionalUNet()
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    model = model.to(device)

    train_conditional_ddpm(model, dataloader, num_timesteps=200, num_epochs=600, lr=2e-4)

    plt.plot(losses)
    plt.title("Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

    print("saving the model...")
    torch.save(model.state_dict(), "complex_unet.pth")
    print("model saved")
