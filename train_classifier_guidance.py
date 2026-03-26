import os
import math
import glob
import argparse
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import AutoencoderKL, DDPMScheduler
from tqdm.auto import tqdm
import random

# ==========================================
# 1. DATASET & DATALOADER
# ==========================================
class DIS5KDepthDataset(Dataset):
    def __init__(self, data_root, size=1024):
        self.data_root = data_root
        self.size = size
        
        self.img_dir = os.path.join(data_root, 'DIS5K', 'DIS-TR', 'im')
        self.mask_dir = os.path.join(data_root, 'DIS5K', 'DIS-TR', 'gt')
        self.depth_dir = os.path.join(data_root, 'depths', 'DIS-TR')
        
        self.img_names = [os.path.splitext(f)[0] for f in os.listdir(self.img_dir) if not f.startswith('.')]
        
        self.transform_rgb = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) 
        ])
        
        self.transform_gray = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST), 
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) 
        ])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        name = self.img_names[idx]
        
        img_path = glob.glob(os.path.join(self.img_dir, f"{name}.*"))[0]
        mask_path = glob.glob(os.path.join(self.mask_dir, f"{name}.*"))[0]
        depth_path = glob.glob(os.path.join(self.depth_dir, f"{name}.*"))[0]
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        depth = Image.open(depth_path).convert('L')
        
        if random.random() > 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        
        return {
            "image": self.transform_rgb(image),
            "mask": self.transform_gray(mask),
            "depth": self.transform_gray(depth)
        }

# ==========================================
# 2. MODEL ARCHITECTURE (f_phi)
# ==========================================
def get_timestep_embedding(timesteps, embedding_dim):
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

class LatentDepthPredictor(nn.Module):
    def __init__(self, in_channels=8 + 256, out_channels=4, time_dim=256): 
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.GroupNorm(32, 128), nn.SiLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.GroupNorm(32, 256), nn.SiLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.GroupNorm(32, 128), nn.SiLU(),
            nn.Conv2d(128, out_channels, 3, padding=1)
        )

    def forward(self, z_t, latent_image, timesteps):
        t_emb = get_timestep_embedding(timesteps, 256)
        t_emb = self.time_embed(t_emb) 
        t_map = t_emb[:, :, None, None].repeat(1, 1, z_t.shape[2], z_t.shape[3])
        x = torch.cat([latent_image, z_t, t_map], dim=1) 
        return self.net(x)

def encode_to_latent(vae, tensor):
    if tensor.shape[1] == 1:
        tensor = tensor.repeat(1, 3, 1, 1) 
    latent = vae.encode(tensor).latent_dist.sample()
    return latent * 0.18215

# ==========================================
# 3. TRAINING LOOP
# ==========================================
def train_guidance_model(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load từ local path (pretrained_model) để tiết kiệm thời gian tải trên Colab
    print(f"Loading VAE and Scheduler from {args.model_path}...")
    vae = AutoencoderKL.from_pretrained(args.model_path, subfolder="vae").to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_path, subfolder="scheduler")
    
    vae.enable_slicing()
    
    vae.eval()
    vae.requires_grad_(False) 

    print("Initializing LatentDepthPredictor...")
    f_phi = LatentDepthPredictor().to(device)

    start_epoch = 0
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"🔄 Đang load trọng số cũ từ: {args.resume_from} để train tiếp...")
        f_phi.load_state_dict(torch.load(args.resume_from))
        
        try:
            filename = os.path.basename(args.resume_from)
            start_epoch = int(filename.split('_')[-1].split('.')[0])
            print(f"Sẽ bắt đầu train tiếp từ Epoch {start_epoch + 1}")
        except:
            print("Không xác định được số epoch cũ, bắt đầu tính từ 1 nhưng dùng trọng số cũ.")

    f_phi.train()
    
    optimizer = torch.optim.AdamW(f_phi.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    
    dataset = DIS5KDepthDataset(data_root=args.data_root, size=1024)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    print(f"Bắt đầu Training! Số lượng ảnh: {len(dataset)}")
    print(f"Config: Epochs={args.num_epochs}, Batch Size={args.batch_size}, LR={args.learning_rate}")
    
    best_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        epoch_loss = 0.0
        
        for batch in progress_bar:
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            depth = batch["depth"].to(device)
            
            with torch.no_grad():
                latent_img = encode_to_latent(vae, image)
                latent_mask = encode_to_latent(vae, mask)
                latent_depth = encode_to_latent(vae, depth) 
            
            noise = torch.randn_like(latent_mask)
            bsz = latent_mask.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
            z_t = noise_scheduler.add_noise(latent_mask, noise, timesteps)
            
            optimizer.zero_grad()
            pred_depth_latent = f_phi(z_t, latent_img, timesteps)
            
            loss = F.mse_loss(pred_depth_latent, latent_depth)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"👉 Epoch {epoch+1} hoàn tất | Average Loss: {avg_loss:.4f}")
        
        save_path = os.path.join(args.output_dir, f"f_phi_epoch_{epoch+1}.pth")
        torch.save(f_phi.state_dict(), save_path)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_save_path = os.path.join(args.output_dir, "best_f_phi.pth")
            torch.save(f_phi.state_dict(), best_save_path)
            print(f"Đã cập nhật best_checkpoint với loss kỷ lục mới: {best_loss:.4f}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Latent Depth Predictor for LawDIS Guidance")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=15)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Thư mục lưu model")
    parser.add_argument("--model_path", type=str, default="pretrained_model", help="Đường dẫn tới thư mục chứa SDv2 offline")
    
    args = parser.parse_args()
    train_guidance_model(args)