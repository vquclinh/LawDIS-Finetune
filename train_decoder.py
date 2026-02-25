import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import argparse

from diffusers_lawdis.lawdis.lawdis_macro_pipeline import LawDISMacroPipeline


# =========================
# Dataset for DIS-TR
# =========================
class SegDepthDataset(Dataset):
    def __init__(self, root_dir, image_size=512):

        self.image_dir = os.path.join(root_dir, "DIS5K", "DIS-TR", "im")
        self.mask_dir  = os.path.join(root_dir, "DIS5K", "DIS-TR", "gt")
        self.depth_dir = os.path.join(root_dir, "depths")

        self.images = sorted(os.listdir(self.image_dir))
        self.image_size = image_size

        self.img_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

        self.mask_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

        self.resize_tf = transforms.Resize((image_size, image_size))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        name = self.images[idx]
        base = os.path.splitext(name)[0]

        img_path   = os.path.join(self.image_dir, name)
        mask_path  = os.path.join(self.mask_dir, base + ".png")
        depth_path = os.path.join(self.depth_dir, base + ".png")

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        depth = Image.open(depth_path)

        img = self.img_tf(img)
        mask = self.mask_tf(mask)

        depth = np.array(depth).astype(np.float32)

        # Nếu depth là 16-bit
        if depth.max() > 255:
            depth = depth / 65535.0
        else:
            depth = depth / 255.0

        depth = torch.from_numpy(depth).unsqueeze(0)
        depth = self.resize_tf(depth)

        return {
            "image": img,
            "mask": mask,
            "depth": depth
        }


# =========================
# Gradient function
# =========================
def gradient_map(x):
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return dx, dy


# =========================
# Training
# =========================
def train(args):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipeline = LawDISMacroPipeline.from_pretrained(args.model_path)
    pipeline = pipeline.to(device)

    # Freeze UNet
    for p in pipeline.unet.parameters():
        p.requires_grad = False

    # Freeze encoder
    for p in pipeline.vae.encoder.parameters():
        p.requires_grad = False

    # Train decoder + post_quant_conv
    for p in pipeline.vae.decoder.parameters():
        p.requires_grad = True

    for p in pipeline.vae.post_quant_conv.parameters():
        p.requires_grad = True

    dataset = SegDepthDataset(args.data_path, args.image_size)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    optimizer = optim.Adam(
        list(pipeline.vae.decoder.parameters()) +
        list(pipeline.vae.post_quant_conv.parameters()),
        lr=args.lr
    )

    bce = nn.BCEWithLogitsLoss()

    # scaling factor (important)
    if hasattr(pipeline, "rgb_latent_scale_factor"):
        scale_factor = pipeline.rgb_latent_scale_factor
    else:
        scale_factor = pipeline.vae.config.scaling_factor

    for epoch in range(args.epochs):

        pipeline.train()
        total_loss = 0

        for batch in tqdm(loader):

            rgb = batch["image"].to(device)
            mask_gt = batch["mask"].to(device)
            depth = batch["depth"].to(device)

            # Encode (no grad because encoder frozen)
            with torch.no_grad():
                rgb_latent, features = pipeline.encode_rgb(rgb)

            # Decode
            z = pipeline.vae.post_quant_conv(
                rgb_latent / scale_factor
            )

            mask_pred = pipeline.vae.decoder(z, features)

            # Mask loss
            mask_loss = bce(mask_pred, mask_gt)

            # Depth boundary loss
            prob = torch.sigmoid(mask_pred)

            dx_m, dy_m = gradient_map(prob)
            dx_d, dy_d = gradient_map(depth)

            depth_loss = (
                (dx_m - dx_d).abs().mean() +
                (dy_m - dy_d).abs().mean()
            )
         
            print(
                f"mask: {mask_loss.item():.4f} | "
                f"depth: {depth_loss.item():.4f}"
            )

            loss = mask_loss + args.lambda_depth * depth_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {total_loss/len(loader):.6f}")

        torch.save(
            pipeline.vae.state_dict(),
            os.path.join(args.output_path, f"decoder_epoch_{epoch+1}.pt")
        )


# =========================
# Main
# =========================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="./checkpoints")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda_depth", type=float, default=0.3)

    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    train(args)