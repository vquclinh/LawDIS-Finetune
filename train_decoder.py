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
import time

from lawdis.lawdis_macro_pipeline import LawDISMacroPipeline


# =========================
# Dataset
# =========================
class SegDataset(Dataset):
    def __init__(self, root_dir, split="DIS-TR", image_size=512, use_depth=False):

        self.image_dir = os.path.join(root_dir, "DIS5K", split, "im")
        self.mask_dir  = os.path.join(root_dir, "DIS5K", split, "gt")

        self.use_depth = use_depth
        if use_depth:
            self.depth_dir = os.path.join(root_dir, "depths", "DIS-TR")

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

        img_path  = os.path.join(self.image_dir, name)
        mask_path = os.path.join(self.mask_dir, base + ".png")

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        img = self.img_tf(img)
        mask = self.mask_tf(mask)

        sample = {"image": img, "mask": mask}

        if self.use_depth:
            depth_path = os.path.join(self.depth_dir, base + ".png")
            depth = Image.open(depth_path)
            depth = np.array(depth).astype(np.float32)

            if depth.max() > 255:
                depth = depth / 65535.0
            else:
                depth = depth / 255.0

            depth = torch.from_numpy(depth).unsqueeze(0)
            depth = self.resize_tf(depth)
            sample["depth"] = depth

        return sample


# =========================
# Utils
# =========================
def gradient_map(x):
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return dx, dy


def dice_score(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    intersection = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))

    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean()


# =========================
# Training
# =========================
def train(args):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    pipeline = LawDISMacroPipeline.from_pretrained(args.model_path)
    pipeline = pipeline.to(device)

    # ================= Freeze / Unfreeze =================
    for p in pipeline.unet.parameters():
        p.requires_grad = False

    for p in pipeline.vae.encoder.parameters():
        p.requires_grad = False

    for p in pipeline.vae.decoder.parameters():
        p.requires_grad = True

    for p in pipeline.vae.post_quant_conv.parameters():
        p.requires_grad = True

    pipeline.unet.eval()
    pipeline.vae.encoder.eval()
    pipeline.vae.decoder.train()
    pipeline.vae.post_quant_conv.train()

    # ================= Dataset =================
    train_dataset = SegDataset(args.data_path, "DIS-TR", args.image_size, use_depth=True)
    val_dataset   = SegDataset(args.data_path, "DIS-VD", args.image_size, use_depth=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"Train images: {len(train_dataset)}")
    print(f"Val images  : {len(val_dataset)}")

    optimizer = optim.Adam(
        list(pipeline.vae.decoder.parameters()) +
        list(pipeline.vae.post_quant_conv.parameters()),
        lr=args.lr
    )

    bce = nn.BCEWithLogitsLoss()

    if hasattr(pipeline, "rgb_latent_scale_factor"):
        scale_factor = pipeline.rgb_latent_scale_factor
    else:
        scale_factor = pipeline.vae.config.scaling_factor

    best_dice = 0.0

    # ================= Training Loop =================
    for epoch in range(args.epochs):

        start_time = time.time()

        train_loss = 0
        train_mask_loss = 0
        train_depth_loss = 0

        pipeline.vae.decoder.train()
        pipeline.vae.post_quant_conv.train()

        for batch in tqdm(train_loader, desc=f"Train Epoch {epoch+1}"):

            rgb = batch["image"].to(device)
            mask_gt = batch["mask"].to(device)
            depth = batch["depth"].to(device)

            with torch.no_grad():
                rgb_latent, features = pipeline.encode_rgb(rgb)

            z = pipeline.vae.post_quant_conv(rgb_latent / scale_factor)
            mask_pred = pipeline.vae.decoder(z, features)

            mask_loss = bce(mask_pred, mask_gt)

            prob = torch.sigmoid(mask_pred)
            dx_m, dy_m = gradient_map(prob)
            dx_d, dy_d = gradient_map(depth)

            depth_loss = (
                (dx_m - dx_d).abs().mean() +
                (dy_m - dy_d).abs().mean()
            )

            loss = mask_loss + args.lambda_depth * depth_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_mask_loss += mask_loss.item()
            train_depth_loss += depth_loss.item()

        train_loss /= len(train_loader)
        train_mask_loss /= len(train_loader)
        train_depth_loss /= len(train_loader)

        # ================= Validation =================
        pipeline.vae.decoder.eval()
        pipeline.vae.post_quant_conv.eval()

        val_loss = 0
        val_dice = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):

                rgb = batch["image"].to(device)
                mask_gt = batch["mask"].to(device)

                rgb_latent, features = pipeline.encode_rgb(rgb)
                z = pipeline.vae.post_quant_conv(rgb_latent / scale_factor)
                mask_pred = pipeline.vae.decoder(z, features)

                loss = bce(mask_pred, mask_gt)

                val_loss += loss.item()
                val_dice += dice_score(mask_pred, mask_gt).item()

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)

        epoch_time = time.time() - start_time

        print("\n==============================")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Time           : {epoch_time:.2f}s")
        print(f"Train Loss     : {train_loss:.6f}")
        print(f"  ├─ Mask Loss : {train_mask_loss:.6f}")
        print(f"  └─ Depth Loss: {train_depth_loss:.6f}")
        print(f"Val Loss       : {val_loss:.6f}")
        print(f"Val Dice       : {val_dice:.6f}")
        print(f"Best Val Dice  : {best_dice:.6f}")
        print("==============================\n")

        # Save every epoch
        torch.save(
            pipeline.vae.state_dict(),
            os.path.join(args.output_path, f"decoder_epoch_{epoch+1}.pt")
        )

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(
                pipeline.vae.state_dict(),
                os.path.join(args.output_path, "best_decoder.pt")
            )
            print("🔥 New BEST model saved!\n")


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