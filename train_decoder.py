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
import wandb
from dotenv import load_dotenv

from lawdis.lawdis_macro_pipeline import LawDISMacroPipeline

Image.MAX_IMAGE_PIXELS = None

load_dotenv("/content/drive/MyDrive/lawdis/.env")


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

        self.images     = sorted(os.listdir(self.image_dir))
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

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            name = self.images[idx]
            base = os.path.splitext(name)[0]

            img = Image.open(os.path.join(self.image_dir, name))
            img.draft('RGB', (self.image_size, self.image_size))
            img = img.convert("RGB")

            mask = Image.open(os.path.join(self.mask_dir, base + ".png")).convert("L")

            img  = self.img_tf(img)
            mask = self.mask_tf(mask)

            sample = {"image": img, "mask": mask}

            if self.use_depth:
                depth = Image.open(
                    os.path.join(self.depth_dir, base + ".png")
                ).convert("I")
                depth = depth.resize(
                    (self.image_size, self.image_size),
                    Image.Resampling.BILINEAR
                )
                depth = np.array(depth, dtype=np.float32)
                if depth.max() > 255:
                    depth /= 65535.0
                else:
                    depth /= 255.0
                depth = torch.from_numpy(depth).unsqueeze(0)
                sample["depth"] = depth

            return sample

        except Exception as e:
            print(f"⚠️ Skipping image {self.images[idx]}: {e}")
            return None


# =========================
# Safe Collate
# =========================
def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


# =========================
# Utils
# =========================
def gradient_map(x):
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return dx, dy


# [FIX] Soft Dice — not use hard threshold (pred > 0.5)
def dice_score(pred, target, eps=1e-6):
    pred         = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union        = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2 * intersection + eps) / (union + eps)).mean()


# =========================
# Sanitization helpers
# =========================
def sanitize_features(rgb_latent, features, latent_clamp=4.0, feat_clamp=10.0):
    rgb_latent = torch.nan_to_num(rgb_latent, nan=0.0,
                                  posinf=latent_clamp, neginf=-latent_clamp)
    rgb_latent = torch.clamp(rgb_latent, -latent_clamp, latent_clamp)

    clean = []
    for f in features:
        f = torch.nan_to_num(f, nan=0.0, posinf=feat_clamp, neginf=-feat_clamp)
        f = torch.clamp(f, -feat_clamp, feat_clamp)
        clean.append(f)
    return rgb_latent, clean

def sanitize_logit(t, clamp_val=10.0):
    t = torch.nan_to_num(t, nan=-clamp_val, posinf=clamp_val, neginf=-clamp_val)
    t = torch.clamp(t, -clamp_val, clamp_val)
    return t


# =========================
# Training
# =========================
def train(args):

    wandb.init(
        project="lawdis-decoder",
        name=f"decoder_lr{args.lr}_ld{args.lambda_depth}",
        config=vars(args)
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    pipeline = LawDISMacroPipeline.from_pretrained(args.model_path)
    pipeline = pipeline.to(device)
    
    if args.resume_epoch > 0:
        resume_path = os.path.join(args.output_path, f"decoder_epoch_{args.resume_epoch}.pt")
        state_dict = torch.load(resume_path, map_location=device)
        pipeline.vae.load_state_dict(state_dict)
        print(f"Resumed from epoch {args.resume_epoch}: {resume_path}")

    original_params = {
        name: param.clone().detach()
        for name, param in pipeline.vae.decoder.named_parameters()
    }

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

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, persistent_workers=True,
        collate_fn=safe_collate
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True, persistent_workers=True,
        collate_fn=safe_collate
    )

    print(f"Train images: {len(train_dataset)}")
    print(f"Val images  : {len(val_dataset)}")

    optimizer = optim.Adam(
        list(pipeline.vae.decoder.parameters()) +
        list(pipeline.vae.post_quant_conv.parameters()),
        lr=args.lr
    )

    bce = nn.BCEWithLogitsLoss()

    AMP_DTYPE = torch.bfloat16

    if hasattr(pipeline, "rgb_latent_scale_factor"):
        scale_factor = pipeline.rgb_latent_scale_factor
    else:
        scale_factor = pipeline.vae.config.scaling_factor

    best_dice = args.best_dice
    global_step = 0

    # ================= Training Loop =================
    for epoch in range(args.resume_epoch, args.epochs):

        start_time = time.time()

        train_loss       = 0.0
        train_mask_loss  = 0.0
        train_reg_loss = 0.0
        n_batches        = 0
        n_skipped        = 0

        pipeline.vae.decoder.train()
        pipeline.vae.post_quant_conv.train()

        for batch in tqdm(train_loader, desc=f"Train Epoch {epoch+1}"):

            if batch is None:
                continue

            rgb     = batch["image"].to(device)
            mask_gt = batch["mask"].to(device)
            depth   = batch["depth"].to(device)

            with torch.no_grad():
                rgb_latent, features = pipeline.encode_rgb(rgb)

            rgb_latent, features = sanitize_features(rgb_latent, features)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', dtype=AMP_DTYPE):
                z         = pipeline.vae.post_quant_conv(rgb_latent / scale_factor)
                mask_pred = pipeline.vae.decoder(z, features)
                mask_pred = sanitize_logit(mask_pred)

                depth_n = depth.float()
                depth_n = depth_n - depth_n.amin(dim=(-2,-1), keepdim=True)
                depth_n = depth_n / (depth_n.amax(dim=(-2,-1), keepdim=True) + 1e-8)

                dx_d, dy_d = gradient_map(depth_n)
                dx_d = torch.nn.functional.pad(dx_d, (0, 1, 0, 0))
                dy_d = torch.nn.functional.pad(dy_d, (0, 0, 0, 1))
                boundary_map = (dx_d.abs() + dy_d.abs())
                boundary_map = boundary_map / (boundary_map.amax(dim=(-2,-1), keepdim=True) + 1e-8)
                weight = 1.0 + args.lambda_depth * boundary_map

                mask_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    mask_pred, mask_gt, weight=weight.to(mask_pred.dtype)
                )

                reg_loss = torch.stack([
                    (p - original_params[n].to(p.dtype)).pow(2).mean()
                    for n, p in pipeline.vae.decoder.named_parameters()
                ]).sum()
                reg_loss = torch.clamp(reg_loss, 0.0, 10.0)

                loss = mask_loss + args.lambda_reg * reg_loss

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️  NaN/Inf tại step {global_step} | "
                      f"mask={mask_loss.item():.4f} | reg={reg_loss.item():.4f}")
                optimizer.zero_grad()
                n_skipped += 1
                continue

            # Not use scaler.scale(loss).backward()
            # bf16 not need loss scaling → backward 
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                list(pipeline.vae.decoder.parameters()) +
                list(pipeline.vae.post_quant_conv.parameters()),
                max_norm=1.0
            )

            total_norm = 0.0
            for p in (list(pipeline.vae.decoder.parameters()) +
                      list(pipeline.vae.post_quant_conv.parameters())):
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5

            optimizer.step()

            wandb.log({
                "train/loss_step":       loss.item(),
                "train/mask_loss_step":  mask_loss.item(),
                "train/reg_loss_step":   reg_loss.item(),
                "train/grad_norm":       total_norm,
            }, step=global_step)

            global_step      += 1
            train_loss       += loss.item()
            train_mask_loss  += mask_loss.item()
            train_reg_loss   += reg_loss.item()
            n_batches        += 1

        if n_batches > 0:
            train_loss       /= n_batches
            train_mask_loss  /= n_batches
            train_reg_loss /= n_batches

        # ================= Validation =================
        pipeline.vae.decoder.eval()
        pipeline.vae.post_quant_conv.eval()

        val_loss      = 0.0
        val_dice      = 0.0
        n_val_batches = 0

        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=AMP_DTYPE):
                for batch in tqdm(val_loader, desc="Validation"):

                    if batch is None:
                        continue

                    rgb     = batch["image"].to(device)
                    mask_gt = batch["mask"].to(device)

                    rgb_latent, features = pipeline.encode_rgb(rgb)
                    rgb_latent, features = sanitize_features(rgb_latent, features)

                    z         = pipeline.vae.post_quant_conv(rgb_latent / scale_factor)
                    mask_pred = pipeline.vae.decoder(z, features)
                    mask_pred = sanitize_logit(mask_pred)

                    val_loss += bce(mask_pred, mask_gt).item()
                    val_dice += dice_score(mask_pred, mask_gt).item()
                    n_val_batches += 1

        if n_val_batches > 0:
            val_loss /= n_val_batches
            val_dice /= n_val_batches

        epoch_time = time.time() - start_time

        print("\n==============================")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Time           : {epoch_time:.2f}s")
        print(f"Train Loss     : {train_loss:.6f}")
        print(f"  ├─ Mask Loss : {train_mask_loss:.6f}")
        print(f"  └─ Reg Loss: {train_reg_loss:.6f}")
        print(f"Val Loss       : {val_loss:.6f}")
        print(f"Val Dice       : {val_dice:.6f}")
        print(f"Best Val Dice  : {best_dice:.6f}")
        print(f"Skipped batches: {n_skipped}")
        print("==============================\n")

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
            if wandb.run is not None:
                wandb.run.summary["best_val_dice"] = val_dice
            print("New BEST model saved!\n")

        wandb.log({
            "train/loss_epoch":       train_loss,
            "train/mask_loss_epoch":  train_mask_loss,
            "train/reg_loss_epoch":   train_reg_loss,
            "val/loss":               val_loss,
            "val/dice":               val_dice,
            "val/best_dice":          best_dice,
            "train/skipped_batches":  n_skipped,
        }, step=global_step)

    wandb.finish()


# =========================
# Main
# =========================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",   type=str,   required=True)
    parser.add_argument("--data_path",    type=str,   required=True)
    parser.add_argument("--output_path",  type=str,   default="./checkpoints")
    parser.add_argument("--image_size",   type=int,   default=512)
    parser.add_argument("--batch_size",   type=int,   default=4)
    parser.add_argument("--epochs",       type=int,   default=20)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--lambda_depth", type=float, default=0.3)
    parser.add_argument("--resume_epoch", type=int,   default=0)
    parser.add_argument("--best_dice",    type=float, default=0.0)
    parser.add_argument("--lambda_reg",   type=float, default=0.1)
 
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    train(args)