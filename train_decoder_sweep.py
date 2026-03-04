import os
import copy
import time
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import dataset + utils từ file cũ
from train_decoder import SegDataset, gradient_map, dice_score
from lawdis.lawdis_macro_pipeline import LawDISMacroPipeline


# =========================
# Train for sweep
# =========================
def train_sweep(args, lr, lambda_depth, sweep_epochs=5):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipeline = LawDISMacroPipeline.from_pretrained(args.model_path)
    pipeline = pipeline.to(device)

    # Freeze
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

    # Dataset
    train_dataset = SegDataset(args.data_path, "DIS-TR", args.image_size, use_depth=True)
    val_dataset   = SegDataset(args.data_path, "DIS-VD", args.image_size, use_depth=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)

    optimizer = optim.Adam(
        list(pipeline.vae.decoder.parameters()) +
        list(pipeline.vae.post_quant_conv.parameters()),
        lr=lr
    )

    bce = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler()

    if hasattr(pipeline, "rgb_latent_scale_factor"):
        scale_factor = pipeline.rgb_latent_scale_factor
    else:
        scale_factor = pipeline.vae.config.scaling_factor

    best_dice = 0.0

    for epoch in range(sweep_epochs):

        pipeline.vae.decoder.train()
        pipeline.vae.post_quant_conv.train()

        for batch in train_loader:

            rgb = batch["image"].to(device)
            mask_gt = batch["mask"].to(device)
            depth = batch["depth"].to(device)

            with torch.no_grad():
                rgb_latent, features = pipeline.encode_rgb(rgb)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():

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

                loss = mask_loss + lambda_depth * depth_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Validation
        pipeline.vae.decoder.eval()
        pipeline.vae.post_quant_conv.eval()

        val_dice = 0
        with torch.no_grad():
            for batch in val_loader:

                rgb = batch["image"].to(device)
                mask_gt = batch["mask"].to(device)

                rgb_latent, features = pipeline.encode_rgb(rgb)
                z = pipeline.vae.post_quant_conv(rgb_latent / scale_factor)
                mask_pred = pipeline.vae.decoder(z, features)

                val_dice += dice_score(mask_pred, mask_gt).item()

        val_dice /= len(val_loader)
        best_dice = max(best_dice, val_dice)

    return best_dice


# =========================
# Optuna Objective
# =========================
def objective(trial):

    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    lambda_depth = trial.suggest_float("lambda_depth", 0.05, 1.0)

    print(f"\nTrial: lr={lr:.6f}, lambda_depth={lambda_depth:.4f}")

    best_dice = train_sweep(args, lr, lambda_depth, sweep_epochs=5)

    return best_dice


# =========================
# Main
# =========================
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)

    args = parser.parse_args()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=15)

    print("\n=============================")
    print("BEST RESULT")
    print("Best Dice :", study.best_value)
    print("Best Params:", study.best_params)
    print("=============================")