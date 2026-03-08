import os
import time
import optuna
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dotenv import load_dotenv

from train_decoder import (
    SegDataset, safe_collate, gradient_map, dice_score,
    sanitize_features, sanitize_logit
)
from lawdis.lawdis_macro_pipeline import LawDISMacroPipeline
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

load_dotenv("/content/drive/MyDrive/lawdis/.env")

# =========================
# Train for sweep
# =========================
def train_sweep(args, trial, lr, lambda_depth, sweep_epochs=5):

    wandb.init(
        project=os.getenv("WANDB_PROJECT"),
        config={
            "lr":           lr,
            "lambda_depth": lambda_depth,
            "batch_size":   args.batch_size,
            "image_size":   args.image_size,
        },
        name=f"trial_{trial.number}",
        reinit=True
    )

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

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, collate_fn=safe_collate
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=safe_collate
    )

    optimizer = optim.Adam(
        list(pipeline.vae.decoder.parameters()) +
        list(pipeline.vae.post_quant_conv.parameters()),
        lr=lr
    )

    bce = nn.BCEWithLogitsLoss()

    AMP_DTYPE = torch.bfloat16

    if hasattr(pipeline, "rgb_latent_scale_factor"):
        scale_factor = pipeline.rgb_latent_scale_factor
    else:
        scale_factor = pipeline.vae.config.scaling_factor

    best_dice = 0.0

    for epoch in range(sweep_epochs):

        print("\n" + "="*60)
        print(f"Trial {trial.number} | Epoch {epoch+1}/{sweep_epochs}")
        print("="*60)

        pipeline.vae.decoder.train()
        pipeline.vae.post_quant_conv.train()

        epoch_loss       = 0.0
        epoch_mask_loss  = 0.0
        epoch_depth_loss = 0.0
        n_batches        = 0
        n_skipped        = 0

        start_time   = time.time()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=True)

        for batch_idx, batch in enumerate(progress_bar):

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
                mask_pred = sanitize_logit(mask_pred)   # nan → -10.0

                mask_loss = bce(mask_pred, mask_gt)

                prob  = torch.sigmoid(mask_pred).clamp(1e-6, 1 - 1e-6)

                depth = depth.float()
                depth = depth - depth.amin(dim=(-2, -1), keepdim=True)
                depth = depth / (depth.amax(dim=(-2, -1), keepdim=True) + 1e-8)

                dx_m, dy_m = gradient_map(prob)
                dx_d, dy_d = gradient_map(depth)

                depth_loss = (
                    (dx_m - dx_d).abs().mean() +
                    (dy_m - dy_d).abs().mean()
                )
                depth_loss = torch.clamp(depth_loss, 0.0, 10.0)

                loss = mask_loss + lambda_depth * depth_loss

            # Only skip
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️  NaN/Inf tại batch {batch_idx} | "
                      f"mask={mask_loss.item():.4f} | depth={depth_loss.item():.4f}")
                optimizer.zero_grad()
                n_skipped += 1
                continue

            # backward, no scaler
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                list(pipeline.vae.decoder.parameters()) +
                list(pipeline.vae.post_quant_conv.parameters()),
                max_norm=1.0
            )

            # Grad norm all of 2 module
            total_norm = 0.0
            for p in (list(pipeline.vae.decoder.parameters()) +
                      list(pipeline.vae.post_quant_conv.parameters())):
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5

            optimizer.step()

            epoch_loss       += loss.item()
            epoch_mask_loss  += mask_loss.item()
            epoch_depth_loss += depth_loss.item()
            n_batches        += 1

            progress_bar.set_postfix({
                "loss":  f"{loss.item():.4f}",
                "mask":  f"{mask_loss.item():.4f}",
                "depth": f"{depth_loss.item():.4f}",
            })

            wandb.log({
                "train_loss":   loss.item(),
                "mask_loss":    mask_loss.item(),
                "depth_loss":   depth_loss.item(),
                "grad_norm":    total_norm,
                "epoch":        epoch + 1,
            })

        # End epoch stats
        if n_batches > 0:
            epoch_loss       /= n_batches
            epoch_mask_loss  /= n_batches
            epoch_depth_loss /= n_batches

        elapsed = time.time() - start_time
        print(f"\n⏱ Epoch time : {elapsed:.2f}s")
        print(f"   Train Loss : {epoch_loss:.4f}")
        print(f"   Mask Loss  : {epoch_mask_loss:.4f}")
        print(f"   Depth Loss : {epoch_depth_loss:.4f}")
        print(f"   Skipped    : {n_skipped} batches")

        # =============== Validation ===============
        pipeline.vae.decoder.eval()
        pipeline.vae.post_quant_conv.eval()

        val_dice      = 0.0
        n_val_batches = 0

        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=AMP_DTYPE):
                for batch in val_loader:

                    if batch is None:
                        continue

                    rgb     = batch["image"].to(device)
                    mask_gt = batch["mask"].to(device)

                    rgb_latent, features = pipeline.encode_rgb(rgb)
                    rgb_latent, features = sanitize_features(rgb_latent, features)

                    z         = pipeline.vae.post_quant_conv(rgb_latent / scale_factor)
                    mask_pred = pipeline.vae.decoder(z, features)
                    mask_pred = sanitize_logit(mask_pred)

                    val_dice += dice_score(mask_pred, mask_gt).item()
                    n_val_batches += 1

        if n_val_batches > 0:
            val_dice /= n_val_batches
        best_dice = max(best_dice, val_dice)

        trial.report(val_dice, epoch)
        if trial.should_prune():
            wandb.finish()
            raise optuna.exceptions.TrialPruned()

        print(f"Val Dice  : {val_dice:.4f}")
        print(f"Best Dice : {best_dice:.4f}")

        wandb.log({
            "epoch_train_loss":  epoch_loss,
            "epoch_mask_loss":   epoch_mask_loss,
            "epoch_depth_loss":  epoch_depth_loss,
            "val_dice":          val_dice,
            "best_dice_so_far":  best_dice,
            "learning_rate":     lr,
            "epoch":             epoch + 1,
            "skipped_batches":   n_skipped,
        })

    del pipeline
    torch.cuda.empty_cache()
    wandb.finish()
    return best_dice


# =========================
# Optuna Objective
# =========================
def objective(trial):
    lr           = trial.suggest_float("lr", 1e-5, 5e-5, log=True)
    lambda_depth = trial.suggest_float("lambda_depth", 0.01, 1.0, log=True)

    print(f"\nTrial {trial.number}: lr={lr:.6f}, lambda_depth={lambda_depth:.4f}")

    try:
        best_dice = train_sweep(args, trial, lr, lambda_depth, sweep_epochs=5)
        return best_dice
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        wandb.finish()
        raise optuna.exceptions.TrialPruned()


# =========================
# Main
# =========================
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",  type=str, required=True)
    parser.add_argument("--data_path",   type=str, required=True)
    parser.add_argument("--image_size",  type=int, default=512)
    parser.add_argument("--batch_size",  type=int, default=4)

    args = parser.parse_args()

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2)
    )
    study.optimize(objective, n_trials=15)

    print("\n=============================")
    print("BEST RESULT")
    print("Best Dice  :", study.best_value)
    print("Best Params:", study.best_params)
    print("=============================")