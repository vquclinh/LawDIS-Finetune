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
    sanitize_features, sanitize_logit, combo_loss
)
from lawdis.lawdis_macro_pipeline import LawDISMacroPipeline
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

load_dotenv("/content/drive/MyDrive/lawdis/.env")

def get_layerwise_params(decoder, post_quant_conv, base_lr, decay=0.1):

    named_params = list(decoder.named_parameters())
    n = len(named_params)
    
    early = [p for _, p in named_params[:n//2]]
    late  = [p for _, p in named_params[n//2:]]

    return [
        {"params": early,                          "lr": base_lr * decay},
        {"params": late,                           "lr": base_lr},
        {"params": post_quant_conv.parameters(),   "lr": base_lr},
    ]

# =========================
# Train for sweep
# =========================
def train_sweep(args, trial, lr, lambda_depth, lambda_reg, sweep_epochs=3):

    wandb.init(
        project=os.getenv("WANDB_PROJECT"),
        config={
            "lr":           lr,
            "lambda_depth": lambda_depth,
            "lambda_reg":   lambda_reg,
            "batch_size":   args.batch_size,
            "image_size":   args.image_size,
        },
        name=f"trial_{trial.number}",
        reinit=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipeline = LawDISMacroPipeline.from_pretrained(args.model_path)
    pipeline = pipeline.to(device)
    original_params = {
        name: param.clone().detach()
        for name, param in pipeline.vae.decoder.named_parameters()
    } 

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
        get_layerwise_params(
           pipeline.vae.decoder,
           pipeline.vae.post_quant_conv,
           base_lr=lr,
           decay=0.1
        )
    )

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
        epoch_reg_loss   = 0.0
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
                mask_pred = sanitize_logit(mask_pred)

                depth_n = depth.float()
                depth_n = depth_n - depth_n.amin(dim=(-2,-1), keepdim=True)
                depth_n = depth_n / (depth_n.amax(dim=(-2,-1), keepdim=True) + 1e-8)

                dx_d, dy_d = gradient_map(depth_n)
                dx_d = torch.nn.functional.pad(dx_d, (0, 1, 0, 0))
                dy_d = torch.nn.functional.pad(dy_d, (0, 0, 0, 1))
                boundary_map = (dx_d.abs() + dy_d.abs())
                boundary_map = boundary_map / (boundary_map.amax(dim=(-2,-1), keepdim=True) + 1e-8)
                weight = 1.0 + lambda_depth * boundary_map

                mask_loss = combo_loss(
                    mask_pred, mask_gt,
                    weight=weight.to(mask_pred.dtype)
                )

                reg_loss = torch.stack([
                    (p - original_params[n].to(p.dtype)).pow(2).mean()
                    for n, p in pipeline.vae.decoder.named_parameters()
                ]).sum()
                reg_loss = torch.clamp(reg_loss, 0.0, 10.0)

                loss = mask_loss + lambda_reg * reg_loss


            # Only skip
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️  NaN/Inf tại batch {batch_idx} | "
                      f"mask={mask_loss.item():.4f} | reg={reg_loss.item():.4f}")
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
            epoch_reg_loss += reg_loss.item()
            n_batches        += 1

            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "mask": f"{mask_loss.item():.4f}",
                "reg":  f"{reg_loss.item():.4f}",
            })

            wandb.log({
                "train_loss":   loss.item(),
                "mask_loss":    mask_loss.item(),
                "reg_loss":     reg_loss.item(),
                "grad_norm":    total_norm,
                "epoch":        epoch + 1,
            })

        # End epoch stats
        if n_batches > 0:
            epoch_loss       /= n_batches
            epoch_mask_loss  /= n_batches
            epoch_reg_loss /= n_batches

        elapsed = time.time() - start_time
        print(f"\n⏱ Epoch time : {elapsed:.2f}s")
        print(f"   Train Loss : {epoch_loss:.4f}")
        print(f"   Mask Loss  : {epoch_mask_loss:.4f}")
        print(f"   Reg Loss : {epoch_reg_loss:.4f}")
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
            "epoch_reg_loss":    epoch_reg_loss,
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
    lr           = trial.suggest_float("lr", 3e-6, 5e-5, log=True)
    lambda_depth = trial.suggest_float("lambda_depth", 0.05, 3.0, log=True)
    lambda_reg   = trial.suggest_float("lambda_reg", 1.0, 20.0, log=True)

    print(f"\nTrial {trial.number}: lr={lr:.2e}, "
          f"lambda_depth={lambda_depth:.4f}, lambda_reg={lambda_reg:.2f}")

    try:
        best_dice = train_sweep(args, trial, lr, lambda_depth, lambda_reg,
                                sweep_epochs=3)
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
    parser.add_argument("--image_size",  type=int, default=1024)
    parser.add_argument("--batch_size",  type=int, default=32)

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