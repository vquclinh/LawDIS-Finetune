import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import logging
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
from torchvision import transforms
from lawdis.lawdis_macro_pipeline import LawDISMacroPipeline

from train_classifier_guidance import LatentDepthPredictor

def run_guided_inference(pipe, f_phi, image_pil, target_depth_pil, prompt, guidance_scale, denoise_steps, device, dtype):
    orig_w, orig_h = image_pil.size

    transform_rgb = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    transform_depth = transforms.Compose([
        transforms.Resize((1024, 1024), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    img_tensor = transform_rgb(image_pil).unsqueeze(0).to(device, dtype=dtype)
    depth_tensor = transform_depth(target_depth_pil).unsqueeze(0).to(device, dtype=dtype)
    
    with torch.no_grad():
        latent_img = pipe.vae.encode(img_tensor).latent_dist.sample() * pipe.vae.config.scaling_factor
        
        if depth_tensor.shape[1] == 1:
            depth_tensor = depth_tensor.repeat(1, 3, 1, 1)
        target_depth_latent = pipe.vae.encode(depth_tensor).latent_dist.sample() * pipe.vae.config.scaling_factor
        
        text_inputs = pipe.tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True)
        text_embeddings = pipe.text_encoder(text_inputs.input_ids.to(device))[0]

    pipe.scheduler.set_timesteps(denoise_steps, device=device)
    z_t = torch.randn_like(latent_img, device=device, dtype=dtype)

    for i, t in enumerate(tqdm(pipe.scheduler.timesteps, desc="Guided Sampling", leave=False)):
        
        with torch.enable_grad():
            z_t_in = z_t.detach().requires_grad_(True)
            
            pred_depth_latent = f_phi(z_t_in, latent_img, t.unsqueeze(0))
            
            loss = F.mse_loss(pred_depth_latent, target_depth_latent)
            
            grad = torch.autograd.grad(loss, z_t_in)[0]

        with torch.no_grad():
            latent_model_input = torch.cat([z_t, latent_img], dim=1) 
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
            
            noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        alpha_bar_t = pipe.scheduler.alphas_cumprod[t]
        std_t = torch.sqrt(1 - alpha_bar_t)
        
        guided_noise_pred = noise_pred + guidance_scale * std_t * grad
        
        with torch.no_grad():
            z_t = pipe.scheduler.step(guided_noise_pred, t, z_t).prev_sample

    with torch.no_grad():
        mask_decoded = pipe.vae.decode(z_t / pipe.vae.config.scaling_factor).sample
        mask_decoded = (mask_decoded / 2 + 0.5).clamp(0, 1) # Normalize [0, 1]
        mask_decoded = mask_decoded.mean(dim=1, keepdim=True)
    
    mask_decoded = F.interpolate(mask_decoded, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
    
    return mask_decoded

if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Use LawDIS in macro mode for language-guided DIS with Classifier Guidance.")
    
    parser.add_argument("--checkpoint", type=str, default="stable-diffusion-2", help="Checkpoint path or hub name.")
    parser.add_argument("--input_rgb_dir", type=str, default="data/DIS5K", help="Path to the input image folder.")
    parser.add_argument("--subset_name", type=str, default="DIS-TE4", help="Name of the DIS subset to process.")
    parser.add_argument("--prompt_dir", type=str, default='data/json', help="Path to the prompt json folder.")
    parser.add_argument("--output_dir", type=str, default="output/output-macro", help="Output directory.")
    parser.add_argument("--denoise_steps", type=int, default=50, help="Diffusion denoising steps.")
    parser.add_argument("--half_precision", "--fp16", action="store_true", help="Run with half-precision (16-bit float).")
    parser.add_argument("--processing_res", type=int, default=1024, help="Resolution of processing.")
    parser.add_argument("--resample_method", choices=["bilinear", "bicubic", "nearest"], default="bilinear")
    parser.add_argument("--seed", type=int, default=None, help="Reproducibility seed.")
    parser.add_argument("--batch_size", type=int, default=0, help="Inference batch size.")
    parser.add_argument("--vae_checkpoint", type=str, default=None, help="Path to finetuned VAE checkpoint (.pt file).")

    parser.add_argument("--custom_guidance", type=str, default=None, help="Path to best_f_phi.pth")
    parser.add_argument("--input_depth_dir", type=str, required=True, help="Path to the target depth maps folder.")
    parser.add_argument("--guidance_scale", type=float, default=5.0, help="Classifier Guidance Scale (w).")

    args = parser.parse_args()

    # Setup directories
    output_dir = os.path.join(args.output_dir, args.subset_name)
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output dir = {output_dir}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device = {device}")

    # Read prompts
    prompt_path = os.path.join(args.prompt_dir, args.subset_name + '.json')
    try:
        with open(prompt_path, "r") as f:
            filenames = [(json.loads(line)['img'], json.loads(line)['prompt']) for line in f]
    except FileNotFoundError:
        logging.error(f"Cannot find prompt JSON at: {prompt_path}")
        exit(1)

    if len(filenames) > 0:
        logging.info(f"Found {len(filenames)} images to process.")
    else:
        logging.error("No image found.")
        exit(1)

    # Initialize Base Model
    dtype = torch.float16 if args.half_precision else torch.float32
    variant = "fp16" if args.half_precision else None

    pipe = LawDISMacroPipeline.from_pretrained(args.checkpoint, variant=variant, torch_dtype=dtype)
    
    if args.vae_checkpoint:
        state_dict = torch.load(args.vae_checkpoint, map_location=device)
        pipe.vae.load_state_dict(state_dict)
        pipe.vae.eval()
        logging.info(f"Loaded VAE checkpoint: {args.vae_checkpoint}")

    pipe = pipe.to(device)

    # Initialize Guidance Model
    model_f_phi = LatentDepthPredictor().to(device, dtype=dtype)
    if args.custom_guidance and os.path.exists(args.custom_guidance):
        model_f_phi.load_state_dict(torch.load(args.custom_guidance, map_location=device))
        model_f_phi.eval()
        logging.info("✅ Đã nạp Guidance Model f_phi thành công.")
    else:
        logging.warning("⚠️ Không tìm thấy hoặc chưa nạp custom_guidance, model sẽ chạy với trọng số f_phi ngẫu nhiên!")

    # Start Inference Loop
    for idx, filename in enumerate(tqdm(filenames, desc="Estimating DIS with Guidance")):
        rgb_path = os.path.join(args.input_rgb_dir, filename[0])
        prompt = filename[1]

        rgb_name_base = os.path.splitext(os.path.basename(rgb_path))[0]
        png_save_path = os.path.join(output_dir, f"{rgb_name_base}.png")

        if os.path.exists(png_save_path):
            continue

        try:
            input_image = Image.open(rgb_path).convert("RGB")
        except FileNotFoundError:
            logging.error(f"❌ Không tìm thấy ảnh RGB: {rgb_path}")
            continue
            
        depth_name = os.path.splitext(filename[0])[0] + ".png" # Tuỳ chỉnh đuôi file depth nếu cần
        depth_path = os.path.join(args.input_depth_dir, depth_name)
        try:
            target_depth = Image.open(depth_path).convert("L")
        except FileNotFoundError:
            logging.error(f"❌ Không tìm thấy ảnh Depth: {depth_path}")
            continue

        generator = None
        if args.seed is not None:
            generator = torch.Generator(device=device).manual_seed(args.seed)

        dis_pred_tensor = run_guided_inference(
            pipe=pipe,
            f_phi=model_f_phi,
            image_pil=input_image,
            target_depth_pil=target_depth,
            prompt=prompt,
            guidance_scale=args.guidance_scale,
            denoise_steps=args.denoise_steps,
            device=device,
            dtype=dtype,
            generator=generator
        )

        dis_to_save = (dis_pred_tensor.squeeze().cpu().numpy() * 255.0).astype(np.uint8)
        Image.fromarray(dis_to_save).save(png_save_path, mode="L")