from typing import Dict, Optional, Union
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import (
    AutoencoderKlLawDIS,
    DDIMScheduler,
    DiffusionPipeline,
    TCDScheduler,
    UNet2DConditionModel,
)
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import pil_to_tensor, resize
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from .util.batchsize import find_batch_size
from .util.image_util import (
    get_tv_resample_method
)

class LawDISMacroPipeline(DiffusionPipeline):
    rgb_latent_scale_factor = 0.18215
    dis_latent_scale_factor = 0.18215
    
    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKlLawDIS,
        scheduler: Union[DDIMScheduler, TCDScheduler],
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        depth_predictor: torch.nn.Module = None
    ):
        super().__init__()
        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            depth_predictor=depth_predictor
        )

    @torch.no_grad()
    def __call__(
        self,
        input_image: Union[Image.Image, torch.Tensor],
        target_depth: Union[Image.Image, torch.Tensor] = None,
        prompt: Optional[str] = None,
        denoising_steps: Optional[int] = 20,
        guidance_scale: float = 100.0,
        processing_res: Optional[int] = None,
        resample_method: str = "bilinear",
        batch_size: int = 0,
        generator: Union[torch.Generator, None] = None,
        show_progress_bar: bool = True
    ) -> np.ndarray:
        
        assert processing_res >= 0
        resample_method: InterpolationMode = get_tv_resample_method(resample_method) 

        # ----------------- Image Preprocess -----------------
        if isinstance(input_image, Image.Image):
            input_image = input_image.convert("RGB")
            rgb = pil_to_tensor(input_image) 
            rgb = rgb.unsqueeze(0)  
        elif isinstance(input_image, torch.Tensor):
            rgb = input_image
        else:
            raise TypeError(f"Unknown input type: {type(input_image) = }")
            
        input_size = rgb.shape
        assert (4 == rgb.dim() and 3 == input_size[-3]), f"Wrong input shape {input_size}"

        if processing_res > 0: 
            rgb = resize(rgb, (processing_res, processing_res), resample_method, antialias=True)
            
        rgb_norm: torch.Tensor = rgb / 255.0 * 2.0 - 1.0  
        rgb_norm = rgb_norm.to(self.dtype)

        # ----------------- Depth Preprocess -----------------
        target_depth_norm = None
        if target_depth is not None:
            if isinstance(target_depth, Image.Image):
                target_depth = target_depth.convert("L")
                depth_tensor = pil_to_tensor(target_depth).unsqueeze(0)
            elif isinstance(target_depth, torch.Tensor):
                depth_tensor = target_depth
            
            if processing_res > 0:
                depth_tensor = resize(depth_tensor, (processing_res, processing_res), InterpolationMode.NEAREST, antialias=True)
            
            target_depth_norm = depth_tensor / 255.0 * 2.0 - 1.0
            target_depth_norm = target_depth_norm.to(self.dtype)

        duplicated_rgb = rgb_norm.expand(1, -1, -1, -1) 
        single_rgb_dataset = TensorDataset(duplicated_rgb)
        if batch_size > 0:
            _bs = batch_size
        else:
            _bs = find_batch_size(input_res=max(rgb_norm.shape[1:]), dtype=self.dtype)

        single_rgb_loader = DataLoader(single_rgb_dataset, batch_size=_bs, shuffle=False)

        dis_pred_ls = []
        iterable = tqdm(single_rgb_loader, desc=" " * 2 + "Inference batches", leave=False) if show_progress_bar else single_rgb_loader
        
        for batch in iterable:
            (batched_img,) = batch   
            dis_pred_raw = self.single_infer(
                rgb_in=batched_img,
                target_depth=target_depth_norm,
                guidance_scale=guidance_scale,
                prompt=prompt,
                num_inference_steps=denoising_steps,
                generator=generator,
            ) 
            dis_pred_ls.append(dis_pred_raw.detach())
            
        dis_pred = torch.concat(dis_pred_ls, dim=0) 
        torch.cuda.empty_cache()

        dis_pred = resize(dis_pred, input_size[-2:], interpolation=resample_method, antialias=True) 
        dis_pred = dis_pred.squeeze().cpu().numpy() 

        return dis_pred

    def encode_text(self,prompt):
        text_inputs = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device) 
        text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype) 
        return text_embed

    def single_infer(
        self,
        rgb_in: torch.Tensor,
        target_depth: Optional[torch.Tensor],
        guidance_scale: float,
        prompt: str,
        num_inference_steps: int,
        generator: Union[torch.Generator, None],
    ) -> torch.Tensor:

        device = self.device
        rgb_in = rgb_in.to(device)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps  

        with torch.no_grad():
            rgb_latent, features_saver = self.encode_rgb(rgb_in) 
            
            target_depth_latent = None
            if target_depth is not None and self.depth_predictor is not None:
                target_depth = target_depth.to(device)
                if target_depth.shape[1] == 1:
                    target_depth = target_depth.repeat(1, 3, 1, 1) # Lặp lại 3 kênh cho VAE như lúc train
                
                h_depth, _ = self.vae.encoder(target_depth)
                moments_depth = self.vae.quant_conv(h_depth)
                mean_depth, _ = torch.chunk(moments_depth, 2, dim=1)
                target_depth_latent = mean_depth * self.dis_latent_scale_factor

            dis_latent = torch.randn(rgb_latent.shape, device=device, dtype=self.dtype, generator=generator)  
            batch_text_embed = self.encode_text(prompt).repeat((rgb_latent.shape[0], 1, 1)).to(device)
            switch_class = torch.tensor([[0., 1.]], device=device, dtype=torch.float32)
            class_embedding = torch.cat([torch.sin(switch_class), torch.cos(switch_class)], dim=-1)

        iterable = enumerate(timesteps)

        for i, t in iterable:
            if target_depth_latent is not None and guidance_scale > 0:
                dis_latent = dis_latent.detach().requires_grad_(True)
                
                with torch.enable_grad():
                    unet_input = torch.cat([rgb_latent, dis_latent], dim=1)  
                    noise_pred = self.unet(unet_input, t, encoder_hidden_states=batch_text_embed, class_labels=class_embedding).sample  
                    
                    t_tensor = torch.tensor([t] * dis_latent.shape[0], device=device, dtype=torch.long)
                    pred_depth_latent = self.depth_predictor(dis_latent, rgb_latent, t_tensor)
                    
                    loss = F.mse_loss(pred_depth_latent, target_depth_latent)
                    
                    grad = torch.autograd.grad(loss, dis_latent)[0]
                
                alpha_prod_t = self.scheduler.alphas_cumprod[t.item()]
                beta_prod_t = 1 - alpha_prod_t
                
                noise_pred = noise_pred + guidance_scale * (beta_prod_t ** 0.5) * grad
                
                dis_latent = dis_latent.detach()
            else:
                with torch.no_grad():
                    unet_input = torch.cat([rgb_latent, dis_latent], dim=1)  
                    noise_pred = self.unet(unet_input, t, encoder_hidden_states=batch_text_embed, class_labels=class_embedding).sample  

            with torch.no_grad():
                dis_latent = self.scheduler.step(noise_pred, t, dis_latent, generator=generator).prev_sample 

        with torch.no_grad():
            dis = self.decode_dis(dis_latent, features_saver).sigmoid() 

        return dis

    def encode_rgb(self, rgb_in: torch.Tensor) -> torch.Tensor:
        h,encoder_features = self.vae.encoder(rgb_in) 
        moments = self.vae.quant_conv(h) 
        mean, logvar = torch.chunk(moments, 2, dim=1) 
        rgb_latent = mean * self.rgb_latent_scale_factor
        return rgb_latent,encoder_features

    def decode_dis(self, dis_latent: torch.Tensor,feature_saver: list) -> torch.Tensor: 
        dis_latent = dis_latent / self.dis_latent_scale_factor 
        z = self.vae.post_quant_conv(dis_latent.cuda()) 
        pred = self.vae.decoder(z,feature_saver) 
        return pred