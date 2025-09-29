import torch
from PIL import Image
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import os
import cv2
from typing import Tuple
from pathlib import Path
from diffusers import DDIMScheduler, UniPCMultistepScheduler
from diffusers.models import UNet2DConditionModel
from .ref_encoder.latent_controlnet import ControlNetModel
from .ref_encoder.adapter import *
from .ref_encoder.reference_unet import ref_unet
from .pipeline import StableHairPipeline
from .pipeline_cn import StableDiffusionControlNetPipeline
import logging

def concatenate_images(image_files, output_file, type="pil"):
    if type == "np":
        image_files = [Image.fromarray(img) for img in image_files]
    images = image_files  # list
    max_height = max(img.height for img in images)
    images = [img.resize((img.width, max_height)) for img in images]
    total_width = sum(img.width for img in images)
    combined = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for img in images:
        combined.paste(img, (x_offset, 0))
        x_offset += img.width
    combined.save(output_file)

class StableHair:
    def __init__(self, config="stable_hair/configs/hair_transfer.yaml", device: str | None = None, weight_dtype: torch.dtype | None = None, logger: logging.Logger | None = None) -> None:
        print("Initializing Stable Hair Pipeline...")
        # Accept path, dict, or OmegaConf object
        if isinstance(config, str):
            self.config = OmegaConf.load(config)
        elif isinstance(config, dict):
            self.config = OmegaConf.create(config)
        else:
            self.config = config

        # Logger
        self.logger = logger or logging.getLogger("hair_swap")
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO)

        # Device preference: CUDA, then MPS, else CPU
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        # Dtype default
        if weight_dtype is None:
            weight_dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
        self.weight_dtype = weight_dtype

        self.logger.info(f"Device: {self.device}, dtype: {self.weight_dtype}")
        self._build_pipelines()
        print("Initialization Done!")

    @classmethod
    def from_repo_defaults(cls, logger: logging.Logger | None = None) -> "StableHair":
        """Construct using default repo weight locations (no YAML required)."""
        repo_root = Path(__file__).resolve().parents[1]
        config = {
            "pretrained_model_path": "runwayml/stable-diffusion-v1-5",
            "pretrained_folder": str(repo_root / "models" / "stage2"),
            "encoder_path": "pytorch_model.bin",
            "adapter_path": "pytorch_model_1.bin",
            "controlnet_path": "pytorch_model_2.bin",
            "bald_converter_path": str(repo_root / "models" / "stage1" / "pytorch_model.bin"),
        }
        return cls(config=config, logger=logger)

    def _build_pipelines(self) -> None:
        device = self.device
        weight_dtype = self.weight_dtype

        try:
            # Load base UNet and create ControlNet
            self.logger.info("Loading UNet and ControlNet...")
            unet = UNet2DConditionModel.from_pretrained(self.config.pretrained_model_path, subfolder="unet").to(device)
            controlnet = ControlNetModel.from_unet(unet).to(device)
            _state_dict = torch.load(os.path.join(self.config.pretrained_folder, self.config.controlnet_path), map_location="cpu")
            controlnet.load_state_dict(_state_dict, strict=False)
            controlnet.to(weight_dtype)
        except Exception as e:
            self.logger.exception("Failed to load UNet/ControlNet or controlnet weights")
            raise

        try:
            # Main pipeline
            self.logger.info("Building main StableHair pipeline...")
            self.pipeline = StableHairPipeline.from_pretrained(
                self.config.pretrained_model_path,
                controlnet=controlnet,
                safety_checker=None,
                torch_dtype=weight_dtype,
            ).to(device)
            self.pipeline.scheduler = UniPCMultistepScheduler.from_config(self.pipeline.scheduler.config)
        except Exception as e:
            self.logger.exception("Failed to construct main pipeline")
            raise

        try:
            # Hair encoder/adapter
            self.logger.info("Loading hair encoder and adapter...")
            self.hair_encoder = ref_unet.from_pretrained(self.config.pretrained_model_path, subfolder="unet").to(device)
            _state_dict = torch.load(os.path.join(self.config.pretrained_folder, self.config.encoder_path), map_location="cpu")
            self.hair_encoder.load_state_dict(_state_dict, strict=False)
            self.hair_adapter = adapter_injection(self.pipeline.unet, device=self.device, dtype=weight_dtype, use_resampler=False)
            _state_dict = torch.load(os.path.join(self.config.pretrained_folder, self.config.adapter_path), map_location="cpu")
            self.hair_adapter.load_state_dict(_state_dict, strict=False)
        except Exception as e:
            self.logger.exception("Failed to load hair encoder/adapter or inject processors")
            raise

        try:
            # Bald converter
            self.logger.info("Loading bald converter (controlnet for hair removal)...")
            bald_converter = ControlNetModel.from_unet(unet).to(device)
            _state_dict = torch.load(self.config.bald_converter_path, map_location="cpu")
            bald_converter.load_state_dict(_state_dict, strict=False)
            bald_converter.to(dtype=weight_dtype)
            del unet
        except Exception as e:
            self.logger.exception("Failed to load bald converter controlnet")
            raise

        try:
            # Hair removal pipeline
            self.logger.info("Building hair removal pipeline...")
            self.remove_hair_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                self.config.pretrained_model_path,
                controlnet=bald_converter,
                safety_checker=None,
                torch_dtype=weight_dtype,
            )
            self.remove_hair_pipeline.scheduler = UniPCMultistepScheduler.from_config(self.remove_hair_pipeline.scheduler.config)
            self.remove_hair_pipeline = self.remove_hair_pipeline.to(device)
        except Exception as e:
            self.logger.exception("Failed to construct hair removal pipeline")
            raise

        # dtype moves
        try:
            self.hair_encoder.to(weight_dtype)
            self.hair_adapter.to(weight_dtype)
        except Exception:
            self.logger.exception("Failed moving modules to target dtype")
            raise

    def Hair_Transfer(self, source_image, reference_image, random_seed, step, guidance_scale, scale, controlnet_conditioning_scale, size=512):
        prompt = ""
        n_prompt = ""
        random_seed = int(random_seed)
        step = int(step)
        guidance_scale = float(guidance_scale)
        scale = float(scale)

        # load imgs
        source_image = Image.open(source_image).convert("RGB").resize((size, size))
        id = np.array(source_image)
        reference_image = np.array(Image.open(reference_image).convert("RGB").resize((size, size)))
        source_image_bald = np.array(self.get_bald(source_image, scale=0.9))
        H, W, C = source_image_bald.shape

        # generate images
        set_scale(self.pipeline.unet, scale)
        generator = torch.Generator(device="cuda")
        generator.manual_seed(random_seed)
        sample = self.pipeline(
            prompt,
            negative_prompt=n_prompt,
            num_inference_steps=step,
            guidance_scale=guidance_scale,
            width=W,
            height=H,
            controlnet_condition=source_image_bald,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=generator,
            reference_encoder=self.hair_encoder,
            ref_image=reference_image,
        ).samples
        return id, sample, source_image_bald, reference_image

    def get_bald(self, id_image, scale):
        H, W = id_image.size
        scale = float(scale)
        image = self.remove_hair_pipeline(
            prompt="",
            negative_prompt="",
            num_inference_steps=30,
            guidance_scale=1.5,
            width=W,
            height=H,
            image=id_image,
            controlnet_conditioning_scale=scale,
            generator=None,
        ).images[0]

        return image

def hair_transfer(source_image: Image.Image, reference_image: Image.Image, logger: logging.Logger) -> Tuple[Image.Image, Image.Image]:
    """Stateless helper that uses StableHair.from_repo_defaults() and reuses its init path."""
    model = StableHair.from_repo_defaults(logger=logger)

    size = 512
    src_pil = source_image.convert("RGB").resize((size, size))
    ref_pil = reference_image.convert("RGB").resize((size, size))

    # Bald source via instance pipeline
    W, H = src_pil.size
    try:
        bald_pil = model.remove_hair_pipeline(
            prompt="",
            negative_prompt="",
            num_inference_steps=30,
            guidance_scale=1.5,
            width=W,
            height=H,
            image=src_pil,
            controlnet_conditioning_scale=0.9,
            generator=None,
        ).images[0]
    except Exception:
        logger.exception("Bald conversion failed")
        raise

    # Hair transfer via instance pipeline
    set_scale(model.pipeline.unet, 1.0)
    gen = torch.Generator(device=model.device)
    gen.manual_seed(42)

    try:
        result_np = model.pipeline(
            prompt="",
            negative_prompt="",
            num_inference_steps=30,
            guidance_scale=1.5,
            width=W,
            height=H,
            controlnet_condition=np.array(bald_pil),
            controlnet_conditioning_scale=1.0,
            generator=gen,
            reference_encoder=model.hair_encoder,
            ref_image=np.array(ref_pil),
        ).samples
    except Exception:
        logger.exception("Hair transfer pipeline failed")
        raise

    result_uint8 = (np.clip(result_np, 0.0, 1.0) * 255.0).astype(np.uint8)
    result_pil = Image.fromarray(result_uint8)
    return bald_pil, result_pil


if __name__ == '__main__':
    model = StableHair(config="./configs/hair_transfer.yaml", weight_dtype=torch.float32)
    kwargs = OmegaConf.to_container(model.config.inference_kwargs)
    id, image, source_image_bald, reference_image = model.Hair_Transfer(**kwargs)
    os.makedirs(model.config.output_path, exist_ok=True)
    output_file = os.path.join(model.config.output_path, model.config.save_name)
    concatenate_images([id, source_image_bald, reference_image, (image*255.).astype(np.uint8)], output_file=output_file, type="np")
