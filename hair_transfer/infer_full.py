import torch
from PIL import Image
import numpy as np
from typing import Tuple
from omegaconf import OmegaConf
import os
import cv2
from .diffusers import DDIMScheduler, UniPCMultistepScheduler
from .diffusers.models import UNet2DConditionModel
from .ref_encoder.latent_controlnet import ControlNetModel
from .ref_encoder.adapter import *
from .ref_encoder.reference_unet import ref_unet
from .utils.pipeline import StableHairPipeline
from .utils.pipeline_cn import StableDiffusionControlNetPipeline
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = True

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
    def __init__(self, config="stable_hair/configs/hair_transfer.yaml", device=None, weight_dtype=torch.float16) -> None:
        print("Initializing Stable Hair Pipeline...")
        self.config = OmegaConf.load(config)
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = torch.device(device)

        repo_path = os.path.dirname(os.path.abspath(__file__))

        ### Load controlnet
        unet = UNet2DConditionModel.from_pretrained(self.config.pretrained_model_path, subfolder="unet").to(self.device)
        controlnet = ControlNetModel.from_unet(unet).to(self.device)
        _state_dict = torch.load(os.path.join(repo_path, self.config.pretrained_folder, self.config.controlnet_path), map_location="cpu")
        controlnet.load_state_dict(_state_dict, strict=False)
        controlnet.to(weight_dtype)

        ### >>> create pipeline >>> ###
        self.pipeline = StableHairPipeline.from_pretrained(
            self.config.pretrained_model_path,
            controlnet=controlnet,
            safety_checker=None,
            torch_dtype=weight_dtype,
        ).to(self.device)
        self.pipeline.scheduler = UniPCMultistepScheduler.from_config(self.pipeline.scheduler.config)

        ### load Hair encoder/adapter
        self.hair_encoder = ref_unet.from_pretrained(self.config.pretrained_model_path, subfolder="unet").to(self.device)
        _state_dict = torch.load(os.path.join(repo_path, self.config.pretrained_folder, self.config.encoder_path), map_location="cpu")
        self.hair_encoder.load_state_dict(_state_dict, strict=False)
        self.hair_adapter = adapter_injection(self.pipeline.unet, device=self.device, dtype=weight_dtype, use_resampler=False)
        _state_dict = torch.load(os.path.join(repo_path, self.config.pretrained_folder, self.config.adapter_path), map_location="cpu")
        self.hair_adapter.load_state_dict(_state_dict, strict=False)

        ### load bald converter
        bald_converter = ControlNetModel.from_unet(unet).to(self.device)
        _state_dict = torch.load(os.path.join(repo_path, self.config.bald_converter_path), map_location="cpu")
        bald_converter.load_state_dict(_state_dict, strict=False)
        bald_converter.to(dtype=weight_dtype)
        del unet

        ### create pipeline for hair removal
        self.remove_hair_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            self.config.pretrained_model_path,
            controlnet=bald_converter,
            safety_checker=None,
            torch_dtype=weight_dtype,
        )
        self.remove_hair_pipeline.scheduler = UniPCMultistepScheduler.from_config(
            self.remove_hair_pipeline.scheduler.config)
        self.remove_hair_pipeline = self.remove_hair_pipeline.to(self.device)

        ### move to fp16
        self.hair_encoder.to(weight_dtype)
        self.hair_adapter.to(weight_dtype)

        print("Initialization Done!")

    def Hair_Transfer(self, source_image, reference_image, random_seed, step, guidance_scale, scale, controlnet_conditioning_scale, size=512):
        print("source_image:", source_image, "reference_image:", reference_image, "random_seed:", random_seed, "step:", step, "guidance_scale:", guidance_scale, "scale:", scale, "controlnet_conditioning_scale:", controlnet_conditioning_scale, "size:", size)

        prompt = ""
        n_prompt = ""
        random_seed = int(random_seed)
        step = int(step)
        guidance_scale = float(guidance_scale)
        scale = float(scale)

        # load imgs
        source_image = Image.open(source_image) if isinstance(source_image, str) else source_image
        reference_image = Image.open(reference_image) if isinstance(reference_image, str) else reference_image

        source_image =  source_image.convert("RGB").resize((size, size))
        id = np.array(source_image)
        reference_image = np.array(reference_image.convert("RGB").resize((size, size)))
        source_image_bald = np.array(self.get_bald(source_image, scale=0.9))
        H, W, C = source_image_bald.shape

        # generate images
        set_scale(self.pipeline.unet, scale)
        generator = None
        if random_seed >= 0:
            gen_device = "cuda" if torch.cuda.is_available() else "cpu"
            generator = torch.Generator(device=gen_device)
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

def hair_transfer(source_image: Image.Image, reference_image: Image.Image) -> Tuple[Image.Image, Image.Image]:
    repo_path = os.path.dirname(os.path.abspath(__file__))
    model = StableHair(config=os.path.join(repo_path, "configs", "hair_transfer.yaml"), weight_dtype=torch.float32)
    kwargs = OmegaConf.to_container(model.config.inference_kwargs)
    kwargs["source_image"] = source_image
    kwargs["reference_image"] = reference_image
    id, image, source_image_bald, reference_image = model.Hair_Transfer(**kwargs)
    
    bald_pil = Image.fromarray(source_image_bald)
    result_pil = Image.fromarray((image * 255.).astype(np.uint8))
    
    return bald_pil, result_pil


if __name__ == '__main__':
    repo_path = os.path.dirname(os.path.abspath(__file__))
    logger.info(f"Repo path: {repo_path}")
    model = StableHair(config=os.path.join(repo_path, "configs", "hair_transfer.yaml"), weight_dtype=torch.float32)
    logger.info(f"Model: {model}")
    kwargs = OmegaConf.to_container(model.config.inference_kwargs)
    kwargs["source_image"] = os.path.join(repo_path, kwargs["source_image"])
    kwargs["reference_image"] = os.path.join(repo_path, kwargs["reference_image"])
    logger.info(f"Kwargs: {kwargs}")
    logger.info(f"Source image: {kwargs['source_image']}")
    logger.info(f"Reference image: {kwargs['reference_image']}")
    id, image, source_image_bald, reference_image = model.Hair_Transfer(**kwargs)
    logger.info(f"Id: {id}")
    logger.info(f"Image: {image}")
    logger.info(f"Source image bald: {source_image_bald}")
    logger.info(f"Reference image: {reference_image}")
    os.makedirs(model.config.output_path, exist_ok=True)
    logger.info(f"Output path: {model.config.output_path}")
    logger.info(f"Save name: {model.config.save_name}")
    output_file = os.path.join(model.config.output_path, model.config.save_name)
    concatenate_images([id, source_image_bald, reference_image, (image*255.).astype(np.uint8)], output_file=output_file, type="np")
    logger.info(f"Output file: {output_file}")
