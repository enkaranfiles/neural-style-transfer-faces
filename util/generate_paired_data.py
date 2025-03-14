import os
import random
import yaml
import torch
import cv2
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionXLPipeline, AutoPipelineForInpainting
from .data_utils import create_beard_mask_custom
from .config import load_config

config_path = "config.yaml"
project_root = os.path.dirname(os.path.abspath(config_path))

def main(number_of_images):
    config = load_config(config_path=config_path)
    pd_config = config["paired_data"]
    inpaint_config = config["inpainting"]
    

    BASE_IMAGE_DIR = os.path.join(project_root,pd_config["base_image_dir"])
    BEARED_DIR = os.path.join(project_root,pd_config["beared_dir"])
    MASK_DIR = os.path.join(project_root,pd_config["mask_dir"])

    PREDICTOR_PATH = os.path.join(project_root,pd_config["predictor_path"])
    POSITIVE_PROMPT = pd_config["positive_prompt"]
    NEGATIVE_PROMPT = pd_config["negative_prompt"]
    MODEL_ID = str(pd_config["model_id"])
    
    if number_of_images:
        NUM_IMAGES = number_of_images
    else:
        NUM_IMAGES = pd_config["num_images"]

    WIDTH = pd_config["width"]
    HEIGHT = pd_config["height"]

    PROMPT_BEARD = inpaint_config["prompt_beard"]
    NUM_INFERENCE_STEPS = inpaint_config["num_inference_steps"]
    GUIDANCE_SCALE = inpaint_config["guidance_scale"]
    STRENGTH = inpaint_config["strength"]

    os.makedirs(BASE_IMAGE_DIR, exist_ok=True)
    os.makedirs(BEARED_DIR, exist_ok=True)
    os.makedirs(MASK_DIR, exist_ok=True)


    print("Generating base images...")
    pipe = StableDiffusionXLPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    pipe.to("cuda")  # Use GPU

    for i in range(NUM_IMAGES):
        seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator("cuda").manual_seed(seed)
        result = pipe(
            prompt=POSITIVE_PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            width=WIDTH,
            height=HEIGHT,
            generator=generator
        )
        image = result.images[0]
        base_image_path = os.path.join(BASE_IMAGE_DIR, f"base_image_{i+1}.png")
        image.save(base_image_path)
        print(f"Saved base image: {base_image_path}")
    print("Base image generation complete.\n")
    

    print("Extracting masks from base images...")
    # List all image files in the base directory
    image_files = [f for f in os.listdir(BASE_IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for img_file in tqdm(image_files, desc="Extracting masks"):
        image_path = os.path.join(BASE_IMAGE_DIR, img_file)
        mask_save_path = os.path.join(MASK_DIR, img_file)  # Use same filename for the mask
        try:
            mask = create_beard_mask_custom(image_path, PREDICTOR_PATH)
            cv2.imwrite(mask_save_path, mask)
        except Exception as e:
            print(f"Failed to process {image_path}: {e}")
    print("Mask extraction complete.\n")
    
    #inpainting
    print("Performing inpainting to generate bearded variants...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe_inpaint = AutoPipelineForInpainting.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", 
        torch_dtype=torch.float16, 
        variant="fp16"
    ).to(device)
    
    for i in range(NUM_IMAGES):
        base_image_path = os.path.join(BASE_IMAGE_DIR, f"base_image_{i+1}.png")
        mask_image_path = os.path.join(MASK_DIR, f"base_image_{i+1}.png")
        
        if not os.path.exists(mask_image_path):
            print(f"Mask file {mask_image_path} not found. Skipping sample {i+1}.")
            continue
        
        base_img_pil = Image.open(base_image_path).convert("RGB").resize((WIDTH, HEIGHT), Image.LANCZOS)
        mask_img_pil = Image.open(mask_image_path).convert("L").resize((WIDTH, HEIGHT), Image.LANCZOS)
        
        try:
            result_beard = pipe_inpaint(
                prompt=PROMPT_BEARD,
                image=base_img_pil,
                mask_image=mask_img_pil,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                strength=STRENGTH
            )
        except Exception as e:
            print(f"Error during inpainting for sample {i+1}: {e}")
            continue
        
        image_beard = result_beard.images[0]
        beard_save_path = os.path.join(BEARED_DIR, f"inpainted_beard_{i+1}.png")
        image_beard.save(beard_save_path)
        print(f"Beard variant saved as: {beard_save_path}")
        
    print("Paired data generation complete.")

if __name__ == "__main__":
    main(number_of_images=0)
