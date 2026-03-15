# Scripts/minecraft-skins-sdxl-img2skin.py

import os
import sys
import random
import logging
import argparse

import torch
import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist
from diffusers import StableDiffusionXLPipeline

# --- 復用現有皮膚提取邏輯（來自 minecraft-skins-sdxl.py） ---

MODEL_NAME = "monadical-labs/minecraft-skin-generator-sdxl"
MASK_IMAGE = "images/half-transparency-mask.png"

IMAGE_WIDTH = 768
IMAGE_HEIGHT = 768

BACKGROUND_REGIONS = [
    (32, 0, 40, 8),
    (56, 0, 64, 8),
]

TRANSPARENT_REGIONS = [
    (40, 0, 48, 8),
    (48, 0, 56, 8),
    (32, 8, 40, 16),
    (40, 8, 48, 16),
    (48, 8, 56, 16),
    (56, 8, 64, 16),
]


def get_background_color(image):
    pixels = []
    for region in BACKGROUND_REGIONS:
        swatch = image.crop(region)
        width, height = swatch.size
        np_swatch = np.array(swatch)
        np_swatch = np_swatch.reshape(width * height, 3)
        if len(pixels) == 0:
            pixels = np_swatch
        else:
            np.concatenate((pixels, np_swatch))
    (r, g, b) = np.mean(np_swatch, axis=0, dtype=int)
    return [(r, g, b)]


def restore_region_transparency(image, region, transparency_color, cutoff=50):
    changed = 0
    for x in range(region[0], region[2]):
        for y in range(region[1], region[3]):
            pixel = [image.getpixel((x, y))]
            pixel = [(pixel[0][0], pixel[0][1], pixel[0][2])]
            dist = cdist(pixel, transparency_color)
            if dist <= cutoff:
                image.putpixel((x, y), (255, 255, 255, 0))
                changed += 1
    return image, changed


def restore_skin_transparency(image, transparency_color, cutoff=50):
    image = image.convert("RGBA")
    total_changed = 0
    for region in TRANSPARENT_REGIONS:
        image, changed = restore_region_transparency(
            image, region, transparency_color, cutoff=cutoff
        )
        total_changed += changed
    return image, total_changed


def extract_minecraft_skin(generated_image, cutoff=50):
    image = generated_image.crop((0, 0, IMAGE_WIDTH, int(IMAGE_HEIGHT / 2)))
    skin = image.resize((64, 32), Image.NEAREST)
    color = get_background_color(skin)
    transparent_skin, _ = restore_skin_transparency(skin, color, cutoff=cutoff)
    mask = Image.open(MASK_IMAGE)
    transparent_skin.alpha_composite(mask)
    return transparent_skin


# --- 新增：圖片轉皮膚功能 ---

def load_pipeline(device, dtype):
    """加載 SDXL 管線並掛載 IP-Adapter"""
    logger.info(f"Loading model: {MODEL_NAME}")
    if device == "cpu":
        pipeline = StableDiffusionXLPipeline.from_pretrained(MODEL_NAME)
    else:
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            MODEL_NAME, torch_dtype=dtype
        )
    pipeline.to(device)

    # 加載 IP-Adapter（diffusers 內建支持，自動處理圖片編碼）
    logger.info("Loading IP-Adapter...")
    pipeline.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="sdxl_models",
        weight_name="ip-adapter_sdxl.bin",
    )

    # 內存優化
    pipeline.enable_attention_slicing()

    return pipeline


def generate_skin_from_image(
    pipeline,
    reference_image,
    prompt="a minecraft skin",
    ip_adapter_scale=0.7,
    num_inference_steps=30,
    guidance_scale=7.5,
    seed=42,
    device="cuda",
):
    """使用參考圖片 + 文字 prompt 生成皮膚

    Args:
        pipeline: 已加載 IP-Adapter 的 SDXL pipeline
        reference_image: 預處理後的 PIL Image (RGB)
        prompt: 文字描述
        ip_adapter_scale: IP-Adapter 影響強度 (0.0-1.0)
        num_inference_steps: 推理步數
        guidance_scale: 文字引導強度
        seed: 隨機種子
        device: 計算設備
    """
    if seed == 0:
        seed = random.randint(1, 100000)

    generator = torch.Generator(device=device).manual_seed(seed)

    # 設置 IP-Adapter 強度
    pipeline.set_ip_adapter_scale(ip_adapter_scale)

    # 生成（直接傳 PIL Image，由 diffusers 內部處理編碼）
    logger.info(f"Generating with prompt='{prompt}', scale={ip_adapter_scale}")
    output = pipeline(
        prompt=prompt,
        ip_adapter_image=reference_image,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        height=IMAGE_HEIGHT,
        width=IMAGE_WIDTH,
        num_images_per_prompt=1,
    )

    return output.images[0]


def main():
    args = parse_args()

    if args.verbose:
        logger.setLevel(logging.INFO)

    # 設置精度
    dtype = torch.float16 if args.model_precision_type == "fp16" else torch.float32

    # 設備檢測
    if torch.cuda.is_available() and torch.backends.cuda.is_built():
        device = "cuda"
        print("CUDA device found, enabling.")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
        print("Apple MPS device found, enabling.")
    else:
        device = "cpu"
        print("No CUDA or MPS devices found, running on CPU.")

    # 1. 預處理參考圖片
    logger.info(f"Preprocessing reference image: {args.input_image}")
    from image_preprocessor import ImagePreprocessor

    preprocessor = ImagePreprocessor()
    reference_image = preprocessor.process(args.input_image)
    logger.info("Reference image preprocessed.")

    # 2. 加載 pipeline + IP-Adapter
    pipeline = load_pipeline(device, dtype)

    # 3. 生成皮膚
    generated_image = generate_skin_from_image(
        pipeline=pipeline,
        reference_image=reference_image,
        prompt=args.prompt,
        ip_adapter_scale=args.ip_adapter_scale,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        device=device,
    )

    # 4. 後處理：裁剪 + 縮放 + 恢復透明度（復用現有邏輯）
    logger.info("Extracting Minecraft skin from generated image.")
    minecraft_skin = extract_minecraft_skin(generated_image)

    # 5. 保存
    os.makedirs("output_minecraft_skins", exist_ok=True)
    output_path = os.path.join("output_minecraft_skins", args.filename)
    minecraft_skin.save(output_path)
    logger.info(f"Skin saved to: {output_path}")

    print(f"Successfully generated skin from image: {output_path}")

    # 6. 可選 3D 模型
    if args.model_3d:
        os.chdir("Scripts")
        os.system(f"python to_3d_model.py '{args.filename}'")
        os.chdir("..")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Minecraft skin from a reference image using IP-Adapter"
    )
    parser.add_argument(
        "input_image", type=str, help="Path to the reference image (PNG/JPG/WebP)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a minecraft skin",
        help="Text prompt to guide generation (default: 'a minecraft skin')",
    )
    parser.add_argument(
        "--ip_adapter_scale",
        type=float,
        default=0.7,
        help="IP-Adapter strength 0.0-1.0 (default: 0.7)",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=30,
        help="Number of denoising steps (default: 30)",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Text guidance scale (default: 7.5)",
    )
    parser.add_argument(
        "--model_precision_type",
        type=str,
        default="fp16",
        choices=["fp16", "fp32"],
        help="Model precision (default: fp16)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed, 0 for random (default: 42)"
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="output-skin-from-image.png",
        help="Output filename (default: output-skin-from-image.png)",
    )
    parser.add_argument(
        "--model_3d",
        action="store_true",
        default=False,
        help="Also generate 3D model preview",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Verbose output",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.ERROR,
        format="[%(asctime)s] %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("minecraft-skins-img2skin")
    main()
