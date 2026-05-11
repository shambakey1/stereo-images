"""
Stereo Image Generator using Stable Diffusion 1.5 + MiDaS Depth Estimation
===========================================================================
Pipeline:
  1. Load input image
  2. Estimate depth map using MiDaS (monocular depth estimation)
  3. Use depth map to shift pixels and synthesize a right-eye view via SD img2img
  4. Combine original (left) + generated (right) into a side-by-side stereo image

Environment: Compatible with Google Colab (CPU) and Kaggle (T4 GPU)
Dependencies: pip install diffusers transformers accelerate torch torchvision Pillow timm
"""

import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
import torchvision.transforms as T
import warnings
import os

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CONFIG = {
    "model_id": "/d/srta/org/llm/stable_diffusion/sd1-5", #"runwayml/stable-diffusion-v1-5",
    "input_image_path": "/d/srta/projects/diffusers/StereoDiffusion/kaggle_stereo_images/pic_2_left_org.jpg", #"input.jpg",          # <-- Change to your image path
    "output_dir": ".",                            # directory to save output files
    "output_stem": "apple_output",                # base filename (no extension)
    "target_size": (512, 512),                # SD works best at 512×512

    # Stereo depth-shift parameters
    "baseline_shift_px": 20,                  # max pixel shift for stereo effect
    "depth_blur_sigma": 2.0,                  # smooth depth map for cleaner shifts

    # SD img2img parameters
    "prompt": "exact copy, preserve all text, no changes", #"a realistic scene, high quality, sharp details",
    "negative_prompt": "blurry, distorted, changed text, different writing, altered letters, duplicate, ugly, low quality",
    "strength": 0.12,                         # low = preserves structure, high = more change
    "guidance_scale": 7.5,
    "num_inference_steps": 15,                # bump to 30-50 on GPU for better quality
    "seed": 42,
}


# ─────────────────────────────────────────────
# DEVICE SETUP
# ─────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.float16
        print(f"[✓] GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        print("[!] No GPU found — running on CPU (slower, ~10 steps recommended)")
        CONFIG["num_inference_steps"] = 10   # CPU override
    return device, dtype


# ─────────────────────────────────────────────
# DEPTH ESTIMATION (MiDaS)
# ─────────────────────────────────────────────
def estimate_depth(image: Image.Image, device: torch.device) -> np.ndarray:
    """
    Returns a normalized depth map (float32, shape HxW, range [0,1])
    where 1.0 = closest to camera, 0.0 = farthest.
    """
    print("[*] Loading MiDaS depth model...")
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
    midas.to(device).eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    transform = midas_transforms.small_transform

    img_np = np.array(image.convert("RGB"))
    input_tensor = transform(img_np).to(device)

    with torch.no_grad():
        depth = midas(input_tensor)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=image.size[::-1],           # (H, W)
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_np = depth.cpu().numpy()

    # Normalize to [0, 1] — closer objects = higher value
    d_min, d_max = depth_np.min(), depth_np.max()
    depth_norm = (depth_np - d_min) / (d_max - d_min + 1e-8)

    print("[✓] Depth map estimated.")
    return depth_norm.astype(np.float32)


# ─────────────────────────────────────────────
# DEPTH-BASED PIXEL SHIFT (Right-Eye View)
# ─────────────────────────────────────────────
def depth_shift_image(image: Image.Image, depth_map: np.ndarray, shift_px: int) -> Image.Image:
    """
    Shifts each pixel horizontally by an amount proportional to its depth,
    simulating the right-eye view of a stereo pair.
    """
    from scipy.ndimage import gaussian_filter

    img_np = np.array(image.convert("RGB"), dtype=np.float32)
    H, W, C = img_np.shape

    # Smooth depth map to reduce edge artifacts
    depth_smooth = gaussian_filter(depth_map, sigma=CONFIG["depth_blur_sigma"])

    # Per-pixel horizontal shift (positive = shift right for right-eye view)
    shift_map = (depth_smooth * shift_px).astype(np.int32)

    shifted = np.zeros_like(img_np)
    mask = np.zeros((H, W), dtype=np.float32)   # track filled pixels

    for y in range(H):
        for x in range(W):
            x_new = x + shift_map[y, x]
            if 0 <= x_new < W:
                shifted[y, x_new] = img_np[y, x]
                mask[y, x_new] = 1.0

    # Fill holes via nearest-neighbor inpainting (simple column fill)
    for y in range(H):
        for x in range(W):
            if mask[y, x] == 0:
                # Search left for a filled neighbor
                for dx in range(1, shift_px + 2):
                    if x - dx >= 0 and mask[y, x - dx] == 1.0:
                        shifted[y, x] = shifted[y, x - dx]
                        break

    shifted_img = Image.fromarray(np.clip(shifted, 0, 255).astype(np.uint8))
    return shifted_img


# ─────────────────────────────────────────────
# STABLE DIFFUSION IMG2IMG REFINEMENT
# ─────────────────────────────────────────────
def load_sd_pipeline(device: torch.device, dtype: torch.dtype) -> StableDiffusionImg2ImgPipeline:
    print("[*] Loading Stable Diffusion 1.5 img2img pipeline...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        CONFIG["model_id"],
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe = pipe.to(device)

    # Memory optimizations
    pipe.enable_attention_slicing()
    if device.type == "cpu":
        pipe.enable_sequential_cpu_offload()
    else:
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("[✓] xformers enabled.")
        except Exception:
            print("[!] xformers not available — using default attention.")
        pipe.enable_vae_slicing()

    print("[✓] Pipeline loaded.")
    return pipe


def refine_with_sd(
    pipe: StableDiffusionImg2ImgPipeline,
    shifted_image: Image.Image,
    device: torch.device,
) -> Image.Image:
    """
    Passes the depth-shifted image through SD img2img to fill artifacts
    and improve realism while preserving the overall composition.
    """
    print("[*] Running SD img2img refinement on right-eye view...")

    generator = torch.Generator(device=device).manual_seed(CONFIG["seed"])

    result = pipe(
        prompt=CONFIG["prompt"],
        negative_prompt=CONFIG["negative_prompt"],
        image=shifted_image,
        strength=CONFIG["strength"],
        guidance_scale=CONFIG["guidance_scale"],
        num_inference_steps=CONFIG["num_inference_steps"],
        generator=generator,
    )

    refined = result.images[0]
    print("[✓] SD refinement complete.")
    return refined


# ─────────────────────────────────────────────
# SAVE STEREO OUTPUTS AS SEPARATE FILES
# ─────────────────────────────────────────────
def save_stereo_outputs(
    left_image: Image.Image,
    right_image: Image.Image,
    depth_map: np.ndarray,
) -> tuple:
    """
    Saves three separate files:
      - <stem>_left.png   — original image (left eye)
      - <stem>_right.png  — SD-generated stereo view (right eye)
      - <stem>_depth.png  — MiDaS depth map (grayscale)

    Returns (left_path, right_path, depth_path).
    """
    out_dir  = CONFIG["output_dir"]
    stem     = CONFIG["output_stem"]
    os.makedirs(out_dir, exist_ok=True)

    W, H = left_image.size

    # Ensure right image matches dimensions of left
    right_image = right_image.resize((W, H), Image.LANCZOS)

    # ── Left eye (original) ──────────────────
    left_path = os.path.join(out_dir, f"{stem}_left.png")
    left_image.save(left_path)
    print(f"[✓] Left eye  saved → {left_path}")

    # ── Right eye (SD stereo view) ───────────
    right_path = os.path.join(out_dir, f"{stem}_right.png")
    right_image.save(right_path)
    print(f"[✓] Right eye saved → {right_path}")

    # ── Depth map (grayscale, full resolution) ─
    depth_uint8  = (depth_map * 255).astype(np.uint8)
    depth_img    = Image.fromarray(depth_uint8, mode="L")
    depth_path   = os.path.join(out_dir, f"{stem}_depth.png")
    depth_img.save(depth_path)
    print(f"[✓] Depth map saved → {depth_path}")

    return left_path, right_path, depth_path


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Stereo Image Generator — SD 1.5 + MiDaS Depth")
    print("=" * 55)

    # 1. Load & resize input image
    if not os.path.exists(CONFIG["input_image_path"]):
        raise FileNotFoundError(f"Input image not found: {CONFIG['input_image_path']}")

    original = Image.open(CONFIG["input_image_path"]).convert("RGB")
    original = original.resize(CONFIG["target_size"], Image.LANCZOS)
    print(f"[✓] Input image loaded — size: {original.size}")

    # 2. Device setup
    device, dtype = get_device()

    # 3. Depth estimation
    depth_map = estimate_depth(original, device)

    # 4. Depth-based pixel shift → rough right-eye view
    print("[*] Generating depth-shifted right-eye view...")
    shifted = depth_shift_image(original, depth_map, CONFIG["baseline_shift_px"])

    # 5. SD img2img refinement to fill holes & improve quality
    pipe = load_sd_pipeline(device, dtype)
    stereo_right = refine_with_sd(pipe, shifted, device)

    # 6. Save left, right, and depth as separate files
    left_path, right_path, depth_path = save_stereo_outputs(
        original, stereo_right, depth_map
    )

    print("\n[✓] Done!")
    print(f"    Left  → {left_path}")
    print(f"    Right → {right_path}")
    print(f"    Depth → {depth_path}")
    return left_path, right_path, depth_path


if __name__ == "__main__":
    main()
