"""
Minimal example: Segmentor pipeline on a single image

This script shows:
1. How to load the required models
2. How to create a Segmentor
3. How to register a keyframe with positive/negative clicks
4. How to run segmentation on one image
"""

import torch
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision.transforms.v2 import Compose, ToImage, ToDtype
from transformers.image_utils import load_image

# ------------------------
# Segmentor imports
# ------------------------
from segmentor.utils.pipeline.main import Segmentor
from segmentor.utils.models.dinov3 import DINOv3ImageEncoder
from segmentor.utils.models.anyup import AnyUp
from segmentor.utils.models.clip import CLIPImageEncoder
from segmentor.helpers.device import DEVICE

# ------------------------
# User-configurable paths
# ------------------------

# Directory where HuggingFace models will be cached
# (e.g., a fast SSD with enough space for ViT models)
HF_CACHE_DIR = Path("/path/to/huggingface/cache")

# Path to a single RGB image (JPEG/PNG)
# This image will be segmented
IMAGE_PATH = "/path/to/your/image.jpg"

sns.set_theme("talk")

# ------------------------
# Load vision models
# ------------------------

# DINOv3: dense visual features
dinov3 = DINOv3ImageEncoder(
    model_name="facebook/dinov3-vith16plus-pretrain-lvd1689m",
    cache_dir=HF_CACHE_DIR,
    attn_implementation="sdpa",
    dtype=torch.float32,
).to(device=DEVICE)

# AnyUp: upsamples DINOv3 patch features to image resolution
anyup = AnyUp(
    use_natten=False,  # set True if you compiled NATTEN
).to(device=DEVICE)

# CLIP: global semantic guidance
clip = CLIPImageEncoder(
    cache_dir=HF_CACHE_DIR,
).to(device=DEVICE)

# ------------------------
# Create Segmentor
# ------------------------

segmentor = Segmentor(
    dinov3=dinov3,
    anyup=anyup,
    clip=clip,
    keyframe_similarity_threshold=0.85,
    device=DEVICE,
)

# ------------------------
# Image loading & preprocessing
# ------------------------

# Converts PIL image -> torch tensor in [0, 1]
img_to_tensor = Compose(
    [
        ToImage(),
        ToDtype(torch.float32, scale=True),
    ]
)

# Load image from disk (or URL)
image = load_image(IMAGE_PATH)

# Resize to something reasonable (Segmentor assumes fixed size per call)
image = image.resize((512, 512))

# Shape: [3, H, W], dtype=float32, range=[0,1]
image_tensor = img_to_tensor(image)

# ------------------------
# Register a keyframe
# ------------------------

# Positive clicks = pixels that belong to the target object/region
# Negative clicks = pixels that definitely do NOT belong
#
# Coordinates are (x, y) in image pixel space
segmentor.register_keyframe(
    image=image_tensor,
    pos_pixel_coords=[
        (100, 300),  # example foreground point
        (200, 320),
        (250, 310),
    ],
    neg_pixel_coords=[
        (50, 50),  # example background point
        (400, 100),
        (450, 450),
    ],
)

# ------------------------
# Run segmentation
# ------------------------

# gamma controls sharpness / confidence amplification
output = segmentor.step(
    image=image_tensor,
    gamma=30,
)

# output.segmentation is a [H, W] tensor
# Higher values = more likely to belong to the target

# ------------------------
# Visualize result
# ------------------------

plt.figure(figsize=(6, 6))
sns.heatmap(
    output.segmentation.cpu(),
    cmap="viridis",
)
plt.title("Segmentor Output (Single Image)")
plt.tight_layout()
plt.show()
