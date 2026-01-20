# Getting Started

This guide provides a walkthrough of a minimal implementation using the **Segmentor** pipeline. We will cover how to load the vision backbones, register a keyframe with interactive clicks (prompts), and generate a segmentation heatmap for a single image.

---

## Setup and Imports

First, we import the necessary libraries. **Segmentor** relies on `torch` for computation, `torchvision` for image transformations, and specific encoders for feature extraction.

```python
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision.transforms.v2 import Compose, ToImage, ToDtype
from transformers.image_utils import load_image

from segmentor.utils.pipeline.main import Segmentor
from segmentor.utils.models.dinov3 import DINOv3ImageEncoder
from segmentor.utils.models.anyup import AnyUp
from segmentor.utils.models.clip import CLIPImageEncoder
from segmentor.helpers.device import DEVICE

```

---

## Initialize Vision Models

The pipeline uses a tripartite architecture to understand images. You should define a `HF_CACHE_DIR` to store the model weights (e.g., on a fast SSD).

* **DINOv3**: Extracts dense, self-supervised visual features. We use the `vits16` variant for extracting patch embeddings.
* **AnyUp**: An upsampling module that takes the coarse DINOv3 patches and interpolates/refines them to match the original image resolution.
* **CLIP**: Provides global semantic context, helping the model understand "what" the object is at a high level.

```python
# Load DINOv3 for dense features
dinov3 = DINOv3ImageEncoder(
    model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
    cache_dir=HF_CACHE_DIR,
    dtype=torch.float32,
).to(device=DEVICE)

# Load AnyUp for resolution upscaling
anyup = AnyUp(use_natten=False).to(device=DEVICE)

# Load CLIP for semantic guidance
clip = CLIPImageEncoder(cache_dir=HF_CACHE_DIR).to(device=DEVICE)

```

---

## Creating the Segmentor object

With the models loaded, we initialize the `Segmentor` object. The `keyframe_similarity_threshold` determines how aggressively the model correlates new frames with the registered keyframe.

```python
segmentor = Segmentor(
    dinov3=dinov3,
    anyup=anyup,
    clip=clip,
    keyframe_similarity_threshold=0.85,
    device=DEVICE,
)

```

---

## Preparing the Image

Segmentor expects a `torch.Tensor` in the range `[0, 1]`. We resize the image to a standard  resolution to ensure consistent feature extraction.

```python
img_to_tensor = Compose([
    ToImage(),
    ToDtype(torch.float32, scale=True),
])

image = load_image(IMAGE_PATH).resize((512, 512))
image_tensor = img_to_tensor(image)

```

---

## Registering a Keyframe

To tell the model what to segment, you must "register" a keyframe. This is done by providing **Positive Clicks** (points on the object) and **Negative Clicks** (points on the background).

> **Note:** Coordinates are provided as `(x, y)` pixel values.

```python
segmentor.register_keyframe(
    image=image_tensor,
    pos_pixel_coords=[
        (100, 300), (200, 320), (250, 310), # Foreground
    ],
    neg_pixel_coords=[
        (50, 50), (400, 100), (450, 450),   # Background
    ],
)

```

---

## Running Inference and Visualization

Finally, we call `segmentor.step()`. The `gamma` parameter controls the sharpness of the result; higher values increase the contrast between the predicted foreground and background.

The output is a 2D heatmap where higher values indicate a higher probability of the pixel belonging to the target object.

```python
# Run the pipeline
output = segmentor.step(image=image_tensor, gamma=30)

# Visualize the heatmap
plt.figure(figsize=(6, 6))
sns.heatmap(output.segmentation.cpu(), cmap="viridis")
plt.title("Segmentor Output")
plt.show()
```

---

> [!TIP]
> The full code for this example can be found in [example.py](example.py)
