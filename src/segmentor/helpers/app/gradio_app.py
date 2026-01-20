import gradio as gr
import numpy as np
import torch
from pathlib import Path
from PIL import Image as PILImage, ImageDraw
from torchvision.transforms.v2 import Compose, ToImage, ToDtype
import matplotlib.pyplot as plt
import traceback
import os

# Enable CUDA error checking
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Import your segmentor components
from segmentor.utils.pipeline.main import Segmentor
from segmentor.utils.models.dinov3 import DINOv3ImageEncoder
from segmentor.utils.models.anyup import AnyUp
from segmentor.utils.models.clip import CLIPImageEncoder
from segmentor.helpers.device import DEVICE

print(f"Using device: {DEVICE}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")


# Global state
class AppState:
    def __init__(self):
        self.images = []
        self.current_idx = 0
        self.pos_points = []
        self.neg_points = []
        self.segmentor = None
        self.current_image = None
        self.display_image = None  # Image actually shown in UI
        self.mode = "define_exemplars"  # 'define_exemplars' or 'navigate'
        self.click_mode = "positive"  # 'positive' or 'negative'
        self.segmentation = None


state = AppState()


def load_models():
    """Load the segmentor models"""
    try:
        WEIGHTS_DIR = Path("/mnt/toshiba_hdd/weights")
        HF_CACHE_DIR = WEIGHTS_DIR / "hf_models"

        print("Loading DINOv3...")
        dinov3 = DINOv3ImageEncoder(
            model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
            cache_dir=HF_CACHE_DIR,
            attn_implementation="sdpa",
        ).to(device=DEVICE)

        print("Loading AnyUp...")
        anyup = AnyUp(use_natten=False).to(device=DEVICE)

        print("Loading CLIP...")
        clip = CLIPImageEncoder(cache_dir=HF_CACHE_DIR).to(device=DEVICE)

        print("Creating Segmentor...")
        segmentor = Segmentor(
            dinov3=dinov3,
            anyup=anyup,
            clip=clip,
            keyframe_similarity_threshold=0.85,
            device=DEVICE,
        )

        print("Models loaded successfully!")
        return segmentor
    except Exception as e:
        print(f"Error loading models: {e}")
        traceback.print_exc()
        raise


def img_to_tensor(image):
    """Convert PIL image to tensor"""
    transform = Compose([ToImage(), ToDtype(torch.float32, scale=True)])
    return transform(image)


def load_images_from_folder(folder_path):
    """Load all images from folder and sort by name"""
    folder = Path(folder_path)
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    image_files = []
    for ext in valid_extensions:
        image_files.extend(folder.glob(f"*{ext}"))
        image_files.extend(folder.glob(f"*{ext.upper()}"))

    # Sort by filename
    image_files = sorted(image_files, key=lambda x: x.name)
    return image_files


def draw_points_on_display_image(image, pos_points, neg_points, scale_x, scale_y):
    """Draw points on display image with proper scaling"""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    radius = 8

    # Draw positive points (green) - scale coordinates back to display
    for orig_x, orig_y in pos_points:
        x = int(orig_x / scale_x)
        y = int(orig_y / scale_y)
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill="green",
            outline="white",
            width=2,
        )

    # Draw negative points (red)
    for orig_x, orig_y in neg_points:
        x = int(orig_x / scale_x)
        y = int(orig_y / scale_y)
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill="red",
            outline="white",
            width=2,
        )

    return img


def draw_points_on_image(image, pos_points, neg_points):
    """Draw points on image at original coordinates"""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    radius = 8

    # Draw positive points (green)
    for x, y in pos_points:
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill="green",
            outline="white",
            width=2,
        )

    # Draw negative points (red)
    for x, y in neg_points:
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill="red",
            outline="white",
            width=2,
        )

    return img


def overlay_segmentation(image, segmentation_mask):
    """Overlay segmentation mask on image"""
    # Convert image to numpy
    img_array = np.array(image)

    # Create segmentation overlay
    mask = segmentation_mask.cpu().numpy()

    # Create a colormap overlay
    cmap = plt.get_cmap("jet")
    colored_mask = cmap(mask)[:, :, :3]  # RGB only
    colored_mask = (colored_mask * 255).astype(np.uint8)

    # Blend with original image
    alpha = 0.5
    blended = (alpha * colored_mask + (1 - alpha) * img_array).astype(np.uint8)

    return PILImage.fromarray(blended)


def load_folder(folder_path):
    """Load images from folder"""
    try:
        image_files = load_images_from_folder(folder_path)
        if not image_files:
            return (
                None,
                "❌ No images found in folder",
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            )

        state.images = image_files
        state.current_idx = 0
        state.pos_points = []
        state.neg_points = []
        state.mode = "define_exemplars"
        state.click_mode = "positive"
        state.segmentor = load_models()

        # Load first image
        state.current_image = PILImage.open(state.images[0])

        # Create display version (resized to fit height while maintaining aspect ratio)
        display_height = 600
        orig_w, orig_h = state.current_image.size
        aspect_ratio = orig_w / orig_h
        display_width = int(display_height * aspect_ratio)

        state.display_image = state.current_image.resize(
            (display_width, display_height), PILImage.Resampling.LANCZOS
        )

        print(f"\n📂 Loaded {len(image_files)} images")
        print(f"📁 First image: {state.images[0].name}")
        print(f"🖼️  Original size: {state.current_image.size}")
        print(f"🖼️  Display size: {state.display_image.size}")

        info = f"✅ Loaded {len(image_files)} images\n📁 Current: {state.images[0].name} (1/{len(state.images)})"

        return (
            state.display_image,
            info,
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
        )
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return (
            None,
            error_msg,
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )


def handle_click(evt: gr.SelectData):
    """Handle click events on the image"""
    if state.mode != "define_exemplars":
        return state.display_image, get_status_text()

    # evt.index gives coordinates on the DISPLAYED image
    display_x, display_y = evt.index[0], evt.index[1]

    # Get original and display dimensions
    orig_w, orig_h = state.current_image.size
    disp_w, disp_h = state.display_image.size

    # Calculate scaling factors
    scale_x = orig_w / disp_w
    scale_y = orig_h / disp_h

    # Map to original image coordinates
    x = int(display_x * scale_x)
    y = int(display_y * scale_y)

    # Clamp to image bounds
    x = max(0, min(x, orig_w - 1))
    y = max(0, min(y, orig_h - 1))

    print(f"\n🖱️ Click at display: ({display_x}, {display_y})")
    print(f"   Display size: {disp_w}x{disp_h}")
    print(f"   Original size: {orig_w}x{orig_h}")
    print(f"   Scale: ({scale_x:.2f}, {scale_y:.2f})")
    print(f"   Mapped to: ({x}, {y})")

    # Add point
    if state.click_mode == "negative":
        state.neg_points.append((x, y))
        print(f"   ❌ Added NEGATIVE point")
    else:
        state.pos_points.append((x, y))
        print(f"   ✅ Added POSITIVE point")

    print(f"   Total: {len(state.pos_points)} pos, {len(state.neg_points)} neg")

    # Draw points on DISPLAY image
    img_with_points = draw_points_on_display_image(
        state.display_image, state.pos_points, state.neg_points, scale_x, scale_y
    )

    return img_with_points, get_status_text()


def toggle_click_mode(mode):
    """Toggle between positive and negative click mode"""
    state.click_mode = mode
    print(f"\n🔄 Click mode changed to: {mode.upper()}")
    return get_status_text()


def reset_points():
    """Reset all points"""
    print(f"\n🔄 Resetting points...")
    print(f"   Cleared {len(state.pos_points)} positive points")
    print(f"   Cleared {len(state.neg_points)} negative points")
    state.pos_points = []
    state.neg_points = []
    return state.display_image, get_status_text()


def register_exemplars():
    """Register the exemplars and generate segmentation"""
    if not state.pos_points:
        return state.display_image, "❌ Please add at least one positive point"

    try:
        print(f"\nRegistering exemplars...")
        print(f"Positive points: {state.pos_points}")
        print(f"Negative points: {state.neg_points}")
        print(f"Image size: {state.current_image.size}")

        # Convert image to tensor
        image_tensor = img_to_tensor(state.current_image)
        print(f"Image tensor shape: {image_tensor.shape}")

        # Validate coordinates
        h, w = state.current_image.size[1], state.current_image.size[0]
        for x, y in state.pos_points + state.neg_points:
            if x < 0 or x >= w or y < 0 or y >= h:
                return (
                    state.display_image,
                    f"❌ Invalid coordinates: ({x}, {y}) - image size is {w}x{h}",
                )

        # Register keyframe with proper coordinate format (y, x) for tensor indexing
        print("Registering keyframe...")
        state.segmentor.register_keyframe(
            image=image_tensor,
            pos_pixel_coords=[(y, x) for x, y in state.pos_points],
            neg_pixel_coords=[(y, x) for x, y in state.neg_points]
            if state.neg_points
            else [],
        )

        # Generate segmentation
        print("Generating segmentation...")
        output = state.segmentor.step(image=image_tensor, gamma=50)

        if output is None:
            return state.display_image, "❌ No keyframes registered"

        state.segmentation = output.segmentation

        print(f"Segmentation shape: {state.segmentation.shape}")
        print(
            f"Segmentation range: [{state.segmentation.min():.3f}, {state.segmentation.max():.3f}]"
        )

        # Overlay segmentation on original image
        img_with_seg = overlay_segmentation(state.current_image, state.segmentation)

        # Create display version
        display_height = 600
        orig_w, orig_h = img_with_seg.size
        aspect_ratio = orig_w / orig_h
        display_width = int(display_height * aspect_ratio)
        state.display_image = img_with_seg.resize(
            (display_width, display_height), PILImage.Resampling.LANCZOS
        )

        # Switch to navigate mode
        state.mode = "navigate"

        return (
            state.display_image,
            "✅ Exemplars registered! Use Prev/Next to navigate.",
        )
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return state.display_image, error_msg


def navigate_prev():
    """Navigate to previous image"""
    if state.current_idx > 0:
        state.current_idx -= 1
        state.current_image = PILImage.open(state.images[state.current_idx])

        # Generate segmentation
        image_tensor = img_to_tensor(state.current_image)
        output = state.segmentor.step(image=image_tensor, gamma=50)

        if output is None:
            # Create display image
            display_height = 600
            orig_w, orig_h = state.current_image.size
            aspect_ratio = orig_w / orig_h
            display_width = int(display_height * aspect_ratio)
            state.display_image = state.current_image.resize(
                (display_width, display_height), PILImage.Resampling.LANCZOS
            )
            return state.display_image, "❌ No keyframes registered"

        state.segmentation = output.segmentation

        # Overlay on original image
        img_with_seg = overlay_segmentation(state.current_image, state.segmentation)

        # Create display version
        display_height = 600
        orig_w, orig_h = img_with_seg.size
        aspect_ratio = orig_w / orig_h
        display_width = int(display_height * aspect_ratio)
        state.display_image = img_with_seg.resize(
            (display_width, display_height), PILImage.Resampling.LANCZOS
        )

        return state.display_image, get_status_text()

    return state.display_image, get_status_text()


def navigate_next():
    """Navigate to next image"""
    if state.current_idx < len(state.images) - 1:
        state.current_idx += 1
        state.current_image = PILImage.open(state.images[state.current_idx])

        # Generate segmentation
        image_tensor = img_to_tensor(state.current_image)
        output = state.segmentor.step(image=image_tensor, gamma=50)

        if output is None:
            # Create display image
            display_height = 600
            orig_w, orig_h = state.current_image.size
            aspect_ratio = orig_w / orig_h
            display_width = int(display_height * aspect_ratio)
            state.display_image = state.current_image.resize(
                (display_width, display_height), PILImage.Resampling.LANCZOS
            )
            return state.display_image, "❌ No keyframes registered"

        state.segmentation = output.segmentation

        # Overlay on original image
        img_with_seg = overlay_segmentation(state.current_image, state.segmentation)

        # Create display version
        display_height = 600
        orig_w, orig_h = img_with_seg.size
        aspect_ratio = orig_w / orig_h
        display_width = int(display_height * aspect_ratio)
        state.display_image = img_with_seg.resize(
            (display_width, display_height), PILImage.Resampling.LANCZOS
        )

        return state.display_image, get_status_text()

    return state.display_image, get_status_text()


def start_define_exemplars():
    """Start defining exemplars mode"""
    state.mode = "define_exemplars"
    state.pos_points = []
    state.neg_points = []
    state.click_mode = "positive"

    # Recreate display image
    display_height = 600
    orig_w, orig_h = state.current_image.size
    aspect_ratio = orig_w / orig_h
    display_width = int(display_height * aspect_ratio)
    state.display_image = state.current_image.resize(
        (display_width, display_height), PILImage.Resampling.LANCZOS
    )

    return state.display_image, get_status_text()


def get_status_text():
    """Get current status text"""
    if not state.images:
        return "No images loaded"

    status = f"📁 Image {state.current_idx + 1}/{len(state.images)}: {state.images[state.current_idx].name}\n"
    status += f"🎯 Mode: {state.mode}\n"

    if state.mode == "define_exemplars":
        status += f"🖱️ Click mode: {state.click_mode.upper()}\n"
        status += f"✅ Positive points: {len(state.pos_points)}\n"
        status += f"❌ Negative points: {len(state.neg_points)}\n"
        status += "💡 Click on image to add points, then click 'Done' to generate segmentation"
    else:
        status += (
            "💡 Use Prev/Next to navigate or 'Define Exemplars' to add more keyframes"
        )

    return status


# Create Gradio interface
with gr.Blocks(title="Interactive Segmentor") as demo:
    gr.Markdown("# 🎯 Interactive Image Segmentor")
    gr.Markdown(
        "Load a folder of images and interactively define segmentation exemplars"
    )

    with gr.Row():
        with gr.Column(scale=3):
            image_display = gr.Image(
                label="Image Viewer",
                type="pil",
                interactive=False,
                height=600,
                image_mode="RGB",
                sources=[],
                show_label=True,
            )

            with gr.Row():
                prev_btn = gr.Button("⬅️ Previous", size="lg", visible=False)
                next_btn = gr.Button("Next ➡️", size="lg", visible=False)

        with gr.Column(scale=1):
            gr.Markdown("### 📂 Load Images")
            folder_input = gr.Textbox(
                label="Folder Path",
                placeholder="/path/to/image/folder",
                value="/mnt/toshiba_hdd/datasets/test_images",
            )
            load_btn = gr.Button("Load Folder", variant="primary")

            gr.Markdown("---")

            status_text = gr.Textbox(label="Status", lines=8, interactive=False)

            gr.Markdown("---")

            with gr.Group(visible=False) as controls_group:
                gr.Markdown("### 🎨 Define Exemplars")

                with gr.Row():
                    pos_mode_btn = gr.Button("✅ Positive Mode", variant="primary")
                    neg_mode_btn = gr.Button("❌ Negative Mode")

                with gr.Row():
                    reset_btn = gr.Button("🔄 Reset Points")
                    done_btn = gr.Button("✅ Done", variant="primary")

            gr.Markdown("---")

            with gr.Group(visible=False) as nav_group:
                gr.Markdown("### 🧭 Navigation")
                define_exemplars_btn = gr.Button("🎯 Define New Exemplars", size="lg")

    # Event handlers
    load_btn.click(
        fn=load_folder,
        inputs=[folder_input],
        outputs=[
            image_display,
            status_text,
            controls_group,
            nav_group,
            prev_btn,
            next_btn,
        ],
    )

    image_display.select(fn=handle_click, outputs=[image_display, status_text])

    pos_mode_btn.click(fn=lambda: toggle_click_mode("positive"), outputs=[status_text])

    neg_mode_btn.click(fn=lambda: toggle_click_mode("negative"), outputs=[status_text])

    reset_btn.click(fn=reset_points, outputs=[image_display, status_text])

    done_btn.click(fn=register_exemplars, outputs=[image_display, status_text])

    prev_btn.click(fn=navigate_prev, outputs=[image_display, status_text])

    next_btn.click(fn=navigate_next, outputs=[image_display, status_text])

    define_exemplars_btn.click(
        fn=start_define_exemplars, outputs=[image_display, status_text]
    )


if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
    )
