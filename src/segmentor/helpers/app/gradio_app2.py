import gradio as gr
import numpy as np
import torch
from pathlib import Path
from PIL import Image as PILImage, ImageDraw
from torchvision.transforms.v2 import Compose, ToImage, ToDtype
import matplotlib.pyplot as plt
import traceback
import os
import gc

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

# --- CONFIGURATION ---
DISPLAY_WIDTH = 512  # For the UI viewer
PROCESSING_MAX_SIDE = (
    DISPLAY_WIDTH  # Max dimension for the actual processing (Prevent OOM)
)


# Global state
class AppState:
    def __init__(self):
        self.images = []
        self.root_folder = None
        self.current_idx = 0
        self.pos_points = []
        self.neg_points = []
        self.segmentor = None
        self.current_image = None  # This will now store the RESIZED (1024px) image
        self.display_image = None
        self.mode = "define_exemplars"
        self.click_mode = "positive"
        self.segmentation = None

        # Stats tracking
        self.visited_indices = set()
        self.labeled_indices = set()


state = AppState()

# --- HELPER FUNCTIONS ---


def get_status_text():
    """
    Moved to top to prevent NameError.
    Returns the status string for the UI.
    """
    if not state.images:
        return "No images loaded"

    status = f"📁 Image {state.current_idx + 1}/{len(state.images)}: {state.images[state.current_idx].name}\n"

    if state.mode == "define_exemplars":
        status += f"🖱️ Mode: Exemplar Definition ({state.click_mode.upper()})\n"
        status += (
            f"   Points: {len(state.pos_points)} (+) / {len(state.neg_points)} (-)"
        )
    else:
        status += "👁️ Mode: Navigation / Review"

    return status


def get_stats_string():
    """Calculate and return formatted statistics"""
    total = len(state.images)
    seen = len(state.visited_indices)
    unseen = total - seen
    labeled = len(state.labeled_indices)
    seen_unlabeled = seen - labeled

    return (
        f"📊 **Statistics**\n"
        f"• Total Images: {total}\n"
        f"• Unseen: {unseen}\n"
        f"• Seen: {seen}\n"
        f"  - Labeled: {labeled}\n"
        f"  - Unlabeled: {seen_unlabeled}"
    )


def cleanup_gpu():
    """Force garbage collection and clear CUDA cache"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_models():
    """Load the segmentor models"""
    try:
        WEIGHTS_DIR = Path("/mnt/toshiba_hdd/weights")
        HF_CACHE_DIR = WEIGHTS_DIR / "hf_models"

        print("Loading DINOv3 (Huge)...")
        # Using Small (vits16) for stability
        dinov3 = DINOv3ImageEncoder(
            # model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
            model_name="facebook/dinov3-vith16plus-pretrain-lvd1689m",
            cache_dir=HF_CACHE_DIR,
            attn_implementation="sdpa",
            dtype=torch.float32,
        ).to(device=DEVICE)

        print("Loading AnyUp...")
        anyup = AnyUp(use_natten=False).to(device=DEVICE)

        print("Loading CLIP...")
        clip = CLIPImageEncoder(cache_dir=HF_CACHE_DIR).to(device=DEVICE)

        print("Creating Segmentor...")
        # Using 0.60 threshold to ensure masks appear
        segmentor = Segmentor(
            dinov3=dinov3,
            anyup=anyup,
            clip=clip,
            keyframe_similarity_threshold=0.60,
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

    image_files = sorted(image_files, key=lambda x: x.name)
    return image_files


def resize_to_limit(image, max_side=PROCESSING_MAX_SIDE):
    """
    Resize image so that the longest side is at most max_side.
    This runs immediately on load to prevent OOM.
    """
    w, h = image.size
    if max(w, h) <= max_side:
        return image

    scale = max_side / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    print(f"📉 Resizing input: {w}x{h} -> {new_w}x{new_h} (Limit: {max_side})")
    return image.resize((new_w, new_h), PILImage.Resampling.LANCZOS)


def resize_for_display(image):
    """Resize image to fixed width of 512 for the UI"""
    orig_w, orig_h = image.size
    w_percent = DISPLAY_WIDTH / float(orig_w)
    h_size = int((float(orig_h) * float(w_percent)))
    return image.resize((DISPLAY_WIDTH, h_size), PILImage.Resampling.LANCZOS)


def draw_points_on_display_image(image, pos_points, neg_points, scale_x, scale_y):
    """Draw points on display image with proper scaling"""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    radius = 5

    # Draw positive points (green)
    for orig_x, orig_y in pos_points:
        x = int(orig_x / scale_x)
        y = int(orig_y / scale_y)
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill="#00FF00",
            outline="white",
            width=2,
        )

    # Draw negative points (red)
    for orig_x, orig_y in neg_points:
        x = int(orig_x / scale_x)
        y = int(orig_y / scale_y)
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill="#FF0000",
            outline="white",
            width=2,
        )

    return img


def overlay_segmentation(image, segmentation_mask):
    """Overlay binary segmentation mask on image."""
    img_array = np.array(image)

    if isinstance(segmentation_mask, torch.Tensor):
        mask = segmentation_mask.detach().cpu().numpy()
    else:
        mask = segmentation_mask

    if mask.max() == 0:
        return image

    # Thresholding
    threshold = np.percentile(mask, 90)
    binary_mask = mask >= threshold

    # Create colored overlay (Red)
    overlay = np.zeros_like(img_array)
    overlay[binary_mask] = [255, 0, 0]

    base_img = PILImage.fromarray(img_array).convert("RGBA")
    overlay_img = PILImage.fromarray(overlay).convert("RGBA")

    # Alpha blending
    alpha_channel = np.zeros(mask.shape, dtype=np.uint8)
    alpha_channel[binary_mask] = 100

    overlay_img.putalpha(PILImage.fromarray(alpha_channel))
    combined = PILImage.alpha_composite(base_img, overlay_img)

    return combined.convert("RGB")


# --- CORE LOGIC ---


def load_folder(folder_path):
    """Load images from folder and resize the first one immediately"""
    try:
        image_files = load_images_from_folder(folder_path)
        if not image_files:
            return (None, "❌ No images found", *[gr.update(visible=False)] * 4, "")

        state.images = image_files
        state.root_folder = Path(folder_path)
        state.current_idx = 0
        state.pos_points = []
        state.neg_points = []
        state.mode = "define_exemplars"
        state.click_mode = "positive"
        state.visited_indices = {0}
        state.labeled_indices = set()

        if state.segmentor is None:
            state.segmentor = load_models()

        # Load and IMMEDIATELY RESIZE
        raw_img = PILImage.open(state.images[0]).convert("RGB")
        state.current_image = resize_to_limit(raw_img)  # <--- Critical Resize

        # Create display version from the already resized image
        state.display_image = resize_for_display(state.current_image)

        info = (
            f"✅ Loaded {len(image_files)} images\n📁 Current: {state.images[0].name}"
        )

        return (
            state.display_image,
            info,
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            get_stats_string(),
        )
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return (None, error_msg, *[gr.update(visible=False)] * 4, "")


def handle_click(evt: gr.SelectData):
    if state.mode != "define_exemplars":
        return state.display_image, get_status_text()

    display_x, display_y = evt.index[0], evt.index[1]

    # Map from Display Size (512px width) -> Processing Size (1024px max side)
    proc_w, proc_h = state.current_image.size
    disp_w, disp_h = state.display_image.size

    scale_x = proc_w / disp_w
    scale_y = proc_h / disp_h

    x = int(display_x * scale_x)
    y = int(display_y * scale_y)

    x = max(0, min(x, proc_w - 1))
    y = max(0, min(y, proc_h - 1))

    if state.click_mode == "negative":
        state.neg_points.append((x, y))
    else:
        state.pos_points.append((x, y))

    img_with_points = draw_points_on_display_image(
        state.display_image, state.pos_points, state.neg_points, scale_x, scale_y
    )

    return img_with_points, get_status_text()


def toggle_click_mode(mode):
    state.click_mode = mode
    return get_status_text()


def reset_points():
    state.pos_points = []
    state.neg_points = []
    state.display_image = resize_for_display(state.current_image)
    return state.display_image, get_status_text()


def register_exemplars():
    """Register exemplars using the resized image"""
    if not state.pos_points:
        return state.display_image, "❌ Please add points", get_stats_string()

    image_tensor = None
    output = None

    try:
        # Convert the already-resized image to tensor
        image_tensor = img_to_tensor(state.current_image)

        state.segmentor.register_keyframe(
            image=image_tensor,
            pos_pixel_coords=[(y, x) for x, y in state.pos_points],
            neg_pixel_coords=[(y, x) for x, y in state.neg_points]
            if state.neg_points
            else [],
        )

        output = state.segmentor.step(image=image_tensor, gamma=50)

        if output is None:
            del image_tensor
            cleanup_gpu()
            return state.display_image, "❌ No keyframes registered", get_stats_string()

        state.segmentation = output.segmentation.detach().cpu()

        max_val = state.segmentation.max().item()
        print(f"DEBUG: Max Score: {max_val:.4f}")

        del image_tensor
        del output
        cleanup_gpu()

        img_with_seg = overlay_segmentation(state.current_image, state.segmentation)
        state.display_image = resize_for_display(img_with_seg)

        state.mode = "navigate"
        state.labeled_indices.add(state.current_idx)

        return (
            state.display_image,
            f"✅ Mask generated (Max: {max_val:.2f})",
            get_stats_string(),
        )
    except Exception as e:
        if image_tensor is not None:
            del image_tensor
        if output is not None:
            del output
        cleanup_gpu()
        error_msg = f"❌ Error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return state.display_image, error_msg, get_stats_string()


def save_mask():
    """Save the mask (at the resized resolution)"""
    if state.segmentation is None:
        return "❌ No segmentation generated."

    try:
        mask_dir = state.root_folder / "segmentor_masks"
        mask_dir.mkdir(exist_ok=True)

        current_filename = state.images[state.current_idx].name
        save_name = Path(current_filename).stem + ".png"
        save_path = mask_dir / save_name

        if isinstance(state.segmentation, torch.Tensor):
            mask_np = state.segmentation.numpy()
        else:
            mask_np = state.segmentation

        threshold = np.percentile(mask_np, 90)
        binary_mask = (mask_np >= threshold).astype(np.uint8) * 255

        PILImage.fromarray(binary_mask, mode="L").save(save_path)

        print(f"Saved mask to: {save_path}")
        return f"💾 Saved: {save_name} (Size: {binary_mask.shape})"

    except Exception as e:
        return f"❌ Save failed: {str(e)}"


def process_current_image():
    """Load next/prev image, RESIZE IMMEDIATELY, then inference"""
    state.visited_indices.add(state.current_idx)

    # Load and IMMEDIATELY RESIZE
    raw_img = PILImage.open(state.images[state.current_idx]).convert("RGB")
    state.current_image = resize_to_limit(raw_img)  # <--- Critical Resize

    image_tensor = None
    output = None

    try:
        image_tensor = img_to_tensor(state.current_image)
        output = state.segmentor.step(image=image_tensor, gamma=50)

        if output is None:
            del image_tensor
            cleanup_gpu()
            state.segmentation = None
            state.display_image = resize_for_display(state.current_image)
            return state.display_image, "⚪ No segmentation (add exemplars)"

        state.segmentation = output.segmentation.detach().cpu()

        del image_tensor
        del output
        cleanup_gpu()

        img_with_seg = overlay_segmentation(state.current_image, state.segmentation)
        state.display_image = resize_for_display(img_with_seg)

        return state.display_image, get_status_text()

    except Exception as e:
        if image_tensor is not None:
            del image_tensor
        if output is not None:
            del output
        cleanup_gpu()
        print(f"Error processing image: {e}")
        return state.display_image, f"Error: {e}"


def navigate_prev():
    if state.current_idx > 0:
        state.current_idx -= 1
        img, status = process_current_image()
        return img, status, get_stats_string()
    return state.display_image, get_status_text(), get_stats_string()


def navigate_next():
    if state.current_idx < len(state.images) - 1:
        state.current_idx += 1
        img, status = process_current_image()
        return img, status, get_stats_string()
    return state.display_image, get_status_text(), get_stats_string()


def start_define_exemplars():
    state.mode = "define_exemplars"
    state.pos_points = []
    state.neg_points = []
    state.click_mode = "positive"
    state.display_image = resize_for_display(state.current_image)
    return state.display_image, get_status_text()


# --- UI LAYOUT ---

with gr.Blocks(title="Interactive Segmentor") as demo:
    gr.Markdown("# 🎯 Interactive Image Segmentor")

    with gr.Row():
        with gr.Column(scale=3):
            image_display = gr.Image(
                label="Image Viewer",
                type="pil",
                interactive=False,
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

            stats_display = gr.Markdown("📊 Stats will appear here...")

            gr.Markdown("---")
            status_text = gr.Textbox(label="Status", lines=4, interactive=False)
            gr.Markdown("---")

            with gr.Group(visible=False) as controls_group:
                gr.Markdown("### 🎨 Define Exemplars")
                with gr.Row():
                    pos_mode_btn = gr.Button("✅ Positive", variant="secondary")
                    neg_mode_btn = gr.Button("❌ Negative", variant="secondary")

                with gr.Row():
                    reset_btn = gr.Button("🔄 Reset")
                    done_btn = gr.Button("⚡ Generate", variant="primary")

            gr.Markdown("---")

            with gr.Group(visible=False) as nav_group:
                gr.Markdown("### 💾 Actions")
                save_btn = gr.Button("💾 Save Mask", size="lg")
                gr.Markdown("---")
                define_exemplars_btn = gr.Button("🎯 Refine Exemplars", size="lg")

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
            stats_display,
        ],
    )

    image_display.select(fn=handle_click, outputs=[image_display, status_text])

    pos_mode_btn.click(fn=lambda: toggle_click_mode("positive"), outputs=[status_text])
    neg_mode_btn.click(fn=lambda: toggle_click_mode("negative"), outputs=[status_text])
    reset_btn.click(fn=reset_points, outputs=[image_display, status_text])

    done_btn.click(
        fn=register_exemplars, outputs=[image_display, status_text, stats_display]
    )

    prev_btn.click(
        fn=navigate_prev, outputs=[image_display, status_text, stats_display]
    )
    next_btn.click(
        fn=navigate_next, outputs=[image_display, status_text, stats_display]
    )

    define_exemplars_btn.click(
        fn=start_define_exemplars, outputs=[image_display, status_text]
    )

    save_btn.click(fn=save_mask, outputs=[status_text])

if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=6862,
    )
