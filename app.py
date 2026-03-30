import io
import streamlit as st
import numpy as np
from PIL import Image
import cv2

# IMPORTANT:
# Install: streamlit-drawable-canvas-fix
# It keeps the same import path:
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Wall Recolor (Internal)", layout="wide")


# -----------------------------
# Image helpers
# -----------------------------
def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    """PIL RGB -> OpenCV BGR uint8"""
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    """OpenCV BGR uint8 -> PIL RGB"""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def resize_keep_aspect(pil_img: Image.Image, max_w: int) -> Image.Image:
    w, h = pil_img.size
    if w <= max_w:
        return pil_img
    new_h = int(round(h * (max_w / w)))
    return pil_img.resize((max_w, new_h), Image.LANCZOS)


# -----------------------------
# Color conversions / recolor
# -----------------------------
def target_rgb_to_lab(rgb_tuple):
    """
    Convert a single RGB color (0..255) to Lab using OpenCV.
    OpenCV expects BGR for cvtColor, and Lab ranges are 0..255 in uint8.
    """
    rgb_arr = np.uint8([[list(rgb_tuple)]])  # (1,1,3) RGB
    bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
    lab_arr = cv2.cvtColor(bgr_arr, cv2.COLOR_BGR2LAB).astype(np.float32)
    return lab_arr[0, 0, :]  # L, a, b


def canvas_mask_to_binary(canvas_rgba: np.ndarray) -> np.ndarray:
    """
    Canvas returns RGBA float array 0..255 (or 0..1 depending on component),
    but typically uint8-like values.
    We treat alpha > 0 as masked.
    Output: uint8 mask with values {0,1}.
    """
    # Handle float 0..1 safely:
    alpha = canvas_rgba[:, :, 3]
    if alpha.max() <= 1.0:
        alpha = alpha * 255.0
    return (alpha > 0).astype(np.uint8)


def apply_lab_recolor(
    bgr_img: np.ndarray,
    mask01: np.ndarray,
    target_rgb: tuple,
    strength: float = 0.9,
    feather_px: int = 4,
):
    """
    Recolor masked region toward target in Lab while preserving L (luminance)
    to keep wall shading + texture.

    mask01: uint8 (H,W) with 0/1
    strength: 0..1, how strongly to move a/b toward target
    feather_px: 0..N, Gaussian blur radius for smooth edges
    """
    strength = float(np.clip(strength, 0.0, 1.0))

    # Feather/soften the mask edges for realism
    mask_f = mask01.astype(np.float32)
    if feather_px and feather_px > 0:
        k = int(feather_px) * 2 + 1
        mask_f = cv2.GaussianBlur(mask_f, (k, k), 0)

    mask_f = np.clip(mask_f, 0.0, 1.0)[..., None]  # (H,W,1)

    # Convert image to Lab
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]
    a = lab[:, :, 1]
    b = lab[:, :, 2]

    # Target in Lab
    _, ta, tb = target_rgb_to_lab(target_rgb)

    # Preserve L, shift chroma a/b
    a_new = a * (1.0 - mask_f[:, :, 0] * strength) + ta * (mask_f[:, :, 0] * strength)
    b_new = b * (1.0 - mask_f[:, :, 0] * strength) + tb * (mask_f[:, :, 0] * strength)

    lab_out = np.stack([L, a_new, b_new], axis=2)
    lab_out = np.clip(lab_out, 0, 255).astype(np.uint8)

    out_bgr = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)
    return out_bgr


# -----------------------------
# UI
# -----------------------------
st.title("Wall Recolor (Internal App)")
st.caption("Upload → paint wall mask → set target RGB → export recolored image")

left, right = st.columns([1, 1.35], gap="large")

with left:
    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])

    st.subheader("Target color (RGB)")
    r = st.number_input("R", min_value=0, max_value=255, value=169, step=1)
    g = st.number_input("G", min_value=0, max_value=255, value=176, step=1)
    b = st.number_input("B", min_value=0, max_value=255, value=180, step=1)
    target_rgb = (int(r), int(g), int(b))

    strength = st.slider("Recolor strength", 0.0, 1.0, 0.90, 0.01)
    feather = st.slider("Edge feather (px)", 0, 20, 4, 1)

    brush = st.slider("Brush size", 3, 60, 18, 1)

    st.markdown(
        "**If you can’t see the change:**\n"
        "- Set **Recolor strength = 1.0**\n"
        "- Reduce **Edge feather = 1–2**\n"
        "- Paint the wall more densely (cover the full wall area)\n"
    )

    export_full = st.checkbox("Export full resolution", value=True)

with right:
    if not uploaded:
        st.info("Upload an image to start.")
        st.stop()

    base_pil = Image.open(uploaded).convert("RGB")

    # Preview to keep UI responsive
    preview_pil = resize_keep_aspect(base_pil, max_w=1100)
    pw, ph = preview_pil.size

    st.subheader("1) Paint the wall (mask)")
    st.write("Paint only the wall. Painted pixels will be recolored.")

    # st_canvas supports background_image (PIL Image). [1](https://pypi.org/project/streamlit-drawable-canvas/)
    canvas_res = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=int(brush),
        stroke_color="rgba(255, 0, 0, 0.35)",
        background_image=preview_pil,
        update_streamlit=True,
        height=ph,
        width=pw,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_res.image_data is None:
        st.stop()

    # Create binary mask from alpha channel
    mask01 = canvas_mask_to_binary(canvas_res.image_data)

    # Preview recolor
    preview_bgr = pil_to_bgr(preview_pil)
    recolored_preview_bgr = apply_lab_recolor(
        preview_bgr,
        mask01=mask01,
        target_rgb=target_rgb,
        strength=strength,
        feather_px=feather,
    )
    recolored_preview_pil = bgr_to_pil(recolored_preview_bgr)

    st.subheader("2) Result preview")
    st.image(recolored_preview_pil, use_container_width=True)

    st.subheader("3) Generate output")
    if st.button("Generate"):
        if export_full:
            # Scale the mask to full resolution
            fw, fh = base_pil.size
            mask_full = cv2.resize(mask01, (fw, fh), interpolation=cv2.INTER_NEAREST)

            # Scale feather proportionally
            feather_full = int(max(1, round(feather * (fw / max(1, pw)))))

            base_bgr = pil_to_bgr(base_pil)
            out_bgr = apply_lab_recolor(
                base_bgr,
                mask01=mask_full,
                target_rgb=target_rgb,
                strength=strength,
                feather_px=feather_full,
            )
            out_pil = bgr_to_pil(out_bgr)
        else:
            out_pil = recolored_preview_pil

        st.success("Done.")
        st.image(out_pil, use_container_width=True)

        # Download PNG
        buf = io.BytesIO()
        out_pil.save(buf, format="PNG")
        st.download_button(
            "Download PNG",
            data=buf.getvalue(),
            file_name="wall_recolored.png",
            mime="image/png",
        )
