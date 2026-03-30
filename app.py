import streamlit as st
import numpy as np
from PIL import Image
import cv2

import base64
import io
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Wall Recolor (Internal)", layout="wide")

# -----------------------------
# Utilities
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
    new_h = int(h * (max_w / w))
    return pil_img.resize((max_w, new_h), Image.LANCZOS)

def pil_to_data_url(pil_img: Image.Image, fmt="PNG") -> str:
    """Convert PIL image to a base64 data URL (bypasses Streamlit image_to_url)."""
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{b64}"

def target_rgb_to_lab(rgb):
    """Convert a single RGB color to Lab using OpenCV."""
    color = np.uint8([[list(rgb)]])  # (1,1,3) RGB
    bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    return lab[0, 0, :]  # L,a,b

def apply_lab_recolor(bgr_img: np.ndarray, mask: np.ndarray, target_rgb,
                      strength=0.85, soften_edges=4):
    """
    Recolor masked region toward target in Lab space, preserving original L for texture/lighting.
    mask: uint8 0/1 (H,W)
    """
    mask_f = mask.astype(np.float32)

    # Feather mask edges
    if soften_edges and soften_edges > 0:
        k = soften_edges * 2 + 1
        mask_f = cv2.GaussianBlur(mask_f, (k, k), 0)

    mask_f = np.clip(mask_f, 0.0, 1.0)[..., None]  # (H,W,1)

    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]
    a = lab[:, :, 1]
    b = lab[:, :, 2]

    tL, ta, tb = target_rgb_to_lab(target_rgb)

    # Preserve L, move only a/b toward target
    a_new = a * (1 - mask_f[:, :, 0] * strength) + ta * (mask_f[:, :, 0] * strength)
    b_new = b * (1 - mask_f[:, :, 0] * strength) + tb * (mask_f[:, :, 0] * strength)

    lab_out = np.stack([L, a_new, b_new], axis=2)
    lab_out = np.clip(lab_out, 0, 255).astype(np.uint8)

    bgr_out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)
    return bgr_out

def canvas_mask_to_binary(canvas_image_data: np.ndarray) -> np.ndarray:
    """
    Canvas gives RGBA. Use alpha>0 as mask.
    """
    alpha = canvas_image_data[:, :, 3]
    return (alpha > 0).astype(np.uint8)

# -----------------------------
# UI
# -----------------------------
st.title("Wall Recolor (Internal App)")
st.caption("Upload a photo → paint the wall area → set target RGB → export recolored image.")

col_left, col_right = st.columns([1, 1.25], gap="large")

with col_left:
    uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "webp"])

    st.subheader("Target RGB")
    r = st.number_input("R", 0, 255, 169)
    g = st.number_input("G", 0, 255, 176)
    b = st.number_input("B", 0, 255, 180)
    target_rgb = (int(r), int(g), int(b))

    strength = st.slider("Recolor strength", 0.0, 1.0, 0.85, 0.01)
    feather = st.slider("Edge feather (px)", 0, 20, 4, 1)

    st.markdown(
        "**Tips**\n"
        "- If you *can’t see the color*, increase **strength** to 0.95–1.0.\n"
        "- If edges look harsh, increase **feather**.\n"
        "- Paint slightly beyond the wall edges; feather will soften it."
    )

with col_right:
    if not uploaded:
        st.info("Upload an image to begin.")
        st.stop()

    base_pil = Image.open(uploaded).convert("RGB")

    # Smaller preview for speed
    preview_pil = resize_keep_aspect(base_pil, max_w=1100)
    preview_w, preview_h = preview_pil.size

    st.subheader("1) Paint the wall area (mask)")
    st.write("Paint only the wall. Painted pixels will be recolored.")

    # ✅ Option A: Provide background as base64 data URL (avoids image_to_url crash)
    bg_url = pil_to_data_url(preview_pil, fmt="PNG")

    canvas_res = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=18,
        stroke_color="rgba(255, 0, 0, 0.35)",
        background_image=None,          # IMPORTANT
        background_image_url=bg_url,    # IMPORTANT (Option A fix)
        update_streamlit=True,
        height=preview_h,
        width=preview_w,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_res.image_data is None:
        st.stop()

    mask_bin = canvas_mask_to_binary(canvas_res.image_data)

    # Preview recolor
    preview_bgr = pil_to_bgr(preview_pil)
    recolored_bgr = apply_lab_recolor(
        preview_bgr,
        mask=mask_bin,
        target_rgb=target_rgb,
        strength=strength,
        soften_edges=feather,
    )
    recolored_pil = bgr_to_pil(recolored_bgr)

    st.subheader("2) Result preview")
    st.image(recolored_pil, caption="Recolored preview", use_container_width=True)

    st.subheader("3) Export")
    export_full = st.checkbox("Export in full resolution", value=True)

    if st.button("Generate output"):
        if export_full:
            full_w, full_h = base_pil.size
            mask_full = cv2.resize(mask_bin, (full_w, full_h), interpolation=cv2.INTER_NEAREST)

            base_bgr = pil_to_bgr(base_pil)
            # scale feather roughly to full-res width
            feather_full = int(max(1, round(feather * (full_w / max(1, preview_w)))))
            out_bgr = apply_lab_recolor(
                base_bgr,
                mask=mask_full,
                target_rgb=target_rgb,
                strength=strength,
                soften_edges=feather_full,
            )
            out_pil = bgr_to_pil(out_bgr)
        else:
            out_pil = recolored_pil

        st.success("Done.")
        st.image(out_pil, caption="Final output", use_container_width=True)

        buf = io.BytesIO()
        out_pil.save(buf, format="PNG")
        st.download_button(
            "Download PNG",
            data=buf.getvalue(),
            file_name="wall_recolored.png",
            mime="image/png"
        )
