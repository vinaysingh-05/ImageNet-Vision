import os
import base64
import streamlit as st
import numpy as np
from PIL import Image
import keras
from keras.applications.resnet50 import preprocess_input, decode_predictions

st.set_page_config(
    page_title="ImageNet Vision | ResNet-50",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

UI_DIR = os.path.join("assets", "ui")
LOGO_PATH = os.path.join(UI_DIR, "imagenet logo.jpg")
RESNET_PATH = os.path.join(UI_DIR, "resnet50.jpg")
BG_PATH = os.path.join(UI_DIR, "theam back groud.jpeg")

bg_b64 = get_base64_image(BG_PATH)
bg_style = f"""
    [data-testid="stAppViewContainer"] {{
        background: linear-gradient(rgba(10, 15, 29, 0.88), rgba(8, 12, 22, 0.94)), url('data:image/jpeg;base64,{bg_b64}') no-repeat center center fixed !important;
        background-size: cover !important;
    }}
    .stAppHeader {{
        background: transparent !important;
    }}
""" if bg_b64 else """
    [data-testid="stAppViewContainer"] {
        background: #0b0f19 !important;
    }
"""

# Custom Dark Theme matching the background image with neon blue/cyan accents
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #f1f5f9;
    }}

    {bg_style}

    /* Long Top Banner */
    .top-banner {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1.5rem;
        background: rgba(15, 23, 42, 0.85);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.2rem 2.2rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.5);
        animation: fadeIn 0.5s ease-out;
    }}

    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(-8px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    .banner-text {{
        flex: 1;
    }}

    .banner-title {{
        font-size: 1.9rem;
        font-weight: 800;
        color: #f8fafc;
        margin: 0;
        letter-spacing: -0.02em;
    }}

    .banner-title span {{
        color: #38bdf8;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}

    .banner-desc {{
        font-size: 0.95rem;
        color: #94a3b8;
        margin: 0.2rem 0 0 0;
    }}

    /* Main Drag and Drop Glass Card */
    .upload-card {{
        background: rgba(15, 23, 42, 0.75);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        text-align: center;
        max-width: 800px;
        margin: 0 auto 2rem auto;
    }}

    /* Compact Image Preview */
    .preview-box {{
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 1.2rem auto;
        padding: 0.8rem;
        background: rgba(2, 6, 23, 0.6);
        border-radius: 14px;
        border: 1px solid rgba(56, 189, 248, 0.25);
        max-width: 320px;
    }}

    .preview-box img {{
        max-height: 190px !important;
        width: auto !important;
        border-radius: 10px;
        object-fit: contain;
    }}

    /* Button Styling */
    .stButton>button {{
        background: linear-gradient(135deg, #0284c7, #2563eb);
        color: #ffffff;
        font-weight: 700;
        font-size: 1rem;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 1.8rem;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.35);
        transition: all 0.2s ease;
    }}

    .stButton>button:hover {{
        background: linear-gradient(135deg, #38bdf8, #3b82f6);
        box-shadow: 0 6px 20px rgba(56, 189, 248, 0.5);
        transform: translateY(-1px);
        color: #ffffff;
    }}

    /* Footer styling */
    .site-footer {{
        margin-top: 4rem;
        padding: 2.5rem 1.5rem 1.8rem 1.5rem;
        background: rgba(10, 15, 29, 0.85);
        backdrop-filter: blur(12px);
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 18px 18px 0 0;
        text-align: center;
    }}

    .footer-links {{
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 0.8rem;
        margin-bottom: 1.6rem;
    }}

    .footer-bottom {{
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1.5rem;
        color: #64748b;
        font-size: 0.85rem;
        margin-top: 1.2rem;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
        padding-top: 1.2rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_vision_model():
    local_weights = os.path.join("models", "resnet50_model.keras")
    if os.path.exists(local_weights):
        try:
            return keras.models.load_model(local_weights)
        except Exception:
            pass
    from keras.applications.resnet50 import ResNet50
    return ResNet50(weights="imagenet")


def classify_image(model, img: Image.Image, top_k=2):
    if img.mode != "RGB":
        img = img.convert("RGB")
    resized = img.resize((224, 224), Image.Resampling.BILINEAR)
    x = np.array(resized, dtype=np.float32)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x, verbose=0)
    decoded = decode_predictions(preds, top=top_k)[0]
    return [(name.replace("_", " "), float(score)) for (_, name, score) in decoded]


# --- SEPARATE DIALOGS (Alag-Alag Popups) ---

@st.dialog("📋 Project Details", width="large")
def show_project_details():
    st.markdown(
        """
        ### 🚀 ImageNet Vision Project
        This web application delivers interactive, instant image recognition using deep Convolutional Neural Networks (CNNs).
        
        - **Objective:** Give users a clean interface to drop any image and classify objects in real-time.
        - **Pipeline:** Image Drag-and-Drop → 224×224 Normalization → ResNet-50 50-layer Feature Extraction → Softmax Probabilities.
        - **Stack:** Python, Streamlit, TensorFlow, Keras 3, NumPy, Pillow.
        - **Local Model:** Serialized ImageNet weights stored at `models/resnet50_model.keras`.
        """
    )
    if st.button("Close Window", key="close_proj"):
        st.rerun()


@st.dialog("🏗️ Architecture Explanation", width="large")
def show_architecture():
    col1, col2 = st.columns([1, 1.4])
    with col1:
        if os.path.exists(RESNET_PATH):
            st.image(RESNET_PATH, caption="ResNet Residual Bottleneck Block", use_container_width=True)
    with col2:
        st.markdown(
            r"""
            ### 🔬 Deep Residual Architecture
            - **Layers:** 50 deep layers comprising 48 Convolutional layers, 1 MaxPool, and 1 Average Pool layer.
            - **Identity Shortcut Mapping:** Standard deep networks hit gradient degradation. ResNet passes $F(x) + x$ through skip connections.
            - **Bottleneck Design:** Uses $1\times 1$, $3\times 3$, and $1\times 1$ convolutions to reduce computational complexity while retaining representational capacity.
            - **Global Average Pooling:** Replaces flattening to prevent overfitting and directly feed into the 1,000-class dense layer.
            """
        )
    if st.button("Close Window", key="close_arch"):
        st.rerun()


@st.dialog("🌐 About ImageNet Dataset", width="large")
def show_about_imagenet():
    col1, col2 = st.columns([1, 1.4])
    with col1:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, caption="ImageNet Benchmark", use_container_width=True)
    with col2:
        st.markdown(
            """
            ### 📚 ImageNet (ILSVRC)
            - **Scale:** Over 14 million annotated images organized according to the WordNet hierarchy.
            - **Challenge Classes:** 1,000 fine-grained object categories spanning animals, vehicles, household items, tools, and instruments.
            - **Significance:** ImageNet Large Scale Visual Recognition Challenge (ILSVRC) sparked the modern deep learning revolution in computer vision.
            """
        )
    if st.button("Close Window", key="close_imgnet"):
        st.rerun()


@st.dialog("⚡ About ResNet-50", width="large")
def show_about_resnet():
    st.markdown(
        """
        ### 🏆 ResNet-50 (Residual Network)
        - **Origin:** Introduced by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun (Microsoft Research, 2015).
        - **Milestone:** Won 1st place in the ILSVRC 2015 classification challenge with a record 3.57% top-5 error rate, outperforming human accuracy.
        - **Key Innovation:** Proved that residual connections enable training of substantially deeper neural networks without vanishing or exploding gradients.
        """
    )
    if st.button("Close Window", key="close_resnet"):
        st.rerun()


# --- POPUP RESULTS SLIDE (Result in one popup dialog) ---
@st.dialog("🎯 Object Classification Results", width="large")
def show_results_popup(predictions, preview_img):
    col_img, col_pred = st.columns([1, 1.5], gap="medium")
    with col_img:
        st.markdown("<p style='font-size:0.85rem; color:#94a3b8; font-weight:600;'>Analyzed Image</p>", unsafe_allow_html=True)
        st.image(preview_img, use_container_width=True)
    
    with col_pred:
        top_name, top_prob = predictions[0]
        st.markdown(
            f"""
            <div style="background: rgba(14, 165, 233, 0.15); border: 1px solid #38bdf8; border-radius: 12px; padding: 1.2rem; margin-bottom: 1rem;">
                <span style="background: #0284c7; color: white; padding: 0.25rem 0.7rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase;">Top Prediction</span>
                <div style="font-size: 1.8rem; font-weight: 800; color: #f8fafc; text-transform: capitalize; margin-top: 0.4rem;">{top_name}</div>
                <div style="font-size: 1.25rem; font-weight: 800; color: #4ade80; margin-top: 0.2rem;">{top_prob * 100:.2f}% Match</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<p style='font-weight:700; color:#cbd5e1; margin-bottom:0.4rem;'>Top 2 Predictions:</p>", unsafe_allow_html=True)
        for rank, (name, prob) in enumerate(predictions[:2], start=1):
            col_lbl, col_p = st.columns([3, 1])
            with col_lbl:
                st.markdown(f"<span style='color:#e2e8f0; font-weight:600;'>#{rank} {name.title()}</span>", unsafe_allow_html=True)
            with col_p:
                st.markdown(f"<span style='color:#38bdf8; font-weight:700;'>{prob * 100:.1f}%</span>", unsafe_allow_html=True)
            st.progress(float(prob))

    if st.button("Close Results", use_container_width=True):
        st.rerun()


# --- LONG TOP BANNER ---
logo_b64 = get_base64_image(LOGO_PATH)
logo_img_tag = f'<img src="data:image/jpeg;base64,{logo_b64}" style="max-height: 58px; width: auto; border-radius: 8px; border: 1px solid rgba(255,255,255,0.15);">' if logo_b64 else ''

st.markdown(
    f"""
    <div class="top-banner">
        <div class="banner-text">
            <h1 class="banner-title">ImageNet <span>Vision</span></h1>
            <p class="banner-desc">Next-generation object classification with ResNet-50 Deep CNN</p>
        </div>
        {logo_img_tag}
    </div>
    """,
    unsafe_allow_html=True,
)

# Load model quietly
with st.spinner("Preparing Neural Network..."):
    model = load_vision_model()

# --- MAIN DRAG & DROP AND COMPACT PREVIEW SECTION ---
col_center = st.columns([1, 2, 1])[1]

with col_center:
    st.markdown(
        """
        <div style="text-align:center; margin-bottom:1rem;">
            <h3 style="color:#f8fafc; font-weight:700; margin-bottom:0.3rem;">Drop or Upload an Image</h3>
            <p style="color:#94a3b8; font-size:0.9rem;">Drop any object or scene image to run the ResNet-50 neural network</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Drop image file here",
        type=["png", "jpg", "jpeg", "webp"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        
        # Compact preview
        st.markdown('<div class="preview-box">', unsafe_allow_html=True)
        st.image(img, use_container_width=False, width=240)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:center; color:#94a3b8; font-size:0.85rem;'>Preview: <strong>{uploaded_file.name}</strong></p>", unsafe_allow_html=True)

        # Check button
        if st.button("🔍 Check / Classify Object", use_container_width=True):
            with st.spinner("Processing through 50 convolutional layers..."):
                predictions = classify_image(model, img, top_k=2)
            show_results_popup(predictions, img)
    else:
        st.markdown(
            """
            <div style="text-align:center; padding:1.5rem; border:2px dashed rgba(255,255,255,0.15); border-radius:14px; background:rgba(15,23,42,0.4); margin-top:0.5rem;">
                <p style="color:#64748b; font-size:0.95rem; margin:0;">📁 Drag and drop any image file above to begin.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# --- WEBSITE FOOTER WITH SEPARATE LINKS & RESNET LOGO ---
st.markdown("<div style='height: 4rem;'></div>", unsafe_allow_html=True)

st.markdown('<div class="site-footer">', unsafe_allow_html=True)

st.markdown("<p style='color:#94a3b8; font-size:0.85rem; font-weight:600; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:1rem;'>Explore Project Documentation</p>", unsafe_allow_html=True)

# 4 Separate Navigation / Information Buttons
f_col1, f_col2, f_col3, f_col4 = st.columns(4)

with f_col1:
    if st.button("📋 Project Details", key="btn_f_proj", use_container_width=True):
        show_project_details()

with f_col2:
    if st.button("🏗️ Architecture", key="btn_f_arch", use_container_width=True):
        show_architecture()

with f_col3:
    if st.button("🌐 About ImageNet", key="btn_f_imgnet", use_container_width=True):
        show_about_imagenet()

with f_col4:
    if st.button("⚡ About ResNet-50", key="btn_f_resnet", use_container_width=True):
        show_about_resnet()

# Footer Bottom with ResNet Logo
resnet_b64 = get_base64_image(RESNET_PATH)
resnet_tag = f'<img src="data:image/jpeg;base64,{resnet_b64}" style="max-height: 38px; width: auto; border-radius: 6px; border: 1px solid rgba(255,255,255,0.15); vertical-align: middle;">' if resnet_b64 else ''

st.markdown(
    f"""
    <div class="footer-bottom">
        <div>ImageNet Vision &bull; ResNet-50 Deep Learning Engine</div>
        {resnet_tag}
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('</div>', unsafe_allow_html=True)
