import base64
import os
import streamlit as st

def set_background(image_path: str):
\
    if not os.path.exists(image_path):
        print(f"[WARNING] Imaginea de fundal nu a fost gasita: {image_path}")
        return

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    extensie = image_path.rsplit(".", 1)[-1].lower()
    mime_type = "image/png" if extensie == "png" else "image/jpeg"

    css = f"""
    <style>
    .stApp {{
        background-image: url("data:{mime_type};base64,{encoded}") !important;
        background-size: cover !important;
        background-position: center center !important;
        background-repeat: no-repeat !important;
        background-attachment: fixed !important;
    }}
    .stApp::before {{
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(245, 239, 224, 0.05);
        z-index: 0;
        pointer-events: none;
    }}

    .block-container {{
        position: relative;
        z-index: 1;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)