import os
import streamlit as st
import base64

PAGE_CSS_MAP = {
    "home":       "styles/home.css",
    "classifier": "styles/classifier.css",
    "admin":      "styles/admin.css",
    "chat":       "styles/chat.css",
    "cerere":     "styles/cerere.css",
}

BASE_CSS = "styles/base.css"
GOOGLE_FONTS_URL = (
    "https://fonts.googleapis.com/css2?"
    "family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&"
    "family=Crimson+Text:ital,wght@0,400;0,600;1,400&"
    "family=DM+Sans:wght@300;400;500&"
    "display=swap"
)


def _read_css(filepath: str) -> str:
    if not os.path.exists(filepath):
        return ""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def _get_background_css() -> str:
    """Incarc imaginea de background si o convertesc in base64"""
    bg_path = "assets/background.png"
    if os.path.exists(bg_path):
        try:
            with open(bg_path, "rb") as f:
                bg_base64 = base64.b64encode(f.read()).decode()
            return f"""
.stApp {{
    background-image: url('data:image/png;base64,{bg_base64}') !important;
    background-size: cover !important;
    background-position: center !important;
    background-attachment: fixed !important;
    background-repeat: no-repeat !important;
}}
"""
        except Exception as e:
            print(f"Eroare la incarcarea background: {e}")
            return ""
    return ""


def load_styles(page: str = "home") -> None:
    st.markdown(
        #f'<link rel="preconnect" href="https://fonts.googleapis.com">'
        #f'<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        f'<link href="{GOOGLE_FONTS_URL}" rel="stylesheet">',
        unsafe_allow_html=True
    )
    base_css = _read_css(BASE_CSS)
    page_css_file = PAGE_CSS_MAP.get(page, "")
    page_css = _read_css(page_css_file) if page_css_file else ""
    bg_css = _get_background_css()

    if base_css or page_css or bg_css:
        st.markdown(
            f"<style>\n{base_css}\n{page_css}\n{bg_css}\n</style>",
            unsafe_allow_html=True
        )


def load_extra_css(filepath: str) -> None:
    css = _read_css(filepath)
    if css:
        st.markdown(f"<style>\n{css}\n</style>", unsafe_allow_html=True)