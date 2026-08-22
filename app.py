import streamlit as st
import os
import torch
import torch.nn as nn
import numpy as np
import time
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tempfile
import logging
from io import BytesIO
from chat_widget import render_chatbot
from torchvision import models, transforms
from PIL import Image
from admin_page import admin_page
from database import save_prediction, update_feedback, save_cerere, get_predictie_details
from styles.styleLoader import load_styles
from styles.background_loader import set_background

st.set_page_config(
    page_title="HeritageFusion",
    page_icon="🎶",
    layout="wide",
    initial_sidebar_state="expanded"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if "page" not in st.session_state:
    st.session_state.page = "home"

if "current_mode" not in st.session_state:
    st.session_state.current_mode = "Audio"

if "admin_logged" not in st.session_state:
    st.session_state.admin_logged = False

load_styles(st.session_state.page)
set_background("docs/Background.png")

st.markdown("""
<style>
[data-testid="stBottom"],
[data-testid="stBottom"] *,
[data-testid="stBottom"] > div,
[data-testid="stBottom"] > div > div,
[data-testid="stBottom"] > div > div > div,
.stChatFloatingInputContainer,
.stChatFloatingInputContainer * {
    background: white !important;
    background-color: white !important;
    box-shadow: none !important;
}
[data-testid="stChatInput"] {
    background: rgba(253, 250, 245, 0.97) !important;
    border: 2px solid rgba(107, 66, 38, 0.32) !important;
    border-radius: 20px !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #2C1810 !important;
}
[data-testid="stChatInput"] button {
    background: #C0634A !important;
    border-radius: 50% !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)


def handle_error(error_type: str, user_message: str, technical_message: str = None):
    logger.error(f"[{error_type}] {technical_message or user_message}")
    st.error(f"❌ {user_message}")


def show_info(message: str):
    st.info(f"ℹ️ {message}")

class AudioClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.network(x)


@st.cache_resource
def load_audio():
    classes = np.load("models/audio_classes.npy", allow_pickle=True)
    model = AudioClassifier(26, len(classes))
    model.load_state_dict(torch.load("models/audio_model.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model, classes


@st.cache_resource
def load_image():
    classes = np.load("models/imagine_classes.npy", allow_pickle=True)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(torch.load("models/image_model.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model, classes


audio_model = None
audio_classes = None
image_model = None
image_classes = None

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


@st.cache_data(show_spinner=False)
def extract_audio_features(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        y, sr = librosa.load(tmp_path)
        duration = librosa.get_duration(y=y, sr=sr)
        n_mfcc = 13
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1) 
        mfcc_std  = np.std(mfcc,  axis=1)  
        features  = np.concatenate([mfcc_mean, mfcc_std])
        return {
            "features": features,
            "y": y,
            "sr": sr,
            "duration": duration,
            "path": tmp_path,
            "mfcc": mfcc,
            "n_mfcc": n_mfcc,
        }
    except Exception as e:
        logger.error(f"Eroare la extragerea caracteristicilor audio: {e}")
        raise


@st.cache_data(show_spinner=False)
def load_image_from_bytes(img_bytes):
    image = Image.open(BytesIO(img_bytes)).convert("RGB")
    return np.array(image, dtype=np.uint8), image


def get_audio_model():
    global audio_model, audio_classes
    if audio_model is None:
        audio_model, audio_classes = load_audio()
    return audio_model, audio_classes


def get_image_model():
    global image_model, image_classes
    if image_model is None:
        image_model, image_classes = load_image()
    return image_model, image_classes


col_space, col_admin = st.columns([10, 2])
with col_admin:
    if st.session_state.page != "admin":
        if st.button("Admin"):
            st.session_state.page = "admin"
            st.rerun()

if st.session_state.page != "admin" and st.session_state.get("admin_logged"):
    st.session_state.admin_logged = False

if st.session_state.page == "chat":
    pass
else:
    st.markdown("<h1 style='text-align:center; margin:0; '>HeritageFusion</h1>", unsafe_allow_html=True)
    if st.session_state.page == "home":
        st.markdown("<p style='text-align:center; color:#9C7B65;"
                    "font-size:1rem; margin-top:-0.3rem; margin-bottom:1.5rem;'>"
                    "Recunoasterea si clasificarea artefactelor culturale</p>", unsafe_allow_html=True)


def show_audio_statistics(detalii, tmp_path):
    st.markdown("---")
    st.markdown("## 📊 Statistici predictie")

    st.markdown("### 🗄️ Detalii tehnice")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Instrument prezis", detalii["predict_label"])
        st.metric("Confidenta model", f"{detalii['confidenta']*100:.2f}%")
    with col2:
        st.metric("Metoda de extractie", detalii["metoda_nume"])
        st.metric("Tip analiza", detalii["metoda_tip"])
    with col3:
        st.metric("Model folosit", detalii["model_nume"])
        st.metric("Acuratete model", f"{detalii['model_acuratete']*100:.1f}%")

    st.markdown("**Despre metoda de extractie folosita:**")
    st.info(detalii["metoda_descriere"])

    st.markdown("**Parametri generali ai metodei:**")
    st.code(detalii["metoda_parametri"], language="text")

    st.markdown("**Parametri calculati pentru acest fisier:**")
    st.code(detalii["parametri_calculati"], language="text")

    if tmp_path and os.path.exists(tmp_path):
        with open(tmp_path, "rb") as f:
            audio_analysis = extract_audio_features(f.read())

        y    = audio_analysis["y"]
        sr   = audio_analysis["sr"]
        mfcc = audio_analysis["mfcc"]
        n_mfcc = audio_analysis["n_mfcc"]

        st.markdown("---")
        st.markdown("## 🎵 Diagrame audio")

        st.markdown("### 📈 Waveform (forma de unda)")
        st.markdown(
            "Waveform-ul reprezinta amplitudinea semnalului audio in timp. "
            "Arata cat de tare sau incet este sunetul in fiecare moment. "
            "Instrumentele cu atac puternic (tobe, percutie) au varfuri ascutite, "
            "pe cand instrumentele de suflat au o forma mai lina si uniforma."
        )
        fig, ax = plt.subplots(figsize=(9, 3))
        librosa.display.waveshow(y, sr=sr, ax=ax, color="#4b0082")
        ax.set_title(f"Waveform — {detalii['predict_label']}")
        ax.set_xlabel("Timp (s)")
        ax.set_ylabel("Amplitudine")
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("### 🎛️ MFCC — Mel-Frequency Cepstral Coefficients")
        st.markdown(
            "MFCC transforma sunetul intr-o reprezentare matematica bazata pe modul "
            "in care urechea umana percepe frecventele. Cei 13 coeficienti surprind "
            "caracteristicile spectrale esentiale ale instrumentului — timbrul, tonul si textura sunetului. "
            "Modelul foloseste atat media cat si deviatia standard a acestor coeficienti (26 valori total) "
            "pentru a captura atat caracteristicile medii cat si variatia temporala a sunetului. "
            "Culorile mai calde (galben/portocaliu) indica valori mai mari ale coeficientilor."
        )
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(mfcc, x_axis="time", sr=sr, ax=ax, cmap="plasma")
        fig.colorbar(img, ax=ax)
        ax.set_title(f"MFCC  — {detalii['predict_label']}")
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("### 🌈 Spectrograma Mel")
        st.markdown(
            "Spectrograma Mel arata distributia energiei sonore pe frecvente in timp, "
            "scalata dupa perceptia umana (scala Mel). Axa verticala reprezinta frecventa "
            "(de la bas la inalte), axa orizontala timpul, iar culoarea intensitatea sunetului in dB. "
            "Fiecare instrument lasa un tipar unic — de exemplu vioara are linii clare sus, "
            "iar toba are energie concentrata in frecventele joase."
        )
        mel    = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(mel_db, x_axis="time", y_axis="mel", sr=sr, ax=ax, cmap="magma")
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        ax.set_title(f"Spectrograma Mel — {detalii['predict_label']}")
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("### 🎼 Chroma Features")
        st.markdown(
            "Chroma Features reprezinta distributia energiei pe cele 15 note muzicale (Do, Re, Mi...) "
            "de-a lungul timpului. Ne arata ce note sunt prezente si cat de puternic, "
            "independent de octava. Este utila mai ales pentru instrumente armonice "
            "ca chitara, pian sau vioara. O coloana luminoasa inseamna ca nota respectiva "
            "este cantata puternic in acel moment."
        )
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(chroma, x_axis="time", y_axis="chroma", sr=sr, ax=ax, cmap="coolwarm")
        fig.colorbar(img, ax=ax)
        ax.set_title(f"Chroma Features — {detalii['predict_label']}")
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("### 📡 Spectral Centroid (Centrul spectral)")
        st.markdown(
            "Centrul spectral reprezinta 'centrul de greutate' al spectrului de frecvente "
            "la fiecare moment in timp — adica unde se concentreaza cel mai mult energia sonora. "
            "O valoare ridicata (Hz mare) inseamna un sunet mai ascutit/stralucitor, "
            "iar o valoare mica inseamna un sunet mai gros/intunecat. "
            "Instrumentele de suflat tind sa aiba centrul spectral mai stabil, "
            "pe cand percutia are variatii mari."
        )
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        frames   = range(len(centroid))
        t        = librosa.frames_to_time(frames, sr=sr)
        fig, ax  = plt.subplots(figsize=(10, 3))
        ax.plot(t, centroid, color="#4b0082")
        ax.fill_between(t, centroid, alpha=0.2, color="#4b0082")
        ax.set_title(f"Spectral Centroid — {detalii['predict_label']}")
        ax.set_xlabel("Timp (s)")
        ax.set_ylabel("Frecventa (Hz)")
        st.pyplot(fig)
        plt.close(fig)


def show_image_statistics(detalii, img_obj):
    st.markdown("---")
    st.markdown("## 📊 Statistici predictie")
    st.markdown("### 🗄️ Detalii tehnice")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Instrument prezis", detalii["predict_label"])
        st.metric("Confidenta model", f"{detalii['confidenta']*100:.2f}%")
    with col2:
        st.metric("Metoda de extractie", detalii["metoda_nume"])
        st.metric("Tip analiza", detalii["metoda_tip"])
    with col3:
        st.metric("Model folosit", detalii["model_nume"])
        st.metric("Acuratete model", f"{detalii['model_acuratete']*100:.1f}%")

    st.markdown("**Despre metoda de extractie folosita:**")
    st.info(detalii["metoda_descriere"])

    st.markdown("**Parametri generali ai metodei:**")
    st.code(detalii["metoda_parametri"], language="text")

    st.markdown("**Parametri calculati pentru aceasta imagine:**")
    st.code(detalii["parametri_calculati"], language="text")

    if img_obj:
        img_array = np.array(img_obj, dtype=np.uint8)

        st.markdown("---")
        st.markdown("## 🖼️ Diagrame imagine")
        st.markdown("### 🖼️ Imaginea originala")
        st.markdown(
            "Imaginea incarcata de utilizator, asa cum a fost primita de model. "
            "Inainte de a fi procesata, aceasta este redimensionata la 224×224 pixeli "
            "si normalizata cu valorile ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) "
            "pentru a fi compatibila cu arhitectura ResNet18 pretrained."
        )
        st.image(img_obj, width=400)

        st.markdown("### 🎨 Histograma RGB")
        st.markdown(
            "Histograma RGB arata distributia intensitatilor pentru fiecare canal de culoare "
            "(Rosu, Verde, Albastru) din imagine. Axa orizontala reprezinta valoarea pixelilor "
            "(0 = negru, 255 = maxim), iar axa verticala numarul de pixeli cu acea valoare. "
            "O histograma uniforma indica o imagine echilibrata, pe cand varfuri inguste "
            "pot indica zone foarte intunecate sau supraexpuse."
        )
        fig, ax = plt.subplots(figsize=(10, 4))
        culori = ["red", "green", "blue"]
        for i, culoare in enumerate(culori):
            ax.hist(img_array[:, :, i].ravel(), bins=256, color=culoare, alpha=0.5, label=culoare.upper())
        ax.set_title(f"Histograma RGB — {detalii['predict_label']}")
        ax.set_xlabel("Valoare pixel (0-255)")
        ax.set_ylabel("Numar pixeli")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("### 🔴🟢🔵 Canale RGB separate")
        st.markdown(
            "Fiecare imagine color este formata din trei canale independente: Rosu (R), Verde (G) si Albastru (B). "
            "Vizualizand fiecare canal separat putem vedea cum contribuie fiecare culoare la imaginea finala. "
            "Zonele mai luminoase inseamna intensitate mare a canalului respectiv in acea regiune. "
            "Modelul AI analizeaza toate cele trei canale simultan pentru a identifica instrumentul."
        )
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        nume_canale = ["Red", "Green", "Blue"]
        cmaps = ["Reds", "Greens", "Blues"]
        for i, (ax, nume, cmap) in enumerate(zip(axes, nume_canale, cmaps)):
            ax.imshow(img_array[:, :, i], cmap=cmap)
            ax.set_title(nume)
            ax.axis("off")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("### ⬛ Imagine Grayscale")
        st.markdown(
            "Versiunea in tonuri de gri a imaginii, obtinuta prin calcularea mediei celor trei canale RGB. "
            "Modelul ResNet18 poate identifica forme, margini si texturi chiar si fara informatie de culoare. "
            "Grayscale-ul evidentiaza structura si conturul instrumentului — elementele esentiale "
            "pe care reteaua neurala le foloseste pentru clasificare."
        )
        gray = np.mean(img_array, axis=2).astype(np.uint8)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.imshow(gray, cmap="gray")
        ax.set_title(f"Grayscale — {detalii['predict_label']}")
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)


if st.session_state.page == "home":
    st.markdown("""
    <div style="text-align: center;">
        <h3 style="margin-bottom: 20px;">Despre aplicatie</h3>
        <p>HeritageFusion este o aplicatie AI care poate identifica instrumente muzicale<br>
        folosind <strong>audio sau imagini</strong>.</p>
        <p style="margin-top: 15px; font-weight: bold; margin-bottom: 5px;">Functionalitati:</p>
        <ul style="list-style-type: none; padding: 0; margin: 0; line-height: 1.6;">
            <li>🔹 clasificare instrumente din fisiere audio</li>
            <li>🔹 clasificare instrumente din imagini</li>
            <li>🔹 chatbot inteligent despre instrumente muzicale</li>
        </ul>
    </div>
    <br>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns([1.5, 1, 2, 1, 1.5])
    
    with col3:
        sub_col1, sub_col2 = st.columns(2)
        with sub_col1:
            if st.button("▶ Start", use_container_width=True):
                st.session_state.page = "classifier"
                st.rerun()
        with sub_col2:
            if st.button("💬 Chat", use_container_width=True):
                st.session_state.page = "chat"
                st.rerun()

elif st.session_state.page == "classifier":
    st.markdown('<div class="back-btn">', unsafe_allow_html=True)
    if st.button("⬅ Inapoi"):
        st.session_state.page = "home"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
                <div style='background:rgba(253, 250, 245, 0.88); border: 1px solid rgba(107, 66, 38, 0.2);
                border-radius:16px; padding:1.5rem 2rem; margin-bottom:1rem;'>
                    <h4>Alege modalitatea dorita</h4>
                </div>
                """, unsafe_allow_html=True)
    
    col_a, col_i, col_ai = st.columns(3)
    with col_a:
        if st.button("🎶 Audio", use_container_width=True):
            st.session_state.current_mode = "Audio"
            st.rerun()
    
    with col_i:
        if st.button("🖼️ Imagine", use_container_width=True):
            st.session_state.current_mode = "Image"
            st.rerun()
            
    with col_ai:
        if st.button("🎶 + 🖼️", use_container_width=True):
            st.session_state.current_mode = "Audio + Imagine"
            st.rerun()
        
    mode = st.session_state.current_mode

    if mode == "Audio":
        st.subheader("🎵 Clasificare Audio")
        show_info("Incarca un fisier audio in format WAV pentru clasificare.")

        uploaded = st.file_uploader("Upload .wav", type=["wav"])

        current_audio_file = uploaded.name if uploaded else None

        if uploaded is None and st.session_state.get("audio_pred_file") is not None:
            st.session_state["audio_pred_id"] = None
            st.session_state["audio_feedback_given"] = False
            st.session_state["audio_pred_file"] = None
            st.session_state["last_pred_id"] = None
            st.session_state["last_predicted_label"] = None
            st.session_state["last_confidence"] = None
            st.session_state["last_tmp_path"] = None

        if current_audio_file is not None and current_audio_file != st.session_state.get("audio_pred_file"):
            st.session_state["audio_pred_id"] = None
            st.session_state["audio_feedback_given"] = False
            st.session_state["audio_pred_file"] = current_audio_file
            st.session_state["last_predicted_label"] = None
            st.session_state["last_confidence"] = None

        if uploaded:
            st.markdown('<div class="analyze-btn">', unsafe_allow_html=True)
            if st.button("Analizeaza Audio"):
                with st.spinner("Se analizeaza fisierul audio..."):
                    audio_data = extract_audio_features(uploaded.getvalue())
                features = audio_data["features"]
                y        = audio_data["y"]
                sr       = audio_data["sr"]
                duration = audio_data["duration"]
                path     = audio_data["path"]
                mfcc     = audio_data["mfcc"]
                n_mfcc   = audio_data["n_mfcc"]

                model, classes = get_audio_model()
                tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    out   = model(tensor)
                    probs = torch.softmax(out, dim=1).cpu().numpy()[0]
                    pred  = np.argmax(probs)

                predicted_label = classes[pred]
                confidence      = probs[pred]

                parametri_calculati = (
                    f"n_mfcc={n_mfcc}, "
                    f"features=mean+std ({n_mfcc*2} valori), "
                    f"duration={duration:.2f}s, "
                    f"sample_rate={sr}Hz, "
                    f"nr_frame-uri={mfcc.shape[1]}, "
                    f"aggregare=mean+std pe axa temporala"
                )

                st.session_state["last_tmp_path"] = path
                st.session_state["last_modalitate"] = "audio"
                raw_bytes = uploaded.getvalue()
                try:
                    pred_id = save_prediction(
                        filename=uploaded.name,
                        file_size=len(raw_bytes) if raw_bytes else 0,
                        modalitate="audio",
                        predicted_label=predicted_label,
                        confidenta=confidence,
                        metoda_nume="MFCC",
                        parametri_calculati=parametri_calculati
                    )
                    st.session_state["audio_pred_id"] = pred_id
                    st.session_state["audio_feedback_given"] = False
                    st.session_state["audio_pred_file"] = uploaded.name
                    st.session_state["last_pred_id"] = pred_id
                    st.session_state["last_predicted_label"] = predicted_label
                    st.session_state["last_confidence"] = confidence
                except Exception as e:
                    handle_error("DB", "Nu s-a putut salva predictia", str(e))
                    st.session_state["last_predicted_label"] = predicted_label
                    st.session_state["last_confidence"] = confidence
                    st.session_state["audio_pred_id"] = None
            st.markdown('</div>', unsafe_allow_html=True)
    if st.session_state.get("last_predicted_label") and st.session_state.get("current_mode") == "Audio":
        st.success(f"🎯 {st.session_state['last_predicted_label']} ({st.session_state['last_confidence']*100:.2f}%)")

    if (st.session_state.get("audio_pred_id")
            and not st.session_state.get("audio_feedback_given")):
        st.markdown("----")
        st.markdown("**Predictia a fost corecta?**")
        col_da, col_nu = st.columns(2)
        with col_da:
            if st.button("✅ Da", key="audio_da"):
                try:
                    update_feedback(st.session_state["audio_pred_id"], True)
                    st.session_state["audio_feedback_given"] = True
                    st.toast("Multumim pentru feedback!", icon="✅")
                    st.rerun()
                except Exception as e:
                    handle_error("DB", "Nu s-a putut salva feedback-ul", str(e))
        with col_nu:
            if st.button("❌ Nu", key="audio_nu"):
                try:
                    update_feedback(st.session_state["audio_pred_id"], False)
                    st.session_state["audio_feedback_given"] = True
                    st.toast("Ne pare rau! Vom imbunatati modelul.", icon="❌")
                    st.rerun()
                except Exception as e:
                    handle_error("DB", "Nu s-a putut salva feedback-ul", str(e))

    if st.session_state.get("audio_pred_id") and st.session_state.get("last_pred_id"):
        try:
            detalii = get_predictie_details(st.session_state["last_pred_id"])
            if detalii:
                show_audio_statistics(detalii, st.session_state.get("last_tmp_path"))
        except Exception as e:
            handle_error("DB", "Nu s-au putut incarca statisticile", str(e))

    if mode == "Image":
        st.subheader("🖼️ Clasificare Imagine")
        show_info("Incarca o imagine pentru clasificare.")

        uploaded = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

        current_img_file = uploaded.name if uploaded else None

        if uploaded is None and st.session_state.get("img_pred_file") is not None:
            st.session_state["img_pred_id"] = None
            st.session_state["img_feedback_given"] = False
            st.session_state["img_pred_file"] = None
            st.session_state["last_pred_id"] = None
            st.session_state["last_predicted_label"] = None
            st.session_state["last_confidence"] = None
            st.session_state["last_image"] = None
            st.session_state["last_image_bytes"] = None
            st.session_state["last_image_name"] = None

        if current_img_file is not None and current_img_file != st.session_state.get("img_pred_file"):
            st.session_state["img_pred_id"] = None
            st.session_state["img_feedback_given"] = False
            st.session_state["img_pred_file"] = current_img_file
            st.session_state["last_predicted_label"] = None
            st.session_state["last_confidence"] = None
            st.session_state["last_image"] = None
            st.session_state["last_image_bytes"] = uploaded.getvalue()
            st.session_state["last_image_name"] = uploaded.name

        if uploaded is None and st.session_state.get("last_image_bytes"):
            img_bytes = st.session_state["last_image_bytes"]
            img_name  = st.session_state.get("last_image_name", "imagine.jpg")
        else:
            img_bytes = uploaded.getvalue() if uploaded else None
            img_name  = uploaded.name if uploaded else None

        if img_bytes and not st.session_state.get("img_pred_id"):
            if st.button("Analizeaza Imagine"):
                with st.spinner("Se analizeaza imaginea..."):
                    img_array, image = load_image_from_bytes(img_bytes)
                    width, height    = image.size
                    tensor           = image_transform(image).unsqueeze(0).to(DEVICE)

                    model, classes = get_image_model()
                    with torch.no_grad():
                        out   = model(tensor)
                        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
                        pred  = np.argmax(probs)

                predicted_label = classes[pred]
                confidence      = probs[pred]

                mean_r = float(np.mean(img_array[:, :, 0]))
                mean_g = float(np.mean(img_array[:, :, 1]))
                mean_b = float(np.mean(img_array[:, :, 2]))
                parametri_calculati = (
                    f"dimensiune_originala={width}x{height}px, "
                    f"dimensiune_input=224x224px, "
                    f"canale=RGB, "
                    f"normalizare=ImageNet(mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225]), "
                    f"medie_R={mean_r:.1f}, "
                    f"medie_G={mean_g:.1f}, "
                    f"medie_B={mean_b:.1f}"
                )

                st.session_state["last_image"]    = image
                st.session_state["last_modalitate"] = "imagine"

                try:
                    pred_id = save_prediction(
                        filename=img_name,
                        file_size=len(img_bytes) if img_bytes else 0,
                        modalitate="imagine",
                        predicted_label=predicted_label,
                        confidenta=confidence,
                        metoda_nume="ResNet18",
                        parametri_calculati=parametri_calculati
                    )
                    st.session_state["img_pred_id"] = pred_id
                    st.session_state["img_feedback_given"] = False
                    st.session_state["img_pred_file"] = img_name
                    st.session_state["last_pred_id"] = pred_id
                    st.session_state["last_predicted_label"] = predicted_label
                    st.session_state["last_confidence"] = confidence
                except Exception as e:
                    handle_error("DB", "Nu s-a putut salva predictia", str(e))
                    st.session_state["last_predicted_label"] = predicted_label
                    st.session_state["last_confidence"] = confidence
                    st.session_state["img_pred_id"] = None

        if st.session_state.get("img_pred_id") and st.session_state.get("last_image"):
            col1, col2 = st.columns(2)
            with col1:
                st.image(st.session_state["last_image"], width=300)
            with col2:
                st.success(f"🎯 {st.session_state['last_predicted_label']} ({st.session_state['last_confidence']*100:.2f}%)")

        if (st.session_state.get("img_pred_id")
                and not st.session_state.get("img_feedback_given")):
            st.markdown("----")
            st.markdown("**Predictia a fost corecta?**")
            col_da, col_nu = st.columns(2)
            with col_da:
                if st.button("✅ Da", key="img_da"):
                    try:
                        update_feedback(st.session_state["img_pred_id"], True)
                        st.session_state["img_feedback_given"] = True
                        st.toast("Multumesc pentru feedback!", icon="✅")
                        st.rerun()
                    except Exception as e:
                        handle_error("DB", "Nu s-a putut salva feedback-ul", str(e))
            with col_nu:
                if st.button("❌ Nu", key="img_nu"):
                    try:
                        update_feedback(st.session_state["img_pred_id"], False)
                        st.session_state["img_feedback_given"] = True
                        st.toast("Ne pare rau! Vom imbunatati modelul.", icon="❌")
                        st.rerun()
                    except Exception as e:
                        handle_error("DB", "Nu s-a putut salva feedback-ul", str(e))

        if st.session_state.get("img_pred_id") and st.session_state.get("last_pred_id"):
            try:
                detalii = get_predictie_details(st.session_state["last_pred_id"])
                if detalii:
                    show_image_statistics(detalii, st.session_state.get("last_image"))
            except Exception as e:
                handle_error("DB", "Nu s-au putut incarca statisticile", str(e))

    if mode == "Audio + Imagine":
        st.subheader("Clasificare prin Fuziune Audio🎶 + Imagine🖼️")
        show_info("Incarca un fisier audio WAV cat si o imagine pentru acelasi instrument")
        
        uploaded_audio = st.file_uploader("Upload .wav", type=["wav"], key="fusion_audio")
        uploaded_image = st.file_uploader("Upload imagine", type=["jpg", "png", "jpeg"], key="fusion_image")
        
        if uploaded_audio and uploaded_image:
            if st.button("Analizeaza prin Fuziune"):
                with st.spinner("Se analizeaza..."):
                    
                    audio_data = extract_audio_features(uploaded_audio.getvalue())
                    features = audio_data["features"]
                    audio_model, audio_classes = get_audio_model()
                    tensor_audio = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        out_audio = audio_model(tensor_audio)
                        probs_audio = torch.softmax(out_audio, dim=1).cpu().numpy()[0]
                        
                    img_bytes = uploaded_image.getvalue()
                    _, image = load_image_from_bytes(img_bytes)
                    tensor_img = image_transform(image).unsqueeze(0).to(DEVICE)
                    image_model_f, image_classes_f = get_image_model()
                    with torch.no_grad():
                        out_img = image_model_f(tensor_img)
                        probs_img = torch.softmax(out_img, dim=1).cpu().numpy()[0]
                        
                    audio_classes_lower = [c.lower() for c in audio_classes]
                    image_classes_lower = [c.lower() for c in image_classes_f]
                    
                    clase_comune = [c for c in audio_classes_lower if c in image_classes_lower]
                    
                    if len(clase_comune) == 0:
                        st.error("Nu exista clase comune intre cele doua modele")
                    else:
                        probs_audio_comune = np.array([probs_audio[audio_classes_lower.index(c)] for c in clase_comune])
                        probs_img_comune = np.array([probs_img[image_classes_lower.index(c)] for c in clase_comune])
                        
                        probs_audio_comune = probs_audio_comune / probs_audio_comune.sum()
                        probs_img_comune = probs_img_comune / probs_img_comune.sum()
                        
                        probs_combined = (probs_audio_comune + probs_img_comune) / 2
                        pred_idx = np.argmax(probs_combined)
                        predicted_label = clase_comune[pred_idx]
                        confidence = probs_combined[pred_idx]
                        
                        st.success(f" Instrumentul identificat: **{predicted_label}** ({confidence*100:.2f}%)")
                        
                        st.markdown("### Comparatie probabilitati")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Model Audio", audio_classes[np.argmax(probs_audio)], f"{probs_audio[np.argmax(probs_audio)]*100:.1f}%")
                        
                        with col2:
                            st.metric("Model Imagine", image_classes_f[np.argmax(probs_img)], f"{probs_img[np.argmax(probs_img)]*100:.1f}%")
                            
                        with col3:
                            st.metric("Fuziune finala", predicted_label, f"{confidence*100:.1f}%")
                            
                        try:
                            pred_id = save_prediction(
                                filename = f"{uploaded_audio.name}+{uploaded_image.name}",
                                file_size = len(uploaded_audio.getvalue())+len(img_bytes),
                                modalitate="audio",
                                predicted_label = predicted_label,
                                confidenta = float(confidence),
                                metoda_nume = "MFCC",
                                parametri_calculati=f"late_fusion=True, weight_audio=0.5, weight_image=0.5"
                            )
                            st.session_state["fusion_pred_id"] = pred_id
                            st.session_state["fusion_feedback_given"] = False
                        except Exception as e:
                            handle_error("DB", "Nu s-a putut salva predictia", str(e))
                    
        if(st.session_state.get("fusion_pred_id") and not st.session_state.get("fusion_feedback_given")):
            st.markdown("----")
            st.markdown("**Predictia a fost corecta?**")
            col_da, col_nu = st.columns(2)
            with col_da:
                if st.button("✅ Da", key="fusion_da"):
                    try:
                        update_feedback(st.session_state["fusion_pred_id"], True)
                        st.session_state["fusion_feedback_given"] = True
                        st.toast("Multumim pentru feedback!", icon="✅")
                        st.rerun()
                    except Exception as e:
                        handle_error("DB", "Nu s-a putut salva feedback-ul", str(e))
            with col_nu:
                if st.button("❌ Nu", key="fusion_nu"):
                    try:
                        update_feedback(st.session_state["fusion_pred_id"], False)
                        st.session_state["fusion_feedback_given"] = True
                        st.toast("Ne pare rau! Vom imbunatati modelul.", icon="❌")
                        st.rerun()
                    except Exception as e:
                        handle_error("DB", "Nu s-a putut salva feedback-ul", str(e))
                        
    st.markdown('<div class="help-develop-btn">', unsafe_allow_html=True)
    if st.button("Ajuta-ne sa ne dezvoltam sistemul"):
        st.session_state.page = "cerere"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == "cerere":
    if st.button("<- Inapoi"):
        st.session_state.page = "classifier"
        st.rerun()

    st.markdown("<h2 style='text-align: center; color: #4b0082;'>Ajuta-ne sa ne crestem</h2>", unsafe_allow_html=True)
    st.markdown("Nu ai gasit instrumentul pe care il cautai? Spune-ne ce instrument doresti sa adaugam.")

    with st.form("form_cerere", clear_on_submit=False):
        nume_instrument_dorit = st.text_input(
            "Numele instrumentului dorit *",
            placeholder="ex: cobza, balalaika, sitar..."
        )
        descriere = st.text_area(
            "Descriere / detalii suplimentare (optional)",
            placeholder="Origine, familia instrumentului, alte detalii utile"
        )
        imagine_upload = st.file_uploader(
            "Adauga o fotografie cu instrumentul",
            type=["jpg", "jpeg", "png"]
        )
        submitted = st.form_submit_button("Trimite cererea")

    if submitted:
        if imagine_upload and imagine_upload.size > 2 * 1024 * 1024:
            st.error("Imaginea este prea mare (max 2MB)")
        else:
            try:
                extensie = "jpg"
                if imagine_upload:
                    extensie = imagine_upload.name.rsplit(".", 1)[-1].lower()

                save_cerere(
                    nume_instrument_dorit=nume_instrument_dorit.strip() if nume_instrument_dorit else None,
                    descriere=descriere.strip() if descriere else None,
                    imagine_bytes=imagine_upload.read() if imagine_upload else None,
                    imagine_extension=extensie
                )
                st.success("✅ Cererea a fost trimisa! Multumim pentru contributie.")
                st.info("Vei fi redirectionat catre pagina principala in cateva secunde....")
                time.sleep(2)
                st.session_state.page = "home"
                st.rerun()
            except Exception as e:
                handle_error("DB", "Nu s-a putut trimite cererea. Incearca din nou", str(e))

elif st.session_state.page == "chat":
    if st.button("⬅ Inapoi"):
        st.session_state.page = "home"
        st.rerun()
    st.markdown("<h1 style='text-align:center; '>Chatbot HeritageFusion 🎶</h1>", unsafe_allow_html=True)
    render_chatbot()

elif st.session_state.page == "admin":
    if not st.session_state.get("admin_logged"):
        if st.button("<--- Inapoi"):
            st.session_state.page = "home"
            st.rerun()
    admin_page()

with st.sidebar:
    st.title("📚 Ajutor")
    with st.expander("🤷‍♂️ Ce este HeritageFusion"):
        st.markdown("""
        **Cu ce te ajuta acest site?**
        - Invatarea usoara a artefactelor culturale prin audio si imagini
        - Te poate ajuta in domenii precum istorie, muzica si patrimoniu cultural
        - Poti afla cum aplicatia stie ce instrument este prin analiza statistica
        """)
    with st.expander("❓ Cum sa folosesc aplicatia?"):
        st.markdown("""
        **Modul Audio:**
        1. Selecteaza "Audio" in radio button
        2. Incarca un fisier WAV
        3. Apasa "Analizeaza Audio"
        4. Rezultatul si statisticile apar automat mai jos

        **Modul Imagine:**
        1. Selecteaza "Image" in radio button
        2. Incarca o imagine (JPG/PNG)
        3. Apasa "Analizeaza Imagine"
        4. Rezultatul si statisticile apar automat mai jos
        """)
    with st.expander("⚠️ Erori"):
        st.markdown("""
        **Problema:** Fisier nu se incarca
        - Verifica formatul (WAV pentru audio, JPG/PNG pentru imagini)
        - Verifica dimensiunea fisierului

        **Problema:** Rezultate inexacte
        - Incearca cu alte fisiere
        - Asigura-te ca imaginea/audio este clara
        """)     