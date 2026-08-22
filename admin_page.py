import streamlit as st 
import os
import time
import threading
import datetime
from PIL import Image
from training.reantrenare_model import reantrenare_audio, reantrenare_imagine, extract_audio_features
from database import (check_admin_login, 
                      get_all_cereri, 
                      delete_cerere, 
                      get_instrument_types, 
                      add_instrument_type, 
                      add_instrument, 
                      save_instrument_files, 
                      create_reantrenare_job, 
                      update_reantrenare_job, 
                      instrument_exists,)

AUDIO_DATASET_PATH = os.environ.get("AUDIO_DATASET_PATH", "DataSetAudio")
IMAGE_DATASET_PATH = os.environ.get("IMAGE_DATASET_PATH", "DatasetFinal")
UPLOADS_DIR = "uploads"

def _run_training_thread(tip: str, job_id: int):
    def log(msg: str):
        st.session_state["training_log"].append(msg)
    
    try:
        st.session_state["training_running"] = True
        st.session_state["training_log"] = []
        st.session_state["training_error"] = None
        
        update_reantrenare_job(job_id, "running")
        
        if tip in ("audio", "ambele"):
            acc = reantrenare_audio(log_fn=log)
            
        if tip in ("imagine", "ambele"):
            acc = reantrenare_imagine(log_fn=log)

        update_reantrenare_job(
            job_id, "done",
            finalizat_la=datetime.datetime.now()
        )
        st.session_state["training_done"] = True
        st.session_state["training_running"] = False

    except Exception as e:
        st.session_state["training_error"] = str(e)
        st.session_state["training_running"] = False
        update_reantrenare_job(
            job_id, "error",
            finalizat_la=datetime.datetime.now()
        )
        
def _pagina_adaugare_instrumente():
    if st.button("<-- Inapoi la Admin Panel"):
        st.session_state["admin_sub_page"] = "main"
        for k in [
            "add_step", "add_tip_modalitate", "add_instrument_id",
            "add_instrument_nume", "add_instrument_tip",
            "add_instrument_name", "add_instrument_tip_selection",
            "add_instrument_new_type", "add_job_id", "training_running",
            "training_done", "training_log", "training_error",
        ]:
            st.session_state.pop(k, None)
        st.rerun()
    st.markdown(
        "<h2 style='color:#4b0082; margin-bottom:0.2rem;'> ➕ Adaugare instruemnt nou</h2>", unsafe_allow_html=True,
    )
    st.markdown("------")
    
    if "add_step" not in st.session_state:
        st.session_state["add_step"] = 1
    
    step = st.session_state["add_step"]
    
    if step == 1:
        st.subheader("Pasul 1 / 4 - Ce date vrei sa adaugi pentru instrument?")
        st.markdown(
            "Alege modalitatea pentru care vei uploada date de antrenare."
            "Acesta va determina si ce model vas fi reantreant ulterior."
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🎶 Doar Audio", use_container_width=True):
                st.session_state["add_tip_modalitate"] = "audio"
                st.session_state["add_step"] = 2
                st.rerun()
        with col2:
            if st.button("🖼️ Doar Imagini", use_container_width=True):
                st.session_state["add_tip_modalitate"] = "imagine"
                st.session_state["add_step"] = 2
                st.rerun()
        with col3:
            if st.button("🎶 Audio + 🖼️ Imagine", use_container_width=True):
                st.session_state["add_tip_modalitate"] = "ambele"
                st.session_state["add_step"] = 2
                st.rerun()
                
    elif step == 2:
        modalitate = st.session_state.get("add_tip_modalitate", "ambele")
        eticheta_mod = {"audio": "Audio", "imagine": "Imagine", "ambele": "Audio + Imagine"}
        st.subheader(f"Pasul 2 / 4 - Detalii instrument - {eticheta_mod[modalitate]}")
        
        tipuri_db = get_instrument_types()
        optiuni = [t[1] for t in tipuri_db] + ["Alt tip (adauga nou)"]

        if "add_instrument_tip_selection" not in st.session_state:
            st.session_state["add_instrument_tip_selection"] = optiuni[0]
        selectie = st.selectbox(
            "Tipul instrumentului *",
            optiuni,
            key="add_instrument_tip_selection"
        )

        tip_nou_text = ""
        if selectie == "Alt tip (adauga nou)":
            tip_nou_text = st.text_input(
                "Scrie tipul nou",
                placeholder="ex: Electroacustic, Etnic...",
                key="add_instrument_new_type"
            )

        with st.form("form_detalii_instrument", clear_on_submit=False):
            nume = st.text_input(
                "Numele instrumentului *",
                placeholder="ex: cobza, sitar, ...",
                key="add_instrument_name"
            )
            submitted = st.form_submit_button("Continua ->")
        
        if submitted:
            if not nume or not nume.strip():
                st.error("Numele instrumentului este obligatoriu")
                st.stop()
            
            if instrument_exists(nume):
                st.error(f"Instrumentul **{nume}** exista deja in baza de date")
                st.stop()
                
            if selectie == "Alt tip (adauga nou)" and not tip_nou_text.strip():
                st.error("Introdu un tip nou sau selecteaza unul existent")
                st.stop()
                
            if selectie == "Alt tip (adauga nou)":
                type_id = add_instrument_type(tip_nou_text.strip())
                tip_ales = tip_nou_text.strip()
            else:
                type_id = next(t[0] for t in tipuri_db if t[1] == selectie)
                tip_ales = selectie
            
            try:
                instr_id = add_instrument(nume.strip(), type_id)
                st.session_state["add_instrument_id"] = instr_id
                st.session_state["add_instrument_nume"] = nume.strip()
                st.session_state["add_instrument_tip"] = tip_ales
                st.session_state["add_step"] = 3
                st.rerun()
            except ValueError as ve:
                st.error(str(ve))
            except Exception as e:
                st.error(f"Eroare la salvarea instrumentului: {e}")
                
    elif step == 3:
        modalitate   = st.session_state.get("add_tip_modalitate", "ambele")
        instr_id     = st.session_state["add_instrument_id"]
        instr_nume   = st.session_state["add_instrument_nume"]
        instr_tip    = st.session_state["add_instrument_tip"]
 
        eticheta_mod = {"audio": "🎵 Audio", "imagine": "🖼️ Imagini", "ambele": "🎵🖼️ Audio + Imagini"}
        st.subheader(f"Pasul 3 / 4 — Upload fisiere  ·  {eticheta_mod[modalitate]}")
        st.success(f"✅ Instrument creat: **{instr_nume}** (tip: {instr_tip})")
        st.markdown(
            "Incarca fisierele de antrenare pentru instrumentul nou. "
            "Cu cat mai multe fisiere, cu atat modelul va fi mai precis."
        )
 
        erori_upload = []
        uploaded_audio_count  = 0
        uploaded_imagine_count = 0
 
        if modalitate in ("audio", "ambele"):
            st.markdown("#### 🎵 Fisiere Audio (.wav)")
            audio_files = st.file_uploader(
                "Selecteaza fisiere WAV",
                type=["wav"],
                accept_multiple_files=True,
                key="upload_audio_files",
            )
            if audio_files:
                st.caption(f"{len(audio_files)} fisier(e) selectate")
 
        if modalitate in ("imagine", "ambele"):
            st.markdown("#### 🖼️ Imagini (.jpg / .png)")
            image_files = st.file_uploader(
                "Selecteaza imagini",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                key="upload_image_files",
            )
            if image_files:
                st.caption(f"{len(image_files)} imagine(i) selectate")
 
        if st.button("💾 Salveaza fisierele si continua →", type="primary"):
            os.makedirs(UPLOADS_DIR, exist_ok=True)
 
            if modalitate in ("audio", "ambele"):
                if not audio_files:
                    st.error("Trebuie sa incarci cel putin un fisier WAV.")
                    st.stop()
 
                dest_dir = os.path.join(AUDIO_DATASET_PATH, instr_nume)
                os.makedirs(dest_dir, exist_ok=True)
 
                fnames, fsizes, fpaths = [], [], []
                for f in audio_files:
                    dest_path = os.path.join(dest_dir, f.name)
                    with open(dest_path, "wb") as out:
                        out.write(f.read())
                    fnames.append(f.name)
                    fsizes.append(f.size)
                    fpaths.append(dest_path)
                    uploaded_audio_count += 1
 
                try:
                    save_instrument_files(instr_id, fnames, fsizes, fpaths, "audio")
                except Exception as e:
                    erori_upload.append(f"Eroare salvare audio in DB: {e}")
 
            if modalitate in ("imagine", "ambele"):
                if not image_files:
                    st.error("Trebuie sa incarci cel putin o imagine.")
                    st.stop()
 
                dest_dir = os.path.join(IMAGE_DATASET_PATH, instr_nume)
                os.makedirs(dest_dir, exist_ok=True)
 
                fnames, fsizes, fpaths = [], [], []
                for f in image_files:
                    dest_path = os.path.join(dest_dir, f.name)
                    with open(dest_path, "wb") as out:
                        out.write(f.read())
                    fnames.append(f.name)
                    fsizes.append(f.size)
                    fpaths.append(dest_path)
                    uploaded_imagine_count += 1
 
                try:
                    save_instrument_files(instr_id, fnames, fsizes, fpaths, "imagine")
                except Exception as e:
                    erori_upload.append(f"Eroare salvare imagini in DB: {e}")
 
            if erori_upload:
                for err in erori_upload:
                    st.error(err)
            else:
                try:
                    job_id = create_reantrenare_job(instr_id, modalitate)
                    st.session_state["add_job_id"] = job_id
                except Exception as e:
                    st.warning(f"Nu s-a putut crea jobul de reantrenare in DB: {e}")
                    st.session_state["add_job_id"] = None
 
                st.session_state["add_audio_count"]  = uploaded_audio_count
                st.session_state["add_image_count"]  = uploaded_imagine_count
                st.session_state["add_step"] = 4
                st.rerun()
 
    elif step == 4:
        modalitate  = st.session_state.get("add_tip_modalitate", "ambele")
        instr_nume  = st.session_state.get("add_instrument_nume", "—")
        instr_tip   = st.session_state.get("add_instrument_tip",  "—")
        audio_cnt   = st.session_state.get("add_audio_count",  0)
        image_cnt   = st.session_state.get("add_image_count",  0)
        job_id      = st.session_state.get("add_job_id")
 
        st.subheader("Pasul 4 / 4 — Reantrenare model")
        st.success(f"✅ Instrumentul **{instr_nume}** a fost adaugat cu succes!")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Instrument", instr_nume)
        col_b.metric("Tip",        instr_tip)
        if modalitate in ("audio", "ambele"):
            col_c.metric("Fisiere audio",  audio_cnt)
        if modalitate in ("imagine", "ambele"):
            col_c.metric("Imagini",        image_cnt)
 
        st.markdown("---")
 
        training_running = st.session_state.get("training_running", False)
        training_done    = st.session_state.get("training_done",    False)
        training_error   = st.session_state.get("training_error")
 
        label_model = {
            "audio":   "modelul **Audio**",
            "imagine": "modelul **Imagine**",
            "ambele":  "ambele modele (**Audio** + **Imagine**)",
        }
 
        if not training_running and not training_done and not training_error:
            st.info(
                f"Instrumentul a fost adaugat. Acum trebuie reantrenat {label_model[modalitate]}. "
                f"Reantrenarea poate dura cateva minute."
            )
            if st.button("🚀 Porneste reantrenarea", type="primary"):
                st.session_state["training_log"]     = []
                st.session_state["training_running"] = True
                st.session_state["training_done"]    = False
                st.session_state["training_error"]   = None
 
                thread = threading.Thread(
                    target=_run_training_thread,
                    args=(modalitate, job_id or 0),
                    daemon=True,
                )
                thread.start()
                st.rerun()
 
        elif training_running:
            st.warning("⏳ Reantrenarea este in curs... Nu inchide pagina.")
 
            log_lines = st.session_state.get("training_log", [])
            if log_lines:
                st.code("\n".join(log_lines[-30:]), language="text")
 
            time.sleep(3)
            st.rerun()
 
        elif training_done:
            st.success("🎉 Reantrenarea s-a terminat cu succes!")
            log_lines = st.session_state.get("training_log", [])
            if log_lines:
                with st.expander("📋 Log antrenare"):
                    st.code("\n".join(log_lines), language="text")
 
            if st.button("✅ Gata — Inapoi la Admin Panel"):
                for k in [
                    "add_step", "add_tip_modalitate", "add_instrument_id",
                    "add_instrument_nume", "add_instrument_tip",
                    "add_audio_count", "add_image_count", "add_job_id",
                    "training_running", "training_done",
                    "training_log", "training_error",
                ]:
                    st.session_state.pop(k, None)
                st.session_state["admin_sub_page"] = "main"
                st.rerun()
 
        elif training_error:
            st.error(f"❌ Eroare la reantrenare: {training_error}")
            log_lines = st.session_state.get("training_log", [])
            if log_lines:
                with st.expander("📋 Log partial"):
                    st.code("\n".join(log_lines), language="text")
 
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔁 Incearca din nou"):
                    st.session_state["training_error"]   = None
                    st.session_state["training_done"]    = False
                    st.session_state["training_running"] = False
                    st.session_state["training_log"]     = []
                    st.rerun()
            with col2:
                if st.button("← Inapoi la Admin Panel"):
                    st.session_state["admin_sub_page"] = "main"
                    st.rerun()
        
def admin_page():
    if "admin_logged" not in st.session_state:
        st.session_state.admin_logged = False
        
    if "admin_sub_page" not in st.session_state:
        st.session_state["admin_sub_page"] = "main"
        
    if not st.session_state.admin_logged:
        st.title("🔐 Admin Login")

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            if check_admin_login(username,password):
                st.session_state.admin_logged = True
                st.success("Autentificare reusita!")
                st.rerun()
            else:
                st.error("Username sau parola gresita!")
        return

    if st.session_state.get("admin_sub_page") == "add_instrument":
        _pagina_adaugare_instrumente()
        return
    
    col_space, col_title, col_logout = st.columns([1,9,2])
    with col_title:
        st.markdown("<h2 style='text-align:center; margin-top:0;'>Admin Panel</h2>", unsafe_allow_html=True)
    with col_logout:
        if st.button("Logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.session_state.page = "home"
            st.rerun()

    st.markdown("----")
    
    col_btn, _ = st.columns([3, 7])
    with col_btn:
        if st.button("➕ Adauga Instrument Nou", type="primary", use_container_width=True):
            st.session_state["admin_sub_page"] = "add_instrument"
            st.session_state["add_step"] = 1
            st.rerun()
            
    st.markdown("-----")
    
    st.subheader("Cereri utilizatori")
    
    cereri = get_all_cereri()
    
    if not cereri:
        st.info("Nu exista cereri in acest moment")
        return

    for c in cereri:
        id_cerere, nume,descriere, imagine_path, data = c
        label = f"{nume if nume else 'Fara nume'} - {data}"
        with st.expander(label):
            if descriere:
                st.write(descriere)
            
            if imagine_path:
                try:
                    img = Image.open(imagine_path)
                    st.image(img, caption="Imagine uploadata", use_column_width=True)
                except Exception:
                    st.warning("Imaginea nu a putut fi incarcata")
            else:
                st.caption("Fara imagine")
            
            if st.button("🗑️ Sterge cererea", key=f"del_{id_cerere}"):
                st.session_state[f"confirm_del_{id_cerere}"] = True
            
            if st.session_state.get(f"confirm_del_{id_cerere}"):
                st.warning("Esti sigur ca vrei sa stergi aceasta cerere?")
                col_da, col_nu =st.columns(2)
                with col_da:
                    if st.button("✅ Da, sterge", key=f"da_{id_cerere}"):
                        try:
                            delete_cerere(id_cerere, imagine_path)
                            st.session_state.pop(f"confirm_del_{id_cerere}", None)
                            st.toast("Cererea a fost stears!", icon="✅")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Eroare la stergere: {e}")
                with col_nu:
                    if st.button("❌ Nu", key=f"nu_{id_cerere}"):
                        st.session_state.pop(f"confirm_del_{id_cerere}", None)
                        st.rerun()
        
        st.markdown("----")