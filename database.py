import psycopg2
import os
import uuid 
import numpy as np

DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "port": int(os.environ.get("DB_PORT", 5432)),
    "database": os.environ.get("DB_NAME", "heritagefusion"),
    "user": os.environ.get("DB_USER", "postgres"),
    "password": os.environ.get("DB_PASSWORD", "0241")
}
UPLOADS_DIR = "uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)

def get_connection():
    return psycopg2.connect(**DB_CONFIG)

def save_prediction(filename, file_size, modalitate, predicted_label, confidenta, metoda_nume, parametri_calculati):
    conn = None
    try:
        conn = get_connection()
        curs = conn.cursor()

        curs.execute(
            "SELECT id_instrument FROM instrument WHERE LOWER(nume)=LOWER(%s)",
            (predicted_label.strip(),)
        )
        rezultat = curs.fetchone()
        if not rezultat:
            raise Exception("Instrumentul prezis nu exista in baza de date")
        instrument_id = rezultat[0]

        curs.execute(
            "SELECT id_file_type FROM file_type WHERE LOWER(type)=LOWER(%s)",
            (modalitate.strip(),)
        )
        rezultat = curs.fetchone()
        if not rezultat:
            raise Exception("FileType nu exista in baza de date")
        file_type_id = rezultat[0]

        path_in_db = f"uploads/{uuid.uuid4().hex}_{filename}"

        curs.execute("""
            INSERT INTO files(filename, path, size, instrument_id, file_type_id)
            VALUES(%s, %s, %s, %s, %s)
            RETURNING id_file
        """, (
            filename,
            path_in_db,
            int(file_size) if file_size is not None else 0,
            instrument_id,
            file_type_id
        ))
        row = curs.fetchone()
        file_id = row[0]

        model_name = f"{modalitate}_model"

        curs.execute(
            "SELECT id_model FROM ml_models WHERE LOWER(nume)=LOWER(%s)",
            (model_name,)
        )
        rezultat = curs.fetchone()
        if not rezultat:
            raise Exception("Modelul nu exista in baza de date")
        model_id = rezultat[0]

        curs.execute(
            "SELECT id_metoda FROM metoda_extractie WHERE LOWER(nume) = LOWER(%s)",
            (metoda_nume.strip(),)
        )
        rezultat = curs.fetchone()
        if not rezultat:
            raise Exception(f"Metoda de extractie pentru '{modalitate}' nu exista in baza de date")
        metoda_id = rezultat[0]

        npy_path = os.path.join("models", f"{modalitate}_acuratete.npy")
        if os.path.exists(npy_path):
            acuratete_val = float(np.load(npy_path)[0])
        else:
            acuratete_val = 0.0

        curs.execute("""
            INSERT INTO rulari (model_id, metoda_id, acuratete)
            VALUES(%s, %s, %s)
            RETURNING id_rulare
        """, (model_id, metoda_id, acuratete_val))
        rulare_id = curs.fetchone()[0]

        curs.execute("""
            INSERT INTO predictii(file_id, rulare_id, predict_label, confidenta, corect, parametri_calculati)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id_predictie
        """, (
            file_id,
            rulare_id,
            predicted_label,
            float(confidenta),
            None,
            parametri_calculati
        ))
        predictie_id = curs.fetchone()[0]
        conn.commit()
        return predictie_id

    except Exception as e:
        if conn:
            conn.rollback()
        raise e
    finally:
        if conn:
            conn.close()

def get_predictie_details(predictie_id: int):
    conn = None
    try:
        conn = get_connection()
        curs = conn.cursor()
        curs.execute("""
            SELECT 
                p.id_predictie,
                p.predict_label,
                p.confidenta,
                p.corect,
                p.parametri_calculati,
                me.nume AS metoda_nume,
                me.tip AS metoda_tip,
                me.descriere AS metoda_descriere,
                me.parametri AS metoda_parametri,
                f.filename,
                f.size,
                m.nume AS model_nume,
                r.acuratete AS model_acuratete
            FROM predictii p
            JOIN rulari r ON p.rulare_id = r.id_rulare
            JOIN metoda_extractie me ON r.metoda_id = me.id_metoda
            JOIN files f ON p.file_id = f.id_file
            JOIN ml_models m ON r.model_id = m.id_model
            WHERE p.id_predictie = %s
        """, (predictie_id,))
        row = curs.fetchone()
        if not row:
            return None
        return {
            "id_predictie": row[0],
            "predict_label": row[1],
            "confidenta": row[2],
            "corect": row[3],
            "parametri_calculati": row[4],
            "metoda_nume": row[5],
            "metoda_tip": row[6],
            "metoda_descriere": row[7],
            "metoda_parametri": row[8],
            "filename": row[9],
            "file_size": row[10],
            "model_nume": row[11],
            "model_acuratete": row[12] if row[12] is not None else 0.0,
        }
    except Exception as e:
        raise e
    finally:
        if conn:
            conn.close()

def update_feedback(predictie_id: int, corect: bool):
    conn = None
    try:
        conn = get_connection()
        curs = conn.cursor()
        curs.execute(
            "UPDATE predictii SET corect=%s WHERE id_predictie=%s",
            (corect, predictie_id)
        )
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        raise e
    finally:
        if conn:
            conn.close()

def save_cerere(nume_instrument_dorit: str = None, descriere: str = None, imagine_bytes: bytes = None, imagine_extension: str = "jpg"):
    conn = None
    imagine_path = None
    try:
        if imagine_bytes:
            unique_name = f"{uuid.uuid4().hex}.{imagine_extension}"
            imagine_path = os.path.join(UPLOADS_DIR, unique_name)
            with open(imagine_path, "wb") as f:
                f.write(imagine_bytes)
        conn = get_connection()
        curs = conn.cursor()
        curs.execute(
            "INSERT INTO cerere(nume_instrument_dorit, descriere, imagine_url) VALUES (%s, %s, %s)",
            (
                nume_instrument_dorit.strip() if nume_instrument_dorit else None,
                descriere.strip() if descriere else None,
                imagine_path
            )
        )
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        if imagine_path and os.path.exists(imagine_path):
            os.remove(imagine_path)
        raise e
    finally:
        if conn:
            conn.close()

def check_admin_login(username: str, password: str) -> bool:
    conn = get_connection()
    try:
        curs = conn.cursor()
        curs.execute(
            "SELECT id_admin FROM admin_user WHERE username = %s AND password = %s",
            (username, password)
        )
        result = curs.fetchone()
    except Exception as e:
        raise e
    finally:
        if conn:
            conn.close()
    return result is not None

def get_all_cereri():
    conn = get_connection()
    try:
        curs = conn.cursor()
        curs.execute("""
            SELECT id_cerere, nume_instrument_dorit, descriere, imagine_url, data_cerere
            FROM cerere
            ORDER BY data_cerere DESC
        """)
        rows = curs.fetchall()
    except Exception as e:
        raise e
    finally:
        if conn:
            conn.close()
    return rows

def delete_cerere(id_cerere: int, imagine_path: str = None):
    conn = None
    try:
        if imagine_path and os.path.exists(imagine_path):
            os.remove(imagine_path)
        conn = get_connection()
        curs = conn.cursor()
        curs.execute("DELETE FROM cerere WHERE id_cerere = %s", (id_cerere,))
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        raise e
    finally:
        if conn:
            conn.close()

def add_instrument_type(tip: str) -> int:
    conn = get_connection()
    try:
        curs = conn.cursor()
        curs.execute(
            "SELECT id_type FROM instrument_type WHERE LOWER(tip) = LOWER(%s)",
            (tip.strip(),)
        )
        row = curs.fetchone()
        if row:
            return row[0]
        curs.execute(
            "INSERT INTO instrument_type(tip) VALUES (%s) RETURNING id_type",
            (tip.strip(),)
        )
        type_id = curs.fetchone()[0]
        conn.commit()
        return type_id
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def add_instrument(nume: str, type_id: int) -> int:
    conn = get_connection()
    try:
        curs = conn.cursor()
        curs.execute(
            "SELECT id_instrument FROM instrument WHERE LOWER(nume) = LOWER(%s)",
            (nume.strip(),)
        )
        row = curs.fetchone()
        if row:
            raise ValueError(f"Instrumentul '{nume}' exista deja in baza de date.")
        curs.execute(
            "INSERT INTO instrument(nume, instrument_type_id) VALUES (%s, %s) RETURNING id_instrument",
            (nume.strip(), type_id)
        )
        instr_id = curs.fetchone()[0]
        conn.commit()
        return instr_id
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def save_instrument_files(instrument_id: int, filenames: list, size: list, paths: list, modalitate: str) -> list:
    conn = get_connection()
    try:
        curs = conn.cursor()
        curs.execute(
            "SELECT id_file_type FROM file_type WHERE LOWER(type) = LOWER(%s)",
            (modalitate.strip(),)
        )
        row = curs.fetchone()
        if not row:
            raise ValueError(f"FileType '{modalitate}' nu exista in baza de date")
        file_type_id = row[0]

        ids = []
        for fname, fsize, fpath in zip(filenames, size, paths):
            curs.execute("""
                INSERT INTO files(path, filename, size, instrument_id, file_type_id)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id_file
            """, (fpath, fname, int(fsize), instrument_id, file_type_id))
            ids.append(curs.fetchone()[0])

        conn.commit()
        return ids
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def create_reantrenare_job(instrument_id: int, tip: str) -> int:
    conn = get_connection()
    try:
        curs = conn.cursor()
        curs.execute("""
            INSERT INTO reantrenare_job(instrument_id, tip, status)
            VALUES (%s, %s, 'pending')
            RETURNING id_job
        """, (instrument_id, tip))
        job_id = curs.fetchone()[0]
        conn.commit()
        return job_id
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def update_reantrenare_job(job_id: int, status: str, finalizat_la=None):
    conn = get_connection()
    try:
        curs = conn.cursor()
        if finalizat_la:
            curs.execute(
                "UPDATE reantrenare_job SET status=%s, finalizat_la=%s WHERE id_job=%s",
                (status, finalizat_la, job_id)
            )
        else:
            curs.execute(
                "UPDATE reantrenare_job SET status=%s WHERE id_job=%s",
                (status, job_id)
            )
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def get_instrument_types() -> list:
    conn = get_connection()
    try:
        curs = conn.cursor()
        curs.execute("SELECT id_type, tip FROM instrument_type ORDER BY tip")
        return curs.fetchall()
    except Exception as e:
        raise e
    finally:
        conn.close()

def instrument_exists(nume: str) -> bool:
    conn = get_connection()
    try:
        curs = conn.cursor()
        curs.execute(
            "SELECT 1 FROM instrument WHERE LOWER(nume) = LOWER(%s)",
            (nume.strip(),)
        )
        return curs.fetchone() is not None
    finally:
        conn.close()