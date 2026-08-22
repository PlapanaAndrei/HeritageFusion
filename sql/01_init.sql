CREATE TABLE IF NOT EXISTS admin_user (
    id_admin SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password VARCHAR(100) NOT NULL
);

CREATE TABLE IF NOT EXISTS ml_models (
    id_model SERIAL PRIMARY KEY,
    nume VARCHAR(100),
    modalitate VARCHAR(50) CHECK (modalitate IN ('audio', 'imagine', 'fuziune')),
    CONSTRAINT unique_ml_models UNIQUE (nume, modalitate)
);

CREATE TABLE IF NOT EXISTS metoda_extractie (
    id_metoda SERIAL PRIMARY KEY,
    nume VARCHAR(100) NOT NULL,
    tip VARCHAR(50) NOT NULL,
    descriere TEXT,
    parametri TEXT,
    CONSTRAINT unique_metoda_extractie UNIQUE (nume)
);

CREATE TABLE IF NOT EXISTS instrument_type (
    id_type SERIAL PRIMARY KEY,
    tip VARCHAR(100) NOT NULL,
    CONSTRAINT unique_instrument_type UNIQUE (tip)
);

CREATE TABLE IF NOT EXISTS instrument (
    id_instrument SERIAL PRIMARY KEY,
    nume VARCHAR(100) NOT NULL,
    instrument_type_id INTEGER REFERENCES instrument_type(id_type) ON DELETE CASCADE,
    CONSTRAINT unique_instrument UNIQUE (nume, instrument_type_id)
);

CREATE TABLE IF NOT EXISTS file_type (
    id_file_type SERIAL PRIMARY KEY,
    type VARCHAR(50) NOT NULL,
    extension VARCHAR(10),
    CONSTRAINT unique_file_type UNIQUE (extension)
);

CREATE TABLE IF NOT EXISTS files (
    id_file SERIAL PRIMARY KEY,
    path TEXT ,
    filename VARCHAR(255),
    size INTEGER CHECK (size >= 0),
    instrument_id INTEGER REFERENCES instrument(id_instrument) ON DELETE SET NULL,
    file_type_id INTEGER REFERENCES file_type(id_file_type),
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_files UNIQUE (filename, uploaded_at)
);

CREATE TABLE IF NOT EXISTS rulari (
    id_rulare SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES ml_models(id_model) ON DELETE CASCADE,
    metoda_id INTEGER REFERENCES metoda_extractie(id_metoda) ON DELETE SET NULL,
    datetime TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    acuratete FLOAT,
    parametri JSONB,
    CONSTRAINT unique_rulari UNIQUE (model_id, metoda_id, datetime)
);

CREATE TABLE IF NOT EXISTS predictii (
    id_predictie SERIAL PRIMARY KEY,
    file_id INTEGER REFERENCES files(id_file) ON DELETE CASCADE,
    rulare_id INTEGER REFERENCES rulari(id_rulare) ON DELETE CASCADE,
    predict_label VARCHAR(100),
    confidenta FLOAT,
    corect BOOLEAN,
    moment_timp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    parametri_calculati TEXT,
    CONSTRAINT unique_predictii UNIQUE (file_id, rulare_id)
);

CREATE TABLE IF NOT EXISTS cerere (
    id_cerere SERIAL PRIMARY KEY,
    nume_instrument_dorit VARCHAR(255),
    admin_id INTEGER REFERENCES admin_user(id_admin) ON DELETE SET NULL,
    descriere TEXT,
    imagine_url TEXT,
    data_cerere TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_cerere UNIQUE (nume_instrument_dorit, data_cerere, imagine_url)
);

CREATE TABLE IF NOT EXISTS reantrenare_job (
    id_job SERIAL PRIMARY KEY,
    instrument_id INTEGER REFERENCES instrument(id_instrument) ON DELETE CASCADE,
    tip VARCHAR(20) NOT NULL CHECK (tip IN ('audio', 'imagine', 'ambele')),
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'done', 'error')),
    creat_la TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    finalizat_la TIMESTAMP,
    CONSTRAINT unique_reantrenare_job UNIQUE (instrument_id, tip, creat_la)
);