INSERT INTO instrument_type (id_type, tip) VALUES
(1, 'Suflat'),
(2, 'Coarde'),
(3, 'Percutie'),
(4, 'Lovit');

SELECT setval('instrument_type_id_type_seq', (SELECT MAX(id_type) FROM instrument_type));

INSERT INTO admin_user (username, password) VALUES ('admin', 'admin123');

INSERT INTO file_type (type, extension) VALUES
('audio', '.wav'),
('imagine', '.jpg');

INSERT INTO ml_models (nume, modalitate) VALUES
('audio_model', 'audio'),
('imagine_model', 'imagine'),
('fuziune_model', 'fuziune');

INSERT INTO metoda_extractie(nume, tip, descriere, parametri) VALUES
('MFCC',
'audio',
'Mel-Frequency Cepstral Coefficients (MFCC) transforma semnalul audio intr o reprezentare compacta bazata pe perceptia umana a sunetului. Se calculeaza FFT, se aplica filtrele Mel, apoi se aplica DCT pentru a obtine coeficientii.',
'n_mfcc = 13, duration =5s, sample_rate=22050, aggregare=mean pe axa temporala'),
('ResNet18',
'imagine',
'Retea neuronala convolutionala cu 18 straturi si conexiuni reziduale (skip connections). Extrage caracteristicile vizuale ierarhice din imagine - margini, texturi, forme, structuri complexe.',
'input_size=224x224, pretained=False, optimizer = Adam, loss= CrossEntropyLoss');

INSERT INTO instrument (nume, instrument_type_id) VALUES
('Accordion', 1),
('Clarinet', 1),
('flute', 1),
('Harmonica', 1),
('Harmonium', 1),
('Horn', 1),
('Saxophone', 1),
('Trombone', 1),
('Trumpet', 1),
('Acoustic_Guitar', 2),
('Banjo', 2),
('Bass_Guitar', 2),
('Dobro', 2),
('Electro_Guitar', 2),
('Mandolin', 2),
('Ukulele', 2),
('Violin', 2),
('Drum_set', 3),
('Floor_Tom', 3),
('cowbell', 4),
('Cymbals', 4),
('Hi_Hats', 4),
('Shakers', 4),
('Tambourine', 4),
('Vibraphone', 4),
('Piano', 2),
('Keyboard', 4),
('Organ', 1),
('Alphorn', 1),
('bagpipes', 1),
('concertina', 1),
('Didgeridoo', 1),
('ocarina', 1),
('tuba', 1),
('clavichord', 2),
('dulcimer', 2),
('guitar', 2),
('harp', 2),
('sitar', 2),
('Bongo_Drum', 3),
('casaba', 4),
('castanets', 4),
('guiro', 4),
('marakas', 4),
('steel_drum', 4),
('Xylophone', 4),
('drums', 3);