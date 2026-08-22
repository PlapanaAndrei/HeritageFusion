import os
import shutil

SOURCE = "E:\\Facultate\\Licenta\\Date\\Imagini"
DEST = "DatasetFinal"

os.makedirs(DEST, exist_ok=True)

for split in ["train", "test", "valid"]:

    split_path = os.path.join(SOURCE, split)

    if not os.path.exists(split_path):
        continue

    for instrument in os.listdir(split_path):

        instrument_src = os.path.join(split_path, instrument)

        if not os.path.isdir(instrument_src):
            continue

        instrument_dest = os.path.join(DEST, instrument)

        os.makedirs(instrument_dest, exist_ok=True)

        for file in os.listdir(instrument_src):

            src = os.path.join(instrument_src, file)
            dst = os.path.join(instrument_dest, file)

            shutil.copy2(src, dst)

print("DONE")
