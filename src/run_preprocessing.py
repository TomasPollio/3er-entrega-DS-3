"""
Aplica preprocess_batch (limpieza regex + normalizacion + lematizacion
spaCy, del Modulo 2) sobre train y test del corpus AG News, por separado.

Nota sobre data leakage: limpiar/lematizar cada fila de texto es una
transformacion fila-a-fila que no usa ninguna estadistica del corpus
(no mira frecuencias ni vocabulario global), por lo que aplicarla a
train y test por separado NO genera fuga de datos. El leakage real
aparece mas adelante, en el fit del TfidfVectorizer (ver
run_tfidf_pipeline.py): por eso ese fit se hace solo sobre train.
"""

import time
import pandas as pd
from preprocess import preprocess_batch

for split in ["train", "test"]:
    print(f"--- Preprocesando {split} ---")
    df = pd.read_csv(f"../data/ag_news_{split}.csv")
    print(f"Filas: {len(df)}")

    t0 = time.time()
    df["text_clean"] = preprocess_batch(df["text"].tolist())
    print(f"Listo en {time.time() - t0:.1f}s")

    df.to_csv(f"../data/{split}_preprocesado.csv", index=False)
    print(f"Guardado en data/{split}_preprocesado.csv\n")
