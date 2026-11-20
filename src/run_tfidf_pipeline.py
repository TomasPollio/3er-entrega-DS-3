"""
Pipeline de Clasificacion Supervisada con TF-IDF (Modulo 3).

1. Carga el corpus ya preprocesado (train/test, generado por
   run_preprocessing.py).
2. Vectoriza con TfidfVectorizer: fit_transform SOLO en train,
   transform en test (evita data leakage de vocabulario).
3. Experimenta con un pequeno grid de (max_features, ngram_range)
   para justificar la configuracion final elegida.
4. Entrena un clasificador de Regresion Logistica sobre la
   configuracion ganadora.
5. Evalua con classification_report y matriz de confusion sobre test.
"""

import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay

OUT = "../outputs"
os.makedirs(OUT, exist_ok=True)

train = pd.read_csv("../data/train_preprocesado.csv")
test = pd.read_csv("../data/test_preprocesado.csv")
train["text_clean"] = train["text_clean"].fillna("")
test["text_clean"] = test["text_clean"].fillna("")

X_train_text, y_train = train["text_clean"], train["label"]
X_test_text, y_test = test["text_clean"], test["label"]

# -----------------------------------------------------------------
# 1) Experimentacion: comparar configuraciones de TfidfVectorizer
# -----------------------------------------------------------------
# max_features acota el vocabulario (evita overfitting / RAM excesiva
# con un corpus chico de 8000 docs). ngram_range=(1,2) suma bigramas
# para capturar frases cortas ("oil price", "prime minister") que un
# unigrama solo no distingue.
configs = [
    {"name": "unigramas, sin limite",      "max_features": None,  "ngram_range": (1, 1)},
    {"name": "unigramas, max_features=5000", "max_features": 5000, "ngram_range": (1, 1)},
    {"name": "uni+bigramas, max_features=5000", "max_features": 5000, "ngram_range": (1, 2)},
    {"name": "uni+bigramas, max_features=10000", "max_features": 10000, "ngram_range": (1, 2)},
]

resultados = []
for cfg in configs:
    vec = TfidfVectorizer(max_features=cfg["max_features"], ngram_range=cfg["ngram_range"])
    X_train = vec.fit_transform(X_train_text)       # fit SOLO en train
    X_test = vec.transform(X_test_text)              # transform en test

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    f1_macro = f1_score(y_test, y_pred, average="macro")
    resultados.append({
        "config": cfg["name"],
        "vocab_size": len(vec.vocabulary_),
        "f1_macro_test": round(f1_macro, 4),
    })
    print(f"{cfg['name']:35s} | vocab={len(vec.vocabulary_):6d} | F1-macro={f1_macro:.4f}")

resultados_df = pd.DataFrame(resultados)
resultados_df.to_csv(f"{OUT}/comparacion_configs_tfidf.csv", index=False)

# -----------------------------------------------------------------
# 2) Configuracion final elegida (la de mejor F1-macro)
# -----------------------------------------------------------------
best_idx = resultados_df["f1_macro_test"].idxmax()
best_cfg = configs[best_idx]
print(f"\nConfiguracion elegida: {best_cfg['name']}")

vectorizer = TfidfVectorizer(
    max_features=best_cfg["max_features"],
    ngram_range=best_cfg["ngram_range"],
    sublinear_tf=True,   # escala log(tf) en vez de tf crudo, ayuda con docs cortos
)
X_train = vectorizer.fit_transform(X_train_text)   # fit SOLO en train
X_test = vectorizer.transform(X_test_text)           # solo transform en test

clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# -----------------------------------------------------------------
# 3) Evaluacion
# -----------------------------------------------------------------
report_dict = classification_report(y_test, y_pred, output_dict=True)
report_txt = classification_report(y_test, y_pred)
print("\n" + report_txt)

with open(f"{OUT}/classification_report.txt", "w") as f:
    f.write(report_txt)
with open(f"{OUT}/classification_report.json", "w") as f:
    json.dump(report_dict, f, indent=2)

labels_sorted = sorted(y_test.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)

fig, ax = plt.subplots(figsize=(6.5, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_sorted)
disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
ax.set_title("Matriz de confusión — TF-IDF + Regresión Logística (test, n=2000)")
plt.tight_layout()
plt.savefig(f"{OUT}/matriz_confusion.png", dpi=150)
plt.close()

# -----------------------------------------------------------------
# 4) Resumen numerico para el README
# -----------------------------------------------------------------
resumen = {
    "config_elegida": best_cfg["name"],
    "max_features": best_cfg["max_features"],
    "ngram_range": list(best_cfg["ngram_range"]),
    "vocab_size": len(vectorizer.vocabulary_),
    "accuracy_test": round(report_dict["accuracy"], 4),
    "f1_macro_test": round(report_dict["macro avg"]["f1-score"], 4),
    "f1_weighted_test": round(report_dict["weighted avg"]["f1-score"], 4),
    "por_clase": {
        clase: {
            "precision": round(report_dict[clase]["precision"], 4),
            "recall": round(report_dict[clase]["recall"], 4),
            "f1": round(report_dict[clase]["f1-score"], 4),
        }
        for clase in labels_sorted
    },
    "matriz_confusion": cm.tolist(),
    "labels_orden": labels_sorted,
}
with open(f"{OUT}/resumen_resultados.json", "w") as f:
    json.dump(resumen, f, indent=2, ensure_ascii=False)

print(f"\nAccuracy: {resumen['accuracy_test']:.4f} | F1-macro: {resumen['f1_macro_test']:.4f}")
print("Resultados guardados en outputs/")
