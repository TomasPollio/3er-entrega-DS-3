"""
Pipeline de Preprocesamiento y Diagnostico (PPD) - Modulo 2
Data Science III: NLP & Deep Learning

Implementa preprocess_text(text): limpieza regex + normalizacion +
lematizacion con spaCy, pensado para el corpus AG News.

Nota de diseno: la limpieza regex/normalizacion y la lematizacion con
spaCy estan separadas en dos funciones (_clean_regex y la lematizacion
via nlp.pipe) para poder procesar el corpus completo en lote (batch)
de forma eficiente, en vez de instanciar el pipeline de spaCy fila por
fila. preprocess_text() expone la version "fila por fila", pensada
para probar el pipeline con un texto suelto.
"""

import re
import html
import spacy

# Cargamos el modelo de ingles una sola vez (es costoso instanciarlo).
# Desactivamos componentes que no necesitamos (parser, ner) para que
# la lematizacion corra mas rapido sobre miles de documentos.
_nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# Patrones de ruido observados en una inspeccion previa del corpus:
# - Tags HTML *escapados* como &lt;b&gt; o &lt;A HREF="..."&gt;
# - Entidades HTML como &#39; (apostrofe) o &amp;
# - Una rareza propia de este corpus (AG News, Zhang et al. 2015):
#   a varias entidades les falta el "&" inicial, quedando como texto
#   literal "quot;", "amp;", "lt;", "gt;", "#39;", "#36;" sueltos en
#   el cuerpo de la noticia. html.unescape() no las detecta porque no
#   tienen el "&", asi que se limpian aparte con regex.
# - URLs completas (http://..., www....)
PSEUDO_ENTITY_RE = re.compile(r"(?<![a-zA-Z])(quot|amp|lt|gt|apos|#\d+);")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]")
MULTI_SPACE_RE = re.compile(r"\s+")


def _clean_regex(text: str) -> str:
    """Pasos 1-5: decodificar HTML, sacar tags/URLs, minusculas, solo letras."""
    if not isinstance(text, str):
        return ""

    # 1) Decodificar entidades HTML bien formadas: &lt;b&gt; -> <b>, &#39; -> ', &amp; -> &
    text = html.unescape(text)

    # 1b) Limpiar pseudo-entidades mal formadas propias de este corpus (sin "&")
    text = PSEUDO_ENTITY_RE.sub(" ", text)

    # 2) Eliminar tags HTML (ya decodificados a < >) y URLs
    text = re.sub(r"<[^>]+>", " ", text)
    text = URL_RE.sub(" ", text)

    # 3) Minusculas
    text = text.lower()

    # 4) Solo letras y espacios (fuera numeros, puntuacion, simbolos)
    text = NON_ALPHA_RE.sub(" ", text)

    # 5) Colapsar espacios
    return MULTI_SPACE_RE.sub(" ", text).strip()


def preprocess_text(text: str, remove_stopwords: bool = True) -> str:
    """
    Limpia, normaliza y lematiza un texto crudo del corpus AG News.

    Pasos:
      1. Decodificar entidades HTML (&lt; -> <, &#39; -> ', &amp; -> &)
      2. Eliminar tags HTML (ya decodificados) y URLs
      3. Pasar a minusculas
      4. Eliminar numeros y puntuacion (solo letras y espacios)
      5. Colapsar espacios multiples
      6. Lematizar con spaCy y (opcionalmente) filtrar stopwords

    Args:
        text: texto crudo de una noticia.
        remove_stopwords: si True, descarta stopwords estandar de spaCy.

    Returns:
        Texto limpio y lematizado, tokens separados por un espacio.
    """
    cleaned = _clean_regex(text)
    doc = _nlp(cleaned)
    if remove_stopwords:
        tokens = [tok.lemma_ for tok in doc if not tok.is_stop and tok.lemma_.strip()]
    else:
        tokens = [tok.lemma_ for tok in doc if tok.lemma_.strip()]
    return " ".join(tokens)


def preprocess_batch(texts, remove_stopwords: bool = True, batch_size: int = 128, n_process: int = 1):
    """
    Version en lote de preprocess_text, para corpus grandes.
    Usa nlp.pipe para lematizar en batch (mucho mas rapido que llamar
    preprocess_text() fila por fila sobre miles de documentos).
    """
    cleaned_texts = [_clean_regex(t) for t in texts]
    results = []
    for doc in _nlp.pipe(cleaned_texts, batch_size=batch_size, n_process=n_process):
        if remove_stopwords:
            tokens = [tok.lemma_ for tok in doc if not tok.is_stop and tok.lemma_.strip()]
        else:
            tokens = [tok.lemma_ for tok in doc if tok.lemma_.strip()]
        results.append(" ".join(tokens))
    return results


if __name__ == "__main__":
    ejemplo = (
        "Myanmar #39;s military govt frees number two political prisoner: "
        "party &lt;b&gt;...&lt;/b&gt;. YANGON : Visit http://example.com for more."
    )
    print("ANTES:", ejemplo)
    print("DESPUES:", preprocess_text(ejemplo))
