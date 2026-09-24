"""
Preprocesamiento reutilizable: lo usa tanto el entrenamiento final (step8) como
el modelo servido (model_wrapper.py), para garantizar que el modelo registrado
reciba texto crudo y aplique EXACTAMENTE la misma transformación que en el
entrenamiento (Sección 5: "recibir texto sin procesar y devolver la etiqueta").

Configuración final seleccionada tras las Secciones 3-4: normalizar
alargamientos (elongation=normalize), NO lematizar.
"""
import re

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
WS_RE = re.compile(r"\s+")
ELONGATION_RE = re.compile(r"(.)\1{2,}")


def clean_base(text: str) -> str:
    t = text.lower()
    t = URL_RE.sub(" url ", t)
    t = MENTION_RE.sub(" user ", t)
    t = WS_RE.sub(" ", t).strip()
    return t


def reduce_elongation(text: str) -> str:
    return ELONGATION_RE.sub(r"\1\1", text)


def full_preprocess(texts: list[str]) -> list[str]:
    return [reduce_elongation(clean_base(t)) for t in texts]
