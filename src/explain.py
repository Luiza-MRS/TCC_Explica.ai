"""Explicação lexical via LIME (mesma abordagem do notebook 04).

Explica a classe "true" (índice 1): score negativo = palavra que empurra
para "fake"; score positivo = palavra que empurra para "true".
"""
from __future__ import annotations

import os

from lime.lime_text import LimeTextExplainer

NUM_FEATURES = 7
NUM_SAMPLES = int(os.environ.get("XPLICA_LIME_SAMPLES", "300"))

_explainer = LimeTextExplainer(class_names=["fake", "true"], random_state=42)


def explain(text: str, predict_proba_fn) -> list[dict]:
    exp = _explainer.explain_instance(
        text,
        predict_proba_fn,
        labels=(1,),
        num_features=NUM_FEATURES,
        num_samples=NUM_SAMPLES,
    )
    features = [
        {"word": word, "score": round(float(weight), 4)}
        for word, weight in exp.as_list(label=1)
    ]
    return sorted(features, key=lambda f: f["score"])
