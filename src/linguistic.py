"""Perfil linguístico via spaCy (nível sintático/estrutural, cf. notebook 04).

Cada métrica compara o texto analisado com as médias de três grupos do
dataset: notícias verdadeiras, fakes humanas e fakes sintéticas.
"""
from __future__ import annotations

import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINES_PATH = os.path.join(BASE_DIR, "data", "baselines.json")

METRICS = [
    {
        "key": "properNouns",
        "label": "Nomes próprios",
        "description": "Proporção de nomes de pessoas, lugares e organizações.",
        "unit": "%",
    },
    {
        "key": "peopleMentioned",
        "label": "Pessoas citadas",
        "description": "Proporção de entidades identificadas como pessoas.",
        "unit": "%",
    },
    {
        "key": "adjectives",
        "label": "Adjetivos",
        "description": "Proporção de palavras que qualificam substantivos.",
        "unit": "%",
    },
    {
        "key": "verbs",
        "label": "Verbos",
        "description": "Proporção de verbos entre os tokens analisados.",
        "unit": "%",
    },
    {
        "key": "wordsPerSentence",
        "label": "Palavras por frase",
        "description": "Comprimento médio das frases do texto.",
        "unit": "number",
    },
    {
        "key": "tokens",
        "label": "Total de tokens",
        "description": "Quantidade de unidades processadas pelo modelo linguístico.",
        "unit": "number",
    },
]

_nlp = None
_baselines = None


def get_nlp():
    global _nlp
    if _nlp is None:
        import spacy

        _nlp = spacy.load("pt_core_news_sm")
    return _nlp


def extract_features(text: str) -> dict:
    """Métricas brutas de um texto (mesma definição usada nos baselines)."""
    doc = get_nlp()(text)
    tokens = [t for t in doc if not t.is_punct and not t.is_space]
    total = len(tokens)
    if total == 0:
        return {m["key"]: 0 for m in METRICS}
    n_sents = max(1, sum(1 for _ in doc.sents))
    n_per = sum(1 for e in doc.ents if e.label_ == "PER")
    return {
        "properNouns": round(100 * sum(1 for t in tokens if t.pos_ == "PROPN") / total, 1),
        "peopleMentioned": round(100 * n_per / total, 1),
        "adjectives": round(100 * sum(1 for t in tokens if t.pos_ == "ADJ") / total, 1),
        "verbs": round(100 * sum(1 for t in tokens if t.pos_ == "VERB") / total, 1),
        "wordsPerSentence": round(total / n_sents, 1),
        "tokens": total,
    }


def get_baselines() -> dict:
    global _baselines
    if _baselines is None:
        if not os.path.exists(BASELINES_PATH):
            raise FileNotFoundError(
                f"Baselines não encontrados em {BASELINES_PATH}. "
                "Rode: python scripts/compute_baselines.py"
            )
        with open(BASELINES_PATH, encoding="utf-8") as f:
            _baselines = json.load(f)
    return _baselines


def analyze(text: str) -> list[dict]:
    feats = extract_features(text)
    base = get_baselines()
    out = []
    for m in METRICS:
        out.append(
            {
                **m,
                "text": feats[m["key"]],
                "trueNews": base["true"][m["key"]],
                "humanFake": base["human_fake"][m["key"]],
                "syntheticFake": base["synthetic_fake"][m["key"]],
            }
        )
    return out
