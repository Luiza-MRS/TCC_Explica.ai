"""Carrega o BERTimbau treinado (cenário 3) e retorna probabilidades fake/true.

O diretório do modelo pode ser ajustado via variável de ambiente
XPLICA_MODEL_DIR (padrão: models/bertimbau_scenario3).
"""
from __future__ import annotations

import os

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.environ.get(
    "XPLICA_MODEL_DIR", os.path.join(BASE_DIR, "models", "bertimbau_scenario3")
)
MAX_LENGTH = 512
LABELS = ("fake", "true")  # label_map do treino: fake=0, true=1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = None
_tokenizer = None


def model_available() -> bool:
    return os.path.exists(os.path.join(MODEL_DIR, "config.json"))


def load_model():
    global _model, _tokenizer
    if _model is None:
        if not model_available():
            raise FileNotFoundError(
                f"Modelo não encontrado em {MODEL_DIR}. "
                "Siga docs/treinamento_colab.md para treinar e copiar os pesos."
            )
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, do_lower_case=False)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        _model.to(DEVICE)
        _model.eval()
    return _model, _tokenizer


def predict_proba(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Retorna array (n, 2) com colunas [P(fake), P(true)]."""
    model, tokenizer = load_model()
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i : i + batch_size])
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            logits = model(**inputs).logits
            all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(all_probs, axis=0)


def predict(text: str) -> dict:
    probs = predict_proba([text])[0]
    p_fake, p_true = float(probs[0]), float(probs[1])
    return {
        "falseProbability": round(p_fake * 100),
        "trueProbability": round(p_true * 100),
        "classification": "Provavelmente falsa"
        if p_fake >= 0.5
        else "Provavelmente verdadeira",
    }
