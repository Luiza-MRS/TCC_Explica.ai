"""Xplica.ai — app Gradio (roda local e no Hugging Face Spaces ZeroGPU).

Reusa src/predictor.py, src/explain.py e src/linguistic.py.
Sem modelo em models/bertimbau_scenario3/ → aviso na UI, erro claro ao analisar.
"""

from __future__ import annotations

import os

os.environ.setdefault("XPLICA_LIME_SAMPLES", "100")

import gradio as gr
import pandas as pd

try:
    import spaces

    gpu = spaces.GPU(duration=120)
except ImportError:  # ponytail: spaces só existe no HF Spaces; local vira no-op

    def gpu(fn):
        return fn

from src import explain, linguistic, predictor

try:
    predictor.load_model()  # warm: cuda (emulação no boot do ZeroGPU) ou cpu local
    linguistic.get_nlp()
    MODEL_MISSING = False
except (FileNotFoundError, OSError):
    MODEL_MISSING = True

MODEL_HINT = (
    "Modelo ainda não treinado. Siga `docs/treinamento_colab.md` e coloque "
    "os pesos em `models/bertimbau_scenario3/` (config.json + model.safetensors + tokenizer)."
)


@gpu
def analyze(text: str):
    text = (text or "").strip()
    if len(text) < 20:
        raise gr.Error("Cole uma notícia com pelo menos 20 caracteres.")
    if MODEL_MISSING:
        raise gr.Error(MODEL_HINT)
    result = predictor.predict(text)
    lime_feats = explain.explain(text, predictor.predict_proba)
    ling = linguistic.analyze(text)

    cls = f"## {result['classification']}\nFalsa: {result['falseProbability']}% · Verdadeira: {result['trueProbability']}%"
    probs = {
        "falsa": result["falseProbability"] / 100,
        "verdadeira": result["trueProbability"] / 100,
    }
    lime_df = pd.DataFrame(
        {"palavra": [f["word"] for f in lime_feats], "influência": [f["score"] for f in lime_feats]}
    )
    ling_df = pd.DataFrame(
        [
            {
                "Métrica": m["label"],
                "Seu texto": m["text"],
                "Verdadeiras": m["trueNews"],
                "Fakes humanas": m["humanFake"],
                "Fakes sintéticas": m["syntheticFake"],
                "Unid.": m["unit"],
            }
            for m in ling
        ]
    )
    return cls, probs, lime_df, ling_df


with gr.Blocks(title="Xplica.ai") as demo:
    gr.Markdown(
        "# 🔍 Xplica.ai\nDetecção explicável de possíveis fake news em português: "
        "BERTimbau + LIME + spaCy. Texto truncado em 512 tokens."
    )
    if MODEL_MISSING:
        gr.Markdown(f"⚠️ {MODEL_HINT}")
    inp = gr.Textbox(
        label="Cole a notícia para analisar",
        lines=8,
        max_lines=20,
        placeholder="Cole aqui o texto da notícia (mín. 20 caracteres)…",
    )
    btn = gr.Button("Analisar", variant="primary")
    cls_out = gr.Markdown()
    probs_out = gr.Label(label="Probabilidades")
    gr.Markdown("### Palavras de influência (LIME)\nScore negativo = evidência de fake · positivo = evidência de verdadeira.")
    lime_out = gr.BarPlot(x="palavra", y="influência")
    gr.Markdown("### Perfil linguístico (spaCy)\nSeu texto vs. médias de notícias verdadeiras, fakes humanas e fakes sintéticas.")
    ling_out = gr.Dataframe()
    btn.click(analyze, inputs=inp, outputs=[cls_out, probs_out, lime_out, ling_out])

if __name__ == "__main__":
    demo.launch()
