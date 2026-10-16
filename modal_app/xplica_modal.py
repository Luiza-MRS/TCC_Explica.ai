"""Xplica.ai no Modal.

Duas partes:
  1. Analisador (GPU T4): roda o core.py da Emilly com o modelo oficial do
     cenario 3 (classificacao + LIME com 1000 amostras/10 palavras + spaCy).
  2. Interface (CPU): tela em Gradio que chama o Analisador.

Uso (dentro da pasta modal_app, com o .venv ativo):
  modal serve xplica_modal.py    -> link temporario para testar (Ctrl+C encerra)
  modal deploy xplica_modal.py   -> link fixo, publicado
"""
from pathlib import Path

import modal

PASTA = Path(__file__).parent
MODEL_REPO = "httpsemilly/xplica-bertimbau-scenario3"
TOKENIZER_NAME = "neuralmind/bert-base-portuguese-cased"
SPACY_WHL = (
    "https://github.com/explosion/spacy-models/releases/download/"
    "pt_core_news_sm-3.8.0/pt_core_news_sm-3.8.0-py3-none-any.whl"
)

app = modal.App("xplica-ai")


def baixar_modelos():
    """Roda uma vez, na montagem da imagem: deixa modelo e tokenizer salvos nela."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    AutoTokenizer.from_pretrained(TOKENIZER_NAME, do_lower_case=False)
    AutoModelForSequenceClassification.from_pretrained(MODEL_REPO)


# ---------- Parte 1: analisador na GPU ----------
imagem_gpu = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "torch==2.13.0",
        "transformers==5.16.1",
        "lime==0.2.0.1",
        "spacy==3.8.16",
        "numpy>=2.1,<3",
        SPACY_WHL,
    )
    .run_function(baixar_modelos)
    .add_local_file(PASTA / "core.py", "/root/core.py")
)


@app.cls(image=imagem_gpu, gpu="T4", scaledown_window=120, timeout=600)
class Analisador:
    @modal.enter()
    def carregar(self):
        from core import Analyzer

        self.analyzer = Analyzer(model_repo=MODEL_REPO)

    @modal.method()
    def analisar(self, texto: str) -> dict:
        return self.analyzer.analyze(texto)


# ---------- Parte 2: interface Gradio ----------
imagem_web = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("gradio==6.30.0", "fastapi[standard]==0.143.2", "pandas>=2.2,<4")
    .add_local_file(PASTA.parent / "data" / "baselines.json", "/root/baselines.json")
)

# (chave no core.py, chave no baselines.json, rotulo, multiplicar por 100)
METRICAS = [
    ("propn_ratio", "properNouns", "Nomes próprios (%)", True),
    ("per_ratio", "peopleMentioned", "Pessoas citadas (%)", True),
    ("adj_ratio", "adjectives", "Adjetivos (%)", True),
    ("verb_ratio", "verbs", "Verbos (%)", True),
    ("avg_sentence_length", "wordsPerSentence", "Palavras por frase", False),
    ("total_tokens", "tokens", "Total de tokens", False),
]


@app.function(image=imagem_web, max_containers=1, scaledown_window=600)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def interface():
    import json

    import gradio as gr
    import pandas as pd
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app

    base = json.loads(Path("/root/baselines.json").read_text(encoding="utf-8"))
    analisador = Analisador()

    def analisar(texto):
        texto = (texto or "").strip()
        if len(texto) < 20:
            raise gr.Error("Cole uma notícia com pelo menos 20 caracteres.")
        try:
            r = analisador.analisar.remote(texto)
        except Exception as erro:
            raise gr.Error(f"Não foi possível analisar o texto: {erro}")

        falsa = round(r["prob_fake"] * 100)
        verdadeira = round(r["prob_true"] * 100)
        titulo = "Provavelmente falsa" if r["prob_fake"] >= 0.5 else "Provavelmente verdadeira"
        resumo = f"## {titulo}\nFalsa: {falsa}% · Verdadeira: {verdadeira}%"
        if r["truncated"]:
            resumo += (
                f"\n\n⚠️ O texto tem {r['n_tokens']} tokens. "
                "O modelo analisou apenas os primeiros 512."
            )

        probs = {"falsa": r["prob_fake"], "verdadeira": r["prob_true"]}

        lime = sorted(r["lime"], key=lambda item: item["weight"])
        lime_df = pd.DataFrame(
            {
                "palavra": [item["word"] for item in lime],
                "influência": [round(item["weight"], 4) for item in lime],
            }
        )

        ling = r["linguistic"]
        linhas = []
        for chave, chave_base, rotulo, porcentagem in METRICAS:
            fator = 100 if porcentagem else 1
            linhas.append(
                {
                    "Métrica": rotulo,
                    "Seu texto": round(ling.get(chave, 0) * fator, 1),
                    "Verdadeiras": base["true"][chave_base],
                    "Fakes humanas": base["human_fake"][chave_base],
                    "Fakes sintéticas": base["synthetic_fake"][chave_base],
                }
            )
        ling_df = pd.DataFrame(linhas)

        return resumo, probs, lime_df, ling_df

    with gr.Blocks(title="Xplica.ai") as demo:
        gr.Markdown(
            "# 🔍 Xplica.ai\n"
            "Detecção explicável de possíveis fake news em português: "
            "BERTimbau + LIME + spaCy. Texto truncado em 512 tokens.\n\n"
            "*O resultado indica padrões de escrita associados a notícias falsas. "
            "Não é uma checagem de fatos.*"
        )
        entrada = gr.Textbox(
            label="Cole a notícia para analisar",
            lines=8,
            max_lines=20,
            placeholder="Cole aqui o texto da notícia (mín. 20 caracteres)…",
        )
        botao = gr.Button("Analisar", variant="primary")
        saida_resumo = gr.Markdown()
        saida_probs = gr.Label(label="Probabilidades")
        gr.Markdown(
            "### Palavras de influência (LIME)\n"
            "Score negativo = evidência de fake · positivo = evidência de verdadeira."
        )
        saida_lime = gr.BarPlot(x="palavra", y="influência")
        gr.Markdown(
            "### Perfil linguístico (spaCy)\n"
            "Seu texto comparado às médias de notícias verdadeiras, "
            "fakes humanas e fakes sintéticas."
        )
        saida_ling = gr.Dataframe()
        botao.click(
            analisar,
            inputs=entrada,
            outputs=[saida_resumo, saida_probs, saida_lime, saida_ling],
        )

    return mount_gradio_app(app=FastAPI(), blocks=demo, path="/")
