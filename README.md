---
title: Xplica.ai
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---

# Xplica.ai

Detecção **explicável** de possíveis fake news em português. O app classifica a notícia com **BERTimbau** e mostra *por que* decidiu: **LIME** destaca as palavras que mais pesaram, e o **spaCy** compara o estilo do texto com o perfil médio de notícias verdadeiras, fakes humanas e fakes sintéticas.

App Gradio único, pronto para rodar localmente e no Hugging Face Spaces (hardware ZeroGPU, gratuito).

## Como funciona

1. **Classificação** (`src/predictor.py`): BERTimbau (`neuralmind/bert-base-portuguese-cased`) com fine-tuning binário fake/true, textos truncados em 512 tokens. Saída: probabilidade de ser falsa/verdadeira.
2. **Explicação lexical** (`src/explain.py`): LIME sobre a classe "verdadeira" — score negativo = palavra que empurra para *fake*, positivo = empurra para *verdadeira*.
3. **Perfil linguístico** (`src/linguistic.py`): spaCy (`pt_core_news_sm`) extrai 6 métricas (nomes próprios, pessoas citadas, adjetivos, verbos, palavras por frase, total de tokens) e compara com as médias de cada grupo em `data/baselines.json`.

Modelo treinado no cenário 3 (treino misto 50/50 humano+sintético, teste em sintéticas): **accuracy 0.9965, F1 0.9965** no conjunto de teste. Detalhes em `docs/treinamento_colab.md`.

## Pré-requisitos

- Python 3.10+ (`python --version`, testado em 3.12)
- Git + Git LFS (`git lfs version`) — só para baixar os pesos do modelo
- ~6 GB livres em disco (pesos do BERTimbau + dependências)

## 1. Instalação

Clone com LFS (traz o `model.safetensors` junto) e instale as dependências:

### Windows (PowerShell ou Git Bash)

```powershell
git lfs install
git clone <url-do-repo>
cd tcc-fake-news-v2
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\pip install -r requirements-cpu.txt
.venv\Scripts\python -c "import gradio, torch, transformers, lime, spacy; print('deps ok')"
```

### Linux

```bash
git lfs install
git clone <url-do-repo>
cd tcc-fake-news-v2
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -r requirements-cpu.txt
.venv/bin/python -c "import gradio, torch, transformers, lime, spacy; print('deps ok')"
```

> `requirements-cpu.txt` é para máquinas locais sem GPU. O `requirements.txt` (usado pelo Spaces) instala o torch com CUDA.
O modelo spaCy `pt_core_news_sm` já vem nos dois (via URL direta).

## 2. Configuração

O app funciona sem nenhuma configuração. Três variáveis de ambiente opcionais:

| Variável | Padrão | Para que serve |
|---|---|---|
| `XPLICA_MODEL_DIR` | `models/bertimbau_scenario3` | Aponta para outra pasta de pesos |
| `XPLICA_LIME_SAMPLES` | `100` (definido no `app.py`) | Nº de perturbações do LIME — 100 é ~3x mais rápido que 300, com qualidade equivalente p/ demo |
| `XPLICA_BASELINE_SAMPLE` | `300` | Amostras por grupo ao regenerar `data/baselines.json` |

Exemplo (Linux): `XPLICA_LIME_SAMPLES=300 .venv/bin/python app.py`.

### Modelo

Os pesos já vêm no repositório em `models/bertimbau_scenario3/` (`config.json` + `model.safetensors` + tokenizer, via Git LFS). Sem o modelo, o app abre normalmente e exibe um aviso — nunca quebra no import.

Para **retreinar do zero** em GPU (Colab), siga `docs/treinamento_colab.md` e extraia o `bertimbau_scenario3.zip` resultante para `models/bertimbau_scenario3/`.

### Baselines linguísticos

`data/baselines.json` já vem pronto. Para regenerar a partir do CSV (usa `XPLICA_BASELINE_SAMPLE` textos por grupo, seed fixa):

```bash
.venv/Scripts/python scripts/compute_baselines.py   # Windows
.venv/bin/python scripts/compute_baselines.py       # Linux
```

## 3. Execução

```powershell
.venv\Scripts\python app.py   # Windows
.venv/bin/python app.py       # Linux
```

Acesse http://localhost:7860, cole uma notícia (mín. 20 caracteres) e clique em **Analisar**. A primeira análise carrega o modelo (~1 min em CPU); as seguintes são mais rápidas.

## Solução de problemas

| Sintoma | Causa provável | Correção |
|---|---|---|
| `model.safetensors` com poucos KB / erro ao carregar pesos | Clone sem Git LFS | `git lfs install && git lfs pull` |
| `Baselines não encontrados` | `data/baselines.json` ausente | Regenere com `scripts/compute_baselines.py` |
| LIME muito lento | `XPLICA_LIME_SAMPLES` alto | Use 100 (padrão do app) |
| `ModuleNotFoundError: gradio/torch/transformers/...` | Instalação parcial (interrompida) ou venv errada | Com o `pip` da `.venv`: `pip install "gradio>=5,<7" "spaces>=0.30"` p/ o módulo faltante, ou reinstale tudo com `pip install -r requirements-cpu.txt`. O pacote `spaces` é opcional localmente (só o Spaces exige); sem ele o app roda normalmente |
| `No module named 'spaces'` | `spaces` não instalado localmente | Normal em versões antigas do `app.py` — atualize o código ou `pip install "spaces>=0.30"`. Nas versões atuais o import é opcional e o aviso pode ser ignorado |

## Publicar no Hugging Face Spaces

1. Crie um Space com SDK **Gradio** (o template inicial é sobrescrito pelo push; hardware ZeroGPU é gratuito).
2. Envie `app.py + src/ + data/baselines.json + models/ + requirements.txt + README.md` (o frontmatter YAML no topo deste README já configura o Space como Gradio). Pesos e CSV via LFS: `git lfs track "models/bertimbau_scenario3/*.safetensors" "data/news_dataset.csv"`.
3. Em Settings → Variables, defina `XPLICA_LIME_SAMPLES=100` (a inferência usa GPU via `@spaces.GPU`, liberada após o build).

## Estrutura

```
├── app.py                  # UI Gradio (local + Spaces)
├── src/                    # predictor (BERTimbau) + explain (LIME) + linguistic (spaCy)
├── data/news_dataset.csv   # dataset único (10.800 linhas: fake/human, fake/synthetic, true)
├── data/baselines.json     # médias linguísticas por grupo (obrigatório no deploy)
├── scripts/compute_baselines.py  # regenera data/baselines.json
├── models/bertimbau_scenario3/   # pesos do Colab (via git-lfs)
└── docs/treinamento_colab.md     # passo a passo do treino
```
