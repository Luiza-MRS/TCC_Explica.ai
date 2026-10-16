# Treinamento do modelo (Colab GPU) — Cenário 3

O modelo da API (`models/bertimbau_scenario3/`) é treinado uma vez em GPU e
depois servido localmente em CPU. Faça este passo a passo no Google Colab
(Tempo de execução → Alterar tipo → GPU T4) e traga a pasta do modelo para o projeto.

## 1. Dependências

```python
!pip install -q transformers datasets scikit-learn
```

## 2. Dados

Suba o `data/news_dataset.csv` deste projeto para o Colab
(arraste o arquivo na barra lateral → ícone de pasta) ou leia do Drive:

```python
import pandas as pd

df_news = pd.read_csv("news_dataset.csv")
print(len(df_news), df_news["label"].value_counts().to_dict())
```

Esperado: 10.800 linhas (`fake/human` 3.600, `fake/synthetic` 3.600, `true/human` 3.600), splits `train` 8.640 / `test` 2.160.

## 3. Código de treino (cenário 3: treino misto 50/50, teste em sintéticas)

```python
import numpy as np, torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
MAX_LENGTH = 512
label_map = {"fake": 0, "true": 1}

class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings, self.labels = encodings, labels
    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

def tokenize_texts(texts, tokenizer):
    return tokenizer(list(texts), padding="max_length", truncation=True,
                     max_length=MAX_LENGTH, return_tensors="pt")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    pred = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, pred)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, pred, average="binary")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

def filter_train_data(df, synthetic_ratio=0.5, seed=42):
    df_true = df[(df["label"] == "true") & (df["split"] == "train")].copy()
    n_fake = len(df[(df["label"] == "fake") & (df["origin"] == "human") & (df["split"] == "train")])
    n_synth, n_human = int(n_fake * synthetic_ratio), n_fake - int(n_fake * synthetic_ratio)
    df_fh = df[(df["label"] == "fake") & (df["origin"] == "human") & (df["split"] == "train")].sample(n=n_human, random_state=seed)
    df_fs = df[(df["label"] == "fake") & (df["origin"] == "synthetic") & (df["split"] == "train")].sample(n=n_synth, random_state=seed)
    out = pd.concat([df_true, df_fh, df_fs]).copy()
    out["label_id"] = out["label"].map(label_map)
    return out

def filter_test_data(df):
    df_true = df[(df["label"] == "true") & (df["split"] == "test")].copy()
    df_fake = df[(df["label"] == "fake") & (df["origin"] == "synthetic") & (df["split"] == "test")].copy()
    out = pd.concat([df_true, df_fake]).copy()
    out["label_id"] = out["label"].map(label_map)
    return out

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)
df_train = filter_train_data(df_news)
df_test = filter_test_data(df_news)
train_ds = NewsDataset(tokenize_texts(df_train["text"], tokenizer), df_train["label_id"].tolist())
test_ds = NewsDataset(tokenize_texts(df_test["text"], tokenizer), df_test["label_id"].tolist())

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
args = TrainingArguments(
    output_dir="./results/scenario3",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    seed=42,
)
trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                  eval_dataset=test_ds, compute_metrics=compute_metrics)
trainer.train()
print(trainer.evaluate(test_ds))
```

## 4. Salvar e baixar

```python
model.save_pretrained("bertimbau_scenario3")
tokenizer.save_pretrained("bertimbau_scenario3")
!zip -qr bertimbau_scenario3.zip bertimbau_scenario3
```

Baixe o `bertimbau_scenario3.zip` (barra lateral → arquivos) e extraia em
`models/bertimbau_scenario3/` (deve conter `config.json`,
`model.safetensors` e os arquivos do tokenizer).

Confira no app:

```
.venv\Scripts\python -m streamlit run app.py
```

O aviso "Modelo ainda não treinado" deve sumir e a análise deve funcionar.
