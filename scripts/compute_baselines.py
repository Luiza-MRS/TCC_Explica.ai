"""Gera data/baselines.json com as médias de cada grupo do dataset.

Grupos: true (label=true), human_fake (label=fake, origin=human),
synthetic_fake (label=fake, origin=synthetic).

Uso (a partir da raiz do projeto):
    .venv/Scripts/python scripts/compute_baselines.py   (Windows)
    .venv/bin/python scripts/compute_baselines.py       (Linux)
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.linguistic import BASELINES_PATH, METRICS, extract_features

SAMPLE_PER_GROUP = int(os.environ.get("XPLICA_BASELINE_SAMPLE", "300"))
CSV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "news_dataset.csv",
)


def group_mean(df: pd.DataFrame) -> dict:
    acc = {m["key"]: [] for m in METRICS}
    for text in df["text"]:
        feats = extract_features(str(text))
        for key, val in feats.items():
            acc[key].append(val)
    return {k: round(float(sum(v) / max(1, len(v))), 1) for k, v in acc.items()}


def main() -> None:
    df = pd.read_csv(CSV_PATH, usecols=["text", "label", "origin"])
    groups = {
        "true": df[df["label"] == "true"],
        "human_fake": df[(df["label"] == "fake") & (df["origin"] == "human")],
        "synthetic_fake": df[
            (df["label"] == "fake") & (df["origin"] == "synthetic")
        ],
    }
    baselines = {}
    for name, g in groups.items():
        sample = g.sample(n=min(SAMPLE_PER_GROUP, len(g)), random_state=42)
        print(f"{name}: {len(g)} textos, usando {len(sample)}")
        baselines[name] = group_mean(sample)
    with open(BASELINES_PATH, "w", encoding="utf-8") as f:
        json.dump(baselines, f, ensure_ascii=False, indent=2)
    print(f"Salvo em {BASELINES_PATH}")


if __name__ == "__main__":
    main()
