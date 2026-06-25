#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
    recall_score,
)


PROJECT_DIR = Path("/Users/widjeisi/project_music")

TRACKS_CSV = PROJECT_DIR / "tracks.csv"
FEATURES_CSV = PROJECT_DIR / "features.csv"
GENRES_CSV = PROJECT_DIR / "genres.csv"
MEL_DIR = PROJECT_DIR / "mel_spectrograms_6_genres_full"

OUTDIR = PROJECT_DIR / "cnn_models" / "final_6genres_artist_split"
SPLIT_JSON = OUTDIR / "final_artist_aware_split_6genres.json"

RANDOM_SEED = 31

TARGET_GENRE_NAMES = [
    "Jazz",
    "Classical",
    "Hip-Hop",
    "Metal",
    "Country",
    "Dubstep",
]

TEST_PER_GENRE = 200
VALID_PER_GENRE = 100

CANDIDATE_STATISTICS = {
    "mean",
    "median",
    "std",
    "max",
}


def seed_everything(seed=31):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def normalize_genre_name(name):
    return (
        str(name)
        .lower()
        .strip()
        .replace("_", "-")
        .replace(" ", "-")
    )


def save_json(obj, path):
    def convert(x):
        if isinstance(x, dict):
            return {
                str(k): convert(v)
                for k, v in x.items()
            }

        if isinstance(x, (list, tuple)):
            return [
                convert(v)
                for v in x
            ]

        if isinstance(x, np.ndarray):
            return x.tolist()

        if isinstance(x, np.integer):
            return int(x)

        if isinstance(x, np.floating):
            return float(x)

        if isinstance(x, np.bool_):
            return bool(x)

        return x

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(convert(obj), f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_columns(columns):
    result = []

    for col in columns:
        if isinstance(col, tuple):
            parts = [
                str(x)
                for x in col
                if str(x) != "" and str(x) != "nan"
            ]
            result.append("__".join(parts))
        else:
            result.append(str(col))

    return result


def read_genres():
    genres_df = pd.read_csv(GENRES_CSV, index_col=0)
    genres_df.index = genres_df.index.astype(int)

    name_to_id = {}

    for genre_id, row in genres_df.iterrows():
        name_to_id[normalize_genre_name(row["title"])] = int(genre_id)

    selected_genre_ids = []

    for name in TARGET_GENRE_NAMES:
        key = normalize_genre_name(name)

        if key not in name_to_id:
            raise ValueError(f"Жанр не найден в genres.csv: {name}")

        selected_genre_ids.append(int(name_to_id[key]))

    genre_id_to_label = {
        int(genre_id): label
        for label, genre_id in enumerate(selected_genre_ids)
    }

    label_to_genre_id = {
        int(label): int(genre_id)
        for genre_id, label in genre_id_to_label.items()
    }

    class_names = [
        str(genres_df.loc[label_to_genre_id[label], "title"])
        for label in sorted(label_to_genre_id)
    ]

    if class_names != TARGET_GENRE_NAMES:
        raise ValueError(
            "Неправильный порядок классов.\n"
            f"Ожидалось: {TARGET_GENRE_NAMES}\n"
            f"Получилось: {class_names}"
        )

    return (
        genres_df,
        selected_genre_ids,
        genre_id_to_label,
        label_to_genre_id,
        class_names,
    )


def read_mel_index(genres_df, selected_genre_ids, genre_id_to_label):
    rows = []

    if not MEL_DIR.exists():
        raise ValueError(f"MEL_DIR не существует: {MEL_DIR}")

    name_to_genre_id = {
        normalize_genre_name(genres_df.loc[genre_id, "title"]): int(genre_id)
        for genre_id in selected_genre_ids
    }

    target_set = {
        normalize_genre_name(name)
        for name in TARGET_GENRE_NAMES
    }

    for genre_dir in sorted(MEL_DIR.iterdir()):
        if not genre_dir.is_dir():
            continue

        key = normalize_genre_name(genre_dir.name)

        if key not in target_set:
            continue

        if key not in name_to_genre_id:
            continue

        genre_id = int(name_to_genre_id[key])
        label = int(genre_id_to_label[genre_id])

        for path in sorted(genre_dir.glob("*.npy")):
            try:
                track_id = int(path.stem)
            except Exception:
                continue

            rows.append({
                "track_id": int(track_id),
                "genre_id": int(genre_id),
                "label": int(label),
                "genre_name": TARGET_GENRE_NAMES[label],
                "mel_path": str(path),
            })

    df = pd.DataFrame(rows)

    if len(df) == 0:
        raise ValueError(f"Не найдено mel в {MEL_DIR}")

    df = df.drop_duplicates(subset=["track_id"]).copy()
    df["track_id"] = df["track_id"].astype(int)
    df["genre_id"] = df["genre_id"].astype(int)
    df["label"] = df["label"].astype(int)

    return df


def read_features_candidates():
    features = pd.read_csv(
        FEATURES_CSV,
        index_col=0,
        header=[0, 1, 2],
    )

    features.index = features.index.astype(int)

    features.columns = pd.MultiIndex.from_tuples(
        [
            (
                str(col[0]),
                str(col[1]),
                str(col[2]).zfill(2),
            )
            for col in features.columns
        ],
        names=features.columns.names,
    )

    candidate_cols = [
        col
        for col in features.columns
        if col[1] in CANDIDATE_STATISTICS
    ]

    features = features[candidate_cols].copy()
    features.columns = flatten_columns(features.columns)
    features = features.apply(pd.to_numeric, errors="coerce")

    return features


def build_master_df():
    genres_df, selected_genre_ids, genre_id_to_label, label_to_genre_id, class_names = read_genres()

    mel_df = read_mel_index(
        genres_df=genres_df,
        selected_genre_ids=selected_genre_ids,
        genre_id_to_label=genre_id_to_label,
    )

    features_df = read_features_candidates()

    mel_meta = mel_df.set_index("track_id")[
        [
            "genre_id",
            "label",
            "genre_name",
            "mel_path",
        ]
    ].copy()

    features_df = features_df.copy()
    features_df.index = features_df.index.astype(int)

    df = mel_meta.join(features_df, how="inner")
    df.index = df.index.astype(int)

    return (
        df,
        genres_df,
        selected_genre_ids,
        genre_id_to_label,
        label_to_genre_id,
        class_names,
    )


def load_final_split():
    if not SPLIT_JSON.exists():
        raise FileNotFoundError(
            f"Сначала создай общий split: {SPLIT_JSON}"
        )

    split = load_json(SPLIT_JSON)

    for key in ["train", "valid", "test"]:
        split[key] = [int(x) for x in split[key]]

    return split


def check_split_against_df(df, split):
    available_ids = set(int(x) for x in df.index)

    for key in ["train", "valid", "test"]:
        missing = sorted(set(split[key]) - available_ids)

        if missing:
            raise ValueError(
                f"В split[{key}] есть id, которых нет в master df. "
                f"Примеры: {missing[:10]}"
            )

    if set(split["train"]) & set(split["valid"]):
        raise ValueError("train пересекается с valid.")

    if set(split["train"]) & set(split["test"]):
        raise ValueError("train пересекается с test.")

    if set(split["valid"]) & set(split["test"]):
        raise ValueError("valid пересекается с test.")


def softmax_np(logits):
    x = np.asarray(logits, dtype=np.float64)
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)


def probs_to_logits(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
    return np.log(np.maximum(probs, 1e-12))


def macro_recall(y_true, y_pred, labels):
    recalls = []
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    for label in labels:
        mask = y_true == label

        if int(mask.sum()) == 0:
            continue

        recalls.append(float((y_pred[mask] == label).mean()))

    return float(np.mean(recalls)) if recalls else 0.0


def min_recall(y_true, y_pred, labels):
    recalls = []
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    for label in labels:
        mask = y_true == label

        if int(mask.sum()) == 0:
            continue

        recalls.append(float((y_pred[mask] == label).mean()))

    return float(np.min(recalls)) if recalls else 0.0


def balanced_objective(y_true, y_pred, labels):
    acc = accuracy_score(y_true, y_pred)
    mac = macro_recall(y_true, y_pred, labels)
    mn = min_recall(y_true, y_pred, labels)

    return 0.45 * acc + 0.30 * mac + 0.25 * mn


def metrics_from_logits(logits, y_true, class_names):
    labels = list(range(len(class_names)))

    probs = softmax_np(logits)
    pred = probs.argmax(axis=1)

    acc = accuracy_score(y_true, pred)
    mac = macro_recall(y_true, pred, labels)
    mn = min_recall(y_true, pred, labels)
    obj = balanced_objective(y_true, pred, labels)

    macro_f1 = f1_score(
        y_true,
        pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    per_class = recall_score(
        y_true,
        pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        pred,
        labels=labels,
    )

    report = classification_report(
        y_true,
        pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": float(acc),
        "accuracy_percent": float(acc * 100),
        "macro_recall": float(mac),
        "min_recall": float(mn),
        "objective": float(obj),
        "macro_f1": float(macro_f1),
        "per_class_recall": {
            class_names[i]: float(per_class[i])
            for i in range(len(class_names))
        },
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def print_metrics(title, logits, y_true, class_names):
    metrics = metrics_from_logits(logits, y_true, class_names)

    probs = softmax_np(logits)
    pred = probs.argmax(axis=1)

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)
    print(
        f"accuracy={metrics['accuracy']:.4f} | "
        f"{metrics['accuracy_percent']:.2f}%"
    )
    print(
        f"macro_recall={metrics['macro_recall']:.4f} | "
        f"min_recall={metrics['min_recall']:.4f} | "
        f"objective={metrics['objective']:.4f} | "
        f"macro_f1={metrics['macro_f1']:.4f}"
    )

    print()
    print("По жанрам:")

    rows = []

    for label, genre_name in enumerate(class_names):
        mask = np.asarray(y_true) == label
        total = int(mask.sum())
        correct = int((pred[mask] == label).sum())
        percent = correct / total * 100 if total else 0.0

        rows.append({
            "genre_name": genre_name,
            "correct": correct,
            "total": total,
            "percent": percent,
        })

    rows_df = pd.DataFrame(rows).sort_values("percent", ascending=False)

    for _, row in rows_df.iterrows():
        print(
            f"{row['genre_name']}: "
            f"{int(row['correct'])}/{int(row['total'])} "
            f"= {row['percent']:.2f}%"
        )

    return metrics


def save_predictions_csv(track_ids, y_true, logits, class_names, outpath):
    probs = softmax_np(logits)
    pred = probs.argmax(axis=1)

    data = {
        "track_id": np.asarray(track_ids).astype(int),
        "y_true": np.asarray(y_true).astype(int),
        "y_pred": pred.astype(int),
        "true_label": [class_names[int(x)] for x in y_true],
        "pred_label": [class_names[int(x)] for x in pred],
        "is_correct": np.asarray(y_true).astype(int) == pred.astype(int),
    }

    for i, name in enumerate(class_names):
        safe_name = normalize_genre_name(name).replace("-", "_")
        data[f"prob_{safe_name}"] = probs[:, i]

    pd.DataFrame(data).to_csv(outpath, index=False)