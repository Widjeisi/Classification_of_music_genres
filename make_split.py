#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import random
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd


# ============================================================
# 0. НАСТРОЙКИ
# ============================================================

PROJECT_DIR = Path("/Users/widjeisi/project_music")

TRACKS_CSV = PROJECT_DIR / "tracks.csv"
FEATURES_CSV = PROJECT_DIR / "features.csv"
GENRES_CSV = PROJECT_DIR / "genres.csv"
MEL_DIR = PROJECT_DIR / "mel_spectrograms_6_genres_full"

OUTDIR = PROJECT_DIR / "cnn_models" / "final_6genres_artist_split"
OUTDIR.mkdir(parents=True, exist_ok=True)

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


# ============================================================
# 1. УТИЛИТЫ
# ============================================================

def normalize_genre_name(name):
    return (
        str(name)
        .lower()
        .strip()
        .replace("_", "-")
        .replace(" ", "-")
    )


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


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


def get_artist_columns(tracks):
    possible_id_cols = [
        ("artist", "id"),
        ("artist", "name"),
    ]

    artist_id_col = None
    artist_name_col = None

    if ("artist", "id") in tracks.columns:
        artist_id_col = ("artist", "id")

    if ("artist", "name") in tracks.columns:
        artist_name_col = ("artist", "name")

    if artist_id_col is None and artist_name_col is None:
        raise ValueError(
            "В tracks.csv не найдены artist/id или artist/name. "
            "Без этого artist-aware split невозможен."
        )

    return artist_id_col, artist_name_col


# ============================================================
# 2. ЧТЕНИЕ ДАННЫХ
# ============================================================

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
            f"Порядок классов не совпал.\n"
            f"Ожидалось: {TARGET_GENRE_NAMES}\n"
            f"Получилось: {class_names}"
        )

    return genres_df, selected_genre_ids, genre_id_to_label, label_to_genre_id, class_names


def read_mel_index(genres_df, selected_genre_ids, genre_id_to_label):
    rows = []

    target_set = {
        normalize_genre_name(name)
        for name in TARGET_GENRE_NAMES
    }

    name_to_genre_id = {
        normalize_genre_name(genres_df.loc[genre_id, "title"]): int(genre_id)
        for genre_id in selected_genre_ids
    }

    if not MEL_DIR.exists():
        raise ValueError(f"MEL_DIR не существует: {MEL_DIR}")

    for genre_dir in sorted(MEL_DIR.iterdir()):
        if not genre_dir.is_dir():
            continue

        normalized = normalize_genre_name(genre_dir.name)

        if normalized not in target_set:
            continue

        if normalized not in name_to_genre_id:
            continue

        genre_id = int(name_to_genre_id[normalized])
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

    mel_df = pd.DataFrame(rows)

    if len(mel_df) == 0:
        raise ValueError(f"Не найдено mel-спектрограмм в {MEL_DIR}")

    mel_df = mel_df.drop_duplicates(subset=["track_id"]).copy()
    mel_df["track_id"] = mel_df["track_id"].astype(int)
    mel_df["genre_id"] = mel_df["genre_id"].astype(int)
    mel_df["label"] = mel_df["label"].astype(int)

    return mel_df


def read_features_index():
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

    return set(int(x) for x in features.index)


def read_tracks_artist_info():
    tracks = pd.read_csv(
        TRACKS_CSV,
        index_col=0,
        header=[0, 1],
    )

    tracks.index = tracks.index.astype(int)

    artist_id_col, artist_name_col = get_artist_columns(tracks)

    rows = []

    for track_id, row in tracks.iterrows():
        artist_id = None
        artist_name = None

        if artist_id_col is not None:
            value = row[artist_id_col]

            if not pd.isna(value):
                try:
                    artist_id = int(value)
                except Exception:
                    artist_id = str(value)

        if artist_name_col is not None:
            value = row[artist_name_col]

            if not pd.isna(value):
                artist_name = str(value)

        if artist_id is not None:
            artist_key = f"id:{artist_id}"
        elif artist_name is not None:
            artist_key = f"name:{normalize_genre_name(artist_name)}"
        else:
            artist_key = f"unknown_track:{int(track_id)}"

        rows.append({
            "track_id": int(track_id),
            "artist_key": artist_key,
            "artist_id": artist_id,
            "artist_name": artist_name,
        })

    artist_df = pd.DataFrame(rows)
    artist_df["track_id"] = artist_df["track_id"].astype(int)

    return artist_df


def build_master_df():
    genres_df, selected_genre_ids, genre_id_to_label, label_to_genre_id, class_names = read_genres()

    mel_df = read_mel_index(
        genres_df=genres_df,
        selected_genre_ids=selected_genre_ids,
        genre_id_to_label=genre_id_to_label,
    )

    feature_ids = read_features_index()
    artist_df = read_tracks_artist_info()

    df = mel_df.copy()
    df = df[df["track_id"].isin(feature_ids)].copy()

    df = df.merge(
        artist_df,
        on="track_id",
        how="inner",
    )

    df = df.drop_duplicates(subset=["track_id"]).copy()
    df["track_id"] = df["track_id"].astype(int)

    return df, genres_df, selected_genre_ids, label_to_genre_id, class_names


# ============================================================
# 3. ARTIST-AWARE SPLIT
# ============================================================

def select_holdout_for_genre(
    df,
    label,
    target_count,
    forbidden_artists,
    forbidden_track_ids,
    rng,
):
    genre_df = df[df["label"] == label].copy()

    genre_df = genre_df[
        ~genre_df["track_id"].isin(forbidden_track_ids)
    ].copy()

    # Сначала пробуем брать артистов, которых ещё нет в holdout.
    clean_df = genre_df[
        ~genre_df["artist_key"].isin(forbidden_artists)
    ].copy()

    selected_ids = []
    selected_artists = set()

    artist_groups = []

    for artist_key, group in clean_df.groupby("artist_key"):
        ids = group["track_id"].astype(int).tolist()
        artist_groups.append((artist_key, ids))

    # Сначала артисты с меньшим числом треков: так проще набрать ровно 200/100.
    rng.shuffle(artist_groups)
    artist_groups = sorted(artist_groups, key=lambda x: len(x[1]))

    for artist_key, ids in artist_groups:
        if len(selected_ids) >= target_count:
            break

        ids = list(ids)
        rng.shuffle(ids)

        remaining = target_count - len(selected_ids)

        take = ids[:remaining]

        if not take:
            continue

        selected_ids.extend([int(x) for x in take])
        selected_artists.add(str(artist_key))

    # Если не хватило, добираем из любых артистов, но всё равно без уже выбранных track_id.
    if len(selected_ids) < target_count:
        already = set(selected_ids)

        fallback_df = genre_df[
            ~genre_df["track_id"].isin(already)
        ].copy()

        fallback_ids = fallback_df["track_id"].astype(int).tolist()
        rng.shuffle(fallback_ids)

        remaining = target_count - len(selected_ids)
        take = fallback_ids[:remaining]

        selected_ids.extend([int(x) for x in take])

        for track_id in take:
            artist_key = str(
                fallback_df[fallback_df["track_id"] == track_id]["artist_key"].iloc[0]
            )
            selected_artists.add(artist_key)

    if len(selected_ids) != target_count:
        raise ValueError(
            f"Не удалось набрать {target_count} треков для label={label}. "
            f"Набрано: {len(selected_ids)}"
        )

    selected_ids = selected_ids[:target_count]

    selected_artist_keys = set(
        df[df["track_id"].isin(selected_ids)]["artist_key"].astype(str).tolist()
    )

    return selected_ids, selected_artist_keys


def build_artist_aware_split(df, class_names):
    rng = random.Random(RANDOM_SEED)

    print()
    print("=" * 80)
    print("СОЗДАЮ ARTIST-AWARE SPLIT")
    print("=" * 80)

    print("test на жанр :", TEST_PER_GENRE)
    print("valid на жанр:", VALID_PER_GENRE)

    test_ids = []
    valid_ids = []

    test_artists = set()
    valid_artists = set()

    forbidden_track_ids = set()

    # 1. Сначала test.
    for label, genre_name in enumerate(class_names):
        selected_ids, selected_artist_keys = select_holdout_for_genre(
            df=df,
            label=label,
            target_count=TEST_PER_GENRE,
            forbidden_artists=test_artists,
            forbidden_track_ids=forbidden_track_ids,
            rng=rng,
        )

        test_ids.extend(selected_ids)
        test_artists.update(selected_artist_keys)
        forbidden_track_ids.update(selected_ids)

        print()
        print(f"TEST {genre_name}: {len(selected_ids)} треков, artists={len(selected_artist_keys)}")

    # 2. Потом validation, избегая test artists.
    forbidden_for_valid_artists = set(test_artists)

    for label, genre_name in enumerate(class_names):
        selected_ids, selected_artist_keys = select_holdout_for_genre(
            df=df,
            label=label,
            target_count=VALID_PER_GENRE,
            forbidden_artists=forbidden_for_valid_artists | valid_artists,
            forbidden_track_ids=forbidden_track_ids,
            rng=rng,
        )

        valid_ids.extend(selected_ids)
        valid_artists.update(selected_artist_keys)
        forbidden_track_ids.update(selected_ids)

        print()
        print(f"VALID {genre_name}: {len(selected_ids)} треков, artists={len(selected_artist_keys)}")

    test_ids = [int(x) for x in test_ids]
    valid_ids = [int(x) for x in valid_ids]

    holdout_artists = set(test_artists) | set(valid_artists)

    # 3. Train: всё, что не test/valid и не принадлежит test/valid artists.
    train_df = df[
        ~df["track_id"].isin(set(test_ids) | set(valid_ids))
    ].copy()

    train_df = train_df[
        ~train_df["artist_key"].isin(holdout_artists)
    ].copy()

    train_ids = train_df["track_id"].astype(int).tolist()

    train_ids = sorted(set(train_ids))
    valid_ids = sorted(set(valid_ids))
    test_ids = sorted(set(test_ids))

    # Проверки.
    if set(train_ids) & set(valid_ids):
        raise ValueError("train пересекается с valid.")

    if set(train_ids) & set(test_ids):
        raise ValueError("train пересекается с test.")

    if set(valid_ids) & set(test_ids):
        raise ValueError("valid пересекается с test.")

    train_artists = set(
        df[df["track_id"].isin(train_ids)]["artist_key"].astype(str).tolist()
    )

    valid_artists_real = set(
        df[df["track_id"].isin(valid_ids)]["artist_key"].astype(str).tolist()
    )

    test_artists_real = set(
        df[df["track_id"].isin(test_ids)]["artist_key"].astype(str).tolist()
    )

    train_test_artist_overlap = train_artists & test_artists_real
    train_valid_artist_overlap = train_artists & valid_artists_real
    valid_test_artist_overlap = valid_artists_real & test_artists_real

    print()
    print("=" * 80)
    print("SPLIT SUMMARY")
    print("=" * 80)

    print("train:", len(train_ids))
    print("valid:", len(valid_ids))
    print("test :", len(test_ids))

    print()
    print("Artist overlap:")
    print("train/test artists overlap:", len(train_test_artist_overlap))
    print("train/valid artists overlap:", len(train_valid_artist_overlap))
    print("valid/test artists overlap:", len(valid_test_artist_overlap))

    print()
    print("Количество по жанрам:")

    for label, genre_name in enumerate(class_names):
        train_count = int(df[df["track_id"].isin(train_ids) & (df["label"] == label)].shape[0])
        valid_count = int(df[df["track_id"].isin(valid_ids) & (df["label"] == label)].shape[0])
        test_count = int(df[df["track_id"].isin(test_ids) & (df["label"] == label)].shape[0])

        print()
        print(genre_name)
        print("train:", train_count)
        print("valid:", valid_count)
        print("test :", test_count)

        if valid_count != VALID_PER_GENRE:
            raise ValueError(f"{genre_name}: valid должен быть {VALID_PER_GENRE}, получилось {valid_count}")

        if test_count != TEST_PER_GENRE:
            raise ValueError(f"{genre_name}: test должен быть {TEST_PER_GENRE}, получилось {test_count}")

    split = {
        "split_name": "final_artist_aware_split_6genres",
        "artist_aware": True,
        "random_seed": RANDOM_SEED,
        "target_genre_names": TARGET_GENRE_NAMES,
        "test_per_genre": TEST_PER_GENRE,
        "valid_per_genre": VALID_PER_GENRE,
        "train": train_ids,
        "valid": valid_ids,
        "test": test_ids,
        "artist_overlap": {
            "train_test_artist_overlap_count": len(train_test_artist_overlap),
            "train_valid_artist_overlap_count": len(train_valid_artist_overlap),
            "valid_test_artist_overlap_count": len(valid_test_artist_overlap),
            "train_test_artist_overlap_examples": sorted(list(train_test_artist_overlap))[:30],
            "train_valid_artist_overlap_examples": sorted(list(train_valid_artist_overlap))[:30],
            "valid_test_artist_overlap_examples": sorted(list(valid_test_artist_overlap))[:30],
        },
        "counts_by_genre": {},
    }

    for label, genre_name in enumerate(class_names):
        split["counts_by_genre"][genre_name] = {
            "train": int(df[df["track_id"].isin(train_ids) & (df["label"] == label)].shape[0]),
            "valid": int(df[df["track_id"].isin(valid_ids) & (df["label"] == label)].shape[0]),
            "test": int(df[df["track_id"].isin(test_ids) & (df["label"] == label)].shape[0]),
        }

    return split


def main():
    df, genres_df, selected_genre_ids, label_to_genre_id, class_names = build_master_df()

    print()
    print("=" * 80)
    print("MASTER DATAFRAME")
    print("=" * 80)
    print("rows:", len(df))

    print()
    print("Количество доступных треков по жанрам:")

    for label, genre_name in enumerate(class_names):
        count = int((df["label"] == label).sum())
        artists = int(df[df["label"] == label]["artist_key"].nunique())
        print(f"{genre_name}: {count} tracks, {artists} artists")

    split = build_artist_aware_split(df, class_names)

    save_json(split, SPLIT_JSON)

    print()
    print("=" * 80)
    print("ГОТОВО")
    print("=" * 80)
    print("Split сохранён:")
    print(SPLIT_JSON)


if __name__ == "__main__":
    main()