import ast
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm


# ============================================================
# 0. НАСТРОЙКИ
# ============================================================

PROJECT_DIR = Path("/Users/widjeisi/project_music")

TRACKS_CSV = PROJECT_DIR / "tracks.csv"
GENRES_CSV = PROJECT_DIR / "genres.csv"

AUDIO_DIR = PROJECT_DIR / "/Users/widjeisi/Downloads/fma_large"

OUTPUT_MEL_DIR = PROJECT_DIR / "mel_spectrograms_6_genres_full"

TARGET_GENRE_NAMES = [
    "Jazz",
    "Classical",
    "Hip-Hop",
    "Metal",
    "Country",
    "Dubstep",
]

# Параметры mel
SAMPLE_RATE = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

# Приводим все mel к одной длине по времени
TARGET_FRAMES = 384

# Если True — не пересоздает уже готовые .npy
SKIP_EXISTING = True

# Если True — трек с несколькими выбранными жанрами пропускается,
# кроме случая Metal: если среди жанров есть Metal, считаем его Metal.
STRICT_SINGLE_GENRE = True

# Если True — если у трека есть Metal среди жанров, он идет в Metal
METAL_PRIORITY = True


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


def parse_genres(x):
    if pd.isna(x):
        return []

    if isinstance(x, str):
        try:
            x = ast.literal_eval(x)
        except Exception:
            return []

    if not isinstance(x, list):
        return []

    result = []

    def flatten(items):
        for item in items:
            if isinstance(item, list):
                flatten(item)
            else:
                result.append(item)

    flatten(x)

    clean = []

    for item in result:
        try:
            clean.append(int(item))
        except Exception:
            pass

    return sorted(set(clean))


def fma_track_path(audio_dir, track_id):
    """
    В FMA аудио обычно лежит так:
    fma_large/000/000002.mp3
    fma_large/001/001486.mp3
    """
    track_id = int(track_id)
    tid = f"{track_id:06d}"
    folder = tid[:3]
    return audio_dir / folder / f"{tid}.mp3"


def fix_length_frames(mel_db, target_frames):
    """
    mel_db shape: [n_mels, frames]
    """
    if mel_db.shape[1] < target_frames:
        pad = target_frames - mel_db.shape[1]
        mel_db = np.pad(
            mel_db,
            pad_width=((0, 0), (0, pad)),
            mode="constant",
            constant_values=mel_db.min(),
        )
    elif mel_db.shape[1] > target_frames:
        start = (mel_db.shape[1] - target_frames) // 2
        mel_db = mel_db[:, start:start + target_frames]

    return mel_db


def make_mel(audio_path):
    y, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True,
    )

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max,
    )

    mel_db = fix_length_frames(
        mel_db,
        target_frames=TARGET_FRAMES,
    )

    return mel_db.astype(np.float32)


# ============================================================
# 2. ЧТЕНИЕ METADATA
# ============================================================

def main():
    print()
    print("Читаю genres.csv и tracks.csv...")

    genres_df = pd.read_csv(
        GENRES_CSV,
        index_col=0,
    )
    genres_df.index = genres_df.index.astype(int)

    tracks = pd.read_csv(
        TRACKS_CSV,
        index_col=0,
        header=[0, 1],
    )
    tracks.index = tracks.index.astype(int)

    if ("track", "genres_all") not in tracks.columns:
        raise ValueError("В tracks.csv не найдена колонка ('track', 'genres_all').")

    target_normalized = {
        normalize_genre_name(name)
        for name in TARGET_GENRE_NAMES
    }

    selected_genre_ids = []

    for genre_id, row in genres_df.iterrows():
        title = row["title"]

        if normalize_genre_name(title) in target_normalized:
            selected_genre_ids.append(int(genre_id))

    selected_genre_ids = sorted(set(selected_genre_ids))
    selected_genre_ids_set = set(selected_genre_ids)

    if len(selected_genre_ids) != len(TARGET_GENRE_NAMES):
        print()
        print("Найдены жанры:")
        for genre_id in selected_genre_ids:
            print(genre_id, genres_df.loc[genre_id, "title"])

        raise ValueError("Найдены не все TARGET_GENRE_NAMES.")

    genre_id_to_name = {
        int(genre_id): str(genres_df.loc[genre_id, "title"])
        for genre_id in selected_genre_ids
    }

    genre_name_to_id = {
        normalize_genre_name(name): genre_id
        for genre_id, name in genre_id_to_name.items()
    }

    metal_genre_id = genre_name_to_id.get("metal")

    print()
    print("Выбранные жанры:")
    for genre_id in selected_genre_ids:
        print(f"{genre_id}: {genre_id_to_name[genre_id]}")

    # ------------------------------------------------------------
    # Разбор жанров треков
    # ------------------------------------------------------------

    track_all_genres = tracks[("track", "genres_all")].apply(parse_genres)

    rows = []

    skipped_no_target = 0
    skipped_multi = 0
    skipped_no_audio = 0

    for track_id, genre_list in track_all_genres.items():
        track_id = int(track_id)

        selected = [
            int(g)
            for g in genre_list
            if int(g) in selected_genre_ids_set
        ]

        selected = sorted(set(selected))

        if len(selected) == 0:
            skipped_no_target += 1
            continue

        final_genre_id = None

        if METAL_PRIORITY and metal_genre_id in selected:
            final_genre_id = int(metal_genre_id)
        elif STRICT_SINGLE_GENRE:
            if len(selected) == 1:
                final_genre_id = int(selected[0])
            else:
                skipped_multi += 1
                continue
        else:
            final_genre_id = int(selected[0])

        audio_path = fma_track_path(AUDIO_DIR, track_id)

        if not audio_path.exists():
            skipped_no_audio += 1
            continue

        rows.append({
            "track_id": track_id,
            "genre_id": final_genre_id,
            "genre_name": genre_id_to_name[final_genre_id],
            "audio_path": str(audio_path),
        })

    selected_df = pd.DataFrame(rows)

    if len(selected_df) == 0:
        raise ValueError(
            "Не найдено ни одного аудиофайла для выбранных жанров. "
            "Проверь AUDIO_DIR."
        )

    print()
    print("=" * 80)
    print("ТРЕКИ ДЛЯ СОЗДАНИЯ MEL")
    print("=" * 80)
    print("Всего найдено:", len(selected_df))
    print("Пропущено без нужного жанра:", skipped_no_target)
    print("Пропущено из-за нескольких выбранных жанров:", skipped_multi)
    print("Пропущено без audio-файла:", skipped_no_audio)

    print()
    print("Количество по жанрам:")

    counts = selected_df["genre_name"].value_counts()

    for genre_name in TARGET_GENRE_NAMES:
        print(f"{genre_name}: {int(counts.get(genre_name, 0))}")

    # ------------------------------------------------------------
    # Создание папок
    # ------------------------------------------------------------

    OUTPUT_MEL_DIR.mkdir(parents=True, exist_ok=True)

    for genre_name in TARGET_GENRE_NAMES:
        genre_dir = OUTPUT_MEL_DIR / genre_name
        genre_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Генерация mel
    # ------------------------------------------------------------

    print()
    print("=" * 80)
    print("СОЗДАЮ MEL-СПЕКТРОГРАММЫ")
    print("=" * 80)

    created = 0
    skipped_existing = 0
    failed = 0

    errors = []

    for _, row in tqdm(selected_df.iterrows(), total=len(selected_df)):
        track_id = int(row["track_id"])
        genre_name = row["genre_name"]
        audio_path = Path(row["audio_path"])

        output_path = OUTPUT_MEL_DIR / genre_name / f"{track_id}.npy"

        if SKIP_EXISTING and output_path.exists():
            skipped_existing += 1
            continue

        try:
            mel = make_mel(audio_path)

            if mel.shape != (N_MELS, TARGET_FRAMES):
                raise ValueError(
                    f"Неверная форма mel: {mel.shape}, ожидалось {(N_MELS, TARGET_FRAMES)}"
                )

            np.save(output_path, mel)
            created += 1

        except Exception as e:
            failed += 1
            errors.append({
                "track_id": track_id,
                "genre_name": genre_name,
                "audio_path": str(audio_path),
                "error": str(e),
            })

    # ------------------------------------------------------------
    # Сохранение индекса и ошибок
    # ------------------------------------------------------------

    index_csv = OUTPUT_MEL_DIR / "mel_index.csv"
    selected_df.to_csv(index_csv, index=False)

    if errors:
        errors_df = pd.DataFrame(errors)
        errors_csv = OUTPUT_MEL_DIR / "mel_errors.csv"
        errors_df.to_csv(errors_csv, index=False)
    else:
        errors_csv = None

    print()
    print("=" * 80)
    print("ГОТОВО")
    print("=" * 80)
    print("Папка mel:")
    print(OUTPUT_MEL_DIR)

    print()
    print("Создано новых .npy:", created)
    print("Пропущено уже существующих:", skipped_existing)
    print("Ошибок:", failed)

    print()
    print("Индекс сохранен:")
    print(index_csv)

    if errors_csv is not None:
        print()
        print("Ошибки сохранены:")
        print(errors_csv)

    print()
    print("Итоговое количество .npy по жанрам:")

    for genre_name in TARGET_GENRE_NAMES:
        genre_dir = OUTPUT_MEL_DIR / genre_name
        count = len(list(genre_dir.glob("*.npy")))
        print(f"{genre_name}: {count}")


if __name__ == "__main__":
    main()
