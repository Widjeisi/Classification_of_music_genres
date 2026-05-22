import ast
import pandas as pd
import numpy as np


# ----------------------------
# 0. Настройки вывода
# ----------------------------

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)


# ----------------------------
# 1. Чтение файлов
# ----------------------------

tracks = pd.read_csv(
    "tracks.csv",
    index_col=0,
    header=[0, 1]
)

features = pd.read_csv(
    "features.csv",
    index_col=0,
    header=[0, 1, 2]
)

genres = pd.read_csv(
    "genres.csv",
    index_col=0
)

tracks.index = tracks.index.astype(int)
features.index = features.index.astype(int)
genres.index = genres.index.astype(int)


# ----------------------------
# 2. Настройки
# ----------------------------

TARGET_GENRES = [
    "Ambient",
    "Classical",
    "Country",
]

N_TRACKS_PER_GENRE = 1000


# ----------------------------
# 3. Функции
# ----------------------------

def normalize_name(name):
    return (
        str(name)
        .lower()
        .strip()
        .replace("_", "-")
        .replace(" ", "-")
    )


def parse_genres_all(value):
    if pd.isna(value):
        return []

    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except Exception:
            return []

    if not isinstance(value, list):
        return []

    result = []

    def flatten(items):
        for item in items:
            if isinstance(item, list):
                flatten(item)
            else:
                result.append(item)

    flatten(value)

    clean_result = []

    for genre_id in result:
        try:
            clean_result.append(int(genre_id))
        except Exception:
            pass

    return sorted(set(clean_result))


# ----------------------------
# 4. Находим genre_id нужных жанров
# ----------------------------

if "title" not in genres.columns:
    raise ValueError(
        "В genres.csv не найдена колонка 'title'. "
        f"Доступные колонки: {list(genres.columns)}"
    )

genre_name_to_id = {}

for genre_id, row in genres.iterrows():
    genre_title = row["title"]
    normalized_title = normalize_name(genre_title)

    for target_genre in TARGET_GENRES:
        if normalized_title == normalize_name(target_genre):
            genre_name_to_id[target_genre] = int(genre_id)

missing_genres = [
    genre_name
    for genre_name in TARGET_GENRES
    if genre_name not in genre_name_to_id
]

if missing_genres:
    raise ValueError(
        "Не найдены жанры в genres.csv: "
        + ", ".join(missing_genres)
    )

print()
print("Найденные genre_id:")

for genre_name in TARGET_GENRES:
    print(f"{genre_name}: {genre_name_to_id[genre_name]}")


# ----------------------------
# 5. Оставляем только треки, у которых есть features
# ----------------------------

if ("track", "genres_all") not in tracks.columns:
    raise ValueError("В tracks.csv не найдена колонка ('track', 'genres_all').")

common_track_ids = tracks.index.intersection(features.index)

tracks = tracks.loc[common_track_ids]
features = features.loc[common_track_ids]

track_genres_all = tracks[("track", "genres_all")].apply(parse_genres_all)


# ----------------------------
# 6. Берем по 3 трека каждого жанра
# ----------------------------

selected_tracks = {}

for genre_name in TARGET_GENRES:
    genre_id = genre_name_to_id[genre_name]

    matching_track_ids = []

    for track_id, genre_ids in track_genres_all.items():
        if genre_id in genre_ids:
            matching_track_ids.append(int(track_id))

    matching_track_ids = sorted(set(matching_track_ids))
    selected_track_ids = matching_track_ids[:N_TRACKS_PER_GENRE]

    selected_tracks[genre_name] = selected_track_ids

    print()
    print(f"{genre_name}: найдено треков = {len(matching_track_ids)}")
    print(f"{genre_name}: выбраны track_id = {selected_track_ids}")


# ----------------------------
# 7. Собираем таблицу выбранных треков
# ----------------------------

rows = []

for genre_name, track_ids in selected_tracks.items():
    for track_id in track_ids:
        row = features.loc[track_id].copy()
        row["genre_name"] = genre_name
        row["track_id"] = track_id
        rows.append(row)

selected_df = pd.DataFrame(rows)

selected_df = selected_df.set_index(["genre_name", "track_id"])

# Оставляем только числовые признаки
numeric_df = selected_df.select_dtypes(include=[np.number])


# ----------------------------
# 8. Средние значения признаков по жанрам
# ----------------------------

genre_means = numeric_df.groupby(level="genre_name").mean()

print()
print("=" * 100)
print("СРЕДНИЕ ЗНАЧЕНИЯ ПРИЗНАКОВ ПО ЖАНРАМ")
print("=" * 100)
print(genre_means.to_string())


# ----------------------------
# 9. Считаем, какие признаки отличаются сильнее всего
# ----------------------------

# range = max среднего по жанрам - min среднего по жанрам
feature_difference = genre_means.max(axis=0) - genre_means.min(axis=0)

feature_difference_df = feature_difference.reset_index()
feature_difference_df.columns = [
    "feature",
    "statistics",
    "number",
    "difference_between_genres",
]

feature_difference_df = feature_difference_df.sort_values(
    "difference_between_genres",
    ascending=False
)

print()
print("=" * 100)
print("ТОП-50 ПРИЗНАКОВ, КОТОРЫЕ СИЛЬНЕЕ ВСЕГО ОТЛИЧАЮТСЯ МЕЖДУ ЖАНРАМИ")
print("=" * 100)
print(feature_difference_df.head(50).to_string(index=False))


# ----------------------------
# 10. Сводка по группам признаков
# ----------------------------

feature_group_difference = (
    feature_difference_df
    .groupby("feature")["difference_between_genres"]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

print()
print("=" * 100)
print("КАКИЕ ГРУППЫ ПРИЗНАКОВ В СРЕДНЕМ ОТЛИЧАЮТСЯ СИЛЬНЕЕ ВСЕГО")
print("=" * 100)
print(feature_group_difference.to_string(index=False))


# ----------------------------
# 11. Более честная нормированная разница
# ----------------------------
# Иногда признаки имеют разные масштабы.
# Поэтому дополнительно считаем z-score различимость.

normalized_df = numeric_df.copy()

normalized_df = (
    normalized_df - normalized_df.mean(axis=0)
) / numeric_df.std(axis=0).replace(0, 1)

normalized_genre_means = normalized_df.groupby(level="genre_name").mean()

normalized_feature_difference = (
    normalized_genre_means.max(axis=0)
    - normalized_genre_means.min(axis=0)
)

normalized_feature_difference_df = normalized_feature_difference.reset_index()
normalized_feature_difference_df.columns = [
    "feature",
    "statistics",
    "number",
    "normalized_difference_between_genres",
]

normalized_feature_difference_df = normalized_feature_difference_df.sort_values(
    "normalized_difference_between_genres",
    ascending=False
)

print()
print("=" * 100)
print("ТОП-50 НОРМИРОВАННЫХ ПРИЗНАКОВ, КОТОРЫЕ ЛУЧШЕ ВСЕГО РАЗДЕЛЯЮТ ЖАНРЫ")
print("=" * 100)
print(normalized_feature_difference_df.head(50).to_string(index=False))


normalized_group_difference = (
    normalized_feature_difference_df
    .groupby("feature")["normalized_difference_between_genres"]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

print()
print("=" * 100)
print("НОРМИРОВАННАЯ СВОДКА ПО ГРУППАМ ПРИЗНАКОВ")
print("=" * 100)
print(normalized_group_difference.to_string(index=False))