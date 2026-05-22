import ast
from collections import defaultdict, Counter

import numpy as np
import pandas as pd


# ----------------------------
# 1. Чтение данных
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

genres_df = pd.read_csv(
    "genres.csv",
    index_col=0
)

tracks.index = tracks.index.astype(int)
features.index = features.index.astype(int)
genres_df.index = genres_df.index.astype(int)

print()
print("Изначальное количество признаков:", features.shape[1])
print("Пример колонок features:")
print(features.columns[:10])


# ----------------------------
# 1.1. Выбираем нужные жанры
# ----------------------------

TARGET_GENRE_NAMES = [
    "Jazz",
    "Classical",
    "Rock",
    "Hip-Hop",
    "Metal",
    "Country",
    "Ambient",
    "Techno",
]


def normalize_genre_name(name):
    return (
        str(name)
        .lower()
        .strip()
        .replace("_", "-")
        .replace(" ", "-")
    )


target_genre_names_normalized = {
    normalize_genre_name(name)
    for name in TARGET_GENRE_NAMES
}

genre_title_column = "title"

if genre_title_column not in genres_df.columns:
    raise ValueError(
        f"В genres.csv не найдена колонка '{genre_title_column}'. "
        f"Доступные колонки: {list(genres_df.columns)}"
    )

selected_genre_ids = []

for genre_id, row in genres_df.iterrows():
    genre_title = row[genre_title_column]
    normalized_title = normalize_genre_name(genre_title)

    if normalized_title in target_genre_names_normalized:
        selected_genre_ids.append(int(genre_id))

selected_genre_ids = sorted(set(selected_genre_ids))
selected_genre_ids_set = set(selected_genre_ids)

print()
print("Выбранные жанры:")

for genre_id in selected_genre_ids:
    print(f"{genre_id}: {genres_df.loc[genre_id, genre_title_column]}")

found_names_normalized = {
    normalize_genre_name(genres_df.loc[genre_id, genre_title_column])
    for genre_id in selected_genre_ids
}

missing_names = target_genre_names_normalized - found_names_normalized

if missing_names:
    print()
    print("Внимание: эти жанры не найдены в genres.csv:")
    for name in sorted(missing_names):
        print(name)

if len(selected_genre_ids) == 0:
    raise ValueError(
        "Не найдено ни одного жанра из TARGET_GENRE_NAMES. "
        "Проверь названия жанров в genres.csv."
    )

print()
print("Количество выбранных genre_id:", len(selected_genre_ids))


# ----------------------------
# 1.2. Находим специальные genre_id
# ----------------------------

genre_name_to_id = {}

for genre_id in selected_genre_ids:
    genre_title = genres_df.loc[genre_id, genre_title_column]
    normalized_title = normalize_genre_name(genre_title)
    genre_name_to_id[normalized_title] = int(genre_id)

rock_genre_id = genre_name_to_id.get("rock")
metal_genre_id = genre_name_to_id.get("metal")
classical_genre_id = genre_name_to_id.get("classical")
hiphop_genre_id = genre_name_to_id.get("hip-hop")
techno_genre_id = genre_name_to_id.get("techno")

if rock_genre_id is None:
    raise ValueError("Не найден жанр Rock среди выбранных жанров.")

if metal_genre_id is None:
    raise ValueError("Не найден жанр Metal среди выбранных жанров.")

if classical_genre_id is None:
    raise ValueError("Не найден жанр Classical среди выбранных жанров.")

if hiphop_genre_id is None:
    raise ValueError("Не найден жанр Hip-Hop среди выбранных жанров.")

if techno_genre_id is None:
    raise ValueError("Не найден жанр Techno среди выбранных жанров.")


# ----------------------------
# 1.3. Настройка модели
# ----------------------------

# Используется только модель 2.
# Она обучается на всех выбранных жанрах.
secondary_genre_ids = selected_genre_ids.copy()
secondary_genre_ids_set = set(secondary_genre_ids)

print()
print("Логика классификации:")
print("Итоговый жанр выставляется только по выводу модели 2.")
print("Модель 1 и модель 3 больше не используются.")
print()
print("Методика разметки Rock / Metal:")
print("если у трека есть Metal, считаем трек Metal;")
print("если Metal нет, но есть Rock, считаем трек Rock.")


# ----------------------------
# 2. Парсинг жанров
# ----------------------------

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

    clean_result = []

    for genre_id in result:
        try:
            clean_result.append(int(genre_id))
        except Exception:
            pass

    return sorted(set(clean_result))


track_all_genres = tracks[("track", "genres_all")].apply(parse_genres)

common_track_ids = tracks.index.intersection(features.index)

tracks = tracks.loc[common_track_ids]
features = features.loc[common_track_ids]
track_all_genres = track_all_genres.loc[common_track_ids]

track_all_genres = track_all_genres[track_all_genres.apply(len) > 0]

tracks = tracks.loc[track_all_genres.index]
features = features.loc[track_all_genres.index]

print()
print("Первые распарсенные genres_all:")
print(track_all_genres.head())


# ----------------------------
# 3. Получаем один итоговый выбранный жанр для каждого трека
# ----------------------------

def get_selected_genres_only(genres):
    return [
        genre_id
        for genre_id in genres
        if genre_id in selected_genre_ids_set
    ]


def get_final_training_genre(selected_genres):
    """
    Методика:
    1. Если среди выбранных жанров есть Metal -> трек считается Metal.
    2. Иначе, если есть Rock -> трек считается Rock.
    3. Иначе оставляем трек только если у него ровно один выбранный жанр.
    4. Если у трека несколько выбранных жанров без Rock/Metal, исключаем его.
    """

    selected_genres = sorted(set(selected_genres))

    if len(selected_genres) == 0:
        return None

    if metal_genre_id in selected_genres:
        return metal_genre_id

    if rock_genre_id in selected_genres:
        return rock_genre_id

    if len(selected_genres) == 1:
        return int(selected_genres[0])

    return None


track_selected_genres = track_all_genres.apply(get_selected_genres_only)
track_final_genre = track_selected_genres.apply(get_final_training_genre)

before_filter_count = len(track_final_genre)

track_final_genre = track_final_genre.dropna().astype(int)

tracks = tracks.loc[track_final_genre.index]
features = features.loc[track_final_genre.index]
track_all_genres = track_all_genres.loc[track_final_genre.index]
track_selected_genres = track_selected_genres.loc[track_final_genre.index]

# Теперь у каждого трека ровно один итоговый жанр.
track_genres = track_final_genre.apply(lambda genre_id: [int(genre_id)])

print()
print("Фильтр и переопределение жанров:")
print(f"Было треков с хотя бы одним genres_all: {before_filter_count}")
print(f"Осталось треков после фильтра: {len(track_genres)}")

rock_metal_original_count = track_selected_genres.apply(
    lambda genres: metal_genre_id in genres and rock_genre_id in genres
).sum()

print()
print(f"Треков, где в исходных выбранных жанрах были и Rock, и Metal: {rock_metal_original_count}")
print("Все такие треки теперь считаются Metal.")


# ----------------------------
# 3.1. Собираем треки по итоговому выбранному жанру
# ----------------------------

genre_to_tracks = defaultdict(list)

for track_id, genres in track_genres.items():
    genre_id = int(genres[0])
    genre_to_tracks[genre_id].append(int(track_id))

print()
print("Количество треков по итоговым жанрам после правила Metal > Rock:")

for genre_id in selected_genre_ids:
    genre_name = genres_df.loc[genre_id, genre_title_column]
    count = len(set(genre_to_tracks.get(genre_id, [])))
    print(f"{genre_id} / {genre_name}: {count} треков")


# ----------------------------
# 4. Выбираем случайные треки на жанр для train
# ----------------------------

def sample_tracks_by_genre(genre_to_tracks, n_per_genre=900, seed=None):
    rng = np.random.default_rng(seed)
    sampled = {}

    print()
    print(f"Формируем train: максимум {n_per_genre} треков на жанр")

    for genre_id in selected_genre_ids:
        track_ids = genre_to_tracks.get(genre_id, [])
        track_ids = np.array(sorted(set(track_ids)), dtype=int)

        if len(track_ids) == 0:
            sampled[int(genre_id)] = set()
            continue

        sample_size = min(n_per_genre, len(track_ids))

        sampled_ids = rng.choice(
            track_ids,
            size=sample_size,
            replace=False
        )

        sampled[int(genre_id)] = set(sampled_ids.tolist())

        genre_name = genres_df.loc[genre_id, genre_title_column]

        print(
            f"{genre_id} / {genre_name}: "
            f"всего {len(track_ids)}, "
            f"train {sample_size}"
        )

    return sampled


genre_samples = sample_tracks_by_genre(
    genre_to_tracks,
    n_per_genre=900,
    seed=None
)


# ----------------------------
# 5. Признаки для модели 2
# ----------------------------

SECONDARY_FEATURE_SPECS = [
    ("mfcc", "median"),
    ("mfcc", "std"),
    ("spectral_centroid", "median"),
    ("spectral_bandwidth", "median"),
    ("spectral_rolloff", "median"),
    ("spectral_contrast", "median"),
    ("zcr", "median"),
    ("rmse", "median"),
]


def is_secondary_model_column(col):
    feature_name = col[0]
    statistic_name = col[1]

    return (feature_name, statistic_name) in SECONDARY_FEATURE_SPECS


secondary_feature_columns = [
    col for col in features.columns
    if is_secondary_model_column(col)
]

features_secondary = features[secondary_feature_columns]

print()
print("Признаки для модели 2:")
print("Модель 2: все выбранные жанры")
print("Признаки: mfcc.median + mfcc.std + spectral median + zcr.median + rmse.median")
print("Количество признаков:", features_secondary.shape[1])
print(features_secondary.columns)

if features_secondary.shape[1] == 0:
    raise ValueError(
        "Не найдены признаки для модели 2. "
        "Проверь наличие mfcc.median, mfcc.std, spectral_*.median, "
        "zcr.median и rmse.median."
    )

found_secondary_pairs = {
    (col[0], col[1])
    for col in secondary_feature_columns
}

missing_secondary_pairs = [
    pair
    for pair in SECONDARY_FEATURE_SPECS
    if pair not in found_secondary_pairs
]

if missing_secondary_pairs:
    print()
    print("Внимание: некоторые признаки для модели 2 не найдены:")
    for pair in missing_secondary_pairs:
        print(pair)


# ----------------------------
# 6. Формируем обучающую выборку для модели 2
# ----------------------------

X_train_secondary = []
y_train_secondary = []
secondary_train_track_ids = []

for genre_id in secondary_genre_ids:
    if genre_id not in genre_samples:
        continue

    for track_id in genre_samples[genre_id]:
        if track_id in features_secondary.index:
            vector = features_secondary.loc[track_id].values.astype(float)

            X_train_secondary.append(vector)
            y_train_secondary.append(int(genre_id))
            secondary_train_track_ids.append(int(track_id))

X_train_secondary = np.array(X_train_secondary)
y_train_secondary = np.array(y_train_secondary)
secondary_train_track_ids = np.array(secondary_train_track_ids)

print()
print("Размер X_train_secondary:", X_train_secondary.shape)
print("Размер y_train_secondary:", y_train_secondary.shape)

if len(X_train_secondary) == 0:
    raise ValueError(
        "X_train_secondary пустой. Проверь, что выбранные жанры есть в genres.csv "
        "и что у них есть признаки модели 2."
    )

print()
print("Состав train для модели 2:")

secondary_train_counter = Counter(y_train_secondary)

for genre_id in secondary_genre_ids:
    genre_name = genres_df.loc[genre_id, genre_title_column]
    print(f"{genre_name}: {secondary_train_counter[genre_id]}")


# ----------------------------
# 7. Создаем сбалансированную тестовую выборку
# ----------------------------

all_train_track_ids = set(secondary_train_track_ids)

test_candidates_by_genre = defaultdict(list)

for track_id, genres in track_genres.items():
    track_id = int(track_id)

    if track_id in all_train_track_ids:
        continue

    if track_id not in features_secondary.index:
        continue

    if len(genres) != 1:
        continue

    genre_id = int(genres[0])

    if genre_id in selected_genre_ids_set:
        test_candidates_by_genre[genre_id].append(track_id)

TEST_SIZE = 3000

rng = np.random.default_rng()

genres_for_test = [
    genre_id
    for genre_id in selected_genre_ids
    if len(test_candidates_by_genre[genre_id]) > 0
]

if len(genres_for_test) == 0:
    raise ValueError(
        "Нет кандидатов для тестовой выборки. "
        "Проверь, что после train остаются треки для test."
    )

base_per_genre = TEST_SIZE // len(genres_for_test)

min_available_per_genre = min(
    len(set(test_candidates_by_genre[genre_id]))
    for genre_id in genres_for_test
)

balanced_per_genre = min(base_per_genre, min_available_per_genre)

if balanced_per_genre == 0:
    raise ValueError(
        "Невозможно сделать сбалансированную тестовую выборку: "
        "хотя бы у одного жанра нет доступных треков."
    )

test_track_ids = []
used_test_track_ids = set()

print()
print("Формируем сбалансированную тестовую выборку:")
print(f"Жанров в тесте: {len(genres_for_test)}")
print(f"План TEST_SIZE: {TEST_SIZE}")
print(f"Реально будет взято по {balanced_per_genre} треков на жанр")
print(f"Итоговый размер теста: максимум {balanced_per_genre * len(genres_for_test)}")

for genre_id in genres_for_test:
    candidates = list(set(test_candidates_by_genre[genre_id]))

    candidates = [
        track_id
        for track_id in candidates
        if track_id not in used_test_track_ids
    ]

    sample_size = min(balanced_per_genre, len(candidates))

    if sample_size == 0:
        print(
            f"Внимание: для жанра {genre_id} / "
            f"{genres_df.loc[genre_id, genre_title_column]} "
            f"не осталось уникальных кандидатов после удаления дублей."
        )
        continue

    sampled_ids = rng.choice(
        np.array(candidates, dtype=int),
        size=sample_size,
        replace=False
    )

    for track_id in sampled_ids:
        track_id = int(track_id)
        test_track_ids.append(track_id)
        used_test_track_ids.add(track_id)

    genre_name = genres_df.loc[genre_id, genre_title_column]

    print(
        f"{genre_id} / {genre_name}: "
        f"выбрано {sample_size} из {len(candidates)} доступных"
    )

rng.shuffle(test_track_ids)

test_track_ids = [int(track_id) for track_id in test_track_ids]

print()
print("Размер тестовой выборки:", len(test_track_ids))


# ----------------------------
# 8. Нормализация модели 2
# ----------------------------

mean_secondary = X_train_secondary.mean(axis=0)
std_secondary = X_train_secondary.std(axis=0)
std_secondary[std_secondary == 0] = 1

X_train_secondary_scaled = (X_train_secondary - mean_secondary) / std_secondary

print()
print("Нормализация модели 2 выполнена.")


# ----------------------------
# 9. kNN-классификатор
# ----------------------------

class GenreKNN:
    def __init__(self, X_train, y_train, k=15):
        self.X_train = np.asarray(X_train, dtype=float)
        self.y_train = np.asarray(y_train)
        self.k = k

    def predict_one(self, x):
        x = np.asarray(x, dtype=float)

        distances = np.linalg.norm(self.X_train - x, axis=1)

        nearest_indices = np.argsort(distances)[:self.k]

        nearest_genres = self.y_train[nearest_indices]

        genre_counts = Counter(nearest_genres)

        predicted_genre = genre_counts.most_common(1)[0][0]

        return {
            "predicted_genre": int(predicted_genre),
            "votes": genre_counts,
            "nearest_indices": nearest_indices,
            "nearest_distances": distances[nearest_indices],
        }


def predict_secondary_genre_for_track(track_id, model):
    raw_vector = features_secondary.loc[track_id].values.astype(float)

    scaled_vector = (raw_vector - mean_secondary) / std_secondary

    return model.predict_one(scaled_vector)


# ----------------------------
# 10. Логика классификации
# ----------------------------

def apply_classification_rule(track_id, secondary_model):
    """
    Итоговый жанр полностью равен выводу модели 2.
    """

    secondary_result = predict_secondary_genre_for_track(
        track_id=track_id,
        model=secondary_model
    )

    secondary_predicted_genre = secondary_result["predicted_genre"]

    return {
        "final_predicted_genre": int(secondary_predicted_genre),
        "decision_type": "secondary_model_only",
        "secondary_predicted_genre": int(secondary_predicted_genre),
    }


# ----------------------------
# 11. Оценка точности и распределений
# ----------------------------

def calculate_per_genre_accuracy(predictions):
    per_genre_total_counter = Counter()
    per_genre_correct_counter = Counter()

    for prediction in predictions:
        true_genre = prediction["true_genre"]
        is_correct = prediction["is_correct"]

        per_genre_total_counter[true_genre] += 1

        if is_correct:
            per_genre_correct_counter[true_genre] += 1

    per_genre_accuracy_rows = []

    for genre_id in selected_genre_ids:
        total_count = per_genre_total_counter[genre_id]
        correct_count = per_genre_correct_counter[genre_id]

        accuracy_percent = (
            correct_count / total_count * 100
            if total_count > 0
            else 0
        )

        per_genre_accuracy_rows.append({
            "genre_id": genre_id,
            "genre_name": genres_df.loc[genre_id, genre_title_column],
            "correct_count": correct_count,
            "total_in_test": total_count,
            "accuracy_percent": accuracy_percent,
            "fraction": f"{correct_count}/{total_count}",
        })

    per_genre_accuracy_df = pd.DataFrame(per_genre_accuracy_rows)

    per_genre_accuracy_df = per_genre_accuracy_df.sort_values(
        "accuracy_percent",
        ascending=False
    )

    return per_genre_accuracy_df


def calculate_prediction_distribution_tables(predictions):
    rows = []

    for prediction in predictions:
        true_genre = prediction["true_genre"]
        predicted_genre = prediction["final_predicted_genre"]

        rows.append({
            "true_genre_id": true_genre,
            "true_genre_name": genres_df.loc[true_genre, genre_title_column],
            "predicted_genre_id": predicted_genre,
            "predicted_genre_name": genres_df.loc[predicted_genre, genre_title_column],
        })

    predictions_df = pd.DataFrame(rows)

    distribution_tables = {}

    for true_genre_id in selected_genre_ids:
        genre_name = genres_df.loc[true_genre_id, genre_title_column]

        genre_predictions = predictions_df[
            predictions_df["true_genre_id"] == true_genre_id
        ]

        total_count = len(genre_predictions)

        table_rows = []

        for predicted_genre_id in selected_genre_ids:
            predicted_genre_name = genres_df.loc[predicted_genre_id, genre_title_column]

            predicted_count = (
                genre_predictions["predicted_genre_id"] == predicted_genre_id
            ).sum()

            predicted_percent = (
                predicted_count / total_count * 100
                if total_count > 0
                else 0
            )

            table_rows.append({
                "присвоенный жанр": predicted_genre_name,
                "количество треков": predicted_count,
                "процент": predicted_percent,
                "доля": f"{predicted_count}/{total_count}",
            })

        table_df = pd.DataFrame(table_rows)

        table_df = table_df.sort_values(
            "процент",
            ascending=False
        )

        distribution_tables[genre_name] = table_df

    return distribution_tables


def evaluate_accuracy(
    test_track_ids,
    secondary_model,
    k,
    save_predictions=False
):
    correct = 0
    total = 0
    predictions = []

    decision_counter = Counter()

    for track_id in test_track_ids:
        true_genres = track_genres.loc[track_id]

        if len(true_genres) != 1:
            continue

        true_genre = int(true_genres[0])

        classification_result = apply_classification_rule(
            track_id=track_id,
            secondary_model=secondary_model
        )

        final_predicted_genre = classification_result["final_predicted_genre"]
        secondary_predicted_genre = classification_result["secondary_predicted_genre"]
        decision_type = classification_result["decision_type"]

        decision_counter[decision_type] += 1

        is_correct = final_predicted_genre == true_genre

        if is_correct:
            correct += 1

        total += 1

        predictions.append({
            "track_id": track_id,

            "true_genre": true_genre,
            "true_genre_name": genres_df.loc[true_genre, genre_title_column],

            "secondary_predicted_genre": secondary_predicted_genre,
            "secondary_predicted_genre_name": genres_df.loc[
                secondary_predicted_genre,
                genre_title_column
            ],

            "final_predicted_genre": final_predicted_genre,
            "final_predicted_genre_name": genres_df.loc[
                final_predicted_genre,
                genre_title_column
            ],

            "decision_type": decision_type,
            "is_correct": is_correct,
        })

    accuracy = correct / total if total > 0 else 0

    per_genre_accuracy_df = calculate_per_genre_accuracy(predictions)

    if save_predictions:
        predictions_to_return = predictions
    else:
        predictions_to_return = []

    return {
        "k": k,
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100,
        "correct": correct,
        "total": total,
        "predictions": predictions_to_return,
        "per_genre_accuracy_df": per_genre_accuracy_df,
        "decision_counter": decision_counter,
    }


# ----------------------------
# 12. Оценка модели
# ----------------------------

K_VALUES = range(15, 26)

results = []
per_k_genre_results = {}

print()
print("Оценка модели:")
print("Итоговый жанр выставляется только по модели 2.")

for k in K_VALUES:
    secondary_model = GenreKNN(
        X_train=X_train_secondary_scaled,
        y_train=y_train_secondary,
        k=k
    )

    result = evaluate_accuracy(
        test_track_ids=test_track_ids,
        secondary_model=secondary_model,
        k=k,
        save_predictions=False
    )

    result_without_predictions = {
        "k": result["k"],
        "accuracy": result["accuracy"],
        "accuracy_percent": result["accuracy_percent"],
        "correct": result["correct"],
        "total": result["total"],
    }

    results.append(result_without_predictions)
    per_k_genre_results[k] = result["per_genre_accuracy_df"]

    print()
    print("-" * 60)
    print(
        f"k={k:2d} | "
        f"accuracy={result['accuracy']:.4f} | "
        f"percent={result['accuracy_percent']:.2f}% | "
        f"correct={result['correct']} / {result['total']}"
    )

    print()
    print("Как принимались решения:")

    for decision_type, count in result["decision_counter"].items():
        print(f"{decision_type}: {count}")

    print()
    print(f"Процент угаданных песен для каждого жанра при k={k}:")

    for _, row in result["per_genre_accuracy_df"].iterrows():
        print(
            f"{row['genre_name']}: "
            f"{row['correct_count']}/{row['total_in_test']} "
            f"= {row['accuracy_percent']:.2f}%"
        )

results_df = pd.DataFrame(results)


# ----------------------------
# 13. Лучшее k
# ----------------------------

best_result = results_df.sort_values("accuracy", ascending=False).iloc[0]
best_k = int(best_result["k"])

print()
print("=" * 60)
print("Лучший результат:")
print(best_result)


# ----------------------------
# 14. Получаем предсказания для лучшего k
# ----------------------------

best_secondary_model = GenreKNN(
    X_train=X_train_secondary_scaled,
    y_train=y_train_secondary,
    k=best_k
)

best_eval = evaluate_accuracy(
    test_track_ids=test_track_ids,
    secondary_model=best_secondary_model,
    k=best_k,
    save_predictions=True
)

predictions = best_eval["predictions"]


# ----------------------------
# 15. Процент угаданных песен для каждого жанра для лучшего k
# ----------------------------

per_genre_accuracy_df = best_eval["per_genre_accuracy_df"]

print()
print("Процент угаданных песен для каждого жанра для лучшего k:")

for _, row in per_genre_accuracy_df.iterrows():
    print(
        f"{row['genre_name']}: "
        f"{row['correct_count']}/{row['total_in_test']} "
        f"= {row['accuracy_percent']:.2f}%"
    )


# ----------------------------
# 16. Таблицы распределения предсказаний по каждому жанру
# ----------------------------

distribution_tables = calculate_prediction_distribution_tables(predictions)

print()
print("=" * 60)
print("ТАБЛИЦЫ: КАКИЕ ЖАНРЫ МОДЕЛЬ 2 ПРИСВАИВАЛА ТРЕКАМ КАЖДОГО ЖАНРА")
print("=" * 60)

for true_genre_name, table_df in distribution_tables.items():
    print()
    print("-" * 60)
    print(f"Настоящий жанр: {true_genre_name}")
    print("-" * 60)

    table_to_print = table_df.copy()
    table_to_print["процент"] = table_to_print["процент"].map(lambda x: f"{x:.2f}%")

    print(table_to_print.to_string(index=False))


# ----------------------------
# 17. Сводная таблица по всем k
# ----------------------------

print()
print("=" * 60)
print("СВОДНАЯ ТАБЛИЦА ПО ВСЕМ k")
print("=" * 60)
print(results_df)