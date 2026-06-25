#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from common_final6 import *


# ============================================================
# 3-MODEL ENSEMBLE
# CNN medium + CNN light + MLP heavy
# ============================================================

CNN_MEDIUM_MODEL = "mel_rescnn_medium"
CNN_LIGHT_MODEL = "mel_rescnn_light_fast_seed7"
MLP_MODEL = "feature_mlp_heavy"

MODEL_NAMES = [
    CNN_MEDIUM_MODEL,
    CNN_LIGHT_MODEL,
    MLP_MODEL,
]

ENSEMBLE_NAME = "final_3model_cnn_medium_light_mlp_tuned_rescue"
ENSEMBLE_OUTDIR = OUTDIR / ENSEMBLE_NAME


# ============================================================
# SEARCH SETTINGS
# ============================================================

TOP_PRINT = 80

# Весовые сетки. Потом веса нормализуются.
CNN_MEDIUM_WEIGHTS = [0.18, 0.22, 0.26, 0.30, 0.34]
CNN_LIGHT_WEIGHTS = [0.08, 0.12, 0.16, 0.20, 0.24, 0.28]
MLP_WEIGHTS = [0.42, 0.48, 0.54, 0.60, 0.66]

CNN_MEDIUM_TEMPS = [1.6, 2.0, 2.4]
CNN_LIGHT_TEMPS = [1.2, 1.6, 2.0]
MLP_TEMPS = [1.0, 1.3, 1.6]

DUBSTEP_BOOSTS = [1.0, 1.1, 1.2, 1.35, 1.5, 1.7]
DUBSTEP_RESCUE_THRESHOLDS = [None, 0.26, 0.30, 0.34]
DUBSTEP_RESCUE_BOOSTS = [1.0, 1.5, 2.0]

# Чтобы не выбрать конфиг, который сильно ухудшает баланс.
MIN_VALID_DUBSTEP_RECALL = 0.62
MIN_VALID_CLASSICAL_RECALL = 0.58
MIN_VALID_HIPHOP_RECALL = 0.58
MIN_VALID_OBJECTIVE = 0.66


# ============================================================
# FAST TARGETED RESCUE SETTINGS
# ============================================================

USE_TARGETED_RESCUE = True

HIP_CNN_MEDIUM_THRESHOLDS = [0.44, 0.50, 0.56]
HIP_FINAL_THRESHOLDS = [0.12, 0.16, 0.20]
HIP_MARGIN_THRESHOLDS = [0.10, 0.14, 0.18]

CLASSICAL_CNN_MAX_THRESHOLDS = [0.34, 0.38, 0.42]
CLASSICAL_MLP_THRESHOLDS = [0.16, 0.20, 0.24]
CLASSICAL_FINAL_THRESHOLDS = [0.12, 0.16, 0.20]
CLASSICAL_MARGIN_THRESHOLDS = [0.10, 0.14, 0.18]

PROTECT_DUBSTEP_IF_FINAL_PROB_ABOVE = [0.50, 0.55, 0.60]

MAX_HIPHOP_RESCUES_OPTIONS = [5, 10, 15]
MAX_CLASSICAL_RESCUES_OPTIONS = [5, 10, 15]
MAX_TOTAL_RESCUES_OPTIONS = [15, 25, 35]


# ============================================================
# UTILS
# ============================================================

def to_builtin(obj):
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def softmax_temperature(logits, temperature):
    logits = np.asarray(logits, dtype=np.float64)
    z = logits / float(temperature)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def normalize_probs(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 1e-12, 1.0)
    return probs / probs.sum(axis=1, keepdims=True)


def probs_to_pred(probs):
    return np.asarray(probs).argmax(axis=1)


def class_recall(y_true, y_pred, label):
    mask = y_true == label
    if int(mask.sum()) == 0:
        return 0.0
    return float((y_pred[mask] == label).sum() / mask.sum())


def metrics_for_pred(y_true, pred, class_names):
    labels = list(range(len(class_names)))

    out = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_recall": float(macro_recall(y_true, pred, labels)),
        "min_recall": float(min_recall(y_true, pred, labels)),
        "objective": float(balanced_objective(y_true, pred, labels)),
        "macro_f1": float(
            f1_score(
                y_true,
                pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
    }

    for label, name in enumerate(class_names):
        out[f"recall_{name}"] = class_recall(y_true, pred, label)

    return out


def print_metrics_from_pred(title, y_true, pred, class_names):
    labels = list(range(len(class_names)))

    acc = accuracy_score(y_true, pred)
    mr = macro_recall(y_true, pred, labels)
    mn = min_recall(y_true, pred, labels)
    obj = balanced_objective(y_true, pred, labels)
    mf1 = f1_score(
        y_true,
        pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)
    print(f"accuracy={acc:.4f} | {acc * 100:.2f}%")
    print(f"macro_recall={mr:.4f} | min_recall={mn:.4f} | objective={obj:.4f} | macro_f1={mf1:.4f}")

    rows = []

    for label, name in enumerate(class_names):
        mask = y_true == label
        total = int(mask.sum())
        correct = int((pred[mask] == label).sum())
        recall = correct / max(total, 1)
        rows.append((name, correct, total, recall))

    rows = sorted(rows, key=lambda x: x[3], reverse=True)

    print()
    print("По жанрам:")
    for name, correct, total, recall in rows:
        print(f"{name}: {correct}/{total} = {recall * 100:.2f}%")

    result = {
        "accuracy": float(acc),
        "accuracy_percent": float(acc * 100),
        "macro_recall": float(mr),
        "min_recall": float(mn),
        "objective": float(obj),
        "macro_f1": float(mf1),
    }

    for name, correct, total, recall in rows:
        result[f"recall_{name}"] = float(recall)

    return result


def margin_to_class(probs, target_label):
    top1 = np.max(probs, axis=1)
    target = probs[:, target_label]
    return top1 - target


# ============================================================
# LOAD
# ============================================================

def load_model_result(model_name, class_names):
    model_dir = OUTDIR / model_name

    required = [
        model_dir / "valid_logits.npy",
        model_dir / "valid_y.npy",
        model_dir / "valid_track_ids.npy",
        model_dir / "test_logits.npy",
        model_dir / "test_y.npy",
        model_dir / "test_track_ids.npy",
    ]

    missing = [path for path in required if not path.exists()]

    if missing:
        raise FileNotFoundError(
            f"\nНет файлов для модели: {model_name}\n\n"
            + "\n".join(str(x) for x in missing)
        )

    valid_logits = np.load(model_dir / "valid_logits.npy")
    valid_y = np.load(model_dir / "valid_y.npy")
    valid_ids = np.load(model_dir / "valid_track_ids.npy")

    test_logits = np.load(model_dir / "test_logits.npy")
    test_y = np.load(model_dir / "test_y.npy")
    test_ids = np.load(model_dir / "test_track_ids.npy")

    valid_metrics = print_metrics(
        f"{model_name} VALIDATION",
        valid_logits,
        valid_y,
        class_names,
    )

    test_metrics = print_metrics(
        f"{model_name} TEST",
        test_logits,
        test_y,
        class_names,
    )

    return {
        "model_name": model_name,

        "valid_logits": valid_logits,
        "valid_y": valid_y,
        "valid_ids": valid_ids,

        "test_logits": test_logits,
        "test_y": test_y,
        "test_ids": test_ids,

        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
    }


def check_alignment(results):
    valid_y = None
    test_y = None
    valid_ids = None
    test_ids = None

    for result in results:
        name = result["model_name"]

        if valid_y is None:
            valid_y = result["valid_y"]
            test_y = result["test_y"]
            valid_ids = result["valid_ids"]
            test_ids = result["test_ids"]
        else:
            if not np.array_equal(valid_y, result["valid_y"]):
                raise ValueError(f"valid_y mismatch: {name}")
            if not np.array_equal(test_y, result["test_y"]):
                raise ValueError(f"test_y mismatch: {name}")
            if not np.array_equal(valid_ids, result["valid_ids"]):
                raise ValueError(f"valid_ids mismatch: {name}")
            if not np.array_equal(test_ids, result["test_ids"]):
                raise ValueError(f"test_ids mismatch: {name}")

    return valid_y, test_y, valid_ids, test_ids


# ============================================================
# ENSEMBLE
# ============================================================

def make_probs_for_config(loaded, split_name, config):
    model_probs = {}

    for model_name in MODEL_NAMES:
        logits_key = f"{split_name}_logits"
        temp = config["temps"][model_name]
        model_probs[model_name] = softmax_temperature(
            loaded[model_name][logits_key],
            temp,
        )

    probs = None

    for model_name in MODEL_NAMES:
        w = config["weights"][model_name]

        if probs is None:
            probs = w * model_probs[model_name]
        else:
            probs = probs + w * model_probs[model_name]

    probs = normalize_probs(probs)

    return probs, model_probs


def apply_dubstep_calibration(probs, model_probs, class_names, config):
    dubstep_label = class_names.index("Dubstep")

    probs = probs.copy()

    probs[:, dubstep_label] *= float(config["dubstep_boost"])

    threshold = config["rescue_threshold"]
    rescue_boost = float(config["rescue_boost"])

    if threshold is not None and rescue_boost > 1.0:
        mlp_dub = model_probs[MLP_MODEL][:, dubstep_label]
        light_dub = model_probs[CNN_LIGHT_MODEL][:, dubstep_label]

        mask = (
            (mlp_dub >= float(threshold))
            | (light_dub >= float(threshold))
        )

        probs[mask, dubstep_label] *= rescue_boost

    probs = normalize_probs(probs)

    return probs


def make_config(
    w_medium,
    w_light,
    w_mlp,
    t_medium,
    t_light,
    t_mlp,
    dubstep_boost,
    rescue_threshold,
    rescue_boost,
):
    weights = {
        CNN_MEDIUM_MODEL: float(w_medium),
        CNN_LIGHT_MODEL: float(w_light),
        MLP_MODEL: float(w_mlp),
    }

    s = sum(weights.values())

    weights = {
        k: float(v / s)
        for k, v in weights.items()
    }

    temps = {
        CNN_MEDIUM_MODEL: float(t_medium),
        CNN_LIGHT_MODEL: float(t_light),
        MLP_MODEL: float(t_mlp),
    }

    return {
        "weights": weights,
        "temps": temps,
        "dubstep_boost": float(dubstep_boost),
        "rescue_threshold": None if rescue_threshold is None else float(rescue_threshold),
        "rescue_boost": float(rescue_boost),
    }


def search_best_3model_config(loaded, valid_y, class_names):
    print()
    print("=" * 80)
    print("SEARCH 3-MODEL ENSEMBLE ON VALIDATION")
    print("=" * 80)

    configs = []

    for (
        w_medium,
        w_light,
        w_mlp,
        t_medium,
        t_light,
        t_mlp,
        dubstep_boost,
        rescue_threshold,
        rescue_boost,
    ) in itertools.product(
        CNN_MEDIUM_WEIGHTS,
        CNN_LIGHT_WEIGHTS,
        MLP_WEIGHTS,
        CNN_MEDIUM_TEMPS,
        CNN_LIGHT_TEMPS,
        MLP_TEMPS,
        DUBSTEP_BOOSTS,
        DUBSTEP_RESCUE_THRESHOLDS,
        DUBSTEP_RESCUE_BOOSTS,
    ):
        if rescue_threshold is None and rescue_boost != 1.0:
            continue

        configs.append(
            make_config(
                w_medium=w_medium,
                w_light=w_light,
                w_mlp=w_mlp,
                t_medium=t_medium,
                t_light=t_light,
                t_mlp=t_mlp,
                dubstep_boost=dubstep_boost,
                rescue_threshold=rescue_threshold,
                rescue_boost=rescue_boost,
            )
        )

    print("Configs:", len(configs))

    rows = []
    best_config = None
    best_row = None
    best_score = -1e9

    for config in tqdm(configs, desc="search 3-model"):
        probs, model_probs = make_probs_for_config(
            loaded=loaded,
            split_name="valid",
            config=config,
        )

        probs = apply_dubstep_calibration(
            probs=probs,
            model_probs=model_probs,
            class_names=class_names,
            config=config,
        )

        pred = probs_to_pred(probs)
        metrics = metrics_for_pred(valid_y, pred, class_names)

        safe = (
            metrics["recall_Dubstep"] >= MIN_VALID_DUBSTEP_RECALL
            and metrics["recall_Classical"] >= MIN_VALID_CLASSICAL_RECALL
            and metrics["recall_Hip-Hop"] >= MIN_VALID_HIPHOP_RECALL
            and metrics["objective"] >= MIN_VALID_OBJECTIVE
        )

        score = metrics["objective"] + 0.015 * metrics["accuracy"]

        row = {
            "safe": bool(safe),
            "score": float(score),
            "accuracy": metrics["accuracy"],
            "macro_recall": metrics["macro_recall"],
            "min_recall": metrics["min_recall"],
            "objective": metrics["objective"],
            "macro_f1": metrics["macro_f1"],

            "recall_Jazz": metrics["recall_Jazz"],
            "recall_Classical": metrics["recall_Classical"],
            "recall_Hip-Hop": metrics["recall_Hip-Hop"],
            "recall_Metal": metrics["recall_Metal"],
            "recall_Country": metrics["recall_Country"],
            "recall_Dubstep": metrics["recall_Dubstep"],

            "w_medium": config["weights"][CNN_MEDIUM_MODEL],
            "w_light": config["weights"][CNN_LIGHT_MODEL],
            "w_mlp": config["weights"][MLP_MODEL],

            "t_medium": config["temps"][CNN_MEDIUM_MODEL],
            "t_light": config["temps"][CNN_LIGHT_MODEL],
            "t_mlp": config["temps"][MLP_MODEL],

            "dubstep_boost": config["dubstep_boost"],
            "rescue_threshold": config["rescue_threshold"],
            "rescue_boost": config["rescue_boost"],
        }

        rows.append(row)

        if safe and score > best_score:
            best_score = score
            best_config = config
            best_row = row

    search_df = pd.DataFrame(rows)

    if best_config is None:
        print()
        print("WARNING: safe configs not found. Using best objective overall.")

        search_df = search_df.sort_values(
            ["objective", "accuracy", "min_recall"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

        best_row = search_df.iloc[0].to_dict()

        best_config = {
            "weights": {
                CNN_MEDIUM_MODEL: float(best_row["w_medium"]),
                CNN_LIGHT_MODEL: float(best_row["w_light"]),
                MLP_MODEL: float(best_row["w_mlp"]),
            },
            "temps": {
                CNN_MEDIUM_MODEL: float(best_row["t_medium"]),
                CNN_LIGHT_MODEL: float(best_row["t_light"]),
                MLP_MODEL: float(best_row["t_mlp"]),
            },
            "dubstep_boost": float(best_row["dubstep_boost"]),
            "rescue_threshold": None if pd.isna(best_row["rescue_threshold"]) else float(best_row["rescue_threshold"]),
            "rescue_boost": float(best_row["rescue_boost"]),
        }

    search_df = search_df.sort_values(
        ["safe", "score", "objective", "accuracy", "min_recall"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)

    print()
    print("=" * 80)
    print(f"TOP {TOP_PRINT} 3-MODEL CONFIGS")
    print("=" * 80)

    show_cols = [
        "safe",
        "score",
        "accuracy",
        "macro_recall",
        "min_recall",
        "objective",
        "recall_Jazz",
        "recall_Classical",
        "recall_Hip-Hop",
        "recall_Metal",
        "recall_Country",
        "recall_Dubstep",
        "w_medium",
        "w_light",
        "w_mlp",
        "t_medium",
        "t_light",
        "t_mlp",
        "dubstep_boost",
        "rescue_threshold",
        "rescue_boost",
    ]

    print(search_df[show_cols].head(TOP_PRINT).to_string(index=False))

    print()
    print("=" * 80)
    print("CHOSEN 3-MODEL CONFIG")
    print("=" * 80)

    for k, v in best_row.items():
        print(f"{k}: {v}")

    return best_config, search_df


# ============================================================
# TARGETED RESCUE
# ============================================================

def apply_targeted_rescue(
    base_probs,
    model_probs,
    class_names,
    params,
):
    dubstep_label = class_names.index("Dubstep")
    hiphop_label = class_names.index("Hip-Hop")
    classical_label = class_names.index("Classical")

    pred = probs_to_pred(base_probs).copy()
    base_pred = pred.copy()

    final_hip = base_probs[:, hiphop_label]
    final_cls = base_probs[:, classical_label]
    final_dub = base_probs[:, dubstep_label]

    cnn_medium_hip = model_probs[CNN_MEDIUM_MODEL][:, hiphop_label]

    cnn_medium_cls = model_probs[CNN_MEDIUM_MODEL][:, classical_label]
    cnn_light_cls = model_probs[CNN_LIGHT_MODEL][:, classical_label]
    mlp_cls = model_probs[MLP_MODEL][:, classical_label]

    hip_margin = margin_to_class(base_probs, hiphop_label)
    cls_margin = margin_to_class(base_probs, classical_label)

    protect_dub = final_dub >= float(params["protect_dubstep_prob"])

    hip_candidates = (
        (pred != hiphop_label)
        & (~protect_dub)
        & (cnn_medium_hip >= float(params["hip_cnn_medium_threshold"]))
        & (final_hip >= float(params["hip_final_threshold"]))
        & (hip_margin <= float(params["hip_margin_threshold"]))
    )

    hip_score = (
        1.70 * cnn_medium_hip
        + 0.80 * final_hip
        - 0.55 * hip_margin
    )

    hip_indices = np.where(hip_candidates)[0]
    hip_indices = hip_indices[np.argsort(-hip_score[hip_indices])]
    hip_indices = hip_indices[:int(params["max_hiphop_rescues"])]

    pred[hip_indices] = hiphop_label

    cnn_cls_max = np.maximum(cnn_medium_cls, cnn_light_cls)

    cls_candidates = (
        (pred != classical_label)
        & (~protect_dub)
        & (cnn_cls_max >= float(params["classical_cnn_max_threshold"]))
        & (mlp_cls >= float(params["classical_mlp_threshold"]))
        & (final_cls >= float(params["classical_final_threshold"]))
        & (cls_margin <= float(params["classical_margin_threshold"]))
    )

    cls_candidates[hip_indices] = False

    cls_score = (
        1.10 * cnn_cls_max
        + 0.90 * mlp_cls
        + 0.70 * final_cls
        - 0.50 * cls_margin
    )

    cls_indices = np.where(cls_candidates)[0]
    cls_indices = cls_indices[np.argsort(-cls_score[cls_indices])]
    cls_indices = cls_indices[:int(params["max_classical_rescues"])]

    pred[cls_indices] = classical_label

    changed = np.where(pred != base_pred)[0]

    if len(changed) > int(params["max_total_rescues"]):
        rescue_score = np.zeros_like(final_hip)
        rescue_score[hip_indices] = hip_score[hip_indices]
        rescue_score[cls_indices] = cls_score[cls_indices]

        keep = changed[
            np.argsort(-rescue_score[changed])[:int(params["max_total_rescues"])]
        ]

        new_pred = base_pred.copy()
        new_pred[keep] = pred[keep]
        pred = new_pred
        changed = np.where(pred != base_pred)[0]

    info = {
        "changed_total": int(len(changed)),
        "changed_to_hiphop": int((pred[changed] == hiphop_label).sum()) if len(changed) > 0 else 0,
        "changed_to_classical": int((pred[changed] == classical_label).sum()) if len(changed) > 0 else 0,
    }

    return pred, info


def make_rescue_tasks():
    tasks = []

    for (
        hip_cnn_medium_threshold,
        hip_final_threshold,
        hip_margin_threshold,
        classical_cnn_max_threshold,
        classical_mlp_threshold,
        classical_final_threshold,
        classical_margin_threshold,
        protect_dubstep_prob,
        max_hiphop_rescues,
        max_classical_rescues,
        max_total_rescues,
    ) in itertools.product(
        HIP_CNN_MEDIUM_THRESHOLDS,
        HIP_FINAL_THRESHOLDS,
        HIP_MARGIN_THRESHOLDS,
        CLASSICAL_CNN_MAX_THRESHOLDS,
        CLASSICAL_MLP_THRESHOLDS,
        CLASSICAL_FINAL_THRESHOLDS,
        CLASSICAL_MARGIN_THRESHOLDS,
        PROTECT_DUBSTEP_IF_FINAL_PROB_ABOVE,
        MAX_HIPHOP_RESCUES_OPTIONS,
        MAX_CLASSICAL_RESCUES_OPTIONS,
        MAX_TOTAL_RESCUES_OPTIONS,
    ):
        if max_total_rescues < max(max_hiphop_rescues, max_classical_rescues):
            continue

        tasks.append(
            {
                "hip_cnn_medium_threshold": float(hip_cnn_medium_threshold),
                "hip_final_threshold": float(hip_final_threshold),
                "hip_margin_threshold": float(hip_margin_threshold),

                "classical_cnn_max_threshold": float(classical_cnn_max_threshold),
                "classical_mlp_threshold": float(classical_mlp_threshold),
                "classical_final_threshold": float(classical_final_threshold),
                "classical_margin_threshold": float(classical_margin_threshold),

                "protect_dubstep_prob": float(protect_dubstep_prob),

                "max_hiphop_rescues": int(max_hiphop_rescues),
                "max_classical_rescues": int(max_classical_rescues),
                "max_total_rescues": int(max_total_rescues),
            }
        )

    return tasks


def search_best_rescue(base_probs, model_probs, valid_y, class_names):
    print()
    print("=" * 80)
    print("SEARCH TARGETED RESCUE ON VALIDATION")
    print("=" * 80)

    tasks = make_rescue_tasks()
    print("Configs:", len(tasks))

    base_pred = probs_to_pred(base_probs)
    base_metrics = metrics_for_pred(valid_y, base_pred, class_names)

    print()
    print("BASE 3-MODEL VALID:")
    print("accuracy :", base_metrics["accuracy"])
    print("objective:", base_metrics["objective"])
    print("Hip-Hop  :", base_metrics["recall_Hip-Hop"])
    print("Classical:", base_metrics["recall_Classical"])
    print("Dubstep  :", base_metrics["recall_Dubstep"])

    rows = []
    best_params = None
    best_row = None
    best_score = -1e9

    for params in tqdm(tasks, desc="search rescue"):
        pred, info = apply_targeted_rescue(
            base_probs=base_probs,
            model_probs=model_probs,
            class_names=class_names,
            params=params,
        )

        metrics = metrics_for_pred(valid_y, pred, class_names)

        safe = (
            metrics["objective"] >= MIN_VALID_OBJECTIVE
            and metrics["recall_Dubstep"] >= MIN_VALID_DUBSTEP_RECALL
            and metrics["recall_Classical"] >= MIN_VALID_CLASSICAL_RECALL
            and metrics["recall_Hip-Hop"] >= MIN_VALID_HIPHOP_RECALL
        )

        score = (
            metrics["objective"]
            + 0.025 * metrics["accuracy"]
            + 0.025 * metrics["recall_Classical"]
            + 0.020 * metrics["recall_Hip-Hop"]
        )

        row = {
            **params,
            **info,
            **metrics,
            "safe": bool(safe),
            "score": float(score),
        }

        rows.append(row)

        if safe and score > best_score:
            best_score = score
            best_params = params
            best_row = row

    search_df = pd.DataFrame(rows)

    if best_params is None:
        print()
        print("WARNING: safe rescue configs not found. Using no rescue.")

        no_rescue_params = {
            "hip_cnn_medium_threshold": 999.0,
            "hip_final_threshold": 999.0,
            "hip_margin_threshold": -999.0,

            "classical_cnn_max_threshold": 999.0,
            "classical_mlp_threshold": 999.0,
            "classical_final_threshold": 999.0,
            "classical_margin_threshold": -999.0,

            "protect_dubstep_prob": 0.0,

            "max_hiphop_rescues": 0,
            "max_classical_rescues": 0,
            "max_total_rescues": 0,
        }

        return no_rescue_params, search_df

    search_df = search_df.sort_values(
        ["safe", "score", "objective", "accuracy", "min_recall"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)

    show_cols = [
        "safe",
        "score",
        "accuracy",
        "macro_recall",
        "min_recall",
        "objective",
        "recall_Jazz",
        "recall_Classical",
        "recall_Hip-Hop",
        "recall_Metal",
        "recall_Country",
        "recall_Dubstep",
        "changed_total",
        "changed_to_hiphop",
        "changed_to_classical",
        "hip_cnn_medium_threshold",
        "hip_final_threshold",
        "hip_margin_threshold",
        "classical_cnn_max_threshold",
        "classical_mlp_threshold",
        "classical_final_threshold",
        "classical_margin_threshold",
        "protect_dubstep_prob",
        "max_hiphop_rescues",
        "max_classical_rescues",
        "max_total_rescues",
    ]

    print()
    print("=" * 80)
    print(f"TOP {TOP_PRINT} TARGETED RESCUE CONFIGS")
    print("=" * 80)

    print(search_df[show_cols].head(TOP_PRINT).to_string(index=False))

    print()
    print("=" * 80)
    print("CHOSEN TARGETED RESCUE")
    print("=" * 80)

    for k in show_cols:
        print(f"{k}: {best_row[k]}")

    return best_params, search_df


# ============================================================
# SAVE
# ============================================================

def save_predictions_csv_simple(track_ids, y_true, pred, class_names, path):
    rows = []

    for tid, yt, yp in zip(track_ids, y_true, pred):
        rows.append(
            {
                "track_id": int(tid),
                "true_label": int(yt),
                "true_genre": class_names[int(yt)],
                "pred_label": int(yp),
                "pred_genre": class_names[int(yp)],
                "correct": bool(int(yt) == int(yp)),
            }
        )

    pd.DataFrame(rows).to_csv(path, index=False)


def preds_to_logits(pred, num_classes):
    probs = np.full(
        (len(pred), num_classes),
        1e-6,
        dtype=np.float64,
    )

    probs[np.arange(len(pred)), pred] = 1.0
    probs = normalize_probs(probs)

    return probs_to_logits(probs)


# ============================================================
# MAIN
# ============================================================

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ENSEMBLE_OUTDIR.mkdir(parents=True, exist_ok=True)

    df, genres_df, selected_genre_ids, genre_id_to_label, label_to_genre_id, class_names = build_master_df()

    split = load_final_split()
    check_split_against_df(df, split)

    print()
    print("=" * 80)
    print("3-MODEL ENSEMBLE")
    print("=" * 80)
    print("SPLIT:", SPLIT_JSON)
    print("OUTDIR:", OUTDIR)
    print("ENSEMBLE_OUTDIR:", ENSEMBLE_OUTDIR)

    loaded = {}
    results_list = []

    for model_name in MODEL_NAMES:
        result = load_model_result(
            model_name=model_name,
            class_names=class_names,
        )

        loaded[model_name] = result
        results_list.append(result)

    valid_y, test_y, valid_ids, test_ids = check_alignment(results_list)

    best_config, ensemble_search_df = search_best_3model_config(
        loaded=loaded,
        valid_y=valid_y,
        class_names=class_names,
    )

    valid_probs, valid_model_probs = make_probs_for_config(
        loaded=loaded,
        split_name="valid",
        config=best_config,
    )

    valid_probs = apply_dubstep_calibration(
        probs=valid_probs,
        model_probs=valid_model_probs,
        class_names=class_names,
        config=best_config,
    )

    test_probs, test_model_probs = make_probs_for_config(
        loaded=loaded,
        split_name="test",
        config=best_config,
    )

    test_probs = apply_dubstep_calibration(
        probs=test_probs,
        model_probs=test_model_probs,
        class_names=class_names,
        config=best_config,
    )

    valid_pred_before_rescue = probs_to_pred(valid_probs)
    test_pred_before_rescue = probs_to_pred(test_probs)

    valid_metrics_before_rescue = print_metrics_from_pred(
        "3-MODEL VALIDATION BEFORE TARGETED RESCUE",
        valid_y,
        valid_pred_before_rescue,
        class_names,
    )

    test_metrics_before_rescue = print_metrics_from_pred(
        "3-MODEL TEST BEFORE TARGETED RESCUE",
        test_y,
        test_pred_before_rescue,
        class_names,
    )

    if USE_TARGETED_RESCUE:
        best_rescue_params, rescue_search_df = search_best_rescue(
            base_probs=valid_probs,
            model_probs=valid_model_probs,
            valid_y=valid_y,
            class_names=class_names,
        )

        valid_pred, valid_rescue_info = apply_targeted_rescue(
            base_probs=valid_probs,
            model_probs=valid_model_probs,
            class_names=class_names,
            params=best_rescue_params,
        )

        test_pred, test_rescue_info = apply_targeted_rescue(
            base_probs=test_probs,
            model_probs=test_model_probs,
            class_names=class_names,
            params=best_rescue_params,
        )

    else:
        best_rescue_params = None
        rescue_search_df = pd.DataFrame()
        valid_pred = valid_pred_before_rescue
        test_pred = test_pred_before_rescue
        valid_rescue_info = {"changed_total": 0, "changed_to_hiphop": 0, "changed_to_classical": 0}
        test_rescue_info = {"changed_total": 0, "changed_to_hiphop": 0, "changed_to_classical": 0}

    valid_metrics = print_metrics_from_pred(
        "FINAL 3-MODEL VALIDATION",
        valid_y,
        valid_pred,
        class_names,
    )

    test_metrics = print_metrics_from_pred(
        "FINAL 3-MODEL TEST",
        test_y,
        test_pred,
        class_names,
    )

    print()
    print("=" * 80)
    print("RESCUE CHANGES")
    print("=" * 80)
    print("VALID:", valid_rescue_info)
    print("TEST :", test_rescue_info)

    valid_logits = preds_to_logits(
        valid_pred,
        num_classes=len(class_names),
    )

    test_logits = preds_to_logits(
        test_pred,
        num_classes=len(class_names),
    )

    np.save(ENSEMBLE_OUTDIR / "valid_logits.npy", valid_logits)
    np.save(ENSEMBLE_OUTDIR / "valid_y.npy", valid_y)
    np.save(ENSEMBLE_OUTDIR / "valid_track_ids.npy", valid_ids)

    np.save(ENSEMBLE_OUTDIR / "test_logits.npy", test_logits)
    np.save(ENSEMBLE_OUTDIR / "test_y.npy", test_y)
    np.save(ENSEMBLE_OUTDIR / "test_track_ids.npy", test_ids)

    save_predictions_csv_simple(
        valid_ids,
        valid_y,
        valid_pred,
        class_names,
        ENSEMBLE_OUTDIR / "valid_predictions.csv",
    )

    save_predictions_csv_simple(
        test_ids,
        test_y,
        test_pred,
        class_names,
        ENSEMBLE_OUTDIR / "test_predictions.csv",
    )

    ensemble_search_df.to_csv(
        ENSEMBLE_OUTDIR / "validation_3model_search_results.csv",
        index=False,
    )

    if len(rescue_search_df) > 0:
        rescue_search_df.to_csv(
            ENSEMBLE_OUTDIR / "validation_3model_rescue_search_results.csv",
            index=False,
        )

    summary = {
        "ensemble_name": ENSEMBLE_NAME,
        "ensemble_type": "3model_cnn_medium_light_mlp_tuned_rescue",
        "split_json": str(SPLIT_JSON),

        "models": MODEL_NAMES,

        "best_config": best_config,
        "best_rescue_params": best_rescue_params,

        "valid_rescue_info": valid_rescue_info,
        "test_rescue_info": test_rescue_info,

        "valid_metrics_before_rescue": valid_metrics_before_rescue,
        "test_metrics_before_rescue": test_metrics_before_rescue,

        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,

        "individual_valid_metrics": {
            model_name: loaded[model_name]["valid_metrics"]
            for model_name in MODEL_NAMES
        },

        "individual_test_metrics": {
            model_name: loaded[model_name]["test_metrics"]
            for model_name in MODEL_NAMES
        },
    }

    save_json(
        to_builtin(summary),
        ENSEMBLE_OUTDIR / "summary.json",
    )

    save_json(
        to_builtin(summary),
        OUTDIR / "final_3model_cnn_medium_light_mlp_summary.json",
    )

    print()
    print("=" * 80)
    print("ГОТОВО")
    print("=" * 80)
    print("Saved to:")
    print(ENSEMBLE_OUTDIR)

    print()
    print("TEST RESULTS:")
    print("test_accuracy:", test_metrics["accuracy"])
    print("test_accuracy_percent:", test_metrics["accuracy_percent"])
    print("test_macro_recall:", test_metrics["macro_recall"])
    print("test_min_recall:", test_metrics["min_recall"])
    print("test_objective:", test_metrics["objective"])

    print()
    print("FINAL REFERENCE:")
    print("final_accuracy_percent: 63.75")
    print("final_objective: 0.628125")
    print("final_min_recall: 0.60")
    print("final_Hip-Hop: 0.61")
    print("final_Classical: 0.60")
    print("final_Dubstep: 0.68")


if __name__ == "__main__":
    main()