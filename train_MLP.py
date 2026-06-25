#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from common_final6 import *


# ============================================================
# 0. НАСТРОЙКИ
# ============================================================

MODEL_NAME = "feature_mlp_heavy"

MODEL_DIR = OUTDIR / MODEL_NAME
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / f"{MODEL_NAME}_best.pt"
SUMMARY_JSON = MODEL_DIR / "summary.json"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.npz"

BATCH_SIZE = 256
NUM_EPOCHS = 140
PATIENCE = 22

LEARNING_RATE = 5e-4
WEIGHT_DECAY = 4e-4

LABEL_SMOOTHING = 0.08

USE_MIXUP = True
MIXUP_ALPHA = 0.20

USE_FEATURE_NOISE = True
FEATURE_NOISE_STD = 0.020

USE_CLASS_WEIGHTS = True

NUM_WORKERS = 0

DEVICE = get_device()


# ============================================================
# 1. FEATURES
# ============================================================

def get_feature_columns(df):
    meta_cols = {
        "genre_id",
        "label",
        "genre_name",
        "mel_path",
    }

    return [
        col
        for col in df.columns
        if col not in meta_cols
    ]


def fit_preprocessor(X_train):
    train_means = np.nanmean(X_train, axis=0)
    train_means = np.where(np.isnan(train_means), 0.0, train_means)

    X_filled = np.where(
        np.isnan(X_train),
        train_means,
        X_train,
    )

    mean = X_filled.mean(axis=0)
    std = X_filled.std(axis=0)

    std[std < 1e-8] = 1.0

    return (
        train_means.astype(np.float32),
        mean.astype(np.float32),
        std.astype(np.float32),
    )


def transform_features(X, train_means, mean, std):
    X_filled = np.where(
        np.isnan(X),
        train_means,
        X,
    )

    X_norm = (X_filled - mean) / std
    X_norm = np.clip(X_norm, -8.0, 8.0)

    return X_norm.astype(np.float32)


def build_arrays(df, split):
    feature_cols = get_feature_columns(df)

    train_ids = np.array(split["train"], dtype=int)
    valid_ids = np.array(split["valid"], dtype=int)
    test_ids = np.array(split["test"], dtype=int)

    X_train_raw = df.loc[train_ids, feature_cols].to_numpy(dtype=np.float32)
    X_valid_raw = df.loc[valid_ids, feature_cols].to_numpy(dtype=np.float32)
    X_test_raw = df.loc[test_ids, feature_cols].to_numpy(dtype=np.float32)

    y_train = df.loc[train_ids, "label"].to_numpy(dtype=np.int64)
    y_valid = df.loc[valid_ids, "label"].to_numpy(dtype=np.int64)
    y_test = df.loc[test_ids, "label"].to_numpy(dtype=np.int64)

    train_means, mean, std = fit_preprocessor(X_train_raw)

    X_train = transform_features(
        X_train_raw,
        train_means,
        mean,
        std,
    )

    X_valid = transform_features(
        X_valid_raw,
        train_means,
        mean,
        std,
    )

    X_test = transform_features(
        X_test_raw,
        train_means,
        mean,
        std,
    )

    return {
        "feature_cols": feature_cols,
        "train_ids": train_ids,
        "valid_ids": valid_ids,
        "test_ids": test_ids,
        "X_train": X_train,
        "X_valid": X_valid,
        "X_test": X_test,
        "y_train": y_train,
        "y_valid": y_valid,
        "y_test": y_test,
        "train_means": train_means,
        "mean": mean,
        "std": std,
    }


def print_class_counts(y, class_names, title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    for label, name in enumerate(class_names):
        print(f"{name}: {int((y == label).sum())}")


def make_class_weights(y, num_classes):
    counts = np.array(
        [
            max(int((y == label).sum()), 1)
            for label in range(num_classes)
        ],
        dtype=np.float32,
    )

    # inverse sqrt — мягкая компенсация дисбаланса.
    weights = 1.0 / np.sqrt(counts)
    weights = weights / weights.mean()

    return weights.astype(np.float32)


# ============================================================
# 2. DATASET
# ============================================================

class FeatureDataset(Dataset):
    def __init__(self, X, y, ids, train=False):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.ids = np.array(ids, dtype=np.int64)
        self.train = bool(train)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]

        if self.train and USE_FEATURE_NOISE and FEATURE_NOISE_STD > 0:
            noise = torch.randn_like(x) * FEATURE_NOISE_STD
            x = x + noise

        y = self.y[idx]
        track_id = int(self.ids[idx])

        return x, y, track_id


# ============================================================
# 3. HEAVY RESIDUAL MLP
# ============================================================

class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, dim),
            nn.BatchNorm1d(dim),
        )

        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.block(x)
        out = x + out
        out = self.act(out)
        out = self.dropout(out)
        return out


class FeatureResidualMLPHeavy(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()

        self.input = nn.Sequential(
            nn.Linear(input_dim, 1536),
            nn.BatchNorm1d(1536),
            nn.SiLU(),
            nn.Dropout(0.35),
        )

        self.blocks = nn.Sequential(
            ResidualMLPBlock(1536, 2048, dropout=0.35),
            ResidualMLPBlock(1536, 2048, dropout=0.35),
            ResidualMLPBlock(1536, 1536, dropout=0.30),
            ResidualMLPBlock(1536, 1536, dropout=0.30),
        )

        self.head = nn.Sequential(
            nn.Linear(1536, 768),
            nn.BatchNorm1d(768),
            nn.SiLU(),
            nn.Dropout(0.30),

            nn.Linear(768, 384),
            nn.BatchNorm1d(384),
            nn.SiLU(),
            nn.Dropout(0.25),

            nn.Linear(384, 192),
            nn.BatchNorm1d(192),
            nn.SiLU(),
            nn.Dropout(0.20),

            nn.Linear(192, num_classes),
        )

    def forward(self, x):
        x = self.input(x)
        x = self.blocks(x)
        x = self.head(x)
        return x


# ============================================================
# 4. MIXUP
# ============================================================

def mixup_data(x, y, alpha=0.20):
    if alpha <= 0:
        return x, y, y, 1.0

    lam = np.random.beta(alpha, alpha)

    index = torch.randperm(
        x.size(0),
        device=x.device,
    )

    mixed_x = lam * x + (1.0 - lam) * x[index]

    return mixed_x, y, y[index], lam


def mixup_loss(criterion, prediction, y_a, y_b, lam):
    return (
        lam * criterion(prediction, y_a)
        + (1.0 - lam) * criterion(prediction, y_b)
    )


# ============================================================
# 5. TRAIN / EVAL
# ============================================================

def run_epoch(
    model,
    loader,
    criterion,
    optimizer=None,
    return_logits=False,
):
    is_train = optimizer is not None

    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    all_true = []
    all_pred = []
    all_ids = []
    all_logits = []

    for x, y, track_ids in tqdm(loader, leave=False):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            if is_train and USE_MIXUP:
                mixed_x, y_a, y_b, lam = mixup_data(
                    x,
                    y,
                    MIXUP_ALPHA,
                )

                logits = model(mixed_x)

                loss = mixup_loss(
                    criterion,
                    logits,
                    y_a,
                    y_b,
                    lam,
                )
            else:
                logits = model(x)
                loss = criterion(logits, y)

            if is_train:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0,
                )

                optimizer.step()

        pred = logits.argmax(dim=1)

        total_loss += float(loss.item()) * len(y)
        total_correct += int((pred == y).sum().item())
        total_count += len(y)

        all_true.extend(y.detach().cpu().numpy().tolist())
        all_pred.extend(pred.detach().cpu().numpy().tolist())
        all_ids.extend([int(t) for t in track_ids])

        if return_logits:
            all_logits.append(logits.detach().cpu().numpy())

    result = (
        total_loss / total_count,
        total_correct / total_count,
        np.array(all_true, dtype=int),
        np.array(all_pred, dtype=int),
        np.array(all_ids, dtype=int),
    )

    if return_logits:
        return result + (np.concatenate(all_logits, axis=0),)

    return result


# ============================================================
# 6. MAIN
# ============================================================

def main():
    seed_everything(RANDOM_SEED)

    print()
    print("DEVICE:", DEVICE)

    df, genres_df, selected_genre_ids, genre_id_to_label, label_to_genre_id, class_names = build_master_df()

    split = load_final_split()
    check_split_against_df(df, split)

    print()
    print("=" * 80)
    print("FEATURE HEAVY RESIDUAL MLP: SAME ARTIST-AWARE SPLIT")
    print("=" * 80)
    print("train:", len(split["train"]))
    print("valid:", len(split["valid"]))
    print("test :", len(split["test"]))

    arrays = build_arrays(
        df=df,
        split=split,
    )

    print()
    print("X_train:", arrays["X_train"].shape)
    print("X_valid:", arrays["X_valid"].shape)
    print("X_test :", arrays["X_test"].shape)

    print_class_counts(
        arrays["y_train"],
        class_names,
        "TRAIN CLASS COUNTS",
    )

    print_class_counts(
        arrays["y_valid"],
        class_names,
        "VALID CLASS COUNTS",
    )

    print_class_counts(
        arrays["y_test"],
        class_names,
        "TEST CLASS COUNTS",
    )

    np.savez(
        PREPROCESSOR_PATH,
        train_means=arrays["train_means"],
        mean=arrays["mean"],
        std=arrays["std"],
        feature_cols=np.array(arrays["feature_cols"], dtype=object),
    )

    train_ds = FeatureDataset(
        arrays["X_train"],
        arrays["y_train"],
        arrays["train_ids"],
        train=True,
    )

    valid_ds = FeatureDataset(
        arrays["X_valid"],
        arrays["y_valid"],
        arrays["valid_ids"],
        train=False,
    )

    test_ds = FeatureDataset(
        arrays["X_test"],
        arrays["y_test"],
        arrays["test_ids"],
        train=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    input_dim = arrays["X_train"].shape[1]
    num_classes = len(class_names)

    model = FeatureResidualMLPHeavy(
        input_dim=input_dim,
        num_classes=num_classes,
    ).to(DEVICE)

    print()
    print(model)

    total_params = sum(p.numel() for p in model.parameters())

    print()
    print(f"Параметров: {total_params:,}")

    if USE_CLASS_WEIGHTS:
        class_weights_np = make_class_weights(
            arrays["y_train"],
            num_classes,
        )

        print()
        print("Class weights:")
        for label, name in enumerate(class_names):
            print(f"{name}: {class_weights_np[label]:.4f}")

        class_weights = torch.tensor(
            class_weights_np,
            dtype=torch.float32,
            device=DEVICE,
        )
    else:
        class_weights = None

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=LABEL_SMOOTHING,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
        eta_min=1e-6,
    )

    labels = list(range(num_classes))

    best_valid_objective = -1.0
    best_valid_acc = -1.0
    best_valid_min = -1.0
    best_epoch = 0
    bad_epochs = 0

    print()
    print("=" * 80)
    print("НАЧИНАЮ ОБУЧЕНИЕ FEATURE HEAVY RESIDUAL MLP")
    print("=" * 80)

    for epoch in range(1, NUM_EPOCHS + 1):
        lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc, train_y, train_pred, _ = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer=optimizer,
        )

        valid_loss, valid_acc, valid_y, valid_pred, _ = run_epoch(
            model,
            valid_loader,
            criterion,
            optimizer=None,
        )

        scheduler.step()

        train_obj = balanced_objective(
            train_y,
            train_pred,
            labels,
        )

        valid_obj = balanced_objective(
            valid_y,
            valid_pred,
            labels,
        )

        valid_min = min_recall(
            valid_y,
            valid_pred,
            labels,
        )

        print()
        print(f"epoch={epoch:03d}/{NUM_EPOCHS} | lr={lr:.8f}")
        print(
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"train_obj={train_obj:.4f}"
        )
        print(
            f"valid_loss={valid_loss:.4f} | "
            f"valid_acc={valid_acc:.4f} | "
            f"valid_obj={valid_obj:.4f} | "
            f"valid_min={valid_min:.4f}"
        )

        current_key = (
            valid_obj,
            valid_min,
            valid_acc,
        )

        best_key = (
            best_valid_objective,
            best_valid_min,
            best_valid_acc,
        )

        if current_key > best_key:
            best_valid_objective = valid_obj
            best_valid_acc = valid_acc
            best_valid_min = valid_min
            best_epoch = epoch
            bad_epochs = 0

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "target_genre_names": class_names,
                "genre_id_to_label": genre_id_to_label,
                "label_to_genre_id": label_to_genre_id,
                "input_dim": int(input_dim),
                "feature_cols": arrays["feature_cols"],
                "best_valid_acc": float(best_valid_acc),
                "best_valid_objective": float(best_valid_objective),
                "best_valid_min_recall": float(best_valid_min),
                "best_epoch": int(best_epoch),
                "split_json": str(SPLIT_JSON),
                "model_name": MODEL_NAME,
                "architecture": "FeatureResidualMLPHeavy",
                "use_class_weights": bool(USE_CLASS_WEIGHTS),
                "use_mixup": bool(USE_MIXUP),
                "use_feature_noise": bool(USE_FEATURE_NOISE),
            }

            torch.save(
                checkpoint,
                BEST_MODEL_PATH,
            )

            print("Новая лучшая HEAVY MLP сохранена:", BEST_MODEL_PATH)
        else:
            bad_epochs += 1
            print(f"Без улучшения: {bad_epochs}/{PATIENCE}")

        if bad_epochs >= PATIENCE:
            print()
            print("Early stopping.")
            break

    print()
    print("=" * 80)
    print("LOAD BEST AND PREDICT VALID/TEST")
    print("=" * 80)

    checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
    )

    valid_loss, valid_acc, valid_y, valid_pred, valid_ids, valid_logits = run_epoch(
        model,
        valid_loader,
        criterion,
        optimizer=None,
        return_logits=True,
    )

    test_loss, test_acc, test_y, test_pred, test_ids, test_logits = run_epoch(
        model,
        test_loader,
        criterion,
        optimizer=None,
        return_logits=True,
    )

    valid_metrics = print_metrics(
        "feature_mlp_heavy VALIDATION",
        valid_logits,
        valid_y,
        class_names,
    )

    test_metrics = print_metrics(
        "feature_mlp_heavy TEST",
        test_logits,
        test_y,
        class_names,
    )

    np.save(MODEL_DIR / "valid_logits.npy", valid_logits)
    np.save(MODEL_DIR / "valid_y.npy", valid_y)
    np.save(MODEL_DIR / "valid_track_ids.npy", valid_ids)

    np.save(MODEL_DIR / "test_logits.npy", test_logits)
    np.save(MODEL_DIR / "test_y.npy", test_y)
    np.save(MODEL_DIR / "test_track_ids.npy", test_ids)

    save_predictions_csv(
        valid_ids,
        valid_y,
        valid_logits,
        class_names,
        MODEL_DIR / "valid_predictions.csv",
    )

    save_predictions_csv(
        test_ids,
        test_y,
        test_logits,
        class_names,
        MODEL_DIR / "test_predictions.csv",
    )

    summary = {
        "model_name": MODEL_NAME,
        "architecture": "FeatureResidualMLPHeavy",
        "model_path": str(BEST_MODEL_PATH),
        "preprocessor_path": str(PREPROCESSOR_PATH),
        "split_json": str(SPLIT_JSON),
        "input_dim": int(input_dim),
        "batch_size": int(BATCH_SIZE),
        "num_epochs": int(NUM_EPOCHS),
        "patience": int(PATIENCE),
        "learning_rate": float(LEARNING_RATE),
        "weight_decay": float(WEIGHT_DECAY),
        "use_class_weights": bool(USE_CLASS_WEIGHTS),
        "use_mixup": bool(USE_MIXUP),
        "mixup_alpha": float(MIXUP_ALPHA),
        "use_feature_noise": bool(USE_FEATURE_NOISE),
        "feature_noise_std": float(FEATURE_NOISE_STD),
        "label_smoothing": float(LABEL_SMOOTHING),
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_valid_acc": float(checkpoint["best_valid_acc"]),
        "best_valid_objective": float(checkpoint["best_valid_objective"]),
        "best_valid_min_recall": float(checkpoint["best_valid_min_recall"]),
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
    }

    save_json(
        summary,
        SUMMARY_JSON,
    )

    print()
    print("=" * 80)
    print("ГОТОВО")
    print("=" * 80)
    print("Saved to:")
    print(MODEL_DIR)
    print()
    print(
        f"TEST accuracy={test_metrics['accuracy']:.4f} | "
        f"{test_metrics['accuracy_percent']:.2f}%"
    )
    print(
        f"TEST macro_recall={test_metrics['macro_recall']:.4f} | "
        f"min_recall={test_metrics['min_recall']:.4f} | "
        f"objective={test_metrics['objective']:.4f}"
    )


if __name__ == "__main__":
    main()