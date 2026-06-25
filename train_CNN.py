#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from common_final6 import *


MODEL_DIR = OUTDIR / "mel_rescnn_medium"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "mel_rescnn_medium_best.pt"
SUMMARY_JSON = MODEL_DIR / "summary.json"

TARGET_MELS = 128
TARGET_FRAMES = 384

BATCH_SIZE = 32
NUM_EPOCHS = 80
PATIENCE = 14

LEARNING_RATE = 7e-4
WEIGHT_DECAY = 2e-4

NUM_WORKERS = 0
USE_RAM_CACHE = True

USE_SPEC_AUGMENT = True
USE_MIXUP = True
MIXUP_ALPHA = 0.25

LABEL_SMOOTHING = 0.08

DEVICE = get_device()


def fix_size(mel, target_mels=128, target_frames=384, random_crop=False):
    mel = np.asarray(mel, dtype=np.float32)

    if mel.ndim == 3 and mel.shape[0] == 1:
        mel = mel[0]

    if mel.ndim != 2:
        raise ValueError(f"Ожидалась 2D mel, получена {mel.shape}")

    if mel.shape[0] < target_mels:
        mel = np.pad(
            mel,
            ((0, target_mels - mel.shape[0]), (0, 0)),
            mode="constant",
        )
    elif mel.shape[0] > target_mels:
        mel = mel[:target_mels, :]

    if mel.shape[1] < target_frames:
        mel = np.pad(
            mel,
            ((0, 0), (0, target_frames - mel.shape[1])),
            mode="edge",
        )
    elif mel.shape[1] > target_frames:
        if random_crop:
            start = np.random.randint(
                0,
                mel.shape[1] - target_frames + 1,
            )
        else:
            start = (mel.shape[1] - target_frames) // 2

        mel = mel[:, start:start + target_frames]

    return mel


def normalize_mel(mel):
    mean = float(mel.mean())
    std = float(mel.std())

    if std < 1e-6:
        std = 1.0

    return ((mel - mean) / std).astype(np.float32)


def load_and_process(path, random_crop=False):
    mel = np.load(path)

    mel = fix_size(
        mel,
        target_mels=TARGET_MELS,
        target_frames=TARGET_FRAMES,
        random_crop=random_crop,
    )

    mel = normalize_mel(mel)

    return mel[None, :, :].astype(np.float32)


class MelDataset(Dataset):
    def __init__(self, df, ids, random_crop=False, use_ram_cache=False):
        self.df = df
        self.ids = np.array(ids, dtype=np.int64)
        self.random_crop = bool(random_crop)
        self.use_ram_cache = bool(use_ram_cache)

        self.paths = df.loc[self.ids, "mel_path"].astype(str).tolist()
        self.y = df.loc[self.ids, "label"].to_numpy(dtype=np.int64)

        self.cache_x = None

        if self.use_ram_cache:
            xs = []

            print()
            print(f"Загружаю в RAM: {len(self.ids)} файлов...")

            for path in tqdm(self.paths):
                xs.append(
                    load_and_process(
                        path,
                        random_crop=False,
                    )
                )

            self.cache_x = torch.tensor(
                np.stack(xs),
                dtype=torch.float32,
            )

            size_mb = self.cache_x.numel() * 4 / 1024 / 1024
            print(f"RAM cache X: {size_mb:.1f} MB")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        if self.cache_x is not None:
            x = self.cache_x[idx]
        else:
            x = torch.tensor(
                load_and_process(
                    self.paths[idx],
                    random_crop=self.random_crop,
                ),
                dtype=torch.float32,
            )

        y = torch.tensor(
            int(self.y[idx]),
            dtype=torch.long,
        )

        track_id = int(self.ids[idx])

        return x, y, track_id


class SpecAugment(nn.Module):
    def __init__(
        self,
        freq_mask_param=14,
        time_mask_param=40,
        num_freq_masks=2,
        num_time_masks=2,
    ):
        super().__init__()

        self.freq_mask_param = int(freq_mask_param)
        self.time_mask_param = int(time_mask_param)
        self.num_freq_masks = int(num_freq_masks)
        self.num_time_masks = int(num_time_masks)

    def forward(self, x):
        if not self.training:
            return x

        x = x.clone()

        batch_size, channels, freq_bins, time_frames = x.shape

        for i in range(batch_size):
            for _ in range(self.num_freq_masks):
                mask_size = random.randint(0, self.freq_mask_param)

                if 0 < mask_size < freq_bins:
                    start = random.randint(0, freq_bins - mask_size)
                    x[i, :, start:start + mask_size, :] = 0

            for _ in range(self.num_time_masks):
                mask_size = random.randint(0, self.time_mask_param)

                if 0 < mask_size < time_frames:
                    start = random.randint(0, time_frames - mask_size)
                    x[i, :, :, start:start + mask_size] = 0

        return x


def mixup_data(x, y, alpha=0.25):
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


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        dropout=0.1,
        downsample=False,
    ):
        super().__init__()

        stride = 2 if downsample else 1

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.SiLU()

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout)
        self.act2 = nn.SiLU()

        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.act2(out)
        out = self.dropout(out)

        return out


class ComplexMelCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.augment = SpecAugment()

        self.stem = nn.Sequential(
            nn.Conv2d(
                1,
                32,
                kernel_size=5,
                stride=1,
                padding=2,
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.SiLU(),
        )

        self.stage1 = nn.Sequential(
            ResidualBlock(32, 32, dropout=0.08, downsample=False),
            ResidualBlock(32, 32, dropout=0.08, downsample=False),
        )

        self.stage2 = nn.Sequential(
            ResidualBlock(32, 64, dropout=0.12, downsample=True),
            ResidualBlock(64, 64, dropout=0.12, downsample=False),
        )

        self.stage3 = nn.Sequential(
            ResidualBlock(64, 128, dropout=0.16, downsample=True),
            ResidualBlock(128, 128, dropout=0.16, downsample=False),
        )

        self.stage4 = nn.Sequential(
            ResidualBlock(128, 256, dropout=0.20, downsample=True),
            ResidualBlock(256, 256, dropout=0.20, downsample=False),
        )

        self.stage5 = nn.Sequential(
            ResidualBlock(256, 384, dropout=0.24, downsample=True),
            ResidualBlock(384, 384, dropout=0.24, downsample=False),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),

            nn.Linear(384, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.45),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.30),

            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        if self.training and USE_SPEC_AUGMENT:
            x = self.augment(x)

        x = self.stem(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)

        return self.head(x)


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


def main():
    seed_everything(RANDOM_SEED)

    print()
    print("DEVICE:", DEVICE)

    df, genres_df, selected_genre_ids, genre_id_to_label, label_to_genre_id, class_names = build_master_df()

    split = load_final_split()
    check_split_against_df(df, split)

    print()
    print("=" * 80)
    print("FINAL CNN: SAME ARTIST-AWARE SPLIT")
    print("=" * 80)
    print("train:", len(split["train"]))
    print("valid:", len(split["valid"]))
    print("test :", len(split["test"]))

    train_ds = MelDataset(
        df,
        split["train"],
        random_crop=True,
        use_ram_cache=USE_RAM_CACHE,
    )

    valid_ds = MelDataset(
        df,
        split["valid"],
        random_crop=False,
        use_ram_cache=USE_RAM_CACHE,
    )

    test_ds = MelDataset(
        df,
        split["test"],
        random_crop=False,
        use_ram_cache=USE_RAM_CACHE,
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

    model = ComplexMelCNN(
        num_classes=len(class_names),
    ).to(DEVICE)

    print()
    print(model)
    print("Параметров:", sum(p.numel() for p in model.parameters()))

    criterion = nn.CrossEntropyLoss(
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

    best_valid_objective = -1.0
    best_valid_acc = -1.0
    best_valid_min = -1.0
    best_epoch = 0
    bad_epochs = 0

    labels = list(range(len(class_names)))

    print()
    print("=" * 80)
    print("НАЧИНАЮ ОБУЧЕНИЕ CNN")
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
                "target_mels": TARGET_MELS,
                "target_frames": TARGET_FRAMES,
                "best_valid_acc": best_valid_acc,
                "best_valid_objective": best_valid_objective,
                "best_valid_min_recall": best_valid_min,
                "best_epoch": best_epoch,
                "split_json": str(SPLIT_JSON),
                "model_name": "ComplexMelCNN",
            }

            torch.save(checkpoint, BEST_MODEL_PATH)

            print("Новая лучшая модель сохранена:", BEST_MODEL_PATH)
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

    model.load_state_dict(checkpoint["model_state_dict"])

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
        "mel_rescnn_medium VALIDATION",
        valid_logits,
        valid_y,
        class_names,
    )

    test_metrics = print_metrics(
        "mel_rescnn_medium TEST",
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

    save_json(
        {
            "model_path": str(BEST_MODEL_PATH),
            "split_json": str(SPLIT_JSON),
            "best_epoch": int(checkpoint["best_epoch"]),
            "best_valid_acc": float(checkpoint["best_valid_acc"]),
            "best_valid_objective": float(checkpoint["best_valid_objective"]),
            "best_valid_min_recall": float(checkpoint["best_valid_min_recall"]),
            "valid_metrics": valid_metrics,
            "test_metrics": test_metrics,
        },
        SUMMARY_JSON,
    )

    print()
    print("Saved to:")
    print(MODEL_DIR)


if __name__ == "__main__":
    main()