#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

from common_final6 import *


# ============================================================
# FAST LIGHT CNN SETTINGS
# ============================================================

MODEL_PREFIX = "mel_rescnn_light_fast"

SEEDS = [7]

TARGET_FRAMES = 192

BATCH_SIZE = 96
NUM_EPOCHS = 18
PATIENCE = 5

LR = 1e-3
WEIGHT_DECAY = 2e-4

USE_RAM_CACHE = True
NUM_WORKERS = 0

USE_WEIGHTED_SAMPLER = True
USE_CLASS_WEIGHTS = True

# Focal loss выключен, потому что он слишком сильно тянул редкие классы.
USE_FOCAL_LOSS = False
FOCAL_GAMMA = 1.5

# Мягкое усиление редких классов.
DUBSTEP_EXTRA_WEIGHT = 1.15
METAL_EXTRA_WEIGHT = 1.05
COUNTRY_EXTRA_WEIGHT = 1.05

USE_SPEC_AUGMENT = True
TIME_MASK_MAX = 24
FREQ_MASK_MAX = 8
SPEC_AUG_PROB = 0.45

GRAD_CLIP = 3.0


# ============================================================
# DEVICE
# ============================================================

def get_torch_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# MEL PATHS
# ============================================================

def build_mel_path_map():
    mel_paths = {}

    for path in MEL_DIR.rglob("*.npy"):
        stem = path.stem
        digits = "".join(ch for ch in stem if ch.isdigit())

        if digits == "":
            continue

        track_id = int(digits)
        mel_paths[track_id] = path

    if len(mel_paths) == 0:
        raise RuntimeError(f"Не найдено .npy mel-файлов в {MEL_DIR}")

    return mel_paths


def resolve_track_mel_path(track_id, mel_path_map):
    track_id = int(track_id)

    if track_id in mel_path_map:
        return mel_path_map[track_id]

    p1 = MEL_DIR / f"{track_id}.npy"
    if p1.exists():
        return p1

    p2 = MEL_DIR / f"{track_id:06d}.npy"
    if p2.exists():
        return p2

    raise FileNotFoundError(f"Не найден mel-файл для track_id={track_id}")


# ============================================================
# DATASET
# ============================================================

def fix_mel_shape(x):
    x = np.asarray(x, dtype=np.float32)

    if x.ndim == 3:
        x = np.squeeze(x)

    if x.ndim != 2:
        raise ValueError(f"Bad mel shape: {x.shape}")

    # Обычно mel: [n_mels, frames].
    if x.shape[0] > x.shape[1]:
        x = x.T

    return x


def crop_or_pad(x, target_frames, train):
    n_mels, frames = x.shape

    if frames == target_frames:
        return x

    if frames > target_frames:
        if train:
            start = random.randint(0, frames - target_frames)
        else:
            start = (frames - target_frames) // 2

        return x[:, start:start + target_frames]

    pad = target_frames - frames
    left = pad // 2
    right = pad - left

    return np.pad(
        x,
        ((0, 0), (left, right)),
        mode="constant",
        constant_values=0.0,
    )


def normalize_mel(x):
    x = np.asarray(x, dtype=np.float32)

    mean = float(np.mean(x))
    std = float(np.std(x))

    if std < 1e-6:
        std = 1.0

    x = (x - mean) / std
    x = np.clip(x, -5.0, 5.0)

    return x


def apply_spec_augment_tensor(x):
    # x: [1, n_mels, frames]
    if random.random() > SPEC_AUG_PROB:
        return x

    _, n_mels, frames = x.shape

    if TIME_MASK_MAX > 0 and frames > 4:
        t = random.randint(0, min(TIME_MASK_MAX, frames // 3))
        if t > 0:
            t0 = random.randint(0, frames - t)
            x[:, :, t0:t0 + t] = 0.0

    if FREQ_MASK_MAX > 0 and n_mels > 4:
        f = random.randint(0, min(FREQ_MASK_MAX, n_mels // 3))
        if f > 0:
            f0 = random.randint(0, n_mels - f)
            x[:, f0:f0 + f, :] = 0.0

    return x


class MelDataset(Dataset):
    def __init__(
        self,
        df,
        track_ids,
        mel_path_map,
        target_frames,
        train=False,
        use_ram_cache=False,
    ):
        self.df = df
        self.track_ids = np.array(track_ids, dtype=int)
        self.mel_path_map = mel_path_map
        self.target_frames = int(target_frames)
        self.train = bool(train)
        self.use_ram_cache = bool(use_ram_cache)

        self.labels = self.df.loc[self.track_ids, "label"].to_numpy(dtype=np.int64)

        self.cache = {}

        if self.use_ram_cache:
            print(f"RAM cache: loading {len(self.track_ids)} mel files...")
            for tid in tqdm(self.track_ids, desc="RAM cache"):
                path = resolve_track_mel_path(tid, self.mel_path_map)
                x = np.load(path)
                x = fix_mel_shape(x)
                self.cache[int(tid)] = x

    def __len__(self):
        return len(self.track_ids)

    def __getitem__(self, idx):
        tid = int(self.track_ids[idx])
        y = int(self.labels[idx])

        if self.use_ram_cache and tid in self.cache:
            x = self.cache[tid].copy()
        else:
            path = resolve_track_mel_path(tid, self.mel_path_map)
            x = np.load(path)
            x = fix_mel_shape(x)

        x = crop_or_pad(
            x,
            target_frames=self.target_frames,
            train=self.train,
        )

        x = normalize_mel(x)

        x = torch.from_numpy(x).float().unsqueeze(0)

        if self.train and USE_SPEC_AUGMENT:
            x = apply_spec_augment_tensor(x)

        y = torch.tensor(y, dtype=torch.long)

        return x, y, tid


# ============================================================
# MODEL
# ============================================================

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class LightResBlock(nn.Module):
    def __init__(self, channels, dropout=0.10):
        super().__init__()

        self.conv1 = ConvBNAct(channels, channels)

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
        )

        self.dropout = nn.Dropout2d(dropout)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.dropout(out)
        out = self.conv2(out)

        out = out + residual
        out = self.act(out)

        return out


class FastLightCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.stem = nn.Sequential(
            ConvBNAct(1, 24),
            ConvBNAct(24, 24),
        )

        self.stage1 = nn.Sequential(
            ConvBNAct(24, 48, stride=2),
            LightResBlock(48, dropout=0.06),
        )

        self.stage2 = nn.Sequential(
            ConvBNAct(48, 72, stride=2),
            LightResBlock(72, dropout=0.08),
        )

        self.stage3 = nn.Sequential(
            ConvBNAct(72, 112, stride=2),
            LightResBlock(112, dropout=0.10),
        )

        self.stage4 = nn.Sequential(
            ConvBNAct(112, 160, stride=2),
            LightResBlock(160, dropout=0.12),
        )

        self.head = nn.Sequential(
            nn.Dropout(0.30),
            nn.Linear(160 * 2, 192),
            nn.BatchNorm1d(192),
            nn.SiLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(192, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        avg = F.adaptive_avg_pool2d(x, 1).flatten(1)
        mx = F.adaptive_max_pool2d(x, 1).flatten(1)

        x = torch.cat([avg, mx], dim=1)
        x = self.head(x)

        return x


# ============================================================
# LOSS / SAMPLER
# ============================================================

class FocalCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, gamma=1.5):
        super().__init__()
        self.weight = weight
        self.gamma = float(gamma)

    def forward(self, logits, target):
        ce = F.cross_entropy(
            logits,
            target,
            weight=self.weight,
            reduction="none",
        )

        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce

        return loss.mean()


def make_class_weights(y_train, class_names, device):
    counts = Counter(y_train.tolist())
    num_classes = len(class_names)

    weights = []

    for label in range(num_classes):
        count = counts[label]
        w = 1.0 / math.sqrt(max(count, 1))
        weights.append(w)

    weights = np.asarray(weights, dtype=np.float32)
    weights = weights / weights.mean()

    label_dubstep = class_names.index("Dubstep")
    label_metal = class_names.index("Metal")
    label_country = class_names.index("Country")

    weights[label_dubstep] *= DUBSTEP_EXTRA_WEIGHT
    weights[label_metal] *= METAL_EXTRA_WEIGHT
    weights[label_country] *= COUNTRY_EXTRA_WEIGHT

    weights = weights / weights.mean()

    print()
    print("Class weights:")
    for i, name in enumerate(class_names):
        print(f"{name:10s}: {weights[i]:.4f}")

    return torch.tensor(weights, dtype=torch.float32, device=device)


def make_weighted_sampler(y_train, class_names):
    counts = Counter(y_train.tolist())
    num_classes = len(class_names)

    class_sample_weights = {}

    for label in range(num_classes):
        class_sample_weights[label] = 1.0 / math.sqrt(max(counts[label], 1))

    sample_weights = np.array(
        [
            class_sample_weights[int(y)]
            for y in y_train
        ],
        dtype=np.float64,
    )

    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )

    return sampler


# ============================================================
# TRAIN / EVAL
# ============================================================

def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    train,
):
    model.train(train)

    total_loss = 0.0
    total_n = 0

    all_logits = []
    all_y = []
    all_ids = []

    pbar = tqdm(
        loader,
        desc="train" if train else "eval",
        leave=False,
    )

    for x, y, tids in pbar:
        x = x.to(device)
        y = y.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)

            if train:
                loss.backward()

                if GRAD_CLIP is not None and GRAD_CLIP > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        GRAD_CLIP,
                    )

                optimizer.step()

        bs = x.size(0)

        total_loss += float(loss.detach().cpu()) * bs
        total_n += bs

        all_logits.append(logits.detach().cpu().numpy())
        all_y.append(y.detach().cpu().numpy())
        all_ids.append(np.asarray(tids, dtype=np.int64))

        pbar.set_postfix(loss=total_loss / max(total_n, 1))

    all_logits = np.concatenate(all_logits, axis=0)
    all_y = np.concatenate(all_y, axis=0)
    all_ids = np.concatenate(all_ids, axis=0)

    avg_loss = total_loss / max(total_n, 1)

    return avg_loss, all_logits, all_y, all_ids


def train_one_seed(seed):
    seed_everything(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = get_torch_device()

    model_name = f"{MODEL_PREFIX}_seed{seed}"
    model_dir = OUTDIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 80)
    print(f"TRAIN FAST LIGHT CNN: {model_name}")
    print("=" * 80)
    print("DEVICE:", device)
    print("MODEL_DIR:", model_dir)

    df, genres_df, selected_genre_ids, genre_id_to_label, label_to_genre_id, class_names = build_master_df()

    split = load_final_split()
    check_split_against_df(df, split)

    train_ids = np.array(split["train"], dtype=int)
    valid_ids = np.array(split["valid"], dtype=int)
    test_ids = np.array(split["test"], dtype=int)

    y_train = df.loc[train_ids, "label"].to_numpy(dtype=np.int64)

    print()
    print("Train counts:")
    for label, name in enumerate(class_names):
        c = int((y_train == label).sum())
        print(f"{name:10s}: {c}")

    mel_path_map = build_mel_path_map()

    train_ds = MelDataset(
        df=df,
        track_ids=train_ids,
        mel_path_map=mel_path_map,
        target_frames=TARGET_FRAMES,
        train=True,
        use_ram_cache=USE_RAM_CACHE,
    )

    valid_ds = MelDataset(
        df=df,
        track_ids=valid_ids,
        mel_path_map=mel_path_map,
        target_frames=TARGET_FRAMES,
        train=False,
        use_ram_cache=USE_RAM_CACHE,
    )

    test_ds = MelDataset(
        df=df,
        track_ids=test_ids,
        mel_path_map=mel_path_map,
        target_frames=TARGET_FRAMES,
        train=False,
        use_ram_cache=USE_RAM_CACHE,
    )

    if USE_WEIGHTED_SAMPLER:
        sampler = make_weighted_sampler(
            y_train=y_train,
            class_names=class_names,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=False,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=False,
        )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    model = FastLightCNN(num_classes=len(class_names)).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print()
    print("Params:", f"{num_params:,}")

    if USE_CLASS_WEIGHTS:
        class_weights = make_class_weights(
            y_train=y_train,
            class_names=class_names,
            device=device,
        )
    else:
        class_weights = None

    if USE_FOCAL_LOSS:
        criterion = FocalCrossEntropyLoss(
            weight=class_weights,
            gamma=FOCAL_GAMMA,
        )
    else:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
        eta_min=LR * 0.05,
    )

    best_obj = -1e9
    best_epoch = -1
    bad_epochs = 0

    history = []

    best_ckpt_path = model_dir / f"{model_name}_best.pt"

    for epoch in range(1, NUM_EPOCHS + 1):
        print()
        print("=" * 80)
        print(f"{model_name} | EPOCH {epoch}/{NUM_EPOCHS}")
        print("=" * 80)

        train_loss, train_logits, train_y_epoch, train_ids_epoch = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            train=True,
        )

        valid_loss, valid_logits, valid_y_epoch, valid_ids_epoch = run_epoch(
            model=model,
            loader=valid_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            train=False,
        )

        scheduler.step()

        valid_metrics = metrics_from_logits(
            valid_logits,
            valid_y_epoch,
            list(range(len(class_names))),
        )

        print()
        print(f"train_loss={train_loss:.5f}")
        print(f"valid_loss={valid_loss:.5f}")

        print_metrics(
            f"{model_name} VALID epoch {epoch}",
            valid_logits,
            valid_y_epoch,
            class_names,
        )

        current_obj = float(valid_metrics["objective"])

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                **valid_metrics,
            }
        )

        if current_obj > best_obj:
            best_obj = current_obj
            best_epoch = epoch
            bad_epochs = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "best_obj": best_obj,
                    "class_names": class_names,
                    "settings": {
                        "target_frames": TARGET_FRAMES,
                        "batch_size": BATCH_SIZE,
                        "lr": LR,
                        "weight_decay": WEIGHT_DECAY,
                        "use_weighted_sampler": USE_WEIGHTED_SAMPLER,
                        "use_class_weights": USE_CLASS_WEIGHTS,
                        "use_focal_loss": USE_FOCAL_LOSS,
                    },
                },
                best_ckpt_path,
            )

            print()
            print(f"BEST SAVED: epoch={epoch}, objective={best_obj:.5f}")
        else:
            bad_epochs += 1
            print()
            print(f"No improvement: {bad_epochs}/{PATIENCE}")

        if bad_epochs >= PATIENCE:
            print()
            print(f"Early stopping at epoch {epoch}")
            break

    print()
    print("=" * 80)
    print("LOAD BEST CHECKPOINT AND EXPORT LOGITS")
    print("=" * 80)
    print("best_epoch:", best_epoch)
    print("best_obj:", best_obj)

    ckpt = torch.load(
        best_ckpt_path,
        map_location=device,
    )

    model.load_state_dict(ckpt["model_state_dict"])

    valid_loss, valid_logits, valid_y_final, valid_ids_final = run_epoch(
        model=model,
        loader=valid_loader,
        criterion=criterion,
        optimizer=None,
        device=device,
        train=False,
    )

    test_loss, test_logits, test_y_final, test_ids_final = run_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        optimizer=None,
        device=device,
        train=False,
    )

    valid_metrics = print_metrics(
        f"{model_name} FINAL VALIDATION",
        valid_logits,
        valid_y_final,
        class_names,
    )

    test_metrics = print_metrics(
        f"{model_name} FINAL TEST",
        test_logits,
        test_y_final,
        class_names,
    )

    np.save(model_dir / "valid_logits.npy", valid_logits)
    np.save(model_dir / "valid_y.npy", valid_y_final)
    np.save(model_dir / "valid_track_ids.npy", valid_ids_final)

    np.save(model_dir / "test_logits.npy", test_logits)
    np.save(model_dir / "test_y.npy", test_y_final)
    np.save(model_dir / "test_track_ids.npy", test_ids_final)

    save_predictions_csv(
        valid_ids_final,
        valid_y_final,
        valid_logits,
        class_names,
        model_dir / "valid_predictions.csv",
    )

    save_predictions_csv(
        test_ids_final,
        test_y_final,
        test_logits,
        class_names,
        model_dir / "test_predictions.csv",
    )

    save_json(
        {
            "model_name": model_name,
            "model_dir": str(model_dir),
            "best_epoch": best_epoch,
            "best_valid_objective": best_obj,
            "valid_metrics": valid_metrics,
            "test_metrics": test_metrics,
            "settings": {
                "seed": seed,
                "target_frames": TARGET_FRAMES,
                "batch_size": BATCH_SIZE,
                "num_epochs": NUM_EPOCHS,
                "patience": PATIENCE,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "use_ram_cache": USE_RAM_CACHE,
                "use_weighted_sampler": USE_WEIGHTED_SAMPLER,
                "use_class_weights": USE_CLASS_WEIGHTS,
                "use_focal_loss": USE_FOCAL_LOSS,
                "use_spec_augment": USE_SPEC_AUGMENT,
            },
            "history": history,
        },
        model_dir / "summary.json",
    )

    print()
    print("=" * 80)
    print(f"DONE: {model_name}")
    print("=" * 80)
    print("Saved to:", model_dir)
    print("TEST accuracy:", test_metrics["accuracy_percent"])
    print("TEST objective:", test_metrics["objective"])

    return model_name, valid_metrics, test_metrics


# ============================================================
# MAIN
# ============================================================

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 80)
    print("TRAIN FAST LIGHT CNN")
    print("=" * 80)
    print("SEEDS:", SEEDS)
    print("TARGET_FRAMES:", TARGET_FRAMES)
    print("BATCH_SIZE:", BATCH_SIZE)
    print("NUM_EPOCHS:", NUM_EPOCHS)
    print("PATIENCE:", PATIENCE)

    all_results = []

    for seed in SEEDS:
        model_name, valid_metrics, test_metrics = train_one_seed(seed)

        all_results.append(
            {
                "model_name": model_name,
                "valid_metrics": valid_metrics,
                "test_metrics": test_metrics,
            }
        )

    save_json(
        {
            "models": all_results,
            "seeds": SEEDS,
        },
        OUTDIR / "light_fast_cnn_training_summary.json",
    )

    print()
    print("=" * 80)
    print("ALL FAST LIGHT CNNs DONE")
    print("=" * 80)

    for item in all_results:
        name = item["model_name"]
        tm = item["test_metrics"]

        print()
        print(name)
        print("accuracy_percent:", tm["accuracy_percent"])
        print("objective:", tm["objective"])
        print("min_recall:", tm["min_recall"])


if __name__ == "__main__":
    main()