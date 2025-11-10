# data_preprocess.py
# Wstępne przetwarzanie danych UCI Beijing (MLP + LSTM) z konfiguracją YAML

from __future__ import annotations
import os
import glob
import yaml
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf


# =========================
# --- Klasa konfiguracji ---
# =========================
@dataclass
class PreprocessConfig:
    data_path: str
    target_col: str = "pm2.5"
    lookback: int = 24
    horizon: int = 1
    test_size: float = 0.15
    val_size: float = 0.15
    scaler_type: str = "standard"
    include_lags: List[int] = None
    include_rollings: Dict[str, List[int]] = None
    random_state: int = 42
    drop_na_threshold: float = 0.4

    def __post_init__(self):
        if self.include_lags is None:
            self.include_lags = [1, 3, 6, 12, 24]
        if self.include_rollings is None:
            self.include_rollings = {"mean": [3, 6, 12, 24], "std": [6, 24]}


# =========================
# --- Funkcje pomocnicze ---
# =========================
def load_yaml_config(path: str = "config.yaml") -> PreprocessConfig:
    """Wczytuje konfigurację z pliku YAML i tworzy obiekt PreprocessConfig."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Nie znaleziono pliku konfiguracyjnego: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    cfg = PreprocessConfig(
        data_path=config["data"]["path"],
        target_col=config["data"].get("target_col", "pm2.5"),
        lookback=config["model"].get("lookback", 24),
        horizon=config["model"].get("horizon", 1),
        test_size=config["model"].get("test_size", 0.15),
        val_size=config["model"].get("val_size", 0.15),
        scaler_type=config["model"].get("scaler_type", "standard"),
        include_lags=config["features"].get("include_lags", [1, 3, 6, 12, 24]),
        include_rollings=config["features"].get(
            "include_rollings", {"mean": [3, 6, 12, 24]}
        ),
        random_state=config["runtime"].get("random_state", 42),
    )
    return cfg


# =========================
# --- Wczytanie danych ---
# =========================
def load_beijing_data(path: str) -> pd.DataFrame:
    """Wczytuje dane z pliku lub katalogu CSV."""
    def _read_one(fp: str) -> pd.DataFrame:
        df = pd.read_csv(fp)
        df.columns = [c.strip().lower() for c in df.columns]

        # Tworzymy kolumnę datetime
        if {"year", "month", "day", "hour"}.issubset(df.columns):
            dt = pd.to_datetime(df[["year", "month", "day", "hour"]])
        elif "date" in df.columns:
            dt = pd.to_datetime(df["date"])
        elif "datetime" in df.columns:
            dt = pd.to_datetime(df["datetime"])
        else:
            dt = pd.to_datetime(df.iloc[:, 0], errors="coerce")

        df.insert(0, "datetime", dt)
        df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

        # Ujednolicenie nazwy celu
        rename_candidates = {
            "pm2.5": "pm2.5",
            "pm2_5": "pm2.5",
            "pm25": "pm2.5",
            "pm2-5": "pm2.5",
        }
        for old, new in rename_candidates.items():
            if old in df.columns and new not in df.columns:
                df = df.rename(columns={old: new})

        return df

    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.csv")))
        if not files:
            raise FileNotFoundError(f"Brak plików CSV w katalogu: {path}")
        frames = [_read_one(fp) for fp in files]
        data = pd.concat(frames, ignore_index=True)
    else:
        data = _read_one(path)

    data = data.set_index("datetime").sort_index()
    data = data[~data.index.duplicated(keep="last")]
    return data


# =========================
# --- Czyszczenie danych ---
# =========================
def clean_impute(df: pd.DataFrame, cfg: PreprocessConfig) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype.kind in "iuf":
            df.loc[df[col] < -50, col] = np.nan

    na_share = df.isna().mean()
    to_drop = na_share[na_share > cfg.drop_na_threshold].index.tolist()
    if to_drop:
        df = df.drop(columns=to_drop)

    df = df.sort_index().ffill().bfill()

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype("category")

    df = pd.get_dummies(df, columns=df.select_dtypes(include=["category"]).columns, drop_first=False)
    return df


# =========================
# --- Cechy czasowe ---
# =========================
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    idx = df.index
    df["hour"] = idx.hour
    df["dayofweek"] = idx.dayofweek
    df["month"] = idx.month
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)
    return df


def add_lag_rolling_features(df: pd.DataFrame, target: str, lags: List[int], rollings: Dict[str, List[int]]) -> pd.DataFrame:
    df = df.copy()
    for l in lags:
        df[f"{target}_lag{l}h"] = df[target].shift(l)

    for stat, wins in rollings.items():
        for w in wins:
            if stat == "mean":
                df[f"{target}_roll{w}h_mean"] = df[target].rolling(w, min_periods=max(1, w//2)).mean()
            elif stat == "std":
                df[f"{target}_roll{w}h_std"] = df[target].rolling(w, min_periods=max(1, w//2)).std()
    return df


# =========================
# --- Podział danych ---
# =========================
def time_based_split(df: pd.DataFrame, test_size: float, val_size: float):
    n = len(df)
    n_test = int(np.floor(n * test_size))
    n_trainval = n - n_test
    n_val = int(np.floor(n_trainval * val_size))
    n_train = n_trainval - n_val
    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train:n_train + n_val].copy()
    test = df.iloc[n_train + n_val:].copy()
    return train, val, test


# =========================
# --- Skalowanie ---
# =========================
def get_scaler(scaler_type: str):
    return MinMaxScaler() if scaler_type == "minmax" else StandardScaler()


def scale_splits(train, val, test, target, scaler_type="standard"):
    X_cols = [c for c in train.columns if c != target]
    y_col = target

    x_scaler = get_scaler(scaler_type)
    y_scaler = get_scaler(scaler_type)

    X_train = x_scaler.fit_transform(train[X_cols])
    y_train = y_scaler.fit_transform(train[[y_col]])
    X_val = x_scaler.transform(val[X_cols])
    y_val = y_scaler.transform(val[[y_col]])
    X_test = x_scaler.transform(test[X_cols])
    y_test = y_scaler.transform(test[[y_col]])

    return X_train, y_train, X_val, y_val, X_test, y_test, X_cols, y_col, x_scaler, y_scaler


# =========================
# --- Dataset dla LSTM ---
# =========================
def windowed_dataset(features, target, lookback, horizon, batch_size=128, shuffle=True, seed=42):
    X, y = [], []
    for end in range(lookback, len(features) - horizon + 1):
        start = end - lookback
        X.append(features[start:end])
        y.append(target[end + horizon - 1])
    X, y = np.stack(X), np.stack(y)
    ds = tf.data.Dataset.from_tensor_slices((X.astype(np.float32), y.astype(np.float32)))
    if shuffle:
        ds = ds.shuffle(min(len(X), 10_000), seed=seed)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# =========================
# --- Dataset dla MLP ---
# =========================
def make_mlp_datasets(cfg: PreprocessConfig):
    df = load_beijing_data(cfg.data_path)
    if cfg.target_col not in df.columns:
        raise KeyError(f"Nie znaleziono kolumny celu: {cfg.target_col}")

    df = clean_impute(df, cfg)
    df = add_time_features(df)
    df = add_lag_rolling_features(df, cfg.target_col, cfg.include_lags, cfg.include_rollings)
    df = df.dropna()

    train, val, test = time_based_split(df, cfg.test_size, cfg.val_size)
    X_train, y_train, X_val, y_val, X_test, y_test, X_cols, y_col, x_scaler, y_scaler = scale_splits(
        train, val, test, cfg.target_col, cfg.scaler_type
    )

    def to_ds(X, y, shuffle=False):
        ds = tf.data.Dataset.from_tensor_slices((X.astype(np.float32), y.astype(np.float32)))
        if shuffle:
            ds = ds.shuffle(min(len(X), 10_000), seed=cfg.random_state)
        return ds.batch(256).prefetch(tf.data.AUTOTUNE)

    return to_ds(X_train, y_train, True), to_ds(X_val, y_val), to_ds(X_test, y_test)


def make_lstm_datasets(cfg: PreprocessConfig):
    df = load_beijing_data(cfg.data_path)
    if cfg.target_col not in df.columns:
        raise KeyError(f"Nie znaleziono kolumny celu: {cfg.target_col}")

    df = clean_impute(df, cfg)
    df = add_time_features(df)
    train, val, test = time_based_split(df, cfg.test_size, cfg.val_size)

    X_cols = [c for c in df.columns if c != cfg.target_col]
    x_scaler, y_scaler = get_scaler(cfg.scaler_type), get_scaler(cfg.scaler_type)

    X_train_scaled = x_scaler.fit_transform(train[X_cols])
    y_train_scaled = y_scaler.fit_transform(train[[cfg.target_col]])
    X_val_scaled = x_scaler.transform(val[X_cols])
    y_val_scaled = y_scaler.transform(val[[cfg.target_col]])
    X_test_scaled = x_scaler.transform(test[X_cols])
    y_test_scaled = y_scaler.transform(test[[cfg.target_col]])

    ds_train = windowed_dataset(X_train_scaled, y_train_scaled, cfg.lookback, cfg.horizon)
    ds_val = windowed_dataset(X_val_scaled, y_val_scaled, cfg.lookback, cfg.horizon, shuffle=False)
    ds_test = windowed_dataset(X_test_scaled, y_test_scaled, cfg.lookback, cfg.horizon, shuffle=False)

    return ds_train, ds_val, ds_test


# =========================
# --- Main ---
# =========================
if __name__ == "__main__":
    cfg = load_yaml_config()

    print("Wczytano konfigurację z YAML:")
    print(f"Ścieżka do danych: {cfg.data_path}")
    print(f"Kolumna celu: {cfg.target_col}")
    print(f"Lookback: {cfg.lookback}, Horizon: {cfg.horizon}")

    # Tworzymy dataset dla MLP
    ds_train, ds_val, ds_test = make_mlp_datasets(cfg)
    print("Dane MLP przygotowane.")

    # Tworzymy dataset dla LSTM
    ds_train_lstm, ds_val_lstm, ds_test_lstm = make_lstm_datasets(cfg)
    print("Dane LSTM przygotowane.")
