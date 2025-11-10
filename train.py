# train.py
# Orkiestracja: wczytanie config.yaml, cache preprocessingu, trening MLP i LSTM
# Autor: ChatGPT (Cyber-fiz)

import os
import json
import numpy as np
import tensorflow as tf
from datetime import datetime

from models import build_mlp, build_lstm
from data_preprocess import load_yaml_config, make_mlp_datasets, make_lstm_datasets

# ----------------- UTIL -----------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def cfg_hash_str(cfg) -> str:
    """Lekki klucz cache oparty o istotne parametry preprocessingu."""
    key = {
        "data_path": cfg.data_path,
        "target_col": cfg.target_col,
        "lookback": cfg.lookback,
        "horizon": cfg.horizon,
        "test_size": cfg.test_size,
        "val_size": cfg.val_size,
        "scaler": cfg.scaler_type,
        "lags": cfg.include_lags,
        "roll": cfg.include_rollings,
        "rand": cfg.random_state,
    }
    s = json.dumps(key, sort_keys=True)
    return str(abs(hash(s)))

def dataset_to_arrays(ds: tf.data.Dataset):
    """Materializuje tf.data.Dataset do (X, y) jako numpy (uwaga na RAM przy bardzo dużych danych)."""
    X_list, y_list = [], []
    for Xb, yb in ds.unbatch():
        X_list.append(Xb.numpy())
        y_list.append(yb.numpy())
    X = np.stack(X_list)
    y = np.stack(y_list)
    return X, y

def save_npz(path: str, **arrays):
    ensure_dir(os.path.dirname(path))
    np.savez_compressed(path, **arrays)

def load_npz(path: str):
    with np.load(path) as data:
        return {k: data[k] for k in data.files}

def arrays_to_dataset(X: np.ndarray, y: np.ndarray, batch: int, shuffle: bool = False, seed: int = 42):
    ds = tf.data.Dataset.from_tensor_slices((X.astype(np.float32), y.astype(np.float32)))
    if shuffle:
        ds = ds.shuffle(min(len(X), 10_000), seed=seed)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)

def common_callbacks(out_dir: str, model_name: str):
    ensure_dir(out_dir)
    # ✅ zapisujemy TYLKO wagi do H5, bez natywnego formatu .keras
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(out_dir, f"{model_name}.best.weights.h5"),
        monitor="val_rmse",
        mode="min",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )
    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_rmse", mode="min", patience=10, restore_best_weights=True, verbose=1
    )
    reduce = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_rmse", mode="min", factor=0.5, patience=5, min_lr=1e-5, verbose=1
    )
    return [ckpt, early, reduce]

# ----------------- MAIN -----------------

def main():
    # 1) Wczytaj konfigurację YAML -> PreprocessConfig
    cfg = load_yaml_config()

    # 2) Katalogi wyników i cache
    runs_dir = os.path.join("runs", datetime.now().strftime("%Y%m%d_%H%M%S"))
    models_dir = os.path.join(runs_dir, "models")
    cache_dir = os.path.join(".cache")
    ensure_dir(runs_dir); ensure_dir(models_dir); ensure_dir(cache_dir)

    # Ustawienia runtime (jeśli nie ma w cfg, użyj domyślnych)
    batch_size = int(getattr(cfg, "batch_size", 128))
    epochs = int(getattr(cfg, "epochs", 20))

    cache_key = cfg_hash_str(cfg)

    mlp_cache = {
        "train": os.path.join(cache_dir, f"mlp_{cache_key}_train.npz"),
        "val":   os.path.join(cache_dir, f"mlp_{cache_key}_val.npz"),
        "test":  os.path.join(cache_dir, f"mlp_{cache_key}_test.npz"),
        "meta":  os.path.join(cache_dir, f"mlp_{cache_key}_meta.json"),
    }
    lstm_cache = {
        "train": os.path.join(cache_dir, f"lstm_{cache_key}_train.npz"),
        "val":   os.path.join(cache_dir, f"lstm_{cache_key}_val.npz"),
        "test":  os.path.join(cache_dir, f"lstm_{cache_key}_test.npz"),
        "meta":  os.path.join(cache_dir, f"lstm_{cache_key}_meta.json"),
    }

    # 3) Dane MLP: z cache albo generuj i zapisz
    if all(os.path.exists(p) for p in mlp_cache.values()):
        print("✅ Ładuję cache MLP...")
        tr = load_npz(mlp_cache["train"]); va = load_npz(mlp_cache["val"]); te = load_npz(mlp_cache["test"])
        with open(mlp_cache["meta"], "r", encoding="utf-8") as f:
            meta_mlp = json.load(f)
        mlp_input_dim = int(meta_mlp["input_dim"])
        ds_tr_mlp = arrays_to_dataset(tr["X"], tr["y"], batch=max(256, batch_size), shuffle=True, seed=cfg.random_state)
        ds_va_mlp = arrays_to_dataset(va["X"], va["y"], batch=max(256, batch_size), shuffle=False)
        ds_te_mlp = arrays_to_dataset(te["X"], te["y"], batch=max(256, batch_size), shuffle=False)
    else:
        print("⚙️ Generuję dane MLP (pierwszy raz) i zapisuję cache...")
        ds_tr_mlp, ds_va_mlp, ds_te_mlp = make_mlp_datasets(cfg)
        Xtr, ytr = dataset_to_arrays(ds_tr_mlp)
        Xva, yva = dataset_to_arrays(ds_va_mlp)
        Xte, yte = dataset_to_arrays(ds_te_mlp)
        save_npz(mlp_cache["train"], X=Xtr, y=ytr)
        save_npz(mlp_cache["val"],   X=Xva, y=yva)
        save_npz(mlp_cache["test"],  X=Xte, y=yte)
        meta_mlp = {"input_dim": int(Xtr.shape[1])}
        with open(mlp_cache["meta"], "w", encoding="utf-8") as f:
            json.dump(meta_mlp, f)
        mlp_input_dim = Xtr.shape[1]
        # re-kreacja datasetów z cache (spójność kolejnych uruchomień)
        ds_tr_mlp = arrays_to_dataset(Xtr, ytr, batch=max(256, batch_size), shuffle=True, seed=cfg.random_state)
        ds_va_mlp = arrays_to_dataset(Xva, yva, batch=max(256, batch_size), shuffle=False)
        ds_te_mlp = arrays_to_dataset(Xte, yte, batch=max(256, batch_size), shuffle=False)

    # 4) Dane LSTM: z cache albo generuj i zapisz
    if all(os.path.exists(p) for p in lstm_cache.values()):
        print("✅ Ładuję cache LSTM...")
        tr = load_npz(lstm_cache["train"]); va = load_npz(lstm_cache["val"]); te = load_npz(lstm_cache["test"])
        with open(lstm_cache["meta"], "r", encoding="utf-8") as f:
            meta_lstm = json.load(f)
        lookback = int(meta_lstm["lookback"]); n_feats = int(meta_lstm["n_features"])
        ds_tr_lstm = arrays_to_dataset(tr["X"], tr["y"], batch=batch_size, shuffle=True, seed=cfg.random_state)
        ds_va_lstm = arrays_to_dataset(va["X"], va["y"], batch=batch_size, shuffle=False)
        ds_te_lstm = arrays_to_dataset(te["X"], te["y"], batch=batch_size, shuffle=False)
    else:
        print("⚙️ Generuję dane LSTM (pierwszy raz) i zapisuję cache...")
        ds_tr_lstm, ds_va_lstm, ds_te_lstm = make_lstm_datasets(cfg)
        Xtr, ytr = dataset_to_arrays(ds_tr_lstm)
        Xva, yva = dataset_to_arrays(ds_va_lstm)
        Xte, yte = dataset_to_arrays(ds_te_lstm)
        save_npz(lstm_cache["train"], X=Xtr, y=ytr)
        save_npz(lstm_cache["val"],   X=Xva, y=yva)
        save_npz(lstm_cache["test"],  X=Xte, y=yte)
        lookback, n_feats = int(Xtr.shape[1]), int(Xtr.shape[2])
        with open(lstm_cache["meta"], "w", encoding="utf-8") as f:
            json.dump({"lookback": lookback, "n_features": n_feats}, f)
        # re-kreacja datasetów z cache
        ds_tr_lstm = arrays_to_dataset(Xtr, ytr, batch=batch_size, shuffle=True, seed=cfg.random_state)
        ds_va_lstm = arrays_to_dataset(Xva, yva, batch=batch_size, shuffle=False)
        ds_te_lstm = arrays_to_dataset(Xte, yte, batch=batch_size, shuffle=False)

    # 5) Trening MLP
    mlp = build_mlp(mlp_input_dim)
    print(mlp.summary())
    cb = common_callbacks(models_dir, "mlp_pm25")
    hist_mlp = mlp.fit(
        ds_tr_mlp,
        validation_data=ds_va_mlp,
        epochs=epochs,
        callbacks=cb,
        verbose=1,
    )
    mlp_eval = mlp.evaluate(ds_te_mlp, verbose=0)
    print(f"MLP test -> loss={mlp_eval[0]:.4f}, mae={mlp_eval[1]:.4f}, rmse={mlp_eval[2]:.4f}")

    # ✅ Zapis finalnego modelu w SavedModel (katalog)
    mlp.save(os.path.join(models_dir, "mlp_pm25.final"))  # utworzy folder SavedModel

    # 6) Trening LSTM
    lstm = build_lstm(lookback=lookback, n_features=n_feats)
    print(lstm.summary())
    cb = common_callbacks(models_dir, "lstm_pm25")
    hist_lstm = lstm.fit(
        ds_tr_lstm,
        validation_data=ds_va_lstm,
        epochs=epochs,
        callbacks=cb,
        verbose=1,
    )
    lstm_eval = lstm.evaluate(ds_te_lstm, verbose=0)
    print(f"LSTM test -> loss={lstm_eval[0]:.4f}, mae={lstm_eval[1]:.4f}, rmse={lstm_eval[2]:.4f}")

    # ✅ Zapis finalnego modelu w SavedModel (katalog)
    lstm.save(os.path.join(models_dir, "lstm_pm25.final"))

    # 7) Zapis wyników
    results = {
        "mlp": {"loss": float(mlp_eval[0]), "mae": float(mlp_eval[1]), "rmse": float(mlp_eval[2])},
        "lstm": {"loss": float(lstm_eval[0]), "mae": float(lstm_eval[1]), "rmse": float(lstm_eval[2])},
        "cache_key": cfg_hash_str(cfg),
        "batch_size": batch_size,
        "epochs": epochs,
    }
    with open(os.path.join(runs_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("✅ Zakończono. Wyniki i modele zapisane w:", runs_dir)

if __name__ == "__main__":
    main()
