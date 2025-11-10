#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd

# === Config ===
SOURCE_PATH = "/Users/arielamishaancohen/Downloads/asl-fingerspelling/supplemental_landmarks/1579345709.parquet"
FRAME_LEN = 128  # ajusta si tu modelo espera otro largo

# Columnas a ignorar para construir la matriz de features
IGNORE_COLS = {"sequence_id", "frame"}

def feature_columns(df: pd.DataFrame):
    cols = [c for c in df.columns if c not in IGNORE_COLS]
    return sorted(cols)  # orden estable y reproducible

def prepare_matrix(seq_df: pd.DataFrame, max_frames: int) -> np.ndarray:
    """Convierte la secuencia a matriz (T,F) y aplica pad/truncate a max_frames."""
    cols = feature_columns(seq_df)
    X = (seq_df[cols]
         .astype(np.float32)
         .replace([np.inf, -np.inf], np.nan)
         .fillna(0.0)
         .to_numpy())
    T, F = X.shape
    if T >= max_frames:
        X = X[:max_frames, :]
    else:
        pad = np.zeros((max_frames - T, F), dtype=np.float32)
        X = np.vstack([X, pad])
    return X

def main():
    # CWD = carpeta que tienes abierta en VS Code (si ejecutas el script desde ahí)
    cwd = os.getcwd()
    print(f"📂 Directorio de trabajo: {cwd}")

    # 1) Cargar parquet
    print(f"📥 Cargando parquet: {SOURCE_PATH}")
    df = pd.read_parquet(SOURCE_PATH).reset_index()
    print(df.columns)
    if "sequence_id" not in df.columns or "frame" not in df.columns:
        raise ValueError("El parquet debe contener las columnas 'sequence_id' y 'frame'.")

    # 2) Tomar la primera sequence_id disponible
    seq_ids = df["sequence_id"].dropna().unique().tolist()
    if not seq_ids:
        raise ValueError("No se encontraron sequence_id en el parquet.")
    first_seq_id = seq_ids[0]
    print(f"🔎 Primera sequence_id detectada: {first_seq_id}")

    # 3) Filtrar y ordenar por frame
    seq_df = df[df["sequence_id"] == first_seq_id].sort_values("frame").reset_index(drop=True)

    # 4) Guardar crudo (CSV)
    raw_parquet_path = os.path.join(cwd, "first_sequence_raw.parquet")
    seq_df.to_parquet(raw_parquet_path, index=False)
    print(f"✅ Guardado crudo: {raw_parquet_path} (filas={len(seq_df)}, columnas={len(seq_df.columns)})")

    # 5) Construir y guardar matriz (NPY) lista para el modelo
    X = prepare_matrix(seq_df, max_frames=FRAME_LEN)
    npy_path = os.path.join(cwd, "first_sequence_matrix.npy")
    np.save(npy_path, X)
    print(f"✅ Guardado matriz: {npy_path} (shape={X.shape})")

if __name__ == "__main__":
    main()
