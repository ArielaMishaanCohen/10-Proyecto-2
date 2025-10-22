import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Funciones
def load_data(parquet_path):
    return pd.read_parquet(parquet_path)

def prepare_data(df):
    X = df.drop(columns=['sequence_id', 'frame', 'landmark_type'], errors='ignore').fillna(0).values
    y = np.zeros(len(X))  # dummy si no hay etiquetas
    X = X.reshape(-1, 1, X.shape[-1])  # (samples, timesteps, features)
    return X, y

def build_bilstm(input_shape, num_classes=26):
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=False, dropout=0.3), input_shape=input_shape),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

DATA_PATH = "/kaggle/input/asl-fingerspelling/train_landmarks/"
files = [os.path.join(DATA_PATH, f) for f in os.listdir(DATA_PATH) if f.endswith('.parquet')]

results = []

for f in files:
    print(f"\n=== Procesando {f} ===")
    df = load_data(f)
    X, y = prepare_data(df)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_bilstm(X.shape[1:], num_classes=26)
    es = EarlyStopping(patience=3, restore_best_weights=True)

    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=10, batch_size=32,
                        callbacks=[es], verbose=1)

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    results.append({'file': f, 'val_loss': val_loss, 'val_acc': val_acc})


# Resultados
results_df = pd.DataFrame(results)
print(results_df)