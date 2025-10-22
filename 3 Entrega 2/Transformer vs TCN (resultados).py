# -*- coding: utf-8 -*-
"""
Comparación de Modelos: Transformer vs TCN para ASL Fingerspelling
Incluye métricas detalladas, visualizaciones y análisis comparativo
"""

import os
import json
import math
import random
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# ==== Reporter utilities (injected) ====
import sys, csv, datetime
from contextlib import contextmanager

class _Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

class Reporter:
    def __init__(self, run_dir="./asl_results", run_name=None):
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = os.path.abspath(run_dir)
        os.makedirs(self.run_dir, exist_ok=True)
        self.run_name = run_name or f"run-{ts}"
        self.run_path = os.path.join(self.run_dir, self.run_name)
        os.makedirs(self.run_path, exist_ok=True)

        self.log_path = os.path.join(self.run_path, "console.log")
        self.metrics_path = os.path.join(self.run_path, "metrics.jsonl")
        self.summary_path = os.path.join(self.run_path, "summary.json")
        self.pred_csv_path = os.path.join(self.run_path, "predictions.csv")

        self._orig_stdout = sys.stdout
        self._log_file = open(self.log_path, "a", encoding="utf-8")
        sys.stdout = _Tee(sys.stdout, self._log_file)

        self._metrics = {}
        print(f"📁 Guardando salida en: {self.run_path}")

    def log(self, msg=""):
        print(msg)

    def add_metrics(self, group, **metrics):
        if group not in self._metrics:
            self._metrics[group] = {}
        self._metrics[group].update(metrics)
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            rec = {"group": group, **metrics, "time": datetime.datetime.now().isoformat()}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def save_predictions(self, rows, fieldnames=("modelo","real","pred")):
        header_needed = not os.path.exists(self.pred_csv_path)
        with open(self.pred_csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if header_needed:
                w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})

    def save_summary(self):
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(self._metrics, f, indent=2, ensure_ascii=False)

    def save_config(self, cfg_obj=None, extra: dict = None):
        """Guarda un snapshot de configuración para la corrida."""
        snap = {}
        if cfg_obj is not None:
            try:
                import dataclasses, json as _json
                if dataclasses.is_dataclass(cfg_obj):
                    snap.update(dataclasses.asdict(cfg_obj))
                else:
                    for k in dir(cfg_obj):
                        if k.startswith('_'): continue
                        try:
                            v = getattr(cfg_obj, k)
                        except Exception:
                            continue
                        if callable(v): continue
                        try:
                            _json.dumps(v); snap[k] = v
                        except Exception:
                            snap[k] = str(v)
            except Exception as _e:
                print(f"[Reporter] No se pudo serializar cfg_obj: {_e}")
        if extra: snap.update(extra)
        snap['run_name'] = self.run_name
        snap['saved_at'] = datetime.datetime.now().isoformat()
        cfg_path = os.path.join(self.run_path, "config.json")
        try:
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(snap, f, indent=2, ensure_ascii=False)
        except Exception as _e:
            print(f"[Reporter] No se pudo guardar config.json: {_e}")
        return snap

    def close(self):
        self.save_summary()
        sys.stdout = self._orig_stdout
        try:
            self._log_file.close()
        except Exception:
            pass

@contextmanager
def reporting(run_dir="./asl_results", run_name=None):
    r = Reporter(run_dir, run_name)
    try:
        yield r
    finally:
        r.close()

# global reporter hook
GLOBAL_REPORTER = None
def _set_global_reporter(r):  # called from __main__
    global GLOBAL_REPORTER
    GLOBAL_REPORTER = r
# ==== End Reporter utilities ====

# ==== Text cleaning & normalization helpers (injected) ====
import unicodedata, re as _re, numpy as _np

_ALLOWED_CHARS = list("abcdefghijklmnopqrstuvwxyz0123456789- '")
def clean_text(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode("ascii")
    s = s.lower()
    s = _re.sub(r"[^a-z0-9\- ']", "", s)
    s = _re.sub(r"\s+", " ", s).strip()
    return s

def normalize_landmarks(arr, eps=1e-6):
    try:
        m = arr.mean(axis=0, keepdims=True)
        s = arr.std(axis=0, keepdims=True)
        s = _np.where(s < eps, eps, s)
        return (arr - m) / s
    except Exception:
        return arr
# ==== End helpers ====

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

@dataclass
class CFG:
    TRAIN_CSV: str = '/Users/arielamishaancohen/Downloads/asl-fingerspelling/supplemental_metadata.csv'
    PARQUET_ROOT: str = '/Users/arielamishaancohen/Downloads/asl-fingerspelling/supplemental_landmarks'
    CHAR_MAP_JSON: str = '/Users/arielamishaancohen/Downloads/asl-fingerspelling/character_to_prediction_index.json'
    OUTPUT_DIR: str = './asl_results'
    RANDOM_STATE: int = 42
    SAMPLE_SEQUENCES: int = 1000
    FRAME_LEN: int = 128
    BATCH_SIZE: int = 8
    EPOCHS: int = 50
    LR: float = 0.01
    DROPOUT: float = 0
    NUM_HID: int = 64
    NUM_HEAD: int = 2
    NUM_FEED_FORWARD: int = 128
    NUM_LAYERS_ENC: int = 2
    NUM_LAYERS_DEC: int = 1
    TARGET_MAXLEN: int = 64
    TCN_FILTERS: int = 64
    TCN_KERNEL_SIZE: int = 3
    TCN_DILATIONS: List[int] = None

CFG = CFG()
CFG.TCN_DILATIONS = [1, 2, 4, 8]
os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)

random.seed(CFG.RANDOM_STATE)
tf.random.set_seed(CFG.RANDOM_STATE)
np.random.seed(CFG.RANDOM_STATE)

LPOSE = [13, 15, 17, 19, 21]
RPOSE = [14, 16, 18, 20, 22]
POSE = LPOSE + RPOSE

X = [f'x_right_hand_{i}' for i in range(21)] + [f'x_left_hand_{i}' for i in range(21)] + [f'x_pose_{i}' for i in POSE]
Y = [f'y_right_hand_{i}' for i in range(21)] + [f'y_left_hand_{i}' for i in range(21)] + [f'y_pose_{i}' for i in POSE]
Z = [f'z_right_hand_{i}' for i in range(21)] + [f'z_left_hand_{i}' for i in range(21)] + [f'z_pose_{i}' for i in POSE]

SEL_COLS = X + Y + Z

RHAND_IDX = [i for i, col in enumerate(SEL_COLS) if "right" in col]
LHAND_IDX = [i for i, col in enumerate(SEL_COLS) if "left" in col]
RPOSE_IDX = [i for i, col in enumerate(SEL_COLS) if "pose" in col and int(col.split('_')[-1]) in RPOSE]
LPOSE_IDX = [i for i, col in enumerate(SEL_COLS) if "pose" in col and int(col.split('_')[-1]) in LPOSE]

PAD_TOKEN = 'P'
START_TOKEN = 'S'
END_TOKEN = 'E'

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

class MetricsTracker:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.predictions = []
        self.ground_truths = []
        self.losses = []
        self.times = []
    
    def add_batch(self, preds, truths, loss=None, time_taken=None):
        self.predictions.extend(preds)
        self.ground_truths.extend(truths)
        if loss is not None:
            self.losses.append(loss)
        if time_taken is not None:
            self.times.append(time_taken)
    
    def calculate_cer(self):
        total_chars = 0
        total_errors = 0
        for pred, truth in zip(self.predictions, self.ground_truths):
            pred = pred.strip()
            truth = truth.strip()
            total_chars += len(truth)
            total_errors += levenshtein_distance(pred, truth)
        return (total_errors / total_chars * 100) if total_chars > 0 else 100.0
    
    def calculate_wer(self):
        total_words = 0
        total_errors = 0
        for pred, truth in zip(self.predictions, self.ground_truths):
            pred_words = pred.strip().split()
            truth_words = truth.strip().split()
            total_words += len(truth_words)
            total_errors += levenshtein_distance(' '.join(pred_words), ' '.join(truth_words))
        return (total_errors / total_words * 100) if total_words > 0 else 100.0
    
    def calculate_accuracy(self):
        correct = sum(1 for p, t in zip(self.predictions, self.ground_truths) if p.strip() == t.strip())
        return (correct / len(self.predictions) * 100) if self.predictions else 0.0
    
    def get_summary(self):
        return {
            'cer': self.calculate_cer(),
            'wer': self.calculate_wer(),
            'accuracy': self.calculate_accuracy(),
            'avg_loss': np.mean(self.losses) if self.losses else 0.0,
            'avg_time': np.mean(self.times) if self.times else 0.0,
            'total_samples': len(self.predictions)
        }

def resize_pad(x):
    if tf.shape(x)[0] < CFG.FRAME_LEN:
        x = tf.pad(x, ([[0, CFG.FRAME_LEN - tf.shape(x)[0]], [0, 0], [0, 0]]))
    else:
        x = tf.image.resize(x, (CFG.FRAME_LEN, tf.shape(x)[1]))
    return x

def pre_process(x):
    if not isinstance(x, tf.Tensor):
        x = tf.constant(x, dtype=tf.float32)
    rhand = tf.gather(x, RHAND_IDX, axis=1)
    lhand = tf.gather(x, LHAND_IDX, axis=1)
    rpose = tf.gather(x, RPOSE_IDX, axis=1)
    lpose = tf.gather(x, LPOSE_IDX, axis=1)
    rnan_idx = tf.reduce_any(tf.math.is_nan(rhand), axis=1)
    lnan_idx = tf.reduce_any(tf.math.is_nan(lhand), axis=1)
    rnans = tf.math.count_nonzero(rnan_idx)
    lnans = tf.math.count_nonzero(lnan_idx)
    if rnans > lnans:
        hand = lhand
        pose = lpose
        hand_len = len(LHAND_IDX) // 3
        hand_x = hand[:, 0*hand_len:1*hand_len]
        hand_y = hand[:, 1*hand_len:2*hand_len] 
        hand_z = hand[:, 2*hand_len:3*hand_len]
        hand = tf.concat([1.0 - hand_x, hand_y, hand_z], axis=1)
        pose_len = len(LPOSE_IDX) // 3
        pose_x = pose[:, 0*pose_len:1*pose_len]
        pose_y = pose[:, 1*pose_len:2*pose_len]
        pose_z = pose[:, 2*pose_len:3*pose_len]
        pose = tf.concat([1.0 - pose_x, pose_y, pose_z], axis=1)
    else:
        hand = rhand
        pose = rpose
    hand_len = len(LHAND_IDX) // 3
    hand_x = hand[:, 0*hand_len:1*hand_len]
    hand_y = hand[:, 1*hand_len:2*hand_len]
    hand_z = hand[:, 2*hand_len:3*hand_len]
    hand = tf.stack([hand_x, hand_y, hand_z], axis=-1)
    mean = tf.math.reduce_mean(hand, axis=1, keepdims=True)
    std = tf.math.reduce_std(hand, axis=1, keepdims=True)
    hand = (hand - mean) / (std + 1e-8)
    pose_len = len(LPOSE_IDX) // 3
    pose_x = pose[:, 0*pose_len:1*pose_len]
    pose_y = pose[:, 1*pose_len:2*pose_len]
    pose_z = pose[:, 2*pose_len:3*pose_len]
    pose = tf.stack([pose_x, pose_y, pose_z], axis=-1)
    x = tf.concat([hand, pose], axis=1)
    x = resize_pad(x)
    x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)
    x = tf.reshape(x, (CFG.FRAME_LEN, -1))
    return x

def read_char_map(path: str):
    with open(path, 'r') as f:
        char_to_num = json.load(f)
    char_to_num[PAD_TOKEN] = max(char_to_num.values()) + 1
    char_to_num[START_TOKEN] = max(char_to_num.values()) + 1
    char_to_num[END_TOKEN] = max(char_to_num.values()) + 1
    num_to_char = {j: i for i, j in char_to_num.items()}
    return char_to_num, num_to_char

CHAR_TO_NUM, NUM_TO_CHAR = read_char_map(CFG.CHAR_MAP_JSON)
VOCAB_SIZE = len(CHAR_TO_NUM)
print(f"📊 Vocabulario: {VOCAB_SIZE} tokens")

def preprocess_phrase(phrase):
    phrase = START_TOKEN + phrase + END_TOKEN
    phrase_tokens = [CHAR_TO_NUM.get(c, 0) for c in phrase]
    if len(phrase_tokens) < CFG.TARGET_MAXLEN:
        phrase_tokens = phrase_tokens + [CHAR_TO_NUM[PAD_TOKEN]] * (CFG.TARGET_MAXLEN - len(phrase_tokens))
    else:
        phrase_tokens = phrase_tokens[:CFG.TARGET_MAXLEN]
    return np.array(phrase_tokens, dtype=np.int32)

def load_parquet_data(file_path: str):
    try:
        df = pd.read_parquet(file_path, columns=SEL_COLS)
        landmarks = df.values.astype(np.float32)
        processed = pre_process(landmarks)
        return processed.numpy()
    except Exception as e:
        return None

def cargar_metadata_un_parquet(csv_path: str, sample_n: int = None):
    print("📂 Cargando metadata...")
    meta = pd.read_csv(csv_path)
    meta = meta.dropna(subset=['path', 'phrase'])
    meta = meta[meta['phrase'].str.len() <= 50]
    parquet_counts = meta['path'].value_counts()
    target_count = sample_n or 100
    suitable_parquets = parquet_counts[parquet_counts >= target_count]
    if len(suitable_parquets) == 0:
        best_parquet = parquet_counts.index[0]
        meta_filtered = meta[meta['path'] == best_parquet]
    else:
        selected_parquet = suitable_parquets.index[0]
        meta_filtered = meta[meta['path'] == selected_parquet]
    if sample_n is not None and sample_n < len(meta_filtered):
        meta_filtered = meta_filtered.sample(sample_n, random_state=CFG.RANDOM_STATE)
    def resolve_path(row):
        p = str(row['path'])
        fname = os.path.basename(p)
        full = os.path.join(CFG.PARQUET_ROOT, fname)
        return full
    meta_filtered = meta_filtered.copy()
    meta_filtered.loc[:, 'abs_path'] = meta_filtered.apply(resolve_path, axis=1)
    print(f"✅ Dataset: {len(meta_filtered)} secuencias")
    return meta_filtered.reset_index(drop=True)

class LandmarkEmbedding(layers.Layer):
    def __init__(self, num_hid=64):
        super().__init__()
        self.dense1 = layers.Dense(num_hid, activation="relu")
        self.dropout1 = layers.Dropout(CFG.DROPOUT)
        self.dense2 = layers.Dense(num_hid, activation="relu")
        self.dropout2 = layers.Dropout(CFG.DROPOUT)
        self.num_hid = num_hid

    def call(self, x, training=False):
        if len(x.shape) == 2:
            x = tf.expand_dims(x, 0)
        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        seq_len = tf.shape(x)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1, dtype=tf.float32)
        positions = positions[:, tf.newaxis]
        depth = self.num_hid / 2
        depths = tf.range(depth, dtype=tf.float32)[tf.newaxis, :] / depth
        angle_rates = 1 / (10000 ** depths)
        angle_rads = positions * angle_rates
        pos_encoding = tf.concat([tf.sin(angle_rads), tf.cos(angle_rads)], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, :, :]
        return x + pos_encoding

class TokenEmbedding(layers.Layer):
    def __init__(self, num_vocab=1000, num_hid=64):
        super().__init__()
        self.emb = layers.Embedding(num_vocab, num_hid)
        self.num_hid = num_hid

    def call(self, x):
        x = self.emb(x)
        seq_len = tf.shape(x)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1, dtype=tf.float32)
        positions = positions[:, tf.newaxis]
        depth = self.num_hid / 2
        depths = tf.range(depth, dtype=tf.float32)[tf.newaxis, :] / depth
        angle_rates = 1 / (10000 ** depths)
        angle_rads = positions * angle_rates
        pos_encoding = tf.concat([tf.sin(angle_rads), tf.cos(angle_rads)], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, :, :]
        return x + pos_encoding

class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=rate)
        self.ffn = keras.Sequential([layers.Dense(feed_forward_dim, activation="relu"), layers.Dropout(rate), layers.Dense(embed_dim)])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, dropout_rate=0.1):
        super().__init__()
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        self.self_att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=dropout_rate)
        self.enc_att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=dropout_rate)
        self.self_dropout = layers.Dropout(dropout_rate)
        self.enc_dropout = layers.Dropout(dropout_rate)
        self.ffn_dropout = layers.Dropout(dropout_rate)
        self.ffn = keras.Sequential([layers.Dense(feed_forward_dim, activation="relu"), layers.Dropout(dropout_rate), layers.Dense(embed_dim)])

    def causal_attention_mask(self, batch_size, n_dest, n_src, dtype):
        i = tf.range(n_dest)[:, None]
        j = tf.range(n_src)
        m = i >= j - n_src + n_dest
        mask = tf.cast(m, dtype)
        mask = tf.reshape(mask, [1, n_dest, n_src])
        mult = tf.concat([tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0)
        return tf.tile(mask, mult)

    def call(self, enc_out, target, training):
        input_shape = tf.shape(target)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = self.causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
        target_att = self.self_att(target, target, attention_mask=causal_mask, training=training)
        target_norm = self.layernorm1(target + self.self_dropout(target_att, training=training))
        enc_out_att = self.enc_att(target_norm, enc_out, training=training)
        enc_out_norm = self.layernorm2(self.enc_dropout(enc_out_att, training=training) + target_norm)
        ffn_out = self.ffn(enc_out_norm)
        ffn_out_norm = self.layernorm3(enc_out_norm + self.ffn_dropout(ffn_out, training=training))
        return ffn_out_norm

class Transformer(keras.Model):
    def __init__(self):
        super().__init__()
        self.loss_metric = keras.metrics.Mean(name="loss")
        self.enc_input = LandmarkEmbedding(num_hid=CFG.NUM_HID)
        self.dec_input = TokenEmbedding(num_vocab=VOCAB_SIZE, num_hid=CFG.NUM_HID)
        self.enc_layers = [TransformerEncoder(CFG.NUM_HID, CFG.NUM_HEAD, CFG.NUM_FEED_FORWARD, rate=CFG.DROPOUT) for _ in range(CFG.NUM_LAYERS_ENC)]
        self.dec_layers = [TransformerDecoder(CFG.NUM_HID, CFG.NUM_HEAD, CFG.NUM_FEED_FORWARD, dropout_rate=CFG.DROPOUT) for _ in range(CFG.NUM_LAYERS_DEC)]
        self.classifier = layers.Dense(VOCAB_SIZE)

    def call(self, inputs, training=False):
        source = inputs[0]
        target = inputs[1]
        x = self.enc_input(source, training=training)
        for enc_layer in self.enc_layers:
            x = enc_layer(x, training=training)
        enc_out = x
        y = self.dec_input(target)
        for dec_layer in self.dec_layers:
            y = dec_layer(enc_out, y, training=training)
        return self.classifier(y)

    @property
    def metrics(self):
        return [self.loss_metric]

    def train_step(self, batch):
        source = batch[0]
        target = batch[1]
        dec_input = target[:, :-1]
        dec_target = target[:, 1:]
        with tf.GradientTape() as tape:
            preds = self([source, dec_input], training=True)
            one_hot = tf.one_hot(dec_target, depth=VOCAB_SIZE)
            mask = tf.math.logical_not(tf.math.equal(dec_target, CHAR_TO_NUM[PAD_TOKEN]))
            loss = self.compiled_loss(one_hot, preds, sample_weight=mask)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        gradients = [tf.clip_by_norm(g, 1.0) for g in gradients]
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.loss_metric.update_state(loss)
        return {"loss": self.loss_metric.result()}

    def generate(self, source, target_start_token_idx, temperature=0.8):
        bs = tf.shape(source)[0]
        enc = self.enc_input(source, training=False)
        for enc_layer in self.enc_layers:
            enc = enc_layer(enc, training=False)
        dec_input = tf.ones((bs, 1), dtype=tf.int32) * target_start_token_idx
        for i in range(CFG.TARGET_MAXLEN - 1):
            dec_out = self.dec_input(dec_input)
            for dec_layer in self.dec_layers:
                dec_out = dec_layer(enc, dec_out, training=False)
            logits = self.classifier(dec_out)
            logits = logits / temperature
            probabilities = tf.nn.softmax(logits[:, -1, :], axis=-1)
            next_token = tf.random.categorical(tf.math.log(probabilities + 1e-10), 1, dtype=tf.int32)
            dec_input = tf.concat([dec_input, next_token], axis=-1)
            if tf.reduce_all(tf.equal(next_token, CHAR_TO_NUM[END_TOKEN])):
                break
        return dec_input

class TCNBlock(layers.Layer):
    def __init__(self, filters, kernel_size, dilation_rate, dropout_rate):
        super().__init__()
        self.conv = layers.Conv1D(filters=filters, kernel_size=kernel_size, padding='causal', dilation_rate=dilation_rate, activation='relu')
        self.dropout = layers.Dropout(dropout_rate)
        self.norm = layers.LayerNormalization()
        self.filters = filters
        
    def call(self, x, training=False):
        out = self.conv(x)
        out = self.dropout(out, training=training)
        out = self.norm(out)
        if x.shape[-1] != out.shape[-1]:
            x = layers.Dense(self.filters)(x)
        out = out + x
        return out

class TCNEncoder(keras.Model):
    def __init__(self):
        super().__init__()
        self.input_proj = layers.Dense(CFG.TCN_FILTERS)
        self.tcn_blocks = [TCNBlock(CFG.TCN_FILTERS, CFG.TCN_KERNEL_SIZE, dilation, CFG.DROPOUT) for dilation in CFG.TCN_DILATIONS]
        self.output_proj = layers.Dense(CFG.NUM_HID)
        
    def call(self, x, training=False):
        x = self.input_proj(x)
        for block in self.tcn_blocks:
            x = block(x, training=training)
        x = self.output_proj(x)
        return x

class TCNModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.loss_metric = keras.metrics.Mean(name="loss")
        self.encoder = TCNEncoder()
        self.dec_input = TokenEmbedding(num_vocab=VOCAB_SIZE, num_hid=CFG.NUM_HID)
        self.dec_layers = [TransformerDecoder(CFG.NUM_HID, CFG.NUM_HEAD, CFG.NUM_FEED_FORWARD, dropout_rate=CFG.DROPOUT) for _ in range(CFG.NUM_LAYERS_DEC)]
        self.classifier = layers.Dense(VOCAB_SIZE)
    
    def call(self, inputs, training=False):
        source = inputs[0]
        target = inputs[1]
        enc_out = self.encoder(source, training=training)
        y = self.dec_input(target)
        for dec_layer in self.dec_layers:
            y = dec_layer(enc_out, y, training=training)
        return self.classifier(y)
    
    @property
    def metrics(self):
        return [self.loss_metric]
    
    def train_step(self, batch):
        source = batch[0]
        target = batch[1]
        dec_input = target[:, :-1]
        dec_target = target[:, 1:]
        with tf.GradientTape() as tape:
            preds = self([source, dec_input], training=True)
            one_hot = tf.one_hot(dec_target, depth=VOCAB_SIZE)
            mask = tf.math.logical_not(tf.math.equal(dec_target, CHAR_TO_NUM[PAD_TOKEN]))
            loss = self.compiled_loss(one_hot, preds, sample_weight=mask)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        gradients = [tf.clip_by_norm(g, 1.0) for g in gradients]
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.loss_metric.update_state(loss)
        return {"loss": self.loss_metric.result()}
    
    def generate(self, source, target_start_token_idx, temperature=0.8):
        bs = tf.shape(source)[0]
        enc = self.encoder(source, training=False)
        dec_input = tf.ones((bs, 1), dtype=tf.int32) * target_start_token_idx
        for i in range(CFG.TARGET_MAXLEN - 1):
            dec_out = self.dec_input(dec_input)
            for dec_layer in self.dec_layers:
                dec_out = dec_layer(enc, dec_out, training=False)
            logits = self.classifier(dec_out)
            logits = logits / temperature
            probabilities = tf.nn.softmax(logits[:, -1, :], axis=-1)
            next_token = tf.random.categorical(tf.math.log(probabilities + 1e-10), 1, dtype=tf.int32)
            dec_input = tf.concat([dec_input, next_token], axis=-1)
            if tf.reduce_all(tf.equal(next_token, CHAR_TO_NUM[END_TOKEN])):
                break
        return dec_input

def crear_dataset(meta, batch_size, shuffle):
    def data_generator():
        for idx, row in tqdm(meta.iterrows(), total=len(meta), desc="📦 Cargando datos", leave=False):
            try:
                landmarks = load_parquet_data(row['abs_path'])
                if landmarks is not None:
                    phrase_processed = preprocess_phrase(row['phrase'])
                    yield landmarks, phrase_processed
            except Exception as e:
                continue
    output_signature = (tf.TensorSpec(shape=(CFG.FRAME_LEN, len(LHAND_IDX) + len(LPOSE_IDX)), dtype=tf.float32), tf.TensorSpec(shape=(CFG.TARGET_MAXLEN,), dtype=tf.int32))
    ds = tf.data.Dataset.from_generator(data_generator, output_signature=output_signature)
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def train_model(model, train_ds, val_ds, model_name, epochs=CFG.EPOCHS):
    print(f"\n{'='*60}\n🏋️ Entrenando modelo: {model_name}\n{'='*60}")
    history = {'train_loss': [], 'val_loss': [], 'epoch_times': [], 'lr': []}
    epoch_pbar = tqdm(range(epochs), desc=f"🔄 Épocas [{model_name}]", position=0)
    for epoch in epoch_pbar:
        epoch_start = time.time()
        train_losses = []
        batch_pbar = tqdm(train_ds, desc=f"  📊 Epoch {epoch+1} - Train", leave=False, position=1)
        for batch_idx, batch in enumerate(batch_pbar):
            loss_dict = model.train_step(batch)
            loss_value = loss_dict['loss'].numpy()
            train_losses.append(loss_value)
            batch_pbar.set_postfix({'loss': f'{loss_value:.4f}', 'avg_loss': f'{np.mean(train_losses):.4f}'})
        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)
        val_losses = []
        val_pbar = tqdm(val_ds, desc=f"  ✅ Epoch {epoch+1} - Val", leave=False, position=1)
        for batch in val_pbar:
            source = batch[0]
            target = batch[1]
            dec_input = target[:, :-1]
            dec_target = target[:, 1:]
            preds = model([source, dec_input], training=False)
            one_hot = tf.one_hot(dec_target, depth=VOCAB_SIZE)
            mask = tf.math.logical_not(tf.math.equal(dec_target, CHAR_TO_NUM[PAD_TOKEN]))
            loss = keras.losses.categorical_crossentropy(one_hot, preds, from_logits=True)
            loss = tf.reduce_sum(loss * tf.cast(mask, tf.float32)) / tf.reduce_sum(tf.cast(mask, tf.float32))
            val_losses.append(loss.numpy())
            val_pbar.set_postfix({'val_loss': f'{loss.numpy():.4f}', 'avg_val_loss': f'{np.mean(val_losses):.4f}'})
        avg_val_loss = np.mean(val_losses)
        history['val_loss'].append(avg_val_loss)
        epoch_time = time.time() - epoch_start
        history['epoch_times'].append(epoch_time)
        current_lr = model.optimizer.learning_rate.numpy()
        if hasattr(model.optimizer.learning_rate, '__call__'):
            current_lr = model.optimizer.learning_rate(model.optimizer.iterations).numpy()
        history['lr'].append(current_lr)
        epoch_pbar.set_postfix({'train_loss': f'{avg_train_loss:.4f}', 'val_loss': f'{avg_val_loss:.4f}', 'time': f'{epoch_time:.1f}s'})
        print(f"\n  📈 Época {epoch+1}/{epochs}:")
        print(f"     Train Loss: {avg_train_loss:.4f}")
        print(f"     Val Loss: {avg_val_loss:.4f}")
        print(f"     Time: {epoch_time:.2f}s")
        print(f"     LR: {current_lr:.2e}")
    return history

def evaluate_model(model, val_ds, val_meta, model_name, temperature=0.8):
    
    global GLOBAL_REPORTER
    print(f"\n{'='*60}\n🔍 Evaluando modelo: {model_name}\n{'='*60}")
    tracker = MetricsTracker()
    eval_pbar = tqdm(val_ds, desc=f"🎯 Evaluando {model_name}", position=0)
    for batch_idx, batch in enumerate(eval_pbar):
        batch_start = time.time()
        source = batch[0]
        target = batch[1].numpy()
        preds = model.generate(source, CHAR_TO_NUM[START_TOKEN], temperature=temperature)
        preds = preds.numpy()
        batch_time = time.time() - batch_start
        batch_preds = []
        batch_truths = []
        for i in range(len(target)):
            gt_tokens = []
            for idx in target[i, :]:
                if idx == CHAR_TO_NUM[END_TOKEN]:
                    break
                if idx != CHAR_TO_NUM[PAD_TOKEN] and idx != CHAR_TO_NUM[START_TOKEN]:
                    gt_tokens.append(NUM_TO_CHAR[idx])
            ground_truth = "".join(gt_tokens)
            batch_truths.append(ground_truth)
            pred_tokens = []
            for idx in preds[i, :]:
                char = NUM_TO_CHAR[idx]
                if char == END_TOKEN:
                    break
                if char != PAD_TOKEN and char != START_TOKEN:
                    pred_tokens.append(char)
            prediction = "".join(pred_tokens)
            batch_preds.append(prediction)
        tracker.add_batch(batch_preds, batch_truths, time_taken=batch_time)
        if len(tracker.predictions) > 0:
            partial_cer = tracker.calculate_cer()
            partial_acc = tracker.calculate_accuracy()
            eval_pbar.set_postfix({'CER': f'{partial_cer:.2f}%', 'Acc': f'{partial_acc:.2f}%', 'samples': len(tracker.predictions)})
    metrics = tracker.get_summary()
    print(f"\n📊 Resultados de {model_name}:")
    print(f"   CER (Character Error Rate): {metrics['cer']:.2f}%")
    print(f"   WER (Word Error Rate): {metrics['wer']:.2f}%")
    print(f"   Accuracy (Exact Match): {metrics['accuracy']:.2f}%")
    print(f"   Avg Loss: {metrics['avg_loss']:.4f}")
    print(f"   Avg Time per batch: {metrics['avg_time']:.3f}s")
    print(f"   Total samples: {metrics['total_samples']}")
    print(f"\n📝 Ejemplos de predicciones:")
    
    for i in range(min(5, len(tracker.predictions))):
        print(f"\n   Ejemplo {i+1}:")
        print(f"   Real: '{tracker.ground_truths[i]}'")
        print(f"   Pred: '{tracker.predictions[i]}'")
    
        # --- Reporter writeback (injected) ---
        if GLOBAL_REPORTER is not None:
            try:
                GLOBAL_REPORTER.add_metrics(model_name,
                                            CER=metrics.get('cer', None),
                                            WER=metrics.get('wer', None),
                                            Acc=metrics.get('accuracy', None),
                                            AvgLoss=metrics.get('avg_loss', None),
                                            AvgTimePerBatch=metrics.get('avg_time', None),
                                            TotalSamples=metrics.get('total_samples', None))
                rows = []
                for gt, pr in zip(tracker.ground_truths, tracker.predictions):
                    rows.append({"modelo": model_name, "real": gt, "pred": pr})
                if rows:
                    GLOBAL_REPORTER.save_predictions(rows)
            except Exception as _e:
                print(f"[Reporter] No se pudo guardar métricas/predicciones: {_e}")
        return metrics, tracker

def plot_training_comparison(histories, model_names, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Comparación de Entrenamiento: Transformer vs TCN', fontsize=16, fontweight='bold')
    ax = axes[0, 0]
    for history, name in zip(histories, model_names):
        ax.plot(history['train_loss'], marker='o', label=name, linewidth=2)
    ax.set_xlabel('Época', fontsize=12)
    ax.set_ylabel('Training Loss', fontsize=12)
    ax.set_title('Pérdida en Entrenamiento', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax = axes[0, 1]
    for history, name in zip(histories, model_names):
        ax.plot(history['val_loss'], marker='s', label=name, linewidth=2)
    ax.set_xlabel('Época', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title('Pérdida en Validación', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax = axes[1, 0]
    x = np.arange(len(histories[0]['epoch_times']))
    width = 0.35
    for i, (history, name) in enumerate(zip(histories, model_names)):
        ax.bar(x + i*width, history['epoch_times'], width, label=name, alpha=0.7)
    ax.set_xlabel('Época', fontsize=12)
    ax.set_ylabel('Tiempo (segundos)', fontsize=12)
    ax.set_title('Tiempo de Entrenamiento por Época', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([f'{i+1}' for i in range(len(histories[0]['epoch_times']))])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax = axes[1, 1]
    for history, name in zip(histories, model_names):
        ax.plot(history['lr'], marker='^', label=name, linewidth=2)
    ax.set_xlabel('Época', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Evolución del Learning Rate', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Gráfica guardada: {save_path}")
    plt.close()

def plot_metrics_comparison(metrics_list, model_names, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Comparación de Métricas de Evaluación', fontsize=16, fontweight='bold')
    metric_names = ['cer', 'wer', 'accuracy']
    metric_labels = ['CER (%)', 'WER (%)', 'Accuracy (%)']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    for idx, (metric, label, color) in enumerate(zip(metric_names, metric_labels, colors)):
        ax = axes[idx]
        values = [m[metric] for m in metrics_list]
        bars = ax.bar(model_names, values, color=color, alpha=0.7, edgecolor='black', linewidth=2)
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{value:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax.set_ylabel(label, fontsize=12)
        ax.set_title(label, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, max(values) * 1.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Gráfica guardada: {save_path}")
    plt.close()

def plot_error_analysis(trackers, model_names, save_path):
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    fig.suptitle('Análisis Detallado de Errores', fontsize=16, fontweight='bold')
    for idx, (tracker, name) in enumerate(zip(trackers, model_names)):
        errors = []
        for pred, truth in zip(tracker.predictions, tracker.ground_truths):
            error = levenshtein_distance(pred.strip(), truth.strip())
            errors.append(error)
        ax = fig.add_subplot(gs[0, idx])
        ax.hist(errors, bins=20, color=f'C{idx}', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Distancia de Levenshtein', fontsize=11)
        ax.set_ylabel('Frecuencia', fontsize=11)
        ax.set_title(f'{name} - Distribución de Errores', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axvline(np.mean(errors), color='red', linestyle='--', linewidth=2, label=f'Media: {np.mean(errors):.2f}')
        ax.legend()
        ax = fig.add_subplot(gs[1, idx])
        pred_lens = [len(p.strip()) for p in tracker.predictions]
        truth_lens = [len(t.strip()) for t in tracker.ground_truths]
        ax.scatter(truth_lens, pred_lens, alpha=0.5, s=30, c=f'C{idx}')
        ax.plot([0, max(truth_lens)], [0, max(truth_lens)], 'r--', linewidth=2, label='Ideal')
        ax.set_xlabel('Longitud Real', fontsize=11)
        ax.set_ylabel('Longitud Predicha', fontsize=11)
        ax.set_title(f'{name} - Longitud Predicciones', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    ax = fig.add_subplot(gs[2, :])
    metrics_to_compare = {'CER': [t.calculate_cer() for t in trackers], 'WER': [t.calculate_wer() for t in trackers], 'Accuracy': [t.calculate_accuracy() for t in trackers]}
    x = np.arange(len(model_names))
    width = 0.25
    for i, (metric_name, values) in enumerate(metrics_to_compare.items()):
        offset = width * (i - 1)
        bars = ax.bar(x + offset, values, width, label=metric_name, alpha=0.8)
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{value:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_xlabel('Modelos', fontsize=12)
    ax.set_ylabel('Porcentaje (%)', fontsize=12)
    ax.set_title('Comparación de Métricas por Modelo', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Gráfica guardada: {save_path}")
    plt.close()

def main():
    print("="*80)
    print("🚀 COMPARACIÓN DE MODELOS: TRANSFORMER vs TCN")
    print("   ASL Fingerspelling Recognition")
    print("="*80)
    print("\n" + "="*80)
    print("📂 FASE 1: CARGA Y PREPARACIÓN DE DATOS")
    print("="*80)
    meta = cargar_metadata_un_parquet(CFG.TRAIN_CSV, sample_n=CFG.SAMPLE_SEQUENCES)
    trn_df, val_df = train_test_split(meta, test_size=0.2, random_state=CFG.RANDOM_STATE)
    print(f"\n✅ Split completado: train={len(trn_df)}, val={len(val_df)}")
    print("\n🔄 Creando datasets...")
    train_ds = crear_dataset(trn_df, CFG.BATCH_SIZE, shuffle=True)
    val_ds = crear_dataset(val_df, CFG.BATCH_SIZE, shuffle=False)
    print("\n" + "="*80)
    print("🤖 FASE 2: ENTRENAMIENTO DEL TRANSFORMER")
    print("="*80)
    transformer = Transformer()
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=CFG.LR, decay_steps=1000, decay_rate=0.9)
    transformer.compile(optimizer=keras.optimizers.Adam(lr_schedule), loss=keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.2))
    transformer_history = train_model(transformer, train_ds, val_ds, "Transformer", CFG.EPOCHS)
    print("\n" + "="*80)
    print("🔥 FASE 3: ENTRENAMIENTO DEL TCN")
    print("="*80)
    tcn_model = TCNModel()
    lr_schedule_tcn = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=CFG.LR, decay_steps=1000, decay_rate=0.9)
    tcn_model.compile(optimizer=keras.optimizers.Adam(lr_schedule_tcn), loss=keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.2))
    tcn_history = train_model(tcn_model, train_ds, val_ds, "TCN", CFG.EPOCHS)
    print("\n" + "="*80)
    print("📊 FASE 4: EVALUACIÓN DE MODELOS")
    print("="*80)
    transformer_metrics, transformer_tracker = evaluate_model(transformer, val_ds, val_df, "Transformer", temperature=0.8)
    tcn_metrics, tcn_tracker = evaluate_model(tcn_model, val_ds, val_df, "TCN", temperature=0.8)
    print("\n" + "="*80)
    print("📈 FASE 5: GENERACIÓN DE VISUALIZACIONES")
    print("="*80)
    plot_training_comparison([transformer_history, tcn_history], ['Transformer', 'TCN'], os.path.join(CFG.OUTPUT_DIR, 'training_comparison.png'))
    plot_metrics_comparison([transformer_metrics, tcn_metrics], ['Transformer', 'TCN'], os.path.join(CFG.OUTPUT_DIR, 'metrics_comparison.png'))
    plot_error_analysis([transformer_tracker, tcn_tracker], ['Transformer', 'TCN'], os.path.join(CFG.OUTPUT_DIR, 'error_analysis.png'))
    print("\n" + "="*80)
    print("🏆 RESUMEN FINAL Y CONCLUSIONES")
    print("="*80)
    print("\n📊 COMPARACIÓN DE MÉTRICAS:")
    print(f"\n{'Métrica':<20} {'Transformer':<15} {'TCN':<15} {'Mejor':<15}")
    print("-" * 65)
    metrics_to_show = [('CER (%)', 'cer', 'menor'), ('WER (%)', 'wer', 'menor'), ('Accuracy (%)', 'accuracy', 'mayor'), ('Avg Loss', 'avg_loss', 'menor')]
    for metric_name, metric_key, comparison in metrics_to_show:
        trans_val = transformer_metrics[metric_key]
        tcn_val = tcn_metrics[metric_key]
        if comparison == 'menor':
            better = 'Transformer' if trans_val < tcn_val else 'TCN'
        else:
            better = 'Transformer' if trans_val > tcn_val else 'TCN'
        print(f"{metric_name:<20} {trans_val:<15.2f} {tcn_val:<15.2f} {better:<15}")
    print("\n⏱️ TIEMPO DE ENTRENAMIENTO:")
    trans_total_time = sum(transformer_history['epoch_times'])
    tcn_total_time = sum(tcn_history['epoch_times'])
    print(f"   Transformer: {trans_total_time:.2f}s ({trans_total_time/60:.2f} min)")
    print(f"   TCN: {tcn_total_time:.2f}s ({tcn_total_time/60:.2f} min)")
    print("\n💾 ARCHIVOS GENERADOS:")
    print(f"   📁 Directorio: {CFG.OUTPUT_DIR}")
    print(f"   📊 training_comparison.png")
    print(f"   📊 metrics_comparison.png")
    print(f"   📊 error_analysis.png")
    print("\n" + "="*80)
    print("✅ ANÁLISIS COMPLETADO")
    if GLOBAL_REPORTER is not None:
        try:
            GLOBAL_REPORTER.save_summary()
        except Exception as _e:
            print(f"[Reporter] Error al guardar summary: {_e}")
    print("="*80)

if __name__ == "__main__":
        run_name = "run-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        with reporting(run_dir=CFG.OUTPUT_DIR, run_name=run_name) as _R:
            _set_global_reporter(_R)
            try:
                _R.save_config(CFG)
                # Ejemplo para registrar hyperparams dinámicos:
                GLOBAL_REPORTER.save_config(None, extra={
                    "optimizer": "AdamW",
                    "lr": 1e-4,          # (Transformer) o 3e-4 para TCN+CTC
                    "warmup_ratio": 0.05,
                    "scheduler": "cosine",
                    "grad_clip": 1.0,
                    "label_smoothing": 0.05,   # si usas CE AR
                    "batch_size": 16,
                    "epochs_planned": 50
                })
            except Exception as _e:
                print(f"[Reporter] Error al guardar configuración: {_e}")
            main()
