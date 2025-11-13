"""
Aplicación ASL Fingerspelling Recognition
Sistema de clasificación y análisis de datos con Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import tensorflow as tf
from tensorflow import keras
import time
import io
import imageio.v2 as imageio

# Rutas
MODEL_PATH = "modelos/my_model.h5"
CHAR_MAP_PATH = "modelos/character_to_prediction_index.json"

# Conexiones de MediaPipe 
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),          # Pulgar
    (0,5),(5,6),(6,7),(7,8),          # Índice
    (0,9),(9,10),(10,11),(11,12),     # Medio
    (0,13),(13,14),(14,15),(15,16),   # Anular
    (0,17),(17,18),(18,19),(19,20)    # Meñique
]


st.set_page_config(
    page_title="ASL Fingerspelling Recognition",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paleta de colores 
COLORS = {
    'primary': '#2E86AB',      # Azul principal
    'secondary': '#A23B72',    # Magenta oscuro
    'accent': '#F18F01',       # Naranja cálido
    'success': '#06A77D',      # Verde esmeralda
    'background': '#F5F5F5',   # Gris claro
    'text': '#2C3E50',         # Gris oscuro
    'card': '#FFFFFF',         # Blanco
    'border': '#E0E0E0'        # Gris borde
}

st.markdown(f"""
<style>
    /* Estilos generales */
    .main {{
        background-color: {COLORS['background']};
    }}
    
    /* Títulos */
    h1 {{
        color: {COLORS['primary']};
        font-weight: 700;
        text-align: center;
        padding: 20px 0;
    }}
    
    h2, h3 {{
        color: {COLORS['text']};
        font-weight: 600;
    }}
    
    /* Tarjetas de métricas */
    .metric-card {{
        background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
    }}
    
    .metric-value {{
        font-size: 2.5em;
        font-weight: bold;
        margin: 10px 0;
    }}
    
    .metric-label {{
        font-size: 1.1em;
        opacity: 0.9;
    }}
    
    /* Botones */
    .stButton > button {{
        background-color: {COLORS['accent']};
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }}
    
    .stButton > button:hover {{
        background-color: {COLORS['secondary']};
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }}
    
    /* Sidebar */
    .css-1d391kg {{
        background-color: {COLORS['card']};
    }}
    
    /* Info boxes */
    .info-box {{
        background-color: {COLORS['card']};
        border-left: 4px solid {COLORS['accent']};
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }}
    
    /* Alertas */
    .success-box {{
        background-color: #d4edda;
        border-left: 4px solid {COLORS['success']};
        padding: 15px;
        border-radius: 8px;
        color: #155724;
    }}
    
    .warning-box {{
        background-color: #fff3cd;
        border-left: 4px solid {COLORS['accent']};
        padding: 15px;
        border-radius: 8px;
        color: #856404;
    }}
</style>
""", unsafe_allow_html=True)

# Landmarks
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

# Configuración
FRAME_LEN = 128
TARGET_MAXLEN = 64
PAD_TOKEN = 'P'
START_TOKEN = 'S'
END_TOKEN = 'E'

# Clases del modelo
class LandmarkEmbedding(keras.layers.Layer):
    def __init__(self, num_hid=64, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_hid = num_hid
        self.dropout_rate = dropout
        self.dense1 = keras.layers.Dense(num_hid, activation="relu")
        self.dropout1 = keras.layers.Dropout(dropout)
        self.dense2 = keras.layers.Dense(num_hid, activation="relu")
        self.dropout2 = keras.layers.Dropout(dropout)

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

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_hid": self.num_hid,
            "dropout": self.dropout_rate
        })
        return config

class TokenEmbedding(keras.layers.Layer):
    def __init__(self, num_vocab=1000, num_hid=64, **kwargs):
        super().__init__(**kwargs)
        self.num_vocab = num_vocab
        self.num_hid = num_hid
        self.emb = keras.layers.Embedding(num_vocab, num_hid)

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

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_vocab": self.num_vocab,
            "num_hid": self.num_hid
        })
        return config

class TransformerEncoder(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.feed_forward_dim = feed_forward_dim
        self.rate = rate
        self.att = keras.layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=embed_dim // num_heads, 
            dropout=rate
        )
        self.ffn = keras.Sequential([
            keras.layers.Dense(feed_forward_dim, activation="relu"),
            keras.layers.Dropout(rate),
            keras.layers.Dense(embed_dim)
        ])
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "feed_forward_dim": self.feed_forward_dim,
            "rate": self.rate
        })
        return config

class TransformerDecoder(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.feed_forward_dim = feed_forward_dim
        self.dropout_rate = dropout_rate
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.self_att = keras.layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=embed_dim // num_heads, 
            dropout=dropout_rate
        )
        self.enc_att = keras.layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=embed_dim // num_heads, 
            dropout=dropout_rate
        )
        self.self_dropout = keras.layers.Dropout(dropout_rate)
        self.enc_dropout = keras.layers.Dropout(dropout_rate)
        self.ffn_dropout = keras.layers.Dropout(dropout_rate)
        self.ffn = keras.Sequential([
            keras.layers.Dense(feed_forward_dim, activation="relu"),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Dense(embed_dim)
        ])

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

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "feed_forward_dim": self.feed_forward_dim,
            "dropout_rate": self.dropout_rate
        })
        return config

class Transformer(keras.Model):
    def __init__(self, num_hid=64, num_head=2, num_feed_forward=128, 
                 num_layers_enc=2, num_layers_dec=1, dropout=0.1, 
                 vocab_size=62, **kwargs):
        super().__init__(**kwargs)
        self.num_hid = num_hid
        self.num_head = num_head
        self.num_feed_forward = num_feed_forward
        self.num_layers_enc = num_layers_enc
        self.num_layers_dec = num_layers_dec
        self.dropout_rate = dropout
        self.vocab_size = vocab_size
        
        self.loss_metric = keras.metrics.Mean(name="loss")
        self.enc_input = LandmarkEmbedding(num_hid=num_hid, dropout=dropout)
        self.dec_input = TokenEmbedding(num_vocab=vocab_size, num_hid=num_hid)
        self.enc_layers = [
            TransformerEncoder(num_hid, num_head, num_feed_forward, rate=dropout) 
            for _ in range(num_layers_enc)
        ]
        self.dec_layers = [
            TransformerDecoder(num_hid, num_head, num_feed_forward, dropout_rate=dropout) 
            for _ in range(num_layers_dec)
        ]
        self.classifier = keras.layers.Dense(vocab_size)

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

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_hid": self.num_hid,
            "num_head": self.num_head,
            "num_feed_forward": self.num_feed_forward,
            "num_layers_enc": self.num_layers_enc,
            "num_layers_dec": self.num_layers_dec,
            "dropout": self.dropout_rate,
            "vocab_size": self.vocab_size
        })
        return config

class TCNBlock(keras.layers.Layer):
    def __init__(self, filters, kernel_size, dilation_rate, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout_rate = dropout_rate
        self.conv = keras.layers.Conv1D(
            filters=filters, 
            kernel_size=kernel_size, 
            padding='causal', 
            dilation_rate=dilation_rate, 
            activation='relu'
        )
        self.dropout = keras.layers.Dropout(dropout_rate)
        self.norm = keras.layers.LayerNormalization()
        
    def call(self, x, training=False):
        out = self.conv(x)
        out = self.dropout(out, training=training)
        out = self.norm(out)
        if x.shape[-1] != out.shape[-1]:
            x = keras.layers.Dense(self.filters)(x)
        out = out + x
        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "dilation_rate": self.dilation_rate,
            "dropout_rate": self.dropout_rate
        })
        return config

class TCNEncoder(keras.Model):
    def __init__(self, tcn_filters=64, tcn_kernel_size=3, 
                 tcn_dilations=[1, 2, 4, 8], dropout=0.1, 
                 num_hid=64, **kwargs):
        super().__init__(**kwargs)
        self.tcn_filters = tcn_filters
        self.tcn_kernel_size = tcn_kernel_size
        self.tcn_dilations = tcn_dilations
        self.dropout_rate = dropout
        self.num_hid = num_hid
        
        self.input_proj = keras.layers.Dense(tcn_filters)
        self.tcn_blocks = [
            TCNBlock(tcn_filters, tcn_kernel_size, dilation, dropout) 
            for dilation in tcn_dilations
        ]
        self.output_proj = keras.layers.Dense(num_hid)
        
    def call(self, x, training=False):
        x = self.input_proj(x)
        for block in self.tcn_blocks:
            x = block(x, training=training)
        x = self.output_proj(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "tcn_filters": self.tcn_filters,
            "tcn_kernel_size": self.tcn_kernel_size,
            "tcn_dilations": self.tcn_dilations,
            "dropout": self.dropout_rate,
            "num_hid": self.num_hid
        })
        return config

class TCNModel(keras.Model):
    def __init__(self, tcn_filters=64, tcn_kernel_size=3, 
                 tcn_dilations=[1, 2, 4, 8], dropout=0.1,
                 num_hid=64, num_head=2, num_feed_forward=128,
                 num_layers_dec=1, vocab_size=62, **kwargs):
        super().__init__(**kwargs)
        self.tcn_filters = tcn_filters
        self.tcn_kernel_size = tcn_kernel_size
        self.tcn_dilations = tcn_dilations
        self.dropout_rate = dropout
        self.num_hid = num_hid
        self.num_head = num_head
        self.num_feed_forward = num_feed_forward
        self.num_layers_dec = num_layers_dec
        self.vocab_size = vocab_size
        
        self.loss_metric = keras.metrics.Mean(name="loss")
        self.encoder = TCNEncoder(tcn_filters, tcn_kernel_size, tcn_dilations, dropout, num_hid)
        self.dec_input = TokenEmbedding(num_vocab=vocab_size, num_hid=num_hid)
        self.dec_layers = [
            TransformerDecoder(num_hid, num_head, num_feed_forward, dropout_rate=dropout) 
            for _ in range(num_layers_dec)
        ]
        self.classifier = keras.layers.Dense(vocab_size)
    
    def call(self, inputs, training=False):
        source = inputs[0]
        target = inputs[1]
        enc_out = self.encoder(source, training=training)
        y = self.dec_input(target)
        for dec_layer in self.dec_layers:
            y = dec_layer(enc_out, y, training=training)
        return self.classifier(y)

    def get_config(self):
        config = super().get_config()
        config.update({
            "tcn_filters": self.tcn_filters,
            "tcn_kernel_size": self.tcn_kernel_size,
            "tcn_dilations": self.tcn_dilations,
            "dropout": self.dropout_rate,
            "num_hid": self.num_hid,
            "num_head": self.num_head,
            "num_feed_forward": self.num_feed_forward,
            "num_layers_dec": self.num_layers_dec,
            "vocab_size": self.vocab_size
        })
        return config


# Funciones 
@st.cache_resource
def load_model(model_path):
    """Carga el modelo entrenado con capas personalizadas"""
    try:
        import h5py
        import json
        
        st.info("🔄 Cargando modelo...")
        
        with h5py.File(model_path, 'r') as f:
            num_hid = None
            num_head = None
            vocab_size = 62
            
            if 'model_weights' in f:
                weight_group = f['model_weights']
                
                for layer_name in weight_group.keys():
                    if 'landmark_embedding' in layer_name or 'token_embedding' in layer_name:
                        layer_group = weight_group[layer_name]
                        weight_names = [n.decode('utf8') if isinstance(n, bytes) else n 
                                       for n in layer_group.attrs.get('weight_names', [])]
                        
                        for weight_name in weight_names:
                            if 'kernel' in weight_name or 'embeddings' in weight_name:
                                if weight_name in layer_group:
                                    weight_shape = layer_group[weight_name].shape
                                    if len(weight_shape) == 2:
                                        num_hid = weight_shape[1]
                                        if 'token_embedding' in layer_name:
                                            vocab_size = weight_shape[0]
                                        st.info(f"🔍 Detectado desde {layer_name}: num_hid={num_hid}")
                                        break
                        if num_hid:
                            break
                
                # Buscar TransformerEncoder 
                for layer_name in weight_group.keys():
                    if 'transformer_encoder' in layer_name:
                        layer_group = weight_group[layer_name]
                        
                        # Buscar pesos de atención multi-head
                        for sub_layer in layer_group.keys():
                            if 'multi_head_attention' in sub_layer:
                                mha_group = layer_group[sub_layer]
                                
                                for weight_name in mha_group.keys():
                                    if 'query' in weight_name and 'kernel' in weight_name:
                                        weight_shape = mha_group[weight_name].shape
                                        if len(weight_shape) == 3:
                                            num_head = weight_shape[1]
                                            st.info(f"🔍 Detectado desde {layer_name}: num_head={num_head}")
                                            break
                                if num_head:
                                    break
                        if num_head:
                            break
            
            if num_hid is None or num_head is None:
                if 'model_config' in f.attrs:
                    model_config_str = f.attrs['model_config']
                    if isinstance(model_config_str, bytes):
                        model_config_str = model_config_str.decode('utf-8')
                    model_config = json.loads(model_config_str)
                    
                    config = model_config.get('config', {})
                    if num_hid is None:
                        num_hid = config.get('num_hid', 128)
                    if num_head is None:
                        num_head = config.get('num_head', 4)
                    vocab_size = config.get('vocab_size', vocab_size)
                    model_class = model_config.get('class_name', 'Transformer')
                else:
                    if num_hid is None:
                        num_hid = 128
                    if num_head is None:
                        num_head = 4
                    model_class = 'Transformer'
            else:
                model_class = 'Transformer'
            
            # Calculo de otros parámetros 
            num_feed_forward = num_hid * 2
            num_layers_enc = 2
            num_layers_dec = 1
            dropout = 0.0
            
            st.info(f"📋 Tipo: {model_class}")
            st.info(f"📊 Parámetros FINALES: vocab={vocab_size}, hid={num_hid}, heads={num_head}, ff={num_feed_forward}")
        
        # Registro de objetos personalizados
        custom_objects = {
            'LandmarkEmbedding': LandmarkEmbedding,
            'TokenEmbedding': TokenEmbedding,
            'TransformerEncoder': TransformerEncoder,
            'TransformerDecoder': TransformerDecoder,
            'Transformer': Transformer,
            'TCNBlock': TCNBlock,
            'TCNEncoder': TCNEncoder,
            'TCNModel': TCNModel
        }
        
        st.info("🔧 Método 1: Carga directa...")
        try:
            with keras.utils.custom_object_scope(custom_objects):
                from tensorflow import keras as tf_keras
                model = tf_keras.models.load_model(model_path, compile=False)
                
            st.success("✅ Modelo cargado exitosamente!")
            return model
            
        except Exception as e1:
            st.warning(f"Método 1 falló: {str(e1)[:150]}")
            
            st.info("🔧 Método 2: Reconstrucción con parámetros detectados...")
            try:
                if model_class == 'Transformer':
                    model = Transformer(
                        num_hid=num_hid,
                        num_head=num_head,
                        num_feed_forward=num_feed_forward,
                        num_layers_enc=num_layers_enc,
                        num_layers_dec=num_layers_dec,
                        dropout=dropout,
                        vocab_size=vocab_size
                    )
                elif model_class == 'TCNModel':
                    model = TCNModel(
                        tcn_filters=config.get('tcn_filters', 64),
                        tcn_kernel_size=config.get('tcn_kernel_size', 3),
                        tcn_dilations=config.get('tcn_dilations', [1, 2, 4, 8]),
                        dropout=dropout,
                        num_hid=num_hid,
                        num_head=num_head,
                        num_feed_forward=num_feed_forward,
                        num_layers_dec=num_layers_dec,
                        vocab_size=vocab_size
                    )
                
                dummy_source = tf.ones((1, FRAME_LEN, 78))
                dummy_target = tf.ones((1, TARGET_MAXLEN), dtype=tf.int32)
                _ = model([dummy_source, dummy_target], training=False)
                
                st.info(f"🏗️ Modelo construido con {len(model.layers)} capas")
                
                with h5py.File(model_path, 'r') as f:
                    if 'model_weights' in f:
                        weight_group = f['model_weights']
                        
                        saved_layers = list(weight_group.keys())
                        st.info(f"💾 Capas guardadas: {len(saved_layers)}")
                        
                        layer_names = [layer.name for layer in model.layers]
                        st.info(f"🔍 Capas del modelo: {layer_names}")
                        
                        # Cargar pesos para cada capa
                        loaded_count = 0
                        skipped_layers = []
                        
                        for layer in model.layers:
                            if layer.name in weight_group:
                                try:
                                    layer_group = weight_group[layer.name]
                                    weight_names = [n.decode('utf8') if isinstance(n, bytes) else n 
                                                   for n in layer_group.attrs['weight_names']]
                                    
                                    weight_values = []
                                    for weight_name in weight_names:
                                        if weight_name in layer_group:
                                            weight_values.append(layer_group[weight_name][()])
                                    
                                    if weight_values:
                                        # Verificar compatibilidad de formas
                                        current_weights = layer.get_weights()
                                        if len(weight_values) == len(current_weights):
                                            compatible = True
                                            for i, (saved_w, current_w) in enumerate(zip(weight_values, current_weights)):
                                                if saved_w.shape != current_w.shape:
                                                    compatible = False
                                                    st.warning(f"⚠️ Forma incompatible en {layer.name}[{i}]: {saved_w.shape} vs {current_w.shape}")
                                                    break
                                            
                                            if compatible:
                                                layer.set_weights(weight_values)
                                                loaded_count += 1
                                            else:
                                                skipped_layers.append(layer.name)
                                        else:
                                            skipped_layers.append(layer.name)
                                            st.warning(f"⚠️ Número de pesos diferente en {layer.name}")
                                except Exception as layer_error:
                                    skipped_layers.append(layer.name)
                                    st.warning(f"⚠️ Error cargando {layer.name}: {str(layer_error)[:100]}")
                        
                        st.success(f"✅ Cargados pesos de {loaded_count}/{len(model.layers)} capas")
                        
                        if skipped_layers:
                            st.warning(f"⚠️ Capas omitidas: {skipped_layers}")
                
                if loaded_count > 0:
                    st.success("✅ Modelo cargado con método 2!")
                    st.info(f"💡 {loaded_count} de {len(model.layers)} capas cargadas correctamente")
                    return model
                else:
                    raise ValueError("No se pudieron cargar pesos")
                    
            except Exception as e2:
                st.error(f"❌ Método 2 falló: {str(e2)}")
                
                st.info("🔧 Método 3: Carga forzada...")
                try:
                    import warnings
                    warnings.filterwarnings('ignore')
                    
                    model = Transformer(
                        num_hid=num_hid,
                        num_head=num_head,
                        num_feed_forward=num_feed_forward,
                        num_layers_enc=num_layers_enc,
                        num_layers_dec=num_layers_dec,
                        dropout=dropout,
                        vocab_size=vocab_size
                    )
                    
                    dummy_source = tf.ones((1, FRAME_LEN, 78))
                    dummy_target = tf.ones((1, TARGET_MAXLEN), dtype=tf.int32)
                    _ = model([dummy_source, dummy_target], training=False)
                    
                    model.load_weights(model_path, skip_mismatch=True, by_name=True)
                    st.warning("⚠️ Modelo cargado parcialmente con skip_mismatch")
                    return model
                        
                except Exception as e3:
                    st.error(f"❌ Todos los métodos fallaron")
                    with st.expander("🔍 Ver detalles del error"):
                        st.text(f"Error método 1: {str(e1)}")
                        st.text(f"Error método 2: {str(e2)}")
                        st.text(f"Error método 3: {str(e3)}")
                    return None
        
    except Exception as e:
        st.error(f"❌ Error crítico al cargar el modelo: {str(e)}")
        import traceback
        with st.expander("🔍 Ver traceback completo"):
            st.code(traceback.format_exc())
        return None

@st.cache_data
def load_char_map(char_map_path):
    """Carga el mapeo de caracteres"""
    try:
        with open(char_map_path, 'r') as f:
            char_to_num = json.load(f)
        
        char_to_num[PAD_TOKEN] = max(char_to_num.values()) + 1
        char_to_num[START_TOKEN] = max(char_to_num.values()) + 1
        char_to_num[END_TOKEN] = max(char_to_num.values()) + 1
        
        num_to_char = {j: i for i, j in char_to_num.items()}
        return char_to_num, num_to_char
    except Exception as e:
        st.error(f"Error al cargar el mapeo de caracteres: {str(e)}")
        return None, None

def resize_pad(x):
    """Redimensiona y rellena secuencias"""
    if tf.shape(x)[0] < FRAME_LEN:
        x = tf.pad(x, ([[0, FRAME_LEN - tf.shape(x)[0]], [0, 0], [0, 0]]))
    else:
        x = tf.image.resize(x, (FRAME_LEN, tf.shape(x)[1]))
    return x

def pre_process(x):
    """Preprocesa los landmarks"""
    if not isinstance(x, tf.Tensor):
        x = tf.constant(x, dtype=tf.float32)
    
    # Extraer manos y poses
    rhand = tf.gather(x, RHAND_IDX, axis=1)
    lhand = tf.gather(x, LHAND_IDX, axis=1)
    rpose = tf.gather(x, RPOSE_IDX, axis=1)
    lpose = tf.gather(x, LPOSE_IDX, axis=1)
    
    # Determinar qué mano usar
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
    
    # Normalizar mano
    hand_len = len(LHAND_IDX) // 3
    hand_x = hand[:, 0*hand_len:1*hand_len]
    hand_y = hand[:, 1*hand_len:2*hand_len]
    hand_z = hand[:, 2*hand_len:3*hand_len]
    hand = tf.stack([hand_x, hand_y, hand_z], axis=-1)
    
    mean = tf.math.reduce_mean(hand, axis=1, keepdims=True)
    std = tf.math.reduce_std(hand, axis=1, keepdims=True)
    hand = (hand - mean) / (std + 1e-8)
    
    # Normalizar pose
    pose_len = len(LPOSE_IDX) // 3
    pose_x = pose[:, 0*pose_len:1*pose_len]
    pose_y = pose[:, 1*pose_len:2*pose_len]
    pose_z = pose[:, 2*pose_len:3*pose_len]
    pose = tf.stack([pose_x, pose_y, pose_z], axis=-1)
    
    x = tf.concat([hand, pose], axis=1)
    x = resize_pad(x)
    x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)
    x = tf.reshape(x, (FRAME_LEN, -1))
    
    return x

def load_parquet_data(file_path):
    """Carga datos de un archivo parquet"""
    try:
        df = pd.read_parquet(file_path, columns=SEL_COLS)
        return df
    except Exception as e:
        st.error(f"Error al cargar archivo: {str(e)}")
        return None

def predict_sequence(model, landmarks, char_to_num, num_to_char, temperature=0.8):
    """Realiza predicción en una secuencia de landmarks"""
    try:
        processed = pre_process(landmarks.values.astype(np.float32))
        processed = tf.expand_dims(processed, 0) 
        
        # Generar predicción
        start_token = char_to_num[START_TOKEN]
        dec_input = tf.ones((1, 1), dtype=tf.int32) * start_token
        
        for i in range(TARGET_MAXLEN - 1):
            predictions = model([processed, dec_input], training=False)
            
            logits = predictions[:, -1, :] / temperature
            probabilities = tf.nn.softmax(logits, axis=-1)
            next_token = tf.argmax(probabilities, axis=-1, output_type=tf.int32)
            next_token = tf.expand_dims(next_token, -1)
            
            dec_input = tf.concat([dec_input, next_token], axis=-1)
            
            if next_token[0, 0].numpy() == char_to_num[END_TOKEN]:
                break
        
        tokens = dec_input[0].numpy()
        result = []
        for token in tokens:
            char = num_to_char[token]
            if char == END_TOKEN:
                break
            if char not in [START_TOKEN, PAD_TOKEN]:
                result.append(char)
        
        return ''.join(result)
    except Exception as e:
        st.error(f"Error en predicción: {str(e)}")
        return None

def create_landmark_visualization(df):
    """Crea visualización 3D de landmarks"""
    frame_0 = df.iloc[0]
    
    x_coords = [frame_0[col] for col in X]
    y_coords = [frame_0[col] for col in Y]
    z_coords = [frame_0[col] for col in Z]
    
    fig = go.Figure()
    
    # Mano derecha
    rhand_x = x_coords[:21]
    rhand_y = y_coords[:21]
    rhand_z = z_coords[:21]
    
    fig.add_trace(go.Scatter3d(
        x=rhand_x, y=rhand_y, z=rhand_z,
        mode='markers+lines',
        name='Mano Derecha',
        marker=dict(size=5, color=COLORS['primary']),
        line=dict(color=COLORS['primary'], width=2)
    ))
    
    # Mano izquierda
    lhand_x = x_coords[21:42]
    lhand_y = y_coords[21:42]
    lhand_z = z_coords[21:42]
    
    fig.add_trace(go.Scatter3d(
        x=lhand_x, y=lhand_y, z=lhand_z,
        mode='markers+lines',
        name='Mano Izquierda',
        marker=dict(size=5, color=COLORS['accent']),
        line=dict(color=COLORS['accent'], width=2)
    ))
    
    # Pose
    pose_x = x_coords[42:]
    pose_y = y_coords[42:]
    pose_z = z_coords[42:]
    
    fig.add_trace(go.Scatter3d(
        x=pose_x, y=pose_y, z=pose_z,
        mode='markers',
        name='Pose',
        marker=dict(size=8, color=COLORS['success'])
    ))
    
    fig.update_layout(
        title='Visualización de Landmarks (Frame 0)',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            bgcolor=COLORS['background']
        ),
        paper_bgcolor=COLORS['card'],
        font=dict(color=COLORS['text']),
        height=600
    )
    
    return fig

def create_temporal_analysis(df):
    """Crea análisis temporal de velocidad 3D de la mano derecha (robusto a NaN)."""

    if len(df) < 2:
        fig = go.Figure()
        fig.update_layout(
            title='Análisis Temporal: (insuficientes frames)',
            paper_bgcolor=COLORS['card'],
            plot_bgcolor=COLORS['background'],
            font=dict(color=COLORS['text']),
            height=400
        )
        return fig

    Rx = df[X[:21]].to_numpy(dtype="float32")
    Ry = df[Y[:21]].to_numpy(dtype="float32")
    Rz = df[Z[:21]].to_numpy(dtype="float32")

    Rx = np.nan_to_num(Rx, nan=0.0)
    Ry = np.nan_to_num(Ry, nan=0.0)
    Rz = np.nan_to_num(Rz, nan=0.0)

    dRx = np.diff(Rx, axis=0)
    dRy = np.diff(Ry, axis=0)
    dRz = np.diff(Rz, axis=0)

    sq = dRx**2 + dRy**2 + dRz**2
    vel = np.sqrt(np.sum(sq, axis=1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.arange(1, len(df)),  
        y=vel,
        mode='lines',
        name='Velocidad',
        line=dict(color=COLORS['primary'], width=2),
        fill='tozeroy'
    ))
    fig.update_layout(
        title='Análisis Temporal: Velocidad de Movimiento (mano derecha)',
        xaxis_title='Frame',
        yaxis_title='Velocidad',
        paper_bgcolor=COLORS['card'],
        plot_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text']),
        height=400
    )
    return fig

def create_coordinate_distribution(df):
    """Crea distribución de coordenadas"""
    all_x = df[X].values.flatten()
    all_x = all_x[~np.isnan(all_x)]
    
    # Histograma
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=all_x,
        nbinsx=50,
        name='Coordenadas X',
        marker_color=COLORS['secondary']
    ))
    
    fig.update_layout(
        title='Distribución de Coordenadas X',
        xaxis_title='Valor de X',
        yaxis_title='Frecuencia',
        paper_bgcolor=COLORS['card'],
        plot_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text']),
        height=400
    )
    
    return fig

def create_correlation_heatmap(df):
    """Crea mapa de calor de correlaciones"""
    sample_cols = X[:10] + Y[:10]  
    
    # Calcular correlaciones
    corr_matrix = df[sample_cols].corr()
    
    # Heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=[col.replace('x_', '').replace('y_', '') for col in sample_cols],
        y=[col.replace('x_', '').replace('y_', '') for col in sample_cols],
        colorscale='RdBu',
        zmid=0
    ))
    
    fig.update_layout(
        title='Correlación entre Landmarks',
        paper_bgcolor=COLORS['card'],
        font=dict(color=COLORS['text']),
        height=500
    )
    
    return fig

# GIF
def _hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def create_landmark_gif(df, fps=10, width=800, height=700, max_frames=300):
    """
    Genera un GIF (bytes) recorriendo todos los frames.
    Requiere: pip install kaleido imageio
    - fps: cuadros por segundo del GIF
    - width/height: tamaño del render
    - max_frames: limita frames para evitar GIFs gigantes (ajusta si quieres)
    """
    n_frames = len(df)
    if n_frames < 1:
        return None

    n_use = min(n_frames, max_frames)

    def _global_range(cols):
        vals = df[cols].to_numpy().astype("float32")
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            return (0.0, 1.0)
        lo, hi = float(vals.min()), float(vals.max())
        if lo == hi:
            lo -= 0.5
            hi += 0.5
        margin = 0.05 * (hi - lo)
        return (lo - margin, hi + margin)

    x_min, x_max = _global_range(X)
    y_min, y_max = _global_range(Y)
    z_min, z_max = _global_range(Z)

    c_r = COLORS['primary']
    c_l = COLORS['accent']
    c_p = COLORS['success']

    frames_imgs = []

    for i in range(n_use):
        row = df.iloc[i]

        def _safe_vals(cols):
            arr = row[cols].to_numpy(dtype="float32")
            arr = np.nan_to_num(arr, nan=0.0)
            return arr

        Rx, Ry, Rz = _safe_vals(X[:21]), _safe_vals(Y[:21]), _safe_vals(Z[:21])
        Lx, Ly, Lz = _safe_vals(X[21:42]), _safe_vals(Y[21:42]), _safe_vals(Z[21:42])
        Px, Py, Pz = _safe_vals(X[42:]), _safe_vals(Y[42:]), _safe_vals(Z[42:])

        fig = go.Figure()

        fig.add_trace(go.Scatter3d(
            x=Rx, y=Ry, z=Rz,
            mode='markers+lines',
            name='Mano Derecha',
            marker=dict(size=5, color=c_r),
            line=dict(color=c_r, width=2)
        ))
        fig.add_trace(go.Scatter3d(
            x=Lx, y=Ly, z=Lz,
            mode='markers+lines',
            name='Mano Izquierda',
            marker=dict(size=5, color=c_l),
            line=dict(color=c_l, width=2)
        ))
        fig.add_trace(go.Scatter3d(
            x=Px, y=Py, z=Pz,
            mode='markers',
            name='Pose',
            marker=dict(size=8, color=c_p)
        ))

        fig.update_layout(
            title=f'Landmarks — Frame {i+1}/{n_use}',
            scene=dict(
                xaxis_title='X', xaxis=dict(range=[x_min, x_max]),
                yaxis_title='Y', yaxis=dict(range=[y_min, y_max]),
                zaxis_title='Z', zaxis=dict(range=[z_min, z_max]),
                bgcolor=COLORS['background']
            ),
            paper_bgcolor=COLORS['card'],
            font=dict(color=COLORS['text']),
            width=width,
            height=height,
            margin=dict(l=0, r=0, b=0, t=50)
        )

        png_bytes = fig.to_image(format="png", width=width, height=height, scale=1)
        frames_imgs.append(imageio.imread(png_bytes))

    # Guardar GIF en la memoria
    buf = io.BytesIO()
    imageio.mimsave(buf, frames_imgs, format="GIF", duration=1.0/float(fps))
    buf.seek(0)
    return buf.read()

def create_landmark_animation_3d(df, step=2, max_frames=300):

    n = len(df)
    if n < 1: 
        return go.Figure()

    idxs = list(range(0, min(n, max_frames), step))

    def _frame_vals(i):
        row = df.iloc[i]
        def _safe(cols):
            arr = row[cols].to_numpy(dtype="float32")
            return np.nan_to_num(arr, nan=0.0)
        Rx, Ry, Rz = _safe([f'x_right_hand_{k}' for k in range(21)]), \
                     _safe([f'y_right_hand_{k}' for k in range(21)]), \
                     _safe([f'z_right_hand_{k}' for k in range(21)])
        Lx, Ly, Lz = _safe([f'x_left_hand_{k}'  for k in range(21)]), \
                     _safe([f'y_left_hand_{k}'  for k in range(21)]), \
                     _safe([f'z_left_hand_{k}'  for k in range(21)])
        return (Rx, Ry, Rz, Lx, Ly, Lz)

    def _rng(cols):
        vals = df[cols].to_numpy(dtype="float32")
        vals = vals[~np.isnan(vals)]
        if vals.size == 0: return (0.0, 1.0)
        lo, hi = float(vals.min()), float(vals.max())
        if lo == hi: lo -= .5; hi += .5
        m = 0.05*(hi-lo)
        return lo-m, hi+m

    x_min, x_max = _rng([f'x_right_hand_{k}' for k in range(21)] + [f'x_left_hand_{k}' for k in range(21)])
    y_min, y_max = _rng([f'y_right_hand_{k}' for k in range(21)] + [f'y_left_hand_{k}' for k in range(21)])
    z_min, z_max = _rng([f'z_right_hand_{k}' for k in range(21)] + [f'z_left_hand_{k}' for k in range(21)])

    def _skeleton_lines(x, y, z):
        Xl, Yl, Zl = [], [], []
        for a,b in HAND_CONNECTIONS:
            Xl += [x[a], x[b], None]
            Yl += [y[a], y[b], None]
            Zl += [z[a], z[b], None]
        return Xl, Yl, Zl

    Rx,Ry,Rz,Lx,Ly,Lz = _frame_vals(idxs[0])
    rXl,rYl,rZl = _skeleton_lines(Rx,Ry,Rz)
    lXl,lYl,lZl = _skeleton_lines(Lx,Ly,Lz)

    fig = go.Figure(
        data=[
            go.Scatter3d(x=Rx, y=Ry, z=Rz, mode='markers', name='Right (pts)',
                         marker=dict(size=4, color=COLORS['primary'])),
            go.Scatter3d(x=Lx, y=Ly, z=Lz, mode='markers', name='Left (pts)',
                         marker=dict(size=4, color=COLORS['accent'])),
            go.Scatter3d(x=rXl, y=rYl, z=rZl, mode='lines', name='Right (skel)',
                         line=dict(color=COLORS['primary'], width=3)),
            go.Scatter3d(x=lXl, y=lYl, z=lZl, mode='lines', name='Left (skel)',
                         line=dict(color=COLORS['accent'], width=3)),
        ],
        layout=go.Layout(
            title=f'Hands — Frame 1/{len(idxs)}',
            scene=dict(
                xaxis=dict(range=[x_min, x_max], title='X'),
                yaxis=dict(range=[y_min, y_max], title='Y'),
                zaxis=dict(range=[z_min, z_max], title='Z'),
                bgcolor=COLORS['background']
            ),
            paper_bgcolor=COLORS['card'],
            font=dict(color=COLORS['text']),
            height=650,
            updatemenus=[dict(
                type="buttons", showactive=False, y=1.08, x=0.0, xanchor="left",
                buttons=[
                    dict(label="▶ Play", method="animate",
                         args=[None, {"frame":{"duration":50,"redraw":True},"fromcurrent":True,"transition":{"duration":0}}]),
                    dict(label="⏸ Pause", method="animate",
                         args=[[None], {"frame":{"duration":0,"redraw":False},"mode":"immediate","transition":{"duration":0}}])
                ]
            )],
            sliders=[{
                "active": 0, "y": -0.05, "x": 0.1, "len": 0.8,
                "currentvalue": {"prefix": "Frame: "},
                "steps": [{"args": [[str(k)], {"frame":{"duration":0,"redraw":True},"mode":"immediate"}],
                           "label": str(i+1), "method": "animate"} for i,k in enumerate(idxs)]
            }]
        ),
        frames=[
            (lambda k: go.Frame(
                name=str(k),
                data=[
                    go.Scatter3d(x=_frame_vals(k)[0], y=_frame_vals(k)[1], z=_frame_vals(k)[2],
                                 mode='markers', marker=dict(size=4, color=COLORS['primary'])),
                    go.Scatter3d(x=_frame_vals(k)[3], y=_frame_vals(k)[4], z=_frame_vals(k)[5],
                                 mode='markers', marker=dict(size=4, color=COLORS['accent'])),
                    (lambda RX,RY,RZ: go.Scatter3d(
                        x=_skeleton_lines(RX,RY,RZ)[0], y=_skeleton_lines(RX,RY,RZ)[1], z=_skeleton_lines(RX,RY,RZ)[2],
                        mode='lines', line=dict(color=COLORS['primary'], width=3)
                    ))(*_frame_vals(k)[:3]),
                    (lambda LX,LY,LZ: go.Scatter3d(
                        x=_skeleton_lines(LX,LY,LZ)[0], y=_skeleton_lines(LX,LY,LZ)[1], z=_skeleton_lines(LX,LY,LZ)[2],
                        mode='lines', line=dict(color=COLORS['accent'], width=3)
                    ))(*_frame_vals(k)[3:6]),
                ],
                layout=go.Layout(title=f'Hands — Frame {idxs.index(k)+1}/{len(idxs)}')
            ))(k)
            for k in idxs
        ]
    )
    return fig

def compute_right_hand_velocity(df):
    Rx = df[[f'x_right_hand_{i}' for i in range(21)]].to_numpy(dtype="float32")
    Ry = df[[f'y_right_hand_{i}' for i in range(21)]].to_numpy(dtype="float32")
    Rz = df[[f'z_right_hand_{i}' for i in range(21)]].to_numpy(dtype="float32")
    Rx, Ry, Rz = np.nan_to_num(Rx, nan=0.0), np.nan_to_num(Ry, nan=0.0), np.nan_to_num(Rz, nan=0.0)
    if len(Rx) < 2: 
        return np.array([], dtype="float32")
    dRx, dRy, dRz = np.diff(Rx, axis=0), np.diff(Ry, axis=0), np.diff(Rz, axis=0)
    vel = np.sqrt(np.sum(dRx**2 + dRy**2 + dRz**2, axis=1))  # (t-1,)
    return vel

def build_corr_df(df, hands=('Right',), axes=('X','Y'), landmarks=range(21), max_features=60):
    sel_cols = []
    axis_map = {'X':'x', 'Y':'y', 'Z':'z'}
    hand_map = {'Right':'right_hand', 'Left':'left_hand'}
    for hand in hands:
        for ax in axes:
            prefix = f"{axis_map[ax]}_{hand_map[hand]}_"
            for i in landmarks:
                col = f"{prefix}{i}"
                if col in df.columns:
                    sel_cols.append(col)

    sel_cols = list(dict.fromkeys(sel_cols))  
    if len(sel_cols) == 0:
        return pd.DataFrame()

    if len(sel_cols) > max_features:
        sel_cols = sel_cols[:max_features]

    sub = df[sel_cols].copy()
    sub = sub.apply(pd.to_numeric, errors='coerce')
    sub = sub.fillna(method='ffill').fillna(method='bfill').fillna(0.0)
    return sub

def build_feature_df(df, hands=('Right',), axes=('X','Y'), landmarks=range(21), max_features=40):
    """Construye un DataFrame con columnas de mano/eje/landmark seleccionados."""
    sel_cols = []
    axis_map = {'X':'x', 'Y':'y', 'Z':'z'}
    hand_map = {'Right':'right_hand', 'Left':'left_hand'}
    for hand in hands:
        for ax in axes:
            prefix = f"{axis_map[ax]}_{hand_map[hand]}_"
            for i in landmarks:
                col = f"{prefix}{i}"
                if col in df.columns:
                    sel_cols.append(col)
    sel_cols = list(dict.fromkeys(sel_cols))
    if len(sel_cols) == 0:
        return pd.DataFrame()

    if len(sel_cols) > max_features:
        sel_cols = sel_cols[:max_features]

    sub = df[sel_cols].copy()
    sub = sub.apply(pd.to_numeric, errors='coerce')
    sub = sub.fillna(method='ffill').fillna(method='bfill').fillna(0.0)
    return sub

def pca_2d_numpy(dataframe):
    """PCA 2D sin dependencias externas (centrado + SVD). Devuelve coords y varianza explicada."""
    X = dataframe.to_numpy(dtype="float64")
    X = np.nan_to_num(X, nan=0.0)
    X = X - X.mean(axis=0, keepdims=True)
    # SVD
    U, S, VT = np.linalg.svd(X, full_matrices=False)
    comps = VT[:2]                 
    coords = X @ comps.T           
    var_total = (S**2).sum()
    var_exp = (S[:2]**2) / var_total if var_total > 0 else np.array([0.0, 0.0])
    return coords, var_exp

def compact_labels(cols):
    """Etiquetas compactas R/L + eje (X/Y/Z) para los gráficos."""
    labels = []
    for c in cols:
        lbl = c.replace('right_hand','R').replace('left_hand','L')
        lbl = lbl.replace('x_','X.').replace('y_','Y.').replace('z_','Z.')
        labels.append(lbl)
    return labels

# Función principal
def main():
    st.markdown("""
    <h1>🤟 ASL Fingerspelling Recognition</h1>
    <p style='text-align: center; color: #7F8C8D; font-size: 1.2em;'>
        Sistema de reconocimiento de deletreo en lenguaje de señas americano
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    with st.sidebar:

        st.markdown(f"""
        <div style='background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%); 
                    padding: 20px; border-radius: 10px; color: white; margin-bottom: 20px;'>
            <h2 style='color: white; margin: 0;'>⚙️ Configuración</h2>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("📑 Navegación")
        page = st.radio(
            "Selecciona una página:",
            ["🏠 Inicio", "📊 Exploración de Datos", "🎯 Predicción", "📈 Métricas del Modelo"],
            label_visibility="collapsed"
        )

        st.subheader("🎨 Configuraciones")
        show_metrics = st.checkbox("Mostrar métricas", value=True)
        temperature = st.slider("Temperatura de predicción", 0.1, 2.0, 0.8, 0.1)

        st.subheader("📦 Modelo (automático)")
        model = None
        try:
            if os.path.exists(MODEL_PATH):
                model = load_model(MODEL_PATH)
                if model:
                    st.success(f"✅ Modelo cargado: {MODEL_PATH}")
            else:
                st.error(f"❌ No se encontró {MODEL_PATH} en la carpeta.")
        except Exception as e:
            st.error(f"❌ Error al cargar el modelo: {e}")

        st.subheader("🔤 Mapeo de Caracteres (automático)")
        char_to_num, num_to_char = None, None
        try:
            if os.path.exists(CHAR_MAP_PATH):
                char_to_num, num_to_char = load_char_map(CHAR_MAP_PATH)
                if char_to_num:
                    st.success(f"✅ Mapeo cargado: {CHAR_MAP_PATH}")
            else:
                st.error(f"❌ No se encontró {CHAR_MAP_PATH} en la carpeta.")
        except Exception as e:
            st.error(f"❌ Error al cargar el mapeo: {e}")

        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #95A5A6; font-size: 0.9em;'>
            <p>Streamlit + TensorFlow</p>
        </div>
        """, unsafe_allow_html=True)
        

    # Páginas
    if page == "🏠 Inicio":
        st.markdown("""
        <div class='info-box'>
            <h2>👋 Bienvenido al Sistema ASL Fingerspelling</h2>
            <p style='font-size: 1.1em;'>
                Esta aplicación permite analizar datos de movimientos de manos y realizar 
                predicciones de deletreo en lenguaje de señas americano (ASL) utilizando 
                deep learning.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div style='background-color: {COLORS['card']}; padding: 20px; border-radius: 10px; 
                        border-left: 4px solid {COLORS['primary']}; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                <h3 style='color: {COLORS['primary']};'>📊 Exploración</h3>
                <p>Visualiza y analiza las características de tus datos con gráficas interactivas.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style='background-color: {COLORS['card']}; padding: 20px; border-radius: 10px; 
                        border-left: 4px solid {COLORS['accent']}; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                <h3 style='color: {COLORS['accent']};'>🎯 Predicción</h3>
                <p>Realiza predicciones en nuevos datos usando el modelo entrenado.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style='background-color: {COLORS['card']}; padding: 20px; border-radius: 10px; 
                        border-left: 4px solid {COLORS['success']}; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                <h3 style='color: {COLORS['success']};'>📈 Métricas</h3>
                <p>Evalúa el rendimiento del modelo con métricas detalladas.</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### 🚀 Para comenzar:")
        st.markdown(f"""
        1. Asegúrate de que **{MODEL_PATH}** y **{CHAR_MAP_PATH}** estén en la misma carpeta que esta app.
        2. Ve a **📊 Exploración de Datos** para analizar un Parquet.
        3. En **🎯 Predicción**, carga un Parquet y obtén el texto predicho.
        """)
        
        st.markdown("### 📋 Características del Sistema:")
        
        features_col1, features_col2 = st.columns(2)
        
        with features_col1:
            st.markdown(f"""
            - ✨ **Visualización 3D** de landmarks
            - 📊 **Análisis temporal** de movimientos
            - 🔄 **Gráficas interactivas** y enlazadas
            - 🎨 **Dashboard intuitivo** con teoría del color
            """)
        
        with features_col2:
            st.markdown(f"""
            - 🤖 **Predicción en tiempo real**
            - 📈 **Métricas de rendimiento** detalladas
            - 💾 **Carga de datos** en formato Parquet
            - 🎛️ **Configuración ajustable** de parámetros
            """)
    
    elif page == "📊 Exploración de Datos":
        st.markdown("<h2>📊 Exploración Interactiva de Datos</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='info-box'>
            <p>Carga un archivo Parquet para explorar las características y variables del conjunto de datos.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Cargar archivo
        uploaded_file = st.file_uploader("📁 Cargar archivo Parquet", type=['parquet'])
        
        if uploaded_file:
            # Cargar datos
            df = load_parquet_data(uploaded_file)
            
            if df is not None:
                st.markdown("<div class='success-box'>✅ Archivo cargado exitosamente</div>", unsafe_allow_html=True)
                
                st.markdown("### 📋 Información General")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-label'>Frames</div>
                        <div class='metric-value'>{len(df)}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-label'>Variables</div>
                        <div class='metric-value'>{len(df.columns)}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    missing = df.isnull().sum().sum()
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-label'>Valores Faltantes</div>
                        <div class='metric-value'>{missing}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    memory = df.memory_usage(deep=True).sum() / 1024**2
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-label'>Memoria (MB)</div>
                        <div class='metric-value'>{memory:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                tab1, tab2, tab3, tab4 = st.tabs(["🎨 Visualización 3D", "⏱️ Análisis Temporal", "📊 Distribuciones", "🔥 Correlaciones"])
                
                with tab1:
                    st.markdown("### 🎨 Visualización 3D de Landmarks")
                    st.markdown("""
                    <div class='info-box'>
                        <p>Animación interactiva de toda la secuencia. Usa Play/Pause o desliza para explorar frames.</p>
                    </div>
                    """, unsafe_allow_html=True)

                    step = 2          
                    max_frames = 300  

                    with st.spinner("Preparando animación…"):
                        anim_fig = create_landmark_animation_3d(df, step=step, max_frames=max_frames)
                        st.plotly_chart(anim_fig, use_container_width=True)
                                
                with tab2:
                    st.markdown("### ⏱️ Análisis Temporal de Movimientos")
                    st.markdown("""
                    <div class='info-box'>
                        <p>Análisis de la velocidad de movimiento a lo largo del tiempo.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.spinner("Calculando velocidades..."):
                        fig_temporal = create_temporal_analysis(df)
                        st.plotly_chart(fig_temporal, use_container_width=True)
                    
                    st.markdown("#### 📊 Estadísticas de Movimiento")
                    vel = compute_right_hand_velocity(df)
                    if vel.size == 0:
                        v_mean = v_max = v_std = 0.0
                    else:
                        v_mean = float(np.nanmean(vel))
                        v_max  = float(np.nanmax(vel))
                        v_std  = float(np.nanstd(vel))

                    stat_col1, stat_col2, stat_col3 = st.columns(3)
                    with stat_col1: st.metric("Velocidad Media", f"{v_mean:.4f}")
                    with stat_col2: st.metric("Velocidad Máxima", f"{v_max:.4f}")
                    with stat_col3: st.metric("Desv. Estándar", f"{v_std:.4f}")
                
                with tab3:
                    st.markdown("### 📊 Distribución de Coordenadas")
                    st.markdown("""
                    <div class='info-box'>
                        <p>Distribución de las coordenadas X de todos los landmarks.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.spinner("Generando histograma..."):
                        fig_dist = create_coordinate_distribution(df)
                        st.plotly_chart(fig_dist, use_container_width=True)
                    
                    coord_type = st.selectbox("Selecciona tipo de coordenada:", ["X", "Y", "Z"])
                    
                    if coord_type == "Y":
                        all_coords = df[Y].values.flatten()
                    elif coord_type == "Z":
                        all_coords = df[Z].values.flatten()
                    else:
                        all_coords = df[X].values.flatten()
                    
                    all_coords = all_coords[~np.isnan(all_coords)]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=all_coords,
                        nbinsx=50,
                        name=f'Coordenadas {coord_type}',
                        marker_color=COLORS['accent']
                    ))
                    fig.update_layout(
                        title=f'Distribución de Coordenadas {coord_type}',
                        xaxis_title=f'Valor de {coord_type}',
                        yaxis_title='Frecuencia',
                        paper_bgcolor=COLORS['card'],
                        plot_bgcolor=COLORS['background'],
                        font=dict(color=COLORS['text']),
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab4:
                    st.markdown("### 🧭 Explorador de Relaciones entre Features (manos)")
                    st.markdown("""
                    <div class='info-box'>
                        <p>Elige mano(s), ejes y landmarks para visualizar relaciones: Matriz de dispersión, Correlaciones o PCA 2D.</p>
                    </div>
                    """, unsafe_allow_html=True)

                    top_c1, top_c2 = st.columns([1.4, 1.0])
                    with top_c1:
                        vis_type = st.selectbox("Tipo de visualización",
                                                ["Matriz de dispersión (SPLOM)", "Mapa de calor de correlación", "PCA 2D"],
                                                index=0)
                    with top_c2:
                        sample_rows = st.slider("Muestreo de filas (para agilizar)", 200, 5000, 1500, 100)

                    c1, c2, c3 = st.columns([1.2,1.2,1.6])
                    with c1:
                        hands_sel = st.multiselect("Manos", options=["Right","Left"], default=["Right","Left"])
                    with c2:
                        axes_sel = st.multiselect("Ejes", options=["X","Y","Z"], default=["X","Y"])
                    with c3:
                        idxs = st.multiselect("Landmarks (0–20)", options=list(range(21)), default=list(range(21)))

                    feat_df = build_feature_df(df, hands=tuple(hands_sel), axes=tuple(axes_sel), landmarks=idxs, max_features=40)

                    if feat_df.empty or feat_df.shape[1] < 2:
                        st.warning("Selecciona al menos 2 features válidas para visualizar.")
                    else:
                        if len(feat_df) > sample_rows:
                            feat_df = feat_df.sample(sample_rows, random_state=42).reset_index(drop=True)

                        feat_df.columns = compact_labels(feat_df.columns)

                        if vis_type == "Matriz de dispersión (SPLOM)":
                            import plotly.express as px
                            max_cols = 9
                            if feat_df.shape[1] > max_cols:
                                st.info(f"Mostrando {max_cols} de {feat_df.shape[1]} features para mantener la gráfica ágil.")
                                feat_df_plot = feat_df.iloc[:, :max_cols]
                            else:
                                feat_df_plot = feat_df

                            fig = px.scatter_matrix(
                                feat_df_plot,
                                dimensions=list(feat_df_plot.columns),
                                opacity=0.6,
                                height=700
                            )
                            fig.update_layout(
                                paper_bgcolor=COLORS['card'],
                                plot_bgcolor=COLORS['background'],
                                font=dict(color=COLORS['text'])
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        elif vis_type == "Mapa de calor de correlación":
                            corr = feat_df.corr()
                            fig = go.Figure(data=go.Heatmap(
                                z=corr.values,
                                x=list(corr.columns),
                                y=list(corr.columns),
                                colorscale='RdBu',
                                zmid=0
                            ))
                            fig.update_layout(
                                title=f'Correlación ({corr.shape[1]} features)',
                                paper_bgcolor=COLORS['card'],
                                font=dict(color=COLORS['text']),
                                height=600
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            st.markdown("### Landmarks de la Mano")
                            st.image("imagenes/hand_landmarks.png", caption="Landmarks", use_column_width=True)
                            st.markdown("### Landmarks del cuerpo")
                            st.image("imagenes/body_landmarks.jpg", caption="Landmarks", use_column_width=True)
                            
                        else:  
                            import plotly.express as px
                            coords, var_exp = pca_2d_numpy(feat_df)
                            pc_df = pd.DataFrame({'PC1': coords[:,0], 'PC2': coords[:,1]})
                            caption = f"Varianza explicada: PC1={var_exp[0]*100:.1f}%  •  PC2={var_exp[1]*100:.1f}%"
                            fig = px.scatter(pc_df, x="PC1", y="PC2", opacity=0.7, height=600)
                            fig.update_layout(
                                title=caption,
                                paper_bgcolor=COLORS['card'],
                                plot_bgcolor=COLORS['background'],
                                font=dict(color=COLORS['text'])
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        with st.expander("🔎 Variables incluidas"):
                            st.write(list(feat_df.columns))

                st.markdown("---")
                st.markdown("### 📄 Vista de Datos")
                
                show_data = st.checkbox("Mostrar datos crudos")
                if show_data:
                    st.dataframe(df.head(100), use_container_width=True, height=400)
                    
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Descargar datos como CSV",
                        csv,
                        "landmarks_data.csv",
                        "text/csv",
                        key='download-csv'
                    )
        
        else:
            st.markdown("""
            <div class='warning-box'>
                ⚠️ Por favor, carga un archivo Parquet para comenzar la exploración.
            </div>
            """, unsafe_allow_html=True)
    
    elif page == "🎯 Predicción":
        st.markdown("<h2>🎯 Sistema de Predicción</h2>", unsafe_allow_html=True)
        
        if model is None or char_to_num is None:
            st.markdown(f"""
            <div class='warning-box'>
                ⚠️ No se pudo cargar el modelo o el mapeo automáticamente.<br>
                Verifica que <code>{MODEL_PATH}</code> y <code>{CHAR_MAP_PATH}</code> estén en la carpeta.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='success-box'>
                ✅ Sistema listo para realizar predicciones
            </div>
            """, unsafe_allow_html=True)
            
            pred_file = st.file_uploader("📁 Cargar archivo Parquet para predicción", type=['parquet'], key='pred_file')
            
            if pred_file:
                df = load_parquet_data(pred_file)
                
                if df is not None:
                    st.markdown(f"""
                    <div class='info-box'>
                        📊 Datos cargados: {len(df)} frames con {len(df.columns)} variables
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        if st.button("🚀 Realizar Predicción", use_container_width=True):
                            with st.spinner("Procesando datos y realizando predicción..."):
                                start_time = time.time()
                                
                                prediction = predict_sequence(model, df, char_to_num, num_to_char, temperature)
                                
                                elapsed_time = time.time() - start_time
                                
                                if prediction:
                                    st.markdown("---")
                                    st.markdown("### 🎉 Resultado de la Predicción")
                                    
                                    st.markdown(f"""
                                    <div style='background: linear-gradient(135deg, {COLORS['success']} 0%, {COLORS['primary']} 100%);
                                                padding: 30px; border-radius: 15px; text-align: center; color: white;
                                                box-shadow: 0 8px 16px rgba(0,0,0,0.2); margin: 20px 0;'>
                                        <h2 style='color: white; margin: 0 0 10px 0;'>Texto Predicho:</h2>
                                        <h1 style='color: white; font-size: 3em; margin: 0; font-weight: bold;'>
                                            "{prediction}"
                                        </h1>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        st.markdown(f"""
                                        <div class='metric-card'>
                                            <div class='metric-label'>Tiempo de Predicción</div>
                                            <div class='metric-value'>{elapsed_time:.2f}s</div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    
                                    with col2:
                                        st.markdown(f"""
                                        <div class='metric-card'>
                                            <div class='metric-label'>Longitud</div>
                                            <div class='metric-value'>{len(prediction)}</div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    
                                    with col3:
                                        st.markdown(f"""
                                        <div class='metric-card'>
                                            <div class='metric-label'>Frames Procesados</div>
                                            <div class='metric-value'>{len(df)}</div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    
                                    # Visualización de los datos
                                    st.markdown("---")
                                    st.markdown("### 📊 Visualización de los Datos de Entrada")
                                    
                                    viz_tab1, viz_tab2 = st.tabs(["🎬 Animación 3D (manos)", "⏱️ Análisis Temporal"])

                                    with viz_tab1:
                                        step = 2          
                                        max_frames = 300  
                                        with st.spinner("Preparando animación 3D…"):
                                            anim_fig = create_landmark_animation_3d(df, step=step, max_frames=max_frames)
                                            st.plotly_chart(anim_fig, use_container_width=True)

                                    with viz_tab2:
                                        fig_temporal = create_temporal_analysis(df)
                                        st.plotly_chart(fig_temporal, use_container_width=True)

                                else:
                                    st.error("❌ Error al realizar la predicción")
                    
                    with st.expander("ℹ️ Información sobre la predicción"):
                        st.markdown(f"""
                        **Configuración utilizada:**
                        - Temperatura: {temperature}
                        - Frame length: {FRAME_LEN}
                        - Target max length: {TARGET_MAXLEN}
                        - Vocabulario: {len(char_to_num)} tokens
                        
                        **Proceso:**
                        1. Los landmarks se preprocesan y normalizan
                        2. Se selecciona la mano con menos valores faltantes
                        3. La secuencia se redimensiona a {FRAME_LEN} frames
                        4. El modelo genera la predicción token por token
                        5. Se decodifica la secuencia de tokens a texto
                        """)
            else:
                st.markdown("""
                <div class='info-box'>
                    👆 Carga un archivo Parquet para realizar una predicción
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("📋 Formato de datos esperado"):
                    st.markdown(f"""
                    El archivo Parquet debe contener las siguientes columnas:
                    
                    **Coordenadas de mano derecha (X, Y, Z):**
                    - x_right_hand_0 hasta x_right_hand_20
                    - y_right_hand_0 hasta y_right_hand_20
                    - z_right_hand_0 hasta z_right_hand_20
                    
                    **Coordenadas de mano izquierda (X, Y, Z):**
                    - x_left_hand_0 hasta x_left_hand_20
                    - y_left_hand_0 hasta y_left_hand_20
                    - z_left_hand_0 hasta z_left_hand_20
                    
                    **Coordenadas de pose (X, Y, Z):**
                    - x_pose_{{13, 15, 17, 19, 21, 14, 16, 18, 20, 22}}
                    - y_pose_{{13, 15, 17, 19, 21, 14, 16, 18, 20, 22}}
                    - z_pose_{{13, 15, 17, 19, 21, 14, 16, 18, 20, 22}}
                    
                    **Total:** {len(SEL_COLS)} columnas
                    """)
    
    elif page == "📈 Métricas del Modelo":
        st.markdown("<h2>📈 Métricas y Rendimiento del Modelo</h2>", unsafe_allow_html=True)
        
        if not show_metrics:
            st.markdown("""
            <div class='info-box'>
                ℹ️ Las métricas están ocultas. Activa "Mostrar métricas" en la barra lateral para verlas.
            </div>
            """, unsafe_allow_html=True)
        else:
            if model is None:
                st.markdown("""
                <div class='warning-box'>
                    ⚠️ No se detectó el modelo. Verifica que el archivo exista: <code>my_model.h5</code>.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='success-box'>
                    ✅ Modelo cargado - Visualizando información
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("### 🤖 Arquitectura del Modelo")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-label'>Total de Capas</div>
                        <div class='metric-value'>{len(model.layers)}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    total_params = model.count_params()
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-label'>Parámetros Totales</div>
                        <div class='metric-value'>{total_params:,}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Resumen del modelo
                st.markdown("#### 📋 Resumen de Capas")
                
                layer_info = []
                for layer in model.layers:
                    layer_info.append({
                        'Nombre': layer.name,
                        'Tipo': layer.__class__.__name__,
                        'Output Shape': str(layer.output_shape) if hasattr(layer, 'output_shape') else 'N/A',
                        'Parámetros': layer.count_params()
                    })
                
                df_layers = pd.DataFrame(layer_info)
                st.dataframe(df_layers, use_container_width=True, height=400)
                
                st.markdown("---")
                st.markdown("### 📊 Distribución de Parámetros")
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=[layer['Nombre'] for layer in layer_info if layer['Parámetros'] > 0],
                    y=[layer['Parámetros'] for layer in layer_info if layer['Parámetros'] > 0],
                    marker_color=COLORS['primary'],
                    text=[f"{p:,}" for p in [layer['Parámetros'] for layer in layer_info if layer['Parámetros'] > 0]],
                    textposition='outside'
                ))
                
                fig.update_layout(
                    title='Parámetros por Capa',
                    xaxis_title='Capa',
                    yaxis_title='Número de Parámetros',
                    paper_bgcolor=COLORS['card'],
                    plot_bgcolor=COLORS['background'],
                    font=dict(color=COLORS['text']),
                    height=500,
                    xaxis_tickangle=-45
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                st.markdown("### 🎯 Métricas de Rendimiento (Ejemplo)")
                
                st.markdown("""
                <div class='info-box'>
                    ℹ️ Para obtener métricas reales, carga un conjunto de datos de prueba con etiquetas conocidas.
                </div>
                """, unsafe_allow_html=True)
                
                example_metrics = {
                    'CER': 84.6,
                    'WER': 147.9,
                    'Accuracy': 0.00,
                    'Tiempo Promedio': 0.125
                }
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-label'>CER (%)</div>
                        <div class='metric-value'>{example_metrics['CER']:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-label'>WER (%)</div>
                        <div class='metric-value'>{example_metrics['WER']:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-label'>Accuracy (%)</div>
                        <div class='metric-value'>{example_metrics['Accuracy']:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-label'>Tiempo (s)</div>
                        <div class='metric-value'>{example_metrics['Tiempo Promedio']:.3f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                fig = go.Figure()
                
                metrics_names = ['CER', 'WER', 'Accuracy']
                metrics_values = [example_metrics['CER'], example_metrics['WER'], example_metrics['Accuracy']]
                colors_list = [COLORS['primary'], COLORS['secondary'], COLORS['success']]
                
                fig.add_trace(go.Bar(
                    x=metrics_names,
                    y=metrics_values,
                    marker_color=colors_list,
                    text=[f"{v:.2f}%" for v in metrics_values],
                    textposition='outside'
                ))
                
                fig.update_layout(
                    title='Comparación de Métricas',
                    yaxis_title='Porcentaje (%)',
                    paper_bgcolor=COLORS['card'],
                    plot_bgcolor=COLORS['background'],
                    font=dict(color=COLORS['text']),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("ℹ️ Explicación de Métricas"):
                    st.markdown("""
                    **CER (Character Error Rate):**
                    - Mide el porcentaje de caracteres incorrectos
                    - Valores más bajos son mejores
                    - Calculado usando la distancia de Levenshtein
                    
                    **WER (Word Error Rate):**
                    - Mide el porcentaje de palabras incorrectas
                    - Similar al CER pero a nivel de palabras
                    - Más estricto que el CER
                    
                    **Accuracy:**
                    - Porcentaje de predicciones exactamente correctas
                    - Métrica más estricta
                    - 100% significa coincidencia perfecta
                    
                    **Tiempo Promedio:**
                    - Tiempo de procesamiento por muestra
                    - Incluye preprocesamiento y predicción
                    - Importante para aplicaciones en tiempo real
                    """)

if __name__ == "__main__":
    main()