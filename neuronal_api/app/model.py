import os
import json
import numpy as np
import tensorflow as tf
from functools import lru_cache

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'conversion.keras')
LOSS_PATH  = os.path.join(os.path.dirname(__file__), 'static', 'loss_history.json')

@lru_cache(maxsize=1)
def load_model() -> tf.keras.Model:
    """Carga el modelo una sola vez y lo cachea en memoria."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Modelo no encontrado en {MODEL_PATH}. "
            "Ejecuta primero: python training/train.py"
        )
    return tf.keras.models.load_model(MODEL_PATH)

def predict_millas(kilometros: list[float]) -> list[float]:
    """Recibe lista de km, devuelve lista de millas predichas por el modelo."""
    model = load_model()
    input_array = np.array(kilometros, dtype=float)
    predictions = model.predict(input_array, verbose=0)
    return [float(p[0]) for p in predictions]

def get_loss_history() -> dict:
    """Lee el historial de pérdida guardado en JSON."""
    if not os.path.exists(LOSS_PATH):
        raise FileNotFoundError(
            "Historial no encontrado. Ejecuta primero: python training/train.py"
        )
    with open(LOSS_PATH) as f:
        return json.load(f)

def get_model_weights() -> tuple[float, float]:
    """Retorna (peso, sesgo) de la única capa densa."""
    model = load_model()
    weights = model.layers[0].get_weights()
    return float(weights[0][0][0]), float(weights[1][0])