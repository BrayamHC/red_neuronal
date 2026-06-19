# -*- coding: utf-8 -*-
"""
Red Neuronal: Kilómetros → Millas
Alumna: Marilu Mendoza Ramírez
Framework: TensorFlow / Keras
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Sin GUI, para generar imagen desde servidor
import matplotlib.pyplot as plt
import tensorflow as tf

# ── Datos de entrenamiento ──────────────────────────────────────────────────
kilometros = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)
millas     = np.array([0.621, 1.242, 1.864, 2.485, 3.106, 3.728, 4.349, 4.970, 5.593], dtype=float)

# ── Arquitectura de la red ──────────────────────────────────────────────────
capa   = tf.keras.layers.Dense(units=1, input_shape=[1])
modelo = tf.keras.Sequential([capa])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

# ── Entrenamiento ───────────────────────────────────────────────────────────
print("▶ Iniciando entrenamiento...")

historial = modelo.fit(
    kilometros,
    millas,
    epochs=1000,
    verbose=False
)

print("✓ Entrenamiento finalizado")

# ── Guardar modelo ──────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'conversion.keras')
modelo.save(MODEL_PATH)
print(f"✓ Modelo guardado en: {MODEL_PATH}")

# ── Guardar historial de pérdida como JSON (para la web) ────────────────────
LOSS_PATH = os.path.join(os.path.dirname(__file__), '..', 'app', 'static', 'loss_history.json')
os.makedirs(os.path.dirname(LOSS_PATH), exist_ok=True)

loss_data = {
    "epochs": list(range(1, len(historial.history["loss"]) + 1)),
    "loss":   [float(v) for v in historial.history["loss"]],
}
with open(LOSS_PATH, 'w') as f:
    json.dump(loss_data, f)
print(f"✓ Historial de pérdida guardado en: {LOSS_PATH}")

# ── Guardar gráfica PNG (fallback estático) ─────────────────────────────────
CHART_PATH = os.path.join(os.path.dirname(__file__), '..', 'app', 'static', 'loss_chart.png')

fig, ax = plt.subplots(figsize=(10, 5), facecolor='#0a0a0a')
ax.set_facecolor('#0a0a0a')
ax.plot(historial.history["loss"], color='#00ff41', linewidth=1.5, alpha=0.9)
ax.fill_between(range(len(historial.history["loss"])), historial.history["loss"], alpha=0.08, color='#00ff41')
ax.set_xlabel("Épocas", color='#00ff41', fontsize=10, labelpad=10)
ax.set_ylabel("Pérdida (MSE)", color='#00ff41', fontsize=10, labelpad=10)
ax.set_title("Curva de Entrenamiento — Red Neuronal km→mi", color='#ffffff', fontsize=12, pad=15)
ax.tick_params(colors='#00ff41', which='both')
ax.spines['bottom'].set_color('#00ff41')
ax.spines['left'].set_color('#00ff41')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, color='#00ff41', alpha=0.07, linestyle='--')
plt.tight_layout()
plt.savefig(CHART_PATH, dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
print(f"✓ Gráfica guardada en: {CHART_PATH}")

# ── Verificación ────────────────────────────────────────────────────────────
resultado = modelo.predict(np.array([20.0]))
print(f"\n▶ Predicción: 20 km → {resultado[0][0]:.4f} millas")
print(f"▶ Referencia real: {20 * 0.621:.4f} millas")
print(f"\n▶ Pesos internos de la capa:")
print(f"  peso (×):  {capa.get_weights()[0][0][0]:.6f}  (esperado ≈ 0.621)")
print(f"  sesgo (+): {capa.get_weights()[1][0]:.6f}  (esperado ≈ 0.0)")