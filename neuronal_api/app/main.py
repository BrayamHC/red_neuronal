# -*- coding: utf-8 -*-
"""
neuronal_api — FastAPI
Red Neuronal: Kilómetros → Millas
Alumna: Marilu Mendoza Ramírez
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from app.schemas import (
    PredictRequest, PredictResponse,
    LossHistoryResponse, ModelInfoResponse,
)
from app.model import predict_millas, get_loss_history, get_model_weights, load_model

# ── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Red Neuronal API — km → millas",
    description="API para predecir millas a partir de kilómetros usando TensorFlow. Alumna: Marilu Mendoza Ramírez.",
    version="1.0.0",
)

# ── CORS (permite peticiones desde Astro en localhost:4321) ──────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4321", "http://localhost:3000", "*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Archivos estáticos (gráfica PNG) ────────────────────────────────────────
STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ── Rutas ───────────────────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
def root():
    return {
        "proyecto": "Red Neuronal — km → millas",
        "alumna": "Marilu Mendoza Ramírez",
        "docs": "/docs",
        "predict": "/predict",
        "loss": "/loss",
        "info": "/info",
    }


@app.post("/predict", response_model=PredictResponse, tags=["Predicción"])
def predict(request: PredictRequest):
    """
    Recibe un array de kilómetros y devuelve las millas
    predichas por el modelo TensorFlow entrenado.
    """
    try:
        millas = predict_millas(request.kilometros)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {e}")

    return PredictResponse(
        kilometros=request.kilometros,
        millas=millas,
    )


@app.get("/loss", response_model=LossHistoryResponse, tags=["Entrenamiento"])
def loss_history():
    """
    Devuelve el historial de pérdida (MSE) por época.
    Usar para graficar la curva de entrenamiento en la web.
    """
    try:
        data = get_loss_history()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    loss_values = data["loss"]
    return LossHistoryResponse(
        epochs=data["epochs"],
        loss=loss_values,
        min_loss=min(loss_values),
        final_loss=loss_values[-1],
        total_epochs=len(loss_values),
    )


@app.get("/info", response_model=ModelInfoResponse, tags=["Modelo"])
def model_info():
    """
    Devuelve metadatos del modelo: pesos internos, alumna, descripción.
    """
    try:
        peso, sesgo = get_model_weights()
        loss_data   = get_loss_history()
        final_loss  = loss_data["loss"][-1]
        total_ep    = len(loss_data["loss"])
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return ModelInfoResponse(
        alumna="Marilu Mendoza Ramírez",
        modelo="conversion.keras",
        descripcion="Red neuronal de 1 capa Dense entrenada para convertir kilómetros a millas.",
        input="Array de kilómetros (float[])",
        output="Array de millas predichas (float[])",
        epochs_entrenados=total_ep,
        loss_final=final_loss,
        peso=peso,
        sesgo=sesgo,
    )