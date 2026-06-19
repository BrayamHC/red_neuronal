from pydantic import BaseModel, Field
from typing import List

class PredictRequest(BaseModel):
    kilometros: List[float] = Field(
        ...,
        min_length=1,
        max_length=100,
        examples=[[1.0, 5.0, 10.0, 20.0]],
        description="Array de kilómetros a convertir"
    )

class PredictResponse(BaseModel):
    kilometros: List[float]
    millas: List[float]
    modelo: str = "conversion.keras"
    alumna: str = "Marilu Mendoza Ramírez"

class LossHistoryResponse(BaseModel):
    epochs: List[int]
    loss: List[float]
    min_loss: float
    final_loss: float
    total_epochs: int

class ModelInfoResponse(BaseModel):
    alumna: str
    modelo: str
    descripcion: str
    input: str
    output: str
    epochs_entrenados: int
    loss_final: float
    peso: float
    sesgo: float