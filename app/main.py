from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any
import sys
from enum import Enum

# Añadir el directorio src al path
sys.path.append("../src")
from prediction import RobustComplaintPredictor

# Crear la aplicación FastAPI
app = FastAPI(
    title="API de Predicción de Respuestas a Quejas",
    description="API para predecir la reacción de empresas a quejas de consumidores",
    version="1.0.0"
)

# Enum para los estados válidos
class StateEnum(str, Enum):
    ME = "ME"
    NH = "NH"
    VT = "VT"
    MA = "MA"
    RI = "RI"
    CT = "CT"
    NY = "NY"
    NJ = "NJ"
    PA = "PA"
    OH = "OH"
    IN = "IN"
    IL = "IL"
    MI = "MI"
    WI = "WI"
    MN = "MN"
    IA = "IA"
    MO = "MO"
    ND = "ND"
    SD = "SD"
    NE = "NE"
    KS = "KS"
    DE = "DE"
    MD = "MD"
    DC = "DC"
    VA = "VA"
    WV = "WV"
    NC = "NC"
    SC = "SC"
    GA = "GA"
    FL = "FL"
    KY = "KY"
    TN = "TN"
    AL = "AL"
    MS = "MS"
    AR = "AR"
    LA = "LA"
    OK = "OK"
    TX = "TX"
    MT = "MT"
    ID = "ID"
    WY = "WY"
    CO = "CO"
    NM = "NM"
    AZ = "AZ"
    UT = "UT"
    NV = "NV"
    CA = "CA"
    OR = "OR"
    WA = "WA"
    AK = "AK"
    HI = "HI"

# Modelo de datos para la request
class ComplaintRequest(BaseModel):
    product: str = Field(..., description="Producto relacionado con la queja")
    issue: str = Field(..., description="Problema o issue de la queja")
    state: StateEnum = Field(..., description="Estado donde se registró la queja")
    zip_code: int = Field(..., ge=501, le=99950, description="Código postal")
    company: str = Field(..., description="Nombre de la empresa")
    
    class Config:
        schema_extra = {
            "example": {
                "product": "Credit card",
                "issue": "Billing disputes",
                "state": "CA",
                "zip_code": 90210,
                "company": "Bank of America"
            }
        }

# Modelo de respuesta
class PredictionResponse(BaseModel):
    prediction_successful: bool
    top_probabilities: Dict[str, float] = None
    error: str = None

# Inicializar el predictor globalmente
predictor = None

@app.on_event("startup")
async def startup_event():
    """Inicializar el modelo al arrancar la aplicación"""
    global predictor
    try:
        predictor = RobustComplaintPredictor()
        print("✅ Modelo cargado correctamente")
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")
        raise e

@app.get("/")
async def root():
    """Endpoint de bienvenida"""
    return {
        "message": "API de Predicción de Respuestas a Quejas",
        "status": "active",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Endpoint para verificar el estado de la API"""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_complaint_response(complaint: ComplaintRequest):
    """
    Predice la respuesta de la empresa a una queja
    
    - **product**: Producto relacionado con la queja
    - **issue**: Problema o issue específico
    - **state**: Estado donde se registró la queja
    - **zip_code**: Código postal (entre 501 y 99950)
    - **company**: Nombre de la empresa
    """
    
    if predictor is None:
        raise HTTPException(
            status_code=500, 
            detail="El modelo no está inicializado. Contacta con el administrador."
        )
    
    try:
        # Preparar las características en el formato esperado
        features = [
            {
                "Product": [complaint.product],
                "Issue": [complaint.issue],
                "State": [complaint.state.value],
                "ZIP code": [complaint.zip_code],
                "Company": [complaint.company]
            }
        ]
        
        # Realizar la predicción
        result = predictor.predict(features)
        
        if result['prediction_successful']:
            # Obtener las top 3 probabilidades
            sorted_probs = sorted(
                result['probabilities'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            top_3_probs = dict(sorted_probs[:3])
            
            return PredictionResponse(
                prediction_successful=True,
                top_probabilities=top_3_probs
            )
        else:
            return PredictionResponse(
                prediction_successful=False,
                error=result.get('error', 'Error desconocido en la predicción')
            )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error interno del servidor: {str(e)}"
        )

@app.get("/states")
async def get_valid_states():
    """Obtiene la lista de estados válidos"""
    return {
        "states": [state.value for state in StateEnum],
        "total": len(StateEnum)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)