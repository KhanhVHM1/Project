from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import pandas as pd
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from .model import BaselineModel, CandidateModel
from .monitoring import MetricsMiddleware, PREDICTIONS

app = FastAPI(title="Country Prediction API")
app.add_middleware(MetricsMiddleware)

# In a real app, load from artifact store; here we “train” quickly on startup demo data.
import numpy as np
DEMO = pd.DataFrame({
    "country": ["US","US","FR","FR","VN","VN"],
    "x1": [1,2,1,2,1,2],
    "x2": [0.5,1.0,0.2,0.3,0.7,1.1],
    "y":  [10,12,6,7,8,9]
})
BASELINE = BaselineModel.train(DEMO, target="y", country_col="country")
CANDIDATE, _ = CandidateModel.train(DEMO, target="y")

class PredictIn(BaseModel):
    features: Dict[str, Any]   # e.g. {"x1": 2, "x2": 0.3}
    country: Optional[str] = "all"  # "US" | "FR" | "VN" | "all"
    model: str = "candidate"   # "baseline" | "candidate"

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict")
def predict(payload: PredictIn):
    X = pd.DataFrame([payload.features])
    country = payload.country or "all"

    if payload.model == "baseline":
        pred = BASELINE.predict(X, country=country)
    elif payload.model == "candidate":
        pred = CANDIDATE.predict(X, country=country)
    else:
        raise HTTPException(400, detail="Unknown model")

    PREDICTIONS.labels(country=country).inc()
    return {"country": country, "model": payload.model, "prediction": float(pred[0])}
