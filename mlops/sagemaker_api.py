from fastapi import Response
from api import app, PredictionRequest, predict_smiles, MODEL_PATH
from predict import predict

# Fail startup if the packaged model cannot perform inference.
predict(MODEL_PATH, "CCO")

@app.get("/ping")
def ping():
    return Response(status_code=200)

@app.post("/invocations")
def invocations(request: PredictionRequest):
    return predict_smiles(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
