from fastapi import APIRouter, HTTPException

from ml.train_pipeline.gru_train import main as gru_main
from ml.train_pipeline.lstm_train import main as lstm_main
from ml.train_pipeline.transformer_train import main as transformer_main


router = APIRouter(
    prefix="/train",
    tags=["train"]
)


models = {
    "gru": gru_main,
    "lstm": lstm_main,
    "transformer": transformer_main
}


companies = {
    "CHCL",
    "CZBIL",
    "BPCL",
    "AHPC",
    "ADBL",
    "ALICL",
    "EBL",
    "NTC",
    "NABIL",
    "PCBL"
}


@router.post("/{model}/{data}")
def train(model: str, data: str):

    if model not in models:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model: {model}"
        )

    if data not in companies:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported company: {data}"
        )

    try:
        models[model](data)

        return {
            "response": f"{model} trained successfully for {data}"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.post("/{data}")
def train_all(data: str):

    if data not in companies:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported company: {data}"
        )

    try:
        for model_name, train_function in models.items():
            train_function(data)

        return {
            "response": f"All models trained successfully for {data}"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )