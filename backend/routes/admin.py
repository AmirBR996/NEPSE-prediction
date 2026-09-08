from fastapi import APIRouter, HTTPException
from ml.train_pipeline.train import main
from ml.model_evaluation import evaluate

router=APIRouter(
    prefix="/admin",
    tags=["admin"]
)

models=["lstm","gru","transformer"]

companies=[
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
]

@router.post("/trainall")
def train_evaluation():
    try:
        results=[]

        for company in companies:
            for model_name in models:
                main(
                    data=company,
                    model_name=model_name,
                    include_test=True
                )

                results.append({
                    "company":company,
                    "model":model_name,
                    "status":"trained"
                })

        return {
            "message":"All evaluation models trained successfully",
            "total_models":len(results),
            "results":results
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@router.post("/{model}/{data}")
def train(model:str,data:str):
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
        main(
            data=data,
            model_name=model,
            include_test=True
        )

        return {
            "response":f"{model} trained successfully for {data}"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@router.post("/{data}")
def train_all(data:str):
    if data not in companies:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported company: {data}"
        )

    try:
        results=[]

        for model_name in models:
            main(
                data=data,
                model_name=model_name,
                include_test=True
            )

            results.append({
                "company":data,
                "model":model_name,
                "status":"trained"
            })

        return {
            "response":f"All models trained successfully for {data}",
            "total_models":len(results),
            "results":results
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@router.post("/evaluate/{data}")
def evaluate_company(data:str):
    if data not in companies:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported company: {data}"
        )

    results=[]

    try:
        for model_name in models:
            result=evaluate(
                data=data,
                model_name=model_name
            )
            results.append(result)

        return {
            "company":data,
            "results":results
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )




