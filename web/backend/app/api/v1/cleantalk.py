from fastapi import APIRouter
from app.schemas.schema import Request,Response
from app.ml.models.model import CleanTalk1

router = APIRouter()
model = CleanTalk1()
@router.post("/CleanTalk1", response_model=Response)
def check(payload: Request):
    result = model.predict(payload.text)
    return result
