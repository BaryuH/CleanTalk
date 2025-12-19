from pydantic import BaseModel
from typing import Dict

class Request(BaseModel):
    text: str


class Response(BaseModel):
    clean_text: str
    labels: Dict[str, int]
    points: int
    final_label: str
