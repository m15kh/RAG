from datetime import datetime
from typing import Annotated, Literal
from uuid import uuid4
from pydantic import BaseModel, Field, HttpUrl, IPvAnyAddress, PositiveInt


class ModelRequest(BaseModel):
    
    prompt: str



class ModelResponse(BaseModel):
    
    request_id: Annotated[str, Field(default_factory=lambda: uuid4().hex)]
    ip: Annotated[str, IPvAnyAddress] | None
    content: Annotated[str | None, Field(min_length=0, max_length=10000)]
    created_at: datetime = datetime.now()


class TextModelRequest(BaseModel):
    
    model: Literal["gpt-3.5-turbo", "gpt-4o"]
    prompt: str
    temperature: float = 0.0
    
    
class TextModelResponse(ModelResponse):
    
    tokens: Annotated[int, Field(ge=0)]