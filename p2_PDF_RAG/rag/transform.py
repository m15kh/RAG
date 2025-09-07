import re
from typing import Any, AsyncGenerator
import aiofiles
from transformers import  AutoTokenizer, AutoModel




DEFAULT_CHUNK_SIZE = 1024 * 1024 * 50 # 50 megabytes


model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True) # trust_remote_code is needed to use the encode method


async def load(filepath: str) -> AsyncGenerator[str, Any]:
    async with aiofiles.open(filepath, "r", encoding="utf-8") as f:
        while chunk := await f.read(DEFAULT_CHUNK_SIZE):
            yield chunk

def clean(text: str) -> str:
    t = text.replace("\n", " ")
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\. ,", "", t)
    t = t.replace("..", ".")
    t = t.replace(". .", ".")
    cleaned_text = t.replace("\n", " ").strip()
    return cleaned_text


def embed(text: str) -> list[float]:
    return model.encode(text).tolist()