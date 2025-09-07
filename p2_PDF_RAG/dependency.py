from rag.service import vector_service
from rag.transform import embed

from schemas import TextModelRequest, TextModelResponse


async def get_rag_content(body: TextModelRequest = Body(...)) -> str:
    
    
    rag_content = await vector_service.search(
    "knowledgebase", embed(body.prompt), 3, 0.7
    )
    rag_content_str = "\n".join(
    [c.payload["original_text"] for c in rag_content]
    )
    return rag_content_str