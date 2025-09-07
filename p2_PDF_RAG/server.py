from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    UploadFile,
    status,
    HTTPException,
    )

from typing import Annotated
from rag.extractor import pdf_text_extractor
from rag.service import vector_service

from uploader import save_file
from dependencies import get_rag_content, get_urls_content


from fastapi import Depends
from schemas import TextModelRequest, TextModelResponse
from fastapi import Body, Request
from llm.models import generate_text, load_text_model

from loguru import logger
app = FastAPI()

@app.post("/upload")
async def file_upload_controller(
    file: Annotated[UploadFile, File(description="A file read as UploadFile")],
    bg_text_processor: BackgroundTasks,
    ):
    
    try:
        filepath = await save_file(file)
        logger.info(f"File saved at {filepath}")
        bg_text_processor.add_task(pdf_text_extractor, filepath)
        
        bg_text_processor.add_task(
        vector_service.store_file_content_in_db,
        filepath.replace("pdf", "txt"),
        512,
        "knowledgebase",
        768,
        )
        
    except Exception as e:
        logger.error(f"Error during file upload: {e}")
        raise HTTPException(
        detail=f"An error occurred while saving file - Error: {e}",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
        
        
    return {"filename": file.filename, "message": "File uploaded successfully"}




@app.post("/generate/text", response_model_exclude_defaults=True)

async def serve_text_to_text_controller(
    request: Request,
    body: TextModelRequest = Body(...),
    urls_content: str = Depends(get_urls_content),
    rag_content: str = Depends(get_rag_content),
    ) -> TextModelResponse:
    
    try:
        prompt = body.prompt + " " + urls_content + rag_content
        logger.info(f"Generating text for prompt: {body.prompt}")
        output = generate_text(load_text_model["text"], prompt, body.temperature)
        logger.info("Text generation successful")
        return TextModelResponse(content=output, ip=request.client.host)
    except Exception as e:
        logger.error(f"Error during text generation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during text generation.",
        )