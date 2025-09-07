from fastapi import FastAPI, HTTPException, status, File
from typing import Annotated
from uploader import save_file
from fastapi import UploadFile

app = FastAPI()


@app.post("/upload")
async def file_upload_controller(
file: Annotated[UploadFile, File(description="Uploaded PDF documents")]
):
    if file.content_type != "application/pdf":    
        raise HTTPException(
        detail=f"Only uploading PDF documents are supported",
        status_code=status.HTTP_400_BAD_REQUEST,
        )
        
    try:
        await save_file(file)
    except Exception as e:
        
        raise HTTPException(
        detail=f"An error occurred while saving file - Error: {e}",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
        
    return {"filename": file.filename, "message": "File uploaded successfully"}