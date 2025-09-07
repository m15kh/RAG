from pypdf import PdfReader
import sys



def pdf_text_extractor(filepath: str) -> None:
    content = ""
    pdf_reader = PdfReader(filepath, strict=True)
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            content += f"{page_text}\n\n"
    with open(filepath.replace("pdf", "txt"), "w", encoding="utf-8") as file:
        file.write(content)
        
        
if __name__ == "__main__":
    pdf_text_extractor('AI_Detection_and_Semantic_Search_Structured.pdf')