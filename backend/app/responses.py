from pydantic import BaseModel



class DocumnetResponse(BaseModel):
    id:str
    name:str
    file_name:str
    document_type:str | None
    department:str | None
    version:int

class AnswerResponse(BaseModel):
    answer: str
    confidence: str