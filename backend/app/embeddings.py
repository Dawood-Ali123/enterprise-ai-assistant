from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()
embedding_model=None
def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        embedding_model=HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5"
        )
    return embedding_model
def create_embedding(text:str) ->list[float]:
    model=get_embedding_model()
    return model.embed_query(text)
def create_embeddings(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    return model.embed_documents(texts)