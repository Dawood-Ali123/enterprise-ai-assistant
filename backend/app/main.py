from fastapi import FastAPI, UploadFile, File
from pathlib import Path
import os
import hashlib
import logfire
from app.database import get_connection
from app.repository import create_document, get_document,create_document_chunk,update_chunk_embedding,search_similar_chunks,get_chunk_by_hash,get_document_by_id,hybrid_search,save_message,get_recent_messages
from app.responses import DocumnetResponse
from app.pdf_processor import extract_text
from app.chunker import split_text
from app.embeddings import create_embedding,create_embeddings
from app.query_transform import transform_query
from app.guardrails_manager import rails
from nemoguardrails.rails.llm.options import RailStatus
from app.answer_generator import generate_answer


UPLOAD_DIR = Path("/app/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


app = FastAPI(
    title="Enterprise AI Assistant API",
    description="Backend API for the Enterprise Company AI Assistant",
    version="1.0.0"
)
logfire.configure(
    token=os.getenv("LOGFIRE_TOKEN"),
    service_name="enterprise_ai_assistant",
    environment="development"
)
logfire.instrument_fastapi(app)

@app.get("/health")
def health_check():
    try:
        connection = get_connection()
        connection.close()

        return {
            "status": "healthy",
            "database": "connected"
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e)
        }


@app.get("/database-test")
def database_tables_test():
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public';
    """)

    tables = cursor.fetchall()

    cursor.close()
    connection.close()

    return {
        "status": "success",
        "tables": [table[0] for table in tables]
    }


@app.post("/database-test")
def database_insert_test():
    document_id = create_document(
        name="Employee Leave Policy",
        file_name="Leave_policy.pdf",
        document_type="policy",
        department="HR"
    )

    return {
        "status": "success",
        "document_id": str(document_id)
    }


@app.get("/documents", response_model=list[DocumnetResponse])
def get_all_documents():
    documents = get_document()

    return [
        {
            "id": str(document[0]),
            "name": document[1],
            "file_name": document[2],
            "document_type": document[3],
            "department": document[4],
            "version": document[5]
        }
        for document in documents
    ]


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...,)):

    file_path = UPLOAD_DIR / file.filename

    content = await file.read()

    with open(file_path, "wb") as f:
        f.write(content)
    document_id=create_document(
        name=file.filename,
        file_name=file.filename,
        document_type="pdf",
        department=None
    )

    return {
        "status": "success",
        "file_name": file.filename,
        "file_path": str(file_path),
        "document_id":str(document_id)
    }
@app.get("/documents/{document_id}/text")
def get_document_text(document_id:str):
    document=get_document_by_id(document_id)
    if document is None:
        return {
            "status":"error",
            "message":"Document not found"
        }
    file_name=document[2]
    file_path=UPLOAD_DIR / file_name
    if not file_path.exists():
        return {
            "status":"error",
            "message":"Uploaded file not found",
            "file_name":file_name
        }
    text=extract_text(str(file_path))
    return {
        "document_id":document_id,
        "file_name":file_name,
        "text":text

    }

@app.get("/documents/{document_id}/chunks")
def get_document_chunks(document_id:str):
    document=get_document_by_id(document_id)
    if document is None:
        return {
            "status":"error",
            "message":"Uploaded file does not found"
        }
    file_name=document[2]
    file_path=UPLOAD_DIR / file_name
    if not file_path.exists():
                return {
            "status":"error",
            "message":"Uploaded file not found",
            "file_name":file_name
        }
    text=extract_text(str(file_path))
    chunks=split_text(text)
    return{
        "document_id": document_id,
        "total_chunks": len(chunks),
        "chunks": chunks

    }
@app.post("/documents/{document_id}/process")
def process_document(document_id: str):

    document = get_document_by_id(document_id)

    if document is None:
        return {
            "status": "error",
            "message": "Document not found"
        }

    file_name = document[2]
    file_path = UPLOAD_DIR / file_name

    if not file_path.exists():
        return {
            "status": "error",
            "message": "Upload file not found",
            "file_name": file_name
        }

    # -----------------------------------
    # 1. Extract text
    # -----------------------------------

    with logfire.span("pdf_extraction"):
        text = extract_text(str(file_path))

    # -----------------------------------
    # 2. Split into chunks
    # -----------------------------------

    with logfire.span("chunking"):
        chunks = split_text(text)

    saved_chunks = []
    pending_embeddings = []

    # -----------------------------------
    # 3. Hash check + DB records
    # -----------------------------------

    with logfire.span("hash_check_and_chunk_records"):

        for chunk_number, chunk in enumerate(
            chunks,
            start=1
        ):

            content_hash = hashlib.sha256(
                chunk.encode("utf-8")
            ).hexdigest()

            existing_chunk = get_chunk_by_hash(
                document_id=document_id,
                chunk_number=chunk_number,
                content_hash=content_hash
            )

            # -----------------------------------
            # Already embedded → skip
            # -----------------------------------

            if existing_chunk and existing_chunk[1] is not None:

                saved_chunks.append({
                    "chunk_id": str(existing_chunk[0]),
                    "chunk_number": chunk_number,
                    "content_hash": content_hash,
                    "embedding_dimensions": 384,
                    "status": "skipped_existing"
                })

                continue

            # -----------------------------------
            # Existing chunk but no embedding
            # -----------------------------------

            if existing_chunk:

                chunk_id = existing_chunk[0]
                status = "embedded_existing"

            # -----------------------------------
            # Completely new chunk
            # -----------------------------------

            else:

                chunk_id = create_document_chunk(
                    document_id=document_id,
                    chunk_number=chunk_number,
                    content=chunk,
                    content_hash=content_hash
                )

                status = "created"

            pending_embeddings.append({
                "chunk_id": str(chunk_id),
                "chunk_number": chunk_number,
                "content": chunk,
                "content_hash": content_hash,
                "status": status
            })

    # -----------------------------------
    # 4. Batch embedding
    # -----------------------------------

    batch_size = 32

    with logfire.span(
        "embedding_generation",
        total_pending=len(pending_embeddings),
        batch_size=batch_size
    ):

        for start in range(
            0,
            len(pending_embeddings),
            batch_size
        ):

            batch = pending_embeddings[
                start:start + batch_size
            ]

            texts = [
                item["content"]
                for item in batch
            ]

            # Generate embeddings for the whole batch
            embeddings = create_embeddings(texts)

            # -----------------------------------
            # 5. Save embeddings
            # -----------------------------------

            for item, embedding in zip(
                batch,
                embeddings
            ):

                update_chunk_embedding(
                    chunk_id=item["chunk_id"],
                    embedding=embedding
                )

                saved_chunks.append({
                    "chunk_id": item["chunk_id"],
                    "chunk_number": item["chunk_number"],
                    "content_hash": item["content_hash"],
                    "embedding_dimensions": len(embedding),
                    "status": item["status"]
                })

    # -----------------------------------
    # 6. Final response
    # -----------------------------------
    return {
        "status": "success",
        "document_id": document_id,
        "total_chunks": len(saved_chunks),
        "embedded_chunks": len(pending_embeddings),
        "skipped_chunks": len(saved_chunks) - len(pending_embeddings),
        "chunks": saved_chunks
    }
@app.get("/embedding-test")
def embedding_test():
    test="Company provides AI soltions"
    embedding=create_embedding(test)
    return{
        "text":test,
        "dimension":len(embedding),
        "embedding":embedding

    }
@app.get("/search")
def search_documents(
    query:str,
    limit:int=5
):
    query_embedding=create_embedding(query)
    results=search_similar_chunks(
        query_embedding=query_embedding,
        limit=limit
    )
    return{
        "query":query,
        "results":[
            {
                "chunk_id": str(result[0]),
                "document_id": str(result[1]),
                "chunk_number": result[2],
                "content": result[3],
                "distance": result[4]
            }
            for result in results
        ]
    }
@app.get("/search/hybrid")
async def hybrid_document_search(
    query: str,
    session_id:str,
    retrieval_limit: int = 10,
    top_k: int = 5,
    department: str | None = None,
    document_type: str | None = None
):
    # 1. Guardrails
    with logfire.span("guardrails"):
        guardrail_result = await rails.check_async(
            messages=[
                {
                    "role": "user",
                    "content": query
                }
            ]
        )

    if guardrail_result.status == RailStatus.BLOCKED:
        return {
            "status": "blocked",
            "message": guardrail_result.content,
            "original_query": query
        }

    with logfire.span("conversation_memory"):
        previous_messages=get_recent_messages(
            session_id=session_id,
            limit=10
        )
    conversation_context="\n".join(
        f"{role}:{content}"
        for role,content in previous_messages
        )
    # 2. Query Transformation
    with logfire.span("query_transformation"):
        rewritten_query = transform_query(query=query,conversation_context=conversation_context)

    # 3. Query Embedding
    with logfire.span("embedding"):
        query_embedding = create_embedding(rewritten_query)

    # 4. Hybrid Search + Reranking
    with logfire.span("hybrid_search_and_reranking"):
        results = hybrid_search(
            query=rewritten_query,
            query_embedding=query_embedding,
            retrieval_limit=retrieval_limit,
            top_k=top_k,
            department=department,
            document_type=document_type
        )

    # 5. Build context
    with logfire.span("context_building"):
        context_parts = []

        for result in results:
            context_parts.append(
                f"""
Source: {result["source"]["document_name"]}
Chunk: {result["chunk_number"]}
Content: {result["content"]}
"""
            )

        context = "\n\n".join(context_parts)

    # 6. Answer Generation
    with logfire.span("answer_generation"):
        answer_result = generate_answer(
            question=query,
            context=context,
            conversation_context=conversation_context
        )

    

    # 7. Sources
    sources = [
        {
            "document_name": result["source"]["document_name"],
            "file_name": result["source"]["file_name"],
            "chunk_number": result["chunk_number"],
            "chunk_id": result["chunk_id"]
        }
        for result in results
    ]
        # 8. Save conversation memory
    with logfire.span("save_conversation_memory"):
        save_message(
            session_id=session_id,
            role="user",
            content=query
        )

        save_message(
            session_id=session_id,
            role="assistant",
            content=answer_result.answer
        )
    # 8. Final response
    return {
        "query": query,
        "session_id": session_id,
        "rewritten_query": rewritten_query,
        "department_filter": department,
        "document_type_filter": document_type,
        "retrieval_limit": retrieval_limit,
        "answer": answer_result.answer,
        "confidence": answer_result.confidence,
        "top_k": top_k,
        "sources": sources,
        "results": results
    }
@app.get("/query-transfrom")
def query_transform(query:str):
    rewritten_query=transform_query(query)
    return {
        "original_query":query,
        "rewritten_query":rewritten_query
    }
@app.get("/guardrails/check")
async def guardrails_check(query:str):
    result=await rails.check_async(
        messages=[{
            "role":"user",
            "content":query
        }
        ]
    )
    return {
        "status":result.status.value,
        "content":result.content
    }
@app.get("/memory/{session_id}")
def get_conversation_memory(
    session_id: str,
    limit: int = 20
):
    messages = get_recent_messages(
        session_id=session_id,
        limit=limit
    )

    return {
        "session_id": session_id,
        "messages": [
            {
                "role": role,
                "content": content
            }
            for role, content in messages
        ]
    }


