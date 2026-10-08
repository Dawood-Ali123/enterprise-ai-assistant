from app.database import get_connection
from app.embeddings import create_embedding
from app.reranker import rerank
from app.query_transform import transform_query
from app.guardrails_manager import rails
def create_document(
    name: str,
    file_name: str,
    document_type: str | None = None,
    department: str | None = None,
):
    connection = get_connection()

    try:
        with connection.cursor() as cursor:

            cursor.execute(
                """
                INSERT INTO documents (
                    name,
                    file_name,
                    document_type,
                    department
                )
                VALUES (%s, %s, %s, %s)
                RETURNING id;
                """,
                (
                    name,
                    file_name,
                    document_type,
                    department,
                ),
            )

            document_id = cursor.fetchone()[0]

        connection.commit()

        return document_id

    finally:
        connection.close()
def get_document():
    connection=get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
SELECT id,name,file_name,document_type,department,version
FROM documents
ORDER BY created_at DESC
""")
            documnets=cursor.fetchall()
            return documnets
    finally:
        connection.close()
def create_document_chunk(
        document_id:str,
        chunk_number:int,
        content:str,
        content_hash:str
):
    connection=get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """INSERT INTO document_chunks (
                document_id,
                chunk_number,
                content,
                content_hash
                )
                VALUES (%s,%s,%s,%s)
                ON CONFLICT(
                document_id,
                chunk_number,
                content_hash
                )
                DO UPDATE SET
                    content_hash=EXCLUDED.content_hash
                RETURNING id;
                """,
                (
                    document_id,
                    chunk_number,
                    content,
                    content_hash
                
            ),
        )
            chunk_id=cursor.fetchone()[0]
            connection.commit()
            return chunk_id
    finally:
        connection.close()

def update_chunk_embedding(
        chunk_id:str,
        embedding:list[float]
):
    connection=get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """UPDATE document_chunks
                SET embedding =%s
                WHERE id = %s;
                """,
                (embedding,chunk_id)
            )
        connection.commit()
    finally:
        connection.close()

def search_similar_chunks(
    query_embedding: list[float],
    limit: int = 5,
    department: str | None = None,
    document_type: str | None = None
):
    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute("""
SELECT
    dc.id,
    dc.document_id,
    dc.chunk_number,
    dc.content,
    dc.embedding <=> %s::vector AS distance,
    d.name,
    d.file_name,
    d.department,
    d.document_type
FROM document_chunks dc
JOIN documents d
    ON dc.document_id = d.id
WHERE dc.embedding IS NOT NULL
  AND (%s::text IS NULL OR d.department = %s)
  AND (%s::text IS NULL OR d.document_type = %s)
ORDER BY dc.embedding <=> %s::vector
LIMIT %s;   
            """,
            (
                query_embedding,
                department,
                department,
                document_type,
                document_type,
                query_embedding,
                limit
            ))

            return cursor.fetchall()

    finally:
        connection.close()
def get_chunk_by_hash(
    document_id: str,
    chunk_number: int,
    content_hash: str
):
    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    id,
                    embedding
                FROM document_chunks
                WHERE document_id = %s
                  AND chunk_number = %s
                  AND content_hash = %s;
                """,
                (
                    document_id,
                    chunk_number,
                    content_hash
                )
            )

            return cursor.fetchone()

    finally:
        connection.close()

def get_document_by_id(document_id:str):
    connection=get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
SELECT id,name,file_name,document_type,department,version
FROM documents
WHERE id=%s
""",
(document_id,)
)
            return cursor.fetchone()
    finally:
        connection.close()

def search_full_text(
    query: str,
    limit: int = 5,
    department: str | None = None,
    document_type: str | None = None
):
    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute("""
               SELECT
    dc.id,
    dc.document_id,
    dc.chunk_number,
    dc.content,
    ts_rank(
        to_tsvector('english', dc.content),
        plainto_tsquery('english', %s)
    ) AS text_score,
    d.name,
    d.file_name,
    d.department,
    d.document_type
FROM document_chunks dc
JOIN documents d
    ON dc.document_id = d.id
WHERE to_tsvector('english', dc.content)
      @@ plainto_tsquery('english', %s)
  AND (%s::text IS NULL OR d.department = %s)
  AND (%s::text IS NULL OR d.document_type = %s)
ORDER BY text_score DESC
LIMIT %s;
            """,
            (
                query,
                query,
                department,
                department,
                document_type,
                document_type,
                limit
            ))

            return cursor.fetchall()

    finally:
        connection.close()

def reciprocal_rank_fusion(
    vector_results,
    text_results,
    k: int = 60
):
    scores = {}

    for rank, result in enumerate(vector_results, start=1):

        chunk_id = str(result[0])

        scores.setdefault(
            chunk_id,
            {
                "rrf_score": 0.0,
                "vector_rank": None,
                "text_rank": None,
                "result": result
            }
        )

        scores[chunk_id]["vector_rank"] = rank

        scores[chunk_id]["rrf_score"] += (
            1 / (k + rank)
        )

    for rank, result in enumerate(text_results, start=1):

        chunk_id = str(result[0])

        scores.setdefault(
            chunk_id,
            {
                "rrf_score": 0.0,
                "vector_rank": None,
                "text_rank": None,
                "result": result
            }
        )

        scores[chunk_id]["text_rank"] = rank

        scores[chunk_id]["rrf_score"] += (
            1 / (k + rank)
        )

    return sorted(
        scores.values(),
        key=lambda item: item["rrf_score"],
        reverse=True
    )
def hybrid_search(query:str,
                  query_embedding:list[float],
                  retrieval_limit :int=10,
                  top_k :int=5,
                  department:str |None=None,
                  document_type:str |None=None):
    vector_results=search_similar_chunks(
        query_embedding=query_embedding,
        limit=retrieval_limit,
        department=department,
        document_type=document_type
    )
    text_results=search_full_text(
        query=query,
        limit=retrieval_limit,
        department=department,
        document_type=document_type
    )

    fused_results=reciprocal_rank_fusion(
        vector_results=vector_results,
        text_results=text_results
    )
    final_results=rerank(
        query=query,
        fused_results=fused_results,
        top_k=top_k
    )

    return final_results
def get_chunk_source(chunk_id:str):
    connection=get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
SELECT 
dc.id,
dc.document_id,
dc.chunk_number,
d.name,
d.file_name,
d.department,
d.documet_type
FROM document_chunks dc
JOIN documents d
ON dc.document_id=d.id
WHERE dc.id=%s;
""",
(chunk_id,)
)
            return cursor.fetchone()
    finally:
        connection.close()

def save_message(
        session_id:str,
        role:str,
        content:str
):
    connection=get_connection()
    cursor=connection.cursor()
    cursor.execute(
        """
INSERT INTO conversation_messages
(session_id,role,content)
VALUES (%s,%s,%s)
""",
(session_id,role,content)
    )
    connection.commit()
    cursor.close()

def get_recent_messages(
        session_id:str,
        limit:int =10
):
    connection=get_connection()
    cursor=connection.cursor()
    cursor.execute(
        """
SELECT role,content
FROM conversation_messages
WHERE session_id=%s
ORDER BY created_at DESC
LIMIT %s
""",
(session_id,limit)
    )
    messages=cursor.fetchall()
    cursor.close()
    connection.close()
    return list(reversed(messages))
