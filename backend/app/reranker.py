from sentence_transformers import CrossEncoder

reranker_model = None


def get_reranker_model():
    global reranker_model

    if reranker_model is None:
        reranker_model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    return reranker_model


def rerank(query: str, fused_results: list, top_k: int = 5):

    model = get_reranker_model()

    pairs = [
        (query, item["result"][3])
        for item in fused_results
    ]

    scores = model.predict(pairs)

    reranked_results = []

    for item, score in zip(fused_results, scores):
        reranked_results.append({
    "chunk_id": str(item["result"][0]),
    "document_id": str(item["result"][1]),
    "chunk_number": item["result"][2],
    "content": item["result"][3],

    "rrf_score": item["rrf_score"],
    "vector_rank": item["vector_rank"],
    "text_rank": item["text_rank"],
    "reranker_score": float(score),

    "source": {
        "document_name": item["result"][5],
        "file_name": item["result"][6],
        "department": item["result"][7],
        "document_type": item["result"][8]
    }
})
    reranked_results.sort(
        key=lambda item: item["reranker_score"],
        reverse=True
    )

    return reranked_results[:top_k]