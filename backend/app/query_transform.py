from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


class QueryRewrite(BaseModel):
    rewritten_query: str


model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

structured_model = model.with_structured_output(QueryRewrite)


prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a query transformation component for an enterprise company
knowledge assistant.

Your job is to rewrite the user's current question into a clear,
self-contained search query for the company's knowledge base.

Use conversation history only when it helps resolve references such as:
- it
- they
- this
- that
- the above
- this product
- this policy

Rules:
1. Preserve the user's original intent.
2. Make the rewritten query self-contained.
3. Use conversation history to resolve ambiguous references.
4. Do not add information that is not present in the question
   or conversation history.
5. If the current question is already clear, rewrite it minimally.
6. Return only the rewritten query.

Conversation History:
{conversation_context}

Current User Query:
{query}
"""
    )
])


def transform_query(
    query: str,
    conversation_context: str = ""
) -> str:

    chain = prompt | structured_model

    result = chain.invoke({
        "query": query,
        "conversation_context": conversation_context
    })

    return result.rewritten_query