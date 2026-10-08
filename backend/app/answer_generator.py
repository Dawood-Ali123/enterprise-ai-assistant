from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from app.responses import AnswerResponse
model=ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)
structure_model=model.with_structured_output(AnswerResponse)
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an enterprise company knowledge assistant.

Your job is to answer the user's question using the provided company context
and conversation history.

Rules:
1. Use company document context for factual company information.
2. Use conversation history only to understand references and follow-up questions.
3. Do not use outside knowledge.
4. Do not invent or assume information.
5. If company context does not contain enough information, say information is not available in provided company docs.
6. Keep the answer concise and factual.
7. Return confidence:
   - high: context directly supports answer.
   - medium: context partially supports answer.
   - low: context does not provide enough information.

Conversation History:
{conversation_context}

Company Context:
{context}
"""
    ),
    (
        "human",
        "Question:\n{question}"
    )
])

def generate_answer(question:str,context:str,conversation_context:str="")->AnswerResponse:
    chain=prompt|structure_model
    response=chain.invoke({
        "question":question,
        "context":context,
        "conversation_context":conversation_context
    })
    return response
