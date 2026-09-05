"""FastAPI application for the Zakat Companion chat service."""
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from chatbot import retriever
from config import API_KEY

logger = logging.getLogger(__name__)
app = FastAPI(title="Zakat Companion API")


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    api_key=API_KEY,
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a Zakat information assistant.

Answer the user's question using ONLY the provided context.

If the context does not contain enough information,
say that you do not have enough information.

Do not invent religious rulings.

Context:
{context}
""",
    ),
    ("human", "{question}"),
])


def format_docs(docs):
    """Turn the retriever's list of Documents into a plain string
    the prompt's {context} slot can actually use."""
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    try:
        answer = rag_chain.invoke(question)
    except Exception as error:
        logger.exception("Unable to answer the question.")
        raise HTTPException(
            status_code=500,
            detail="Unable to answer the question.",
        ) from error

    return QueryResponse(answer=answer)


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(app, host="127.0.0.1", port=8000)