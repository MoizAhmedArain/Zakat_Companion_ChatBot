import logging
from langchain_chroma import Chroma
from config import CHROMA_PATH, COLLECTION_NAME, get_embedding_model

logger = logging.getLogger(__name__)

def get_vector_store():
    try:
        embedding_model = get_embedding_model()
        return Chroma(
            persist_directory=CHROMA_PATH,
            collection_name=COLLECTION_NAME,
            embedding_function=embedding_model,
        )
    except (OSError, RuntimeError, ValueError) as error:
        logger.exception("Unable to load the vector store.")
        raise RuntimeError("Unable to load the vector store.") from error


def create_retriever(vector_store, k=3):
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


# Module-level objects: built once, on first import, and reused by the API.
vector_store = get_vector_store()
retriever = create_retriever(vector_store)
