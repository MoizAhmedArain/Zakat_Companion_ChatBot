"""
Small, dependency-light helpers used by rag.py.
Split out from rag.py so they can be unit tested without importing
chatbot.py's live Qdrant connection.
"""


def format_docs(docs):
    """Turn a list of retrieved Documents into a plain string for the
    prompt's {context} slot."""
    return "\n\n".join(doc.page_content for doc in docs)