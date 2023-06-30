import os
from typing import Any, Dict, List

import pinecone
from decouple import config
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

api_key = config("PINECONE_API_KEY")
environment = config("PINECONE_ENVIRONMENT_REGION")
secret_key = config("OPENAI_API_KEY")

os.environ["OPENAI_API_KEY"] = secret_key
os.environ["PINECONE_API_KEY"] = api_key
os.environ["PINECONE_ENVIRONMENT_REGION"] = environment

pinecone.init(api_key=os.environ["PINECONE_API_KEY"],
              environment=os.environ["PINECONE_ENVIRONMENT_REGION"])


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []) -> Any:
    embeddings = OpenAIEmbeddings()
    docsearch = Pinecone.from_existing_index(
        index_name="langchain-doc-index", embedding=embeddings)
    chat = ChatOpenAI(verbose=True, temperature=0)

    qa = ConversationalRetrievalChain.from_llm(
        llm=chat,
        retriever=docsearch.as_retriever(),
        return_source_documents=True
    )

    return qa({"question": query, "chat_history": chat_history})


if __name__ == "__main__":
    print(run_llm(query="What is LangChain?"))
