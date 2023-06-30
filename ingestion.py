import os

import pinecone
from decouple import config
from langchain.document_loaders import ReadTheDocsLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone

api_key = config("PINECONE_API_KEY")
environment = config("PINECONE_ENVIRONMENT_REGION")
secret_key = config("OPENAI_API_KEY")

os.environ["OPENAI_API_KEY"] = secret_key
os.environ["PINECONE_API_KEY"] = api_key
os.environ["PINECONE_ENVIRONMENT_REGION"] = environment

pinecone.init(api_key=os.environ["PINECONE_API_KEY"],
              environment=os.environ["PINECONE_ENVIRONMENT_REGION"])


def ingest_docs() -> None:
    loader = ReadTheDocsLoader(path=os.path.join(
        os.getcwd(), "langchain-docs/langchain.readthedocs.io/en/latest"))
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""])
    documents = text_splitter.split_documents(documents=raw_documents)
    print(f"Splitted indo {len(documents)} chuncks")

    for doc in documents:
        old_path = doc.metadata["source"]
        new_url = old_path.replace(os.getcwd()+"/langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Going to insert {len(documents)} to Pinecone")
    embeddings = OpenAIEmbeddings()
    Pinecone.from_documents(documents, embeddings,
                            index_name="langchain-doc-index")
    print("----- Added to Pinecone vectorstore vectores ------")


if __name__ == "__main__":
    ingest_docs()
