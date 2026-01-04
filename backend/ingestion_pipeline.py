import os

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_documents(docs_path="docs"):
    print(f"Loading documents from {docs_path}...")

    if not os.path.exists(docs_path):
        raise FileExistsError(f"The directory {docs_path} does not exist.")

    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=lambda file_path: TextLoader(file_path, encoding="utf-8")
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}.")

    # Preview first document
    for i, doc in enumerate(documents[:1]):
        print(f"\nDocument {i+1}:")
        print(f"Source: {doc.metadata['source']}")
        print(f"Content length: {len(doc.page_content)} characters")
        print(f"Content preview: {doc.page_content[:100]}...")
        print(f"Metadata: {doc.metadata}")

    return documents

def load_documents(docs_path="docs"):
    print(f"Loading documents from {docs_path}...")

    if not os.path.exists(docs_path):
        raise FileExistsError(f"The directory {docs_path} does not exist.")

    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}.")

    # Preview first document
    for i, doc in enumerate(documents[:1]):
        print(f"\nDocument {i+1}:")
        print(f"Source: {doc.metadata['source']}")
        print(f"Content length: {len(doc.page_content)} characters")
        print(f"Content preview: {doc.page_content[:100]}...")
        print(f"Metadata: {doc.metadata}")

    return documents


def split_documents(documents, chunk_size=200, chunk_overlap=50):
    """Split documents into smaller chunks with overlap"""
    print("Splitting documents into chunks...")

    text_splitter=CharacterTextSplitter(
        separator="",
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(documents)

    if chunks:

        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} character")
            print(f"Content:")
            print(chunk.page_content)
            print("-" * 50)

        if len(chunks) > 5:
            print(f"\n... and {len(chunks)-5}more chunks")
    return chunks                  


def create_vector_store(chunks, persist_directory="chroma_db"):
    """Create and persist ChromaDB vector store"""
    print("Create embeddings and storing in ChromaDB...")

    embedding_model = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")

    #Create ChromaDB vector store
    print("--- Create vector store ---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space":"cosine"}
    )
    print("--- Finished creating vector store ---")

    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore


def main():
    print("Main Function")

    #1. Loading the files
    documents=load_documents(docs_path="docs")

    #2. Chunking the files
    chunks = split_documents(documents)

    #3. Embedding and storing in VectorDB
    vectorstore=create_vector_store(chunks)

if __name__=="__main__":
    main()
