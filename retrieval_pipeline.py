# from langchain_chroma import Chroma
# from langchain_openai import OpenAIEmbeddings
# from dotenv import load_dotenv

# load_dotenv()

# persist_directory="chroma_db"

# db = Chroma(
#     persist_directory=persist_directory,
#     embedding_function=embeddings_model,
#     collection_metadata={"hnsw:space":"cosine"}
# )

# query="What is the course structure for BCA?"


# retriver= db.as_retriever(search_kwargs={"k":5})


# relevant_docs=retriver.invoke(query)

# print(f"User Query: {query}")
# print("--- Context ---")
# for i, doc in enumerate(relevant_docs,1):
#     print(f"\nDocument {i}:\n{doc.page_content}\n")

# def main():

from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

PERSIST_DIRECTORY = "chroma_db"

def main():
    print("Starting Retrieval Pipeline...")

    # 1. Load embedding model (LOCAL – no API key)
    embeddings_model = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434"
    )

    # 2. Load existing vector database
    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings_model
    )

    # 3. Create retriever
    retriever = db.as_retriever(search_kwargs={"k": 5})

    # 4. User query
    query = "What is the course structure for BCA?"

    # 5. Retrieve relevant documents
    relevant_docs = retriever.invoke(query)

    print(f"\nUser Query: {query}")
    print("\n--- Retrieved Context ---")

    for i, doc in enumerate(relevant_docs, 1):
        print(f"\nDocument {i}:")
        print(doc.page_content)

if __name__ == "__main__":
    main()
