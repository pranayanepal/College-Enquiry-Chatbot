from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

PERSIST_DIRECTORY = "chroma_db"

# Use the same embeddings model as ingestion
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)

# Load Chroma vector store
db = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings
)

retriever = db.as_retriever(search_kwargs={"k": 5})

# Initialize LLM for answer generation
llm = Ollama(
    model="llama3",  # You can change this to your preferred Ollama model (e.g., "llama3", "mistral", "phi3")
    base_url="http://localhost:11434",
    temperature=0.3  # Lower temperature for more focused answers
)

# Prompt template for generating specific answers
def create_prompt(context: str, question: str) -> str:
    """Create a formatted prompt for the LLM"""
    return f"""You are a helpful assistant that provides specific and accurate information about college courses.

Based on the following context, answer the user's question in a clear, concise, and well-structured manner. 
Only use information from the context provided. If the context doesn't contain enough information, say so clearly. 
Do not repeat information unnecessarily. Format your answer in a readable way with proper structure.

Context:
{context}

Question: {question}

Provide a specific and well-formatted answer:"""

def get_answer(question: str) -> str:
    try:
        # Retrieve relevant documents
        docs = retriever.invoke(question)
        if not docs:
            return "No relevant information found."
        
        # Remove duplicate content and combine context
        seen_content = set()
        unique_docs = []
        for doc in docs:
            content = doc.page_content.strip()
            # Use a hash of the content to detect duplicates
            if content and content not in seen_content:
                seen_content.add(content)
                unique_docs.append(content)
        
        # Combine unique contexts
        context = "\n\n".join(unique_docs)
        
        # Generate answer using LLM
        prompt = create_prompt(context, question)
        response = llm.invoke(prompt)
        
        # Handle different response types (string or object with content attribute)
        if isinstance(response, str):
            answer = response.strip()
        else:
            answer = str(response).strip()
        
        return answer if answer else "I couldn't generate a proper answer. Please try rephrasing your question."
        
    except Exception as e:
        print("Error in get_answer:", e)
        return f"Error occurred while retrieving the answer: {str(e)}"
