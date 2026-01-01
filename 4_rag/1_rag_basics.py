import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings


# -----------------------------------
# Paths
# -----------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "odyssey.txt")
persistent_directory = os.path.join(current_dir, "db", "chroma_db")


# -----------------------------------
# Embeddings
# -----------------------------------
embeddings = OllamaEmbeddings(model="embeddinggemma")


# -----------------------------------
# Build DB if not exists
# -----------------------------------
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please check the path."
        )

    # Load file
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(documents)

    print(f"\nNumber of document chunks: {len(docs)}")
    print(f"Sample chunk:\n{docs[0].page_content[:400]}...\n")

    # Create Chroma DB
    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persistent_directory
    )

    print("Vector store created and persisted successfully!\n")
else:
    print("Vector store already exists. Loading it...")


# -----------------------------------
# Load existing DB
# -----------------------------------
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embeddings
)


# -----------------------------------
# Retriever
# -----------------------------------
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 3,
        "score_threshold": 0.8  # higher keeps more results
    },
)

query = "Who is Odysseus' wife?"
relevant_docs = retriever.invoke(query)

# -----------------------------------
# Results
# -----------------------------------
print("\n--- Query ---")
print(query)

print("\n--- Relevant Documents ---")
if not relevant_docs:
    print("No relevant documents found. Try lowering score_threshold.")
else:
    for i, doc in enumerate(relevant_docs, 1):
        print(f"\nDocument {i}:\n{doc.page_content[:800]}")
        if doc.metadata:
            print(f"\nSource: {doc.metadata.get('source', 'Unknown')}")