import os

from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
    TextSplitter,
    TokenTextSplitter,
)
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings


# -----------------------------------
# Paths
# -----------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "romeo_and_juliet.txt")
db_dir = os.path.join(current_dir, "db")

if not os.path.exists(file_path):
    raise FileNotFoundError(f"{file_path} not found")


# -----------------------------------
# Load document
# -----------------------------------
loader = TextLoader(file_path, encoding="utf-8")
documents = loader.load()


# -----------------------------------
# Embeddings
# -----------------------------------
embeddings = OllamaEmbeddings(model="embeddinggemma")


# -----------------------------------
# Helper: Create DB
# -----------------------------------
def create_vector_store(docs, store_name):
    persistent_directory = os.path.join(db_dir, store_name)

    if not os.path.exists(persistent_directory):
        print(f"\n--- Creating vector store {store_name} ---")
        Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persistent_directory,
        )
        print(f"--- Finished creating vector store {store_name} ---")
    else:
        print(f"Vector store {store_name} already exists.")


# -----------------------------------
# 1. Character Split
# -----------------------------------
print("\n--- Character Split ---")
char_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
char_docs = char_splitter.split_documents(documents)
create_vector_store(char_docs, "chroma_db_char")


# -----------------------------------
# 2. Sentence Token Split
# -----------------------------------
print("\n--- Sentence Token Split ---")
sent_splitter = SentenceTransformersTokenTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
)
sent_docs = sent_splitter.split_documents(documents)
create_vector_store(sent_docs, "chroma_db_sent")


# -----------------------------------
# 3. Token Splitter
# -----------------------------------
print("\n--- Token Split ---")
token_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=50)
token_docs = token_splitter.split_documents(documents)
create_vector_store(token_docs, "chroma_db_token")


# -----------------------------------
# 4. Recursive Character Split
# -----------------------------------
print("\n--- Recursive Character Split ---")
rec_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
)
rec_docs = rec_splitter.split_documents(documents)
create_vector_store(rec_docs, "chroma_db_rec_char")


# -----------------------------------
# 5. Custom Splitter
# -----------------------------------
print("\n--- Custom Paragraph Split ---")


class CustomTextSplitter(TextSplitter):
    def split_text(self, text):
        return text.split("\n\n")


custom_splitter = CustomTextSplitter()
custom_docs = custom_splitter.split_documents(documents)
create_vector_store(custom_docs, "chroma_db_custom")


# -----------------------------------
# Query Helper
# -----------------------------------
def query_vector_store(store_name, query):
    persistent_directory = os.path.join(db_dir, store_name)

    if not os.path.exists(persistent_directory):
        print(f"Store {store_name} does not exist.")
        return

    print(f"\n--- Querying {store_name} ---")
    db = Chroma(
        persist_directory=persistent_directory,
        embedding_function=embeddings,
    )

    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 2, "score_threshold": 0.2},
    )

    docs = retriever.invoke(query)

    if not docs:
        print("No relevant documents found.")
        return

    for i, doc in enumerate(docs, 1):
        print(f"\nDocument {i}:")
        print(doc.page_content[:800])
        if doc.metadata:
            print("\nSource:", doc.metadata.get("source", "Unknown"))


# -----------------------------------
# Run Queries
# -----------------------------------
query = "How did Juliet die?"

query_vector_store("chroma_db_char", query)
query_vector_store("chroma_db_sent", query)
query_vector_store("chroma_db_token", query)
query_vector_store("chroma_db_rec_char", query)
query_vector_store("chroma_db_custom", query)