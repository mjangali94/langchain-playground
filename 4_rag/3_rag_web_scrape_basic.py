import os
from dotenv import load_dotenv

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()

# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_apple")

# -------------------------
# 1) Load from Web
# -------------------------
urls = ["https://www.apple.com/"]

loader = WebBaseLoader(web_paths=urls)
documents = loader.load()

# -------------------------
# 2) Split Content
# -------------------------
text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
)

docs = text_splitter.split_documents(documents)

print("\n--- Document Chunks Information ---")
print(f"Number of chunks: {len(docs)}")
print(f"Sample chunk:\n{docs[0].page_content[:500]}...\n")

# -------------------------
# 3) Embeddings
# -------------------------
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# -------------------------
# 4) Chroma DB (Persist)
# -------------------------
if not os.path.exists(persistent_directory):
    print(f"\n--- Creating vector store in {persistent_directory} ---")
    db = Chroma.from_documents(
        docs,
        embedding=embeddings,
        persist_directory=persistent_directory
    )
    print("--- Vector store created ---")
else:
    print(f"Vector store already exists at {persistent_directory}")
    db = Chroma(
        persist_directory=persistent_directory,
        embedding_function=embeddings
    )

# -------------------------
# 5) Query Vector Store
# -------------------------
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

query = "What new products are announced on Apple.com?"
results = retriever.invoke(query)

print("\n--- Relevant Documents ---")
for i, doc in enumerate(results, 1):
    print(f"\nDocument {i}:\n{doc.page_content[:900]}")
    if doc.metadata:
        print(f"\nSource: {doc.metadata.get('source', 'Unknown')}")