import os
from dotenv import load_dotenv

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


# ----------------------------
# Setup
# ----------------------------
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "..", "..", "4_rag", "db", "chroma_db_with_metadata")

if not os.path.exists(DB_PATH):
    raise FileNotFoundError(f"Chroma DB not found at: {DB_PATH}")

print("Loading existing vector store...")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

llm = ChatOpenAI(model="gpt-4o")


# ----------------------------
# Contextualization Prompt
# ----------------------------
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "Rewrite the latest user question so it is standalone and understandable "
         "without chat history. Do NOT answer it."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm=llm,
    retriever=retriever,
    prompt=contextualize_q_prompt,
)


# ----------------------------
# QA Prompt
# ----------------------------
qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer using the retrieved context. "
            "If unknown, say you don't know. "
            "Keep answers concise (max 3 sentences).\n\n{context}"
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(
    history_aware_retriever,
    question_answer_chain
)


# ----------------------------
# ReAct Agent Setup
# ----------------------------
react_prompt = hub.pull("hwchase17/react")

tools = [
    Tool(
        name="Answer Question",
        func=lambda query, **kwargs: rag_chain.invoke(
            {"input": query, "chat_history": kwargs.get("chat_history", [])}
        ),
        description="Use this to answer questions using the RAG knowledge base."
    )
]

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
)


# ----------------------------
# Chat Loop
# ----------------------------
chat_history = []

print("RAG Chat Ready. Type 'exit' to quit.\n")

try:
    while True:
        query = input("You: ").strip()

        if query.lower() == "exit":
            print("Goodbye!")
            break

        response = agent_executor.invoke(
            {"input": query, "chat_history": chat_history}
        )

        answer = response.get("output", "")
        print(f"AI: {answer}")

        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=answer))

except KeyboardInterrupt:
    print("\nSession closed.")