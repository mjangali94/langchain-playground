from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI


# -----------------------
# Setup
# -----------------------
load_dotenv()


# -----------------------
# Tools
# -----------------------
def get_current_time(*args, **kwargs) -> str:
    """Return current time in H:MM AM/PM format."""
    import datetime

    return datetime.datetime.now().strftime("%I:%M %p")


def search_wikipedia(query: str) -> str:
    """Return a short Wikipedia summary."""
    import wikipedia

    try:
        return wikipedia.summary(query, sentences=2)
    except Exception:
        return "I couldn't find information on that."


tools = [
    Tool(
        name="Time",
        func=get_current_time,
        description="Get the current time.",
    ),
    Tool(
        name="Wikipedia",
        func=search_wikipedia,
        description="Get information about topics from Wikipedia.",
    ),
]


# -----------------------
# LLM + Agent
# -----------------------
prompt = hub.pull("hwchase17/structured-chat-agent")
llm = ChatOpenAI(model="gpt-4o")

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
)

agent = create_structured_chat_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
)

memory.chat_memory.add_message(
    SystemMessage(
        content=(
            "You are a helpful AI assistant. "
            "Use Time or Wikipedia tools when helpful. "
            "If unsure, ask clarifying questions."
        )
    )
)


# -----------------------
# Chat Loop
# -----------------------
print("Assistant ready. Type 'exit' to quit.\n")

try:
    while True:
        user_input = input("User: ").strip()
        if user_input.lower() == "exit":
            break

        memory.chat_memory.add_message(HumanMessage(content=user_input))

        response = agent_executor.invoke({"input": user_input})
        output = response.get("output", "")

        print("Bot:", output)
        memory.chat_memory.add_message(AIMessage(content=output))

except KeyboardInterrupt:
    print("\nConversation ended.")