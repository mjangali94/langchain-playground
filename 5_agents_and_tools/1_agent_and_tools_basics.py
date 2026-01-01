from dotenv import load_dotenv
from langchain_classic import hub
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_classic.tools import Tool
from langchain_ollama import OllamaLLM
# from langchain_openai import ChatOpenAI   # Optional if you want OpenAI

import datetime

# Load environment variables
load_dotenv()


def get_current_time(*_args, **_kwargs) -> str:
    """Return the current time in H:MM AM/PM format."""
    return datetime.datetime.now().strftime("%I:%M %p")


tools = [
    Tool(
        name="Time",
        func=get_current_time,
        description="Returns the current local time.",
    ),
]

# Load ReAct prompt
prompt = hub.pull("hwchase17/react")

# Choose your model
llm = OllamaLLM(model="gemma3")
# llm = ChatOpenAI(model="gpt-4o", temperature=0)

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
    stop_sequence=True,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
)

response = agent_executor.invoke({"input": "What time is it?"})
print("Response:", response)