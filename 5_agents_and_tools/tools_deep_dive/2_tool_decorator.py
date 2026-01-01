# Docs: https://python.langchain.com/v0.1/docs/modules/tools/custom_tools/

from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import tool
from langchain_openai import ChatOpenAI


# -----------------------
# Tools
# -----------------------

@tool()
def greet_user(name: str) -> str:
    """Greet the user by name."""
    return f"Hello, {name}!"


class ReverseStringArgs(BaseModel):
    text: str = Field(description="Text to be reversed")


@tool(args_schema=ReverseStringArgs)
def reverse_string(text: str) -> str:
    """Return the reversed text."""
    return text[::-1]


class ConcatenateStringsArgs(BaseModel):
    a: str = Field(description="First string")
    b: str = Field(description="Second string")


@tool(args_schema=ConcatenateStringsArgs)
def concatenate_strings(a: str, b: str) -> str:
    """Concatenate two strings."""
    return a + b


tools = [greet_user, reverse_string, concatenate_strings]


# -----------------------
# LLM + Agent
# -----------------------

llm = ChatOpenAI(model="gpt-4o")
prompt = hub.pull("hwchase17/openai-tools-agent")

agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
)


# -----------------------
# Tests
# -----------------------

tests = [
    "Greet Alice",
    "Reverse the string 'hello'",
    "Concatenate 'hello' and 'world'",
]

for query in tests:
    response = agent_executor.invoke({"input": query})
    print(f"\nQuery: {query}")
    print("Response:", response)