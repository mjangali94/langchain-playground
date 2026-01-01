# Docs: https://python.langchain.com/v0.1/docs/modules/tools/custom_tools/

from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import StructuredTool, Tool
from langchain_openai import ChatOpenAI


# -----------------------
# Tool Functions
# -----------------------
def greet_user(name: str) -> str:
    """Greet the user by name."""
    return f"Hello, {name}!"


def reverse_string(text: str) -> str:
    """Return the reversed string."""
    return text[::-1]


def concatenate_strings(a: str, b: str) -> str:
    """Concatenate two strings."""
    return a + b


# -----------------------
# Structured Tool Schema
# -----------------------
class ConcatenateStringsArgs(BaseModel):
    a: str = Field(description="First string")
    b: str = Field(description="Second string")


# -----------------------
# Tools
# -----------------------
tools = [
    Tool(
        name="GreetUser",
        func=greet_user,
        description="Greets the user by name.",
    ),
    Tool(
        name="ReverseString",
        func=reverse_string,
        description="Reverses the given string.",
    ),
    StructuredTool.from_function(
        func=concatenate_strings,
        name="ConcatenateStrings",
        description="Concatenates two strings.",
        args_schema=ConcatenateStringsArgs,
    ),
]


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

for t in tests:
    response = agent_executor.invoke({"input": t})
    print(f"\nQuery: {t}")
    print("Response:", response)