# Docs: https://python.langchain.com/v0.1/docs/modules/tools/custom_tools/

import os
from typing import Type

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI


# -----------------------
# Setup
# -----------------------
load_dotenv()


# -----------------------
# Schemas
# -----------------------
class SimpleSearchInput(BaseModel):
    query: str = Field(description="Search query text")


class MultiplyNumbersArgs(BaseModel):
    x: float = Field(description="First number")
    y: float = Field(description="Second number")


# -----------------------
# Tools
# -----------------------
class SimpleSearchTool(BaseTool):
    name = "simple_search"
    description = "Useful for answering questions about current events"
    args_schema: Type[BaseModel] = SimpleSearchInput

    def _run(self, query: str) -> str:
        """Execute Tavily search."""
        from tavily import TavilyClient

        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return "TAVILY_API_KEY is missing in environment variables."

        client = TavilyClient(api_key=api_key)
        results = client.search(query=query)
        return f"Search results for '{query}':\n{results}"


class MultiplyNumbersTool(BaseTool):
    name = "multiply_numbers"
    description = "Multiply two numbers"
    args_schema: Type[BaseModel] = MultiplyNumbersArgs

    def _run(self, x: float, y: float) -> str:
        """Return the multiplication result."""
        result = x * y
        return f"The product of {x} and {y} is {result}"


tools = [SimpleSearchTool(), MultiplyNumbersTool()]


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
    "Search Apple Intelligence",
    "Multiply 10 and 20",
]

for query in tests:
    response = agent_executor.invoke({"input": query})
    print(f"\nQuery: {query}")
    print("Response:", response)