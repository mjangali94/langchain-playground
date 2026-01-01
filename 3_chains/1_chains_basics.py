from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama

# Load environment variables from .env
load_dotenv()

# -----------------------------
# Initialize the model
# -----------------------------
# For OpenAI:
# model = ChatOpenAI(model="gpt-4o", temperature=0.7)

# For Ollama:
model = Ollama(model="gemma3", temperature=0.7)

# -----------------------------
# Define the prompt template
# -----------------------------
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}."),
        ("human", "Tell me {joke_count} jokes."),
    ]
)

# -----------------------------
# Combine prompt with model using LCEL (LangChain Expression Language)
# -----------------------------
chain = prompt_template | model | StrOutputParser()

# -----------------------------
# Run the chain
# -----------------------------
result = chain.invoke({"topic": "lawyers", "joke_count": 3})

# -----------------------------
# Output
# -----------------------------
print("\n--- Generated Jokes ---")
print(result)