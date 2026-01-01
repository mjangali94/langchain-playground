from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

# -----------------------------
# Initialize the model
# -----------------------------
# OpenAI example:
# model = ChatOpenAI(model="gpt-4o", temperature=0.7)

# Ollama example:
model = Ollama(model="gemma3", temperature=0.7)

# -----------------------------
# Define prompt template
# -----------------------------
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}."),
        ("human", "Tell me {joke_count} jokes."),
    ]
)

# -----------------------------
# Define additional processing steps
# -----------------------------
# Convert output to uppercase
uppercase_output = RunnableLambda(lambda x: x.upper())

# Count words and prepend the count
count_words = RunnableLambda(lambda x: f"Word count: {len(x.split())}\n{x}")

# -----------------------------
# Combine into LCEL chain
# -----------------------------
chain = prompt_template | model | StrOutputParser() | uppercase_output | count_words

# -----------------------------
# Run the chain
# -----------------------------
result = chain.invoke({"topic": "lawyers", "joke_count": 3})

# -----------------------------
# Output
# -----------------------------
print("\n--- Processed Jokes ---")
print(result)