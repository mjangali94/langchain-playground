from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama

# Load environment variables from .env
load_dotenv()

# -----------------------------
# Initialize the model
# -----------------------------
# OpenAI example:
# model = ChatOpenAI(model="gpt-4", temperature=0.7)

# Ollama example:
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
# Define runnables
# -----------------------------
# Step 1: Format the prompt
format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))

# Step 2: Invoke the model with the messages from the formatted prompt
invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))

# Step 3: Parse the model output (extract content string)
parse_output = RunnableLambda(lambda x: x.content)

# -----------------------------
# Combine into a RunnableSequence
# -----------------------------
chain = RunnableSequence(first=format_prompt, middle=[invoke_model], last=parse_output)

# -----------------------------
# Run the chain
# -----------------------------
response = chain.invoke({"topic": "lawyers", "joke_count": 3})

# -----------------------------
# Output
# -----------------------------
print("\n--- Generated Jokes ---")
print(response)