from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage



# Load environment variables from .env
load_dotenv()

# Create a llm model
# model = ChatOpenAI(model="gpt-4o")
model = Ollama(model="gemma3")


# Invoke the model with a message
result = model.invoke("What is 2 times 2?")
print("Full result:")
print(result)


# Invoke the model with a list of messages
messages = [
    SystemMessage("Answer the question like the user is a 9 years old kid."),
    HumanMessage("What is 2 times 2?")
]
result = model.invoke(messages)
print("Full result:")
print(result)
