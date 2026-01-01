from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch
from langchain_community.llms import Ollama

# Load environment variables from .env
load_dotenv()

# Initialize Ollama model
model = Ollama(model="gemma3", temperature=0.7)

# -----------------------------
# Define prompt templates for feedback types
# -----------------------------
positive_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "Generate a thank you note for this positive feedback: {feedback}."),
    ]
)

negative_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "Generate a response addressing this negative feedback: {feedback}."),
    ]
)

neutral_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "Generate a request for more details for this neutral feedback: {feedback}."),
    ]
)

escalate_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "Generate a message to escalate this feedback to a human agent: {feedback}."),
    ]
)

# -----------------------------
# Define the feedback classification prompt
# -----------------------------
classification_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "Classify the sentiment of this feedback as positive, negative, neutral, or escalate: {feedback}."),
    ]
)

# -----------------------------
# Define the runnable branches for handling feedback
# -----------------------------
branches = RunnableBranch(
    (
        lambda x: "positive" in x.lower(),
        positive_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: "negative" in x.lower(),
        negative_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: "neutral" in x.lower(),
        neutral_feedback_template | model | StrOutputParser()
    ),
    escalate_feedback_template | model | StrOutputParser()
)

# -----------------------------
# Create classification chain
# -----------------------------
classification_chain = classification_template | model | StrOutputParser()

# Combine classification and response generation
chain = classification_chain | branches

# -----------------------------
# Example review
# -----------------------------
review = "The product is terrible. It broke after just one use and the quality is very poor."

# Run the chain
result = chain.invoke({"feedback": review})

# Output the result
print("\n--- Feedback Response ---")
print(result)