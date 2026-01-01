from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel

# -----------------------------
# Initialize the model
# -----------------------------
model = Ollama(model="gemma3", temperature=0.7)

# -----------------------------
# Define the main product feature prompt
# -----------------------------
main_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a car expert who guides customers with their purchase."),
        ("human", "Tell me features about {product}.")
    ]
)

# -----------------------------
# Define pros and cons analysis functions
# -----------------------------
def analyze_pros(features):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "Given all the features, extract the pros features from this list: {features}")
        ]
    )
    return prompt.invoke({"features": features})

def analyze_cons(features):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "Given all the features, extract the cons features from this list: {features}")
        ]
    )
    return prompt.invoke({"features": features})

# -----------------------------
# Function to combine pros and cons
# -----------------------------
def combine_pros_cons(pros, cons):
    return f"Pros:\n{pros}\n\nCons:\n{cons}"

# -----------------------------
# Create Runnable chains for pros and cons
# -----------------------------
pros_chain = RunnableLambda(lambda x: analyze_pros(x)) | model | StrOutputParser()
cons_chain = RunnableLambda(lambda x: analyze_cons(x)) | model | StrOutputParser()

# -----------------------------
# Combine into a main chain with RunnableParallel
# -----------------------------
chain = (
    main_prompt_template
    | model
    | StrOutputParser()
    | RunnableParallel(branches={"pros": pros_chain, "cons": cons_chain})
    | RunnableLambda(lambda x: combine_pros_cons(x["branches"]["pros"], x["branches"]["cons"]))
)

# -----------------------------
# Run the chain
# -----------------------------
result = chain.invoke({"product": "Honda Civic 2025"})

# -----------------------------
# Output
# -----------------------------
print("\n--- Product Pros and Cons ---")
print(result)