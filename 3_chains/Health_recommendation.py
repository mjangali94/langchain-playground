from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_ollama import OllamaLLM


# ---------------------- CONSTANTS ----------------------
SYSTEM_ISSUE_ANALYZER = (
    "You are a life coaching assistant. "
    "Analyze the user's situation and provide up to 5 key difficulties "
    "they are dealing with."
)

SYSTEM_SOLUTION_ASSISTANT = (
    "You are a helpful life coaching assistant. "
    "You are given a list of 5 user difficulties produced by AI based on user's story."
)


# ---------------------- BASE MODEL ----------------------
model = OllamaLLM(model="gemma3")
parser = StrOutputParser()


# ---------------------- PROMPTS ----------------------
issue_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_ISSUE_ANALYZER),
    ("human", "Here is my story: {message}")
])


def build_solution_prompt(request_text: str, request_type: str):
    """Reusable prompt builder for diet & exercise."""
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_SOLUTION_ASSISTANT),
        ("ai", "{issues}"),
        ("human", f"For each difficulty suggest me a {request_type} solution.")
    ]).invoke({"issues": request_text})


# ---------------------- RUNNABLE HELPERS ----------------------
diet_chain = (
    RunnableLambda(lambda issues: build_solution_prompt(issues, "diet"))
    | model
    | parser
)

exercise_chain = (
    RunnableLambda(lambda issues: build_solution_prompt(issues, "exercise"))
    | model
    | parser
)

solution_chain = RunnableParallel({
    "diet": diet_chain,
    "exercise": exercise_chain
})


def combine_solutions(diet: str, exercise: str) -> str:
    return (
        "===== Diet Solutions =====\n"
        f"{diet}\n\n"
        "===== Exercise Solutions =====\n"
        f"{exercise}\n"
    )


combine_chain = RunnableLambda(
    lambda result: combine_solutions(result["diet"], result["exercise"])
)

# ---------------------- MAIN PIPELINE ----------------------
chain = issue_prompt | model | parser | solution_chain | combine_chain


if __name__ == "__main__":
    user_input = input("Dear user, please describe your situation:\n")
    result = chain.invoke({"message": user_input})
    print(result)
