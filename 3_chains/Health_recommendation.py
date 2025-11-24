from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_ollama import OllamaLLM
user_input = input("Dear user please write up the situation you are dealing:")

messages= [
    ("system", "You are a life coaching assistant. "
               "You are supposed to help the user find out more about their situation "
               "so list top (max) 5 issues they are dealing with"),
    ("human", "This is a story about my recent month of life including physical and mental difficulties that I have. {message}")

]
model = OllamaLLM(model="gemma3")
prompt = ChatPromptTemplate.from_messages(messages)


def diet_solution_prompt(x):
    prompt = ChatPromptTemplate.from_messages([
        ("system","You are a life coaching assistant. You are given a list of 5 user's difficulties produced by ai and based on user's story."),
        ("ai","{x}"),
        ("human","for each difficulty suggest me a diet solution.")
    ])
    return prompt.invoke({"x":x})


def exercise_solution_prompt(x):
    prompt = ChatPromptTemplate.from_messages([
        ("system","You are a life coaching assistant. You are given a list of 5 user's difficulties produced by ai and based on user's story."),
        ("ai","{x}"),
        ("human","for each difficulty suggest me an exercise solution.")
    ])
    return prompt.invoke({"x":x})

def combine_physical_mental(physical, metal):
    return f"Diet solutions: {physical} \n\n Exercise solutions: {metal}\n\n"


diet_runnable_chain=(RunnableLambda(lambda x: diet_solution_prompt(x)) | model | StrOutputParser())
exercise_runnable_chain= (RunnableLambda(lambda x: exercise_solution_prompt(x)) | model | StrOutputParser())


physical_or_mental_chain = RunnableParallel({"diet":diet_runnable_chain,"exercise":exercise_runnable_chain})

combine_runnable_lambda = RunnableLambda(lambda x: combine_physical_mental(x["diet"], x["exercise"]))


chain = prompt | model | StrOutputParser() | physical_or_mental_chain | combine_runnable_lambda

# chain = prompt | model | StrOutputParser()


result = chain.invoke({"message":user_input})
print(result)
