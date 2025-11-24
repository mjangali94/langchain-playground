from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda,RunnableSequence, RunnableParallel



prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a car expert how guide costumers with their purchase"),
        ("human", "tell me features about {product}")
    ]
)

model = Ollama(model="gemma3")

def analyze_cons(features):
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("human", "Given all the features extract cons features from this list {features}")
        ]
    )
    return prompt_template.invoke({"features":features})


def analyze_pros(features):
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("human", "Given all the features extract pros features from this list {features}")
        ]
    )
    return prompt_template.invoke({"features":features})

def combine_pros_cons(pros, cons):
    return f"Pros:\n{pros}\n\nCons:\n{cons}"

pros_lambda_chain = (RunnableLambda(lambda x: analyze_pros(x)) | model | StrOutputParser())
cons_lambda_chain = (RunnableLambda(lambda x: analyze_cons(x)) | model | StrOutputParser())


chain = (prompt_template | model | StrOutputParser() |
         RunnableParallel(branches={"pros":pros_lambda_chain,"cons":cons_lambda_chain}) |
         RunnableLambda(lambda x :combine_pros_cons(x["branches"]["pros"],x["branches"]["cons"]))
         )

result = chain.invoke({"product":"Hunda civic 2025"})
print(result)