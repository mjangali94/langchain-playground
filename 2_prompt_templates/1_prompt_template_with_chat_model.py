from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage


joke_count_input=int(input("How many jokes do we require?"))
joke_subject_input=input("What is your subject?")
model = Ollama(model="gemma3")

# # Prompt with Multiple Placeholders
template_multiple = """You are a helpful assistant for adults.
Human: Tell me a {joke_count} story about a {subject}.
Assistant:"""
prompt_multiple = ChatPromptTemplate.from_template(template_multiple)
prompt = prompt_multiple.invoke({"joke_count": joke_count_input, "subject": joke_subject_input})
print("\n----- Prompt with Multiple Placeholders -----\n")
print(prompt)
result = model.invoke(prompt)
print("\n----- Responds from AI -----\n")
print(result)




# Prompt with System and Human Messages (Using Tuples)
messages = [
    ("system", "You are a comedian who tells jokes to kids, about {subject}."),
    ("human", "Tell me {joke_count} jokes."),
]
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"joke_count": joke_count_input, "subject": joke_subject_input})
print("\n----- Prompt with System and Human Messages (Tuple) -----\n")
print(prompt)
result = model.invoke(prompt)
print("\n----- Responds from AI -----\n")
print(result)

