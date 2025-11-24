from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_community.llms import Ollama


load_dotenv()

model=Ollama(model="gemma3")

chat_history =[
    SystemMessage(content="You are a helpful assistant."),
    SystemMessage(content="The user require the assistant to assist them with their mood, could be about their diet, exercise, or anything else."),
    SystemMessage(content="Chat with them to find out the problem and provide solutions.")
]

while True:
    input_user=input("You: ")
    if input_user.lower() == "exit":
        break
    chat_history.append(HumanMessage(content=input_user))
    ai_response=model.invoke(chat_history)
    chat_history.append(AIMessage(ai_response))
    print(f"AI :{ai_response}")

print("---- Message History ----")
print(chat_history)

