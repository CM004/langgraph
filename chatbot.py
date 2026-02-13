from typing import Literal, TypedDict, Annotated
from langgraph.graph import StateGraph, START, END 
from langchain_openai import ChatOpenAI 
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
import os
import operator

load_dotenv()

model = ChatOpenAI(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
    timeout=60
)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages":[response]}

checkpointer = MemorySaver()

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

workflow = graph.compile(checkpointer=checkpointer)
initial_state={
    "messages":[HumanMessage(content="What is the capital of France?")]
}

if __name__ == "__main__":
    thread_id = "1"
    while True:
        user_message=input("User:")
        if user_message.strip().lower() in ["exit","quit","bye","stop"]:
            break
        config = {"configurable":{"thread_id":thread_id}}
        response = workflow.invoke({"messages":HumanMessage(content=user_message)}, config=config)
        print("AI:",response["messages"][-1].content)
        print(workflow.get_state(config=config))
