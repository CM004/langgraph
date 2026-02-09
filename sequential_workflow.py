from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI  #universal OpenAI-compatible LLM connector
from typing import TypedDict
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatOpenAI(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1"
)

class LLMState(TypedDict):
    quest: str
    ans: str

def llma_qa(state: LLMState) -> LLMState:
    quest = state['quest']
    prompt = f"Answer the following question concisely: {quest}"
    ans = model.invoke(prompt).content
    state['ans'] = ans
    return state

graph = StateGraph(LLMState)

graph.add_node('llm_qa', llma_qa)

graph.add_edge(START, 'llm_qa')
graph.add_edge('llm_qa', END)

workflow = graph.compile()

initial_state = {
    'quest': 'what is a clawdbot. explain in easy to understand 5 bullet points. '
}
final_state = workflow.invoke(initial_state)

if __name__ == "__main__":
    print(final_state['ans'])