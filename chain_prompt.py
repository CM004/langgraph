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

class BlogState(TypedDict):
    topic: str
    outline: str
    content : str

def generate_outline(state: BlogState) -> BlogState:
    topic = state['topic']
    prompt = f"Generate a blog post outline for the topic: {topic}"
    outline = model.invoke(prompt).content
    state['outline'] = outline
    return state

def generate_content(state: BlogState) -> BlogState:
    outline = state['outline']
    prompt = f"Generate blog post content based on the following outline: {outline}"
    content = model.invoke(prompt).content
    state['content'] = content
    return state

graph = StateGraph(BlogState)

graph.add_node('generate_outline', generate_outline)
graph.add_node('generate_content', generate_content)

graph.add_edge(START, 'generate_outline')
graph.add_edge('generate_outline', 'generate_content')
graph.add_edge('generate_content', END)

workflow = graph.compile()

initial_state = {
    'topic': 'Rising corruption and class division in India'
}

final_state = workflow.invoke(initial_state)

if __name__ == "__main__":
    print(final_state['outline'])
    print(final_state['content'])