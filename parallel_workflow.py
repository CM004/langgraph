from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI 
from typing import TypedDict
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatOpenAI(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
    timeout=60
)

class LearningState(TypedDict,total=False):
    supervised: str
    unsupervised: str
    mixed : str
    summary : str


def supervised(state: LearningState) -> LearningState:
    supervised = state['supervised']
    prompt = f"Identify this type of learning : {supervised}"
    answer = model.invoke(prompt).content
    #state['answer'] = answer
    return {'supervised':answer} 
    # partial state update in parallel execution, only updating the 'supervised' key

def unsupervised(state: LearningState) -> LearningState:
    unsupervised = state['unsupervised']
    prompt = f"Identify this type of learning : {unsupervised}"
    answer = model.invoke(prompt).content
    #state['answer'] = answer
    return {'unsupervised':answer}

def mixed(state: LearningState) -> LearningState:
    mixed = state['mixed']
    prompt = f"Identify this type of learning : {mixed}"
    answer = model.invoke(prompt).content
    #state['answer'] = answer
    return {'mixed':answer}

def summary(state: LearningState) -> LearningState:
    combined = f"""
{state['supervised']}
{state['unsupervised']}
{state['mixed']}
"""
    prompt = f"Summarize the following content:\n{combined}"
    answer = model.invoke(prompt).content
    return {"summary": answer}

graph = StateGraph(LearningState)

graph.add_node('supervised', supervised)
graph.add_node('unsupervised', unsupervised)
graph.add_node('mixed', mixed)
graph.add_node('summary',summary)

graph.add_edge(START, 'supervised')
graph.add_edge(START, 'unsupervised')
graph.add_edge(START, 'mixed')

graph.add_edge('supervised', 'summary')
graph.add_edge('unsupervised', 'summary')
graph.add_edge('mixed', 'summary')

graph.add_edge('summary', END)

workflow = graph.compile()

initial_state = {
    'supervised': 'A bank uses supervised learning to detect credit card fraud. The model is trained on past transactions labeled as “fraud” or “not fraud.” By learning patterns from labeled data, it predicts whether new transactions are suspicious. Humans verify flagged cases, and their feedback improves future accuracy, making fraud detection faster and more reliable in real time.',
    'unsupervised': 'A retail company uses unsupervised learning to group customers based on shopping behavior. The algorithm analyzes purchase history without labels and discovers hidden patterns, such as budget shoppers, premium buyers, or seasonal customers. These clusters help the company design targeted marketing strategies, personalize offers, and improve inventory planning without needing predefined categories.',
    'mixed': 'A medical imaging system uses mixed learning to diagnose diseases from scans. A small set of images is labeled by doctors, while thousands remain unlabeled. The model learns from both datasets, using labeled examples as guidance and unlabeled data to refine patterns. This reduces expert workload and improves diagnostic accuracy when labeled medical data is limited.',
}

final_state = workflow.invoke(initial_state)

if __name__ == "__main__":
    print(final_state['summary'])