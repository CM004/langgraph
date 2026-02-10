from typing import Literal, TypedDict
from langgraph.graph import StateGraph, START, END 
from langchain_openai import ChatOpenAI 
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
import os

load_dotenv()

model = ChatOpenAI(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
    timeout=60
)

class SentimentSchema(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="Sentiment of the review")

class DiagnosisSchema(BaseModel):
    issue: Literal["bug", "feature_request", "other"] = Field(description="Identified issue from the review")
    tone: Literal["angry","frustrated","disappointed","calm"] = Field(description="The emotional tone of the review")
    severity: Literal["low", "medium", "high"] = Field(description="Severity of the issue")

parser = PydanticOutputParser(pydantic_object=SentimentSchema)
parser2 = PydanticOutputParser(pydantic_object=DiagnosisSchema)
structured_model = model

class ReviewState(TypedDict):
    review_text: str
    sentiment: Literal["positive", "negative"]
    diagnosis: dict
    response:str

def find_sentiment(state: ReviewState)-> ReviewState:
    prompt = f"""What is the sentiment of the following review \n {state['review_text']}
    \n{parser.get_format_instructions()}"""
    response = structured_model.invoke(prompt)
    response = parser.parse(response.content)
    return {'sentiment': response.sentiment}

def router(state: ReviewState) -> Literal["run_diagnosis", "positive_response"]:
    if state['sentiment'].lower() == "positive":
        return "positive_response"
    return "run_diagnosis"

def positive_response(state: ReviewState) -> ReviewState:
    prompt = f"""Generate a short and positive response for the following review: \n {state['review_text']}"""
    response = model.invoke(prompt).content
    return {'response': response}

def run_diagnosis(state: ReviewState) -> ReviewState:
    prompt = f"""Run a diagnosis for the negative review: \n {state['review_text']}
    \n{parser2.get_format_instructions()}"""
    response = model.invoke(prompt)
    response = parser2.parse(response.content)
    return {'diagnosis':response.model_dump()}

def negative_response(state: ReviewState) -> ReviewState:
    prompt = f"""You are a support assistent. 
    The user had a '{state['diagnosis']['issue']} issue, sounded '{state['diagnosis']['tone']}' and had a severity of '{state['diagnosis']['severity']}'.
    Write an short and empathetic response to the user."""
    response = model.invoke(prompt).content
    return {'response': response}

graph = StateGraph(ReviewState)

graph.add_node("find_sentiment", find_sentiment)
graph.add_node("positive_response", positive_response)
graph.add_node("run_diagnosis", run_diagnosis)
graph.add_node("negative_response", negative_response)

graph.add_edge(START, "find_sentiment")
graph.add_conditional_edges('find_sentiment', router)
graph.add_edge("positive_response", END)
graph.add_edge("run_diagnosis", "negative_response")
graph.add_edge("negative_response", END)

workflow = graph.compile()

initial_state = {
    # 'review_text': '''The home cleaning service was excellent. 
    # Staff arrived on time, worked carefully, and left everything spotless.
    # Booking process was simple and customer support responded quickly. 
    # I would definitely use this service again.'''
    'review_text': '''The food delivery service was disappointing. 
    Order arrived late and items were cold. 
    The packaging was damaged and one item was missing. 
    Customer care took too long to respond and offered no helpful solution.'''
}

final_state = workflow.invoke(initial_state)

if __name__ == "__main__":
    print("Review Sentiment:", final_state['sentiment'])
    print("ChatSupport response:", final_state['response'])
    #print("Diagnosis information:", final_state['diagnosis'])
    print("Diagnosis information:", final_state.get('diagnosis', "No diagnosis needed"))
