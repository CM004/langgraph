from typing import Literal, TypedDict, Annotated
from langgraph.graph import StateGraph, START, END 
from langchain_openai import ChatOpenAI 
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
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

class TweetEvaluationSchema(BaseModel):
    evaluation: Literal["approved","needs_improvement"] = Field(description="Evaluation of the tweet")
    feedback: str = Field(description="Feedback for improving the tweet")

parser = PydanticOutputParser(pydantic_object=TweetEvaluationSchema)

class TweetState(TypedDict):
    topic: str
    tweet: str
    evaluation:Literal["approved","needs_improvement"]
    feedback:str
    iteration:int
    max_iterations:int
    tweet_history: Annotated[list[str],operator.add ]
    feedback_history: Annotated[list[str],operator.add ]

def generate_tweet(state: TweetState) -> TweetState:
    prompt = f"""You are funny and clever Twitter/X influencer. Generate a short,original and funny tweet about {state['topic']}
Rules:
1. Keep it under 280 characters.
2. Use humor, irony, sarcasm, cultural reference and wit.
3. Use simple, day-to-day language.
4. This is version {state['iteration']+1}
"""
    response = model.invoke(prompt).content
    return {'tweet': response,
            "tweet_history":[response]}

def evaluate_tweet(state: TweetState) -> TweetState:
    prompt = f"""You are a no laugh, ruthless Twitter/X critic. Evaluate the following tweet:
{state['tweet']}
Rules:
1. Evaluate on originality, humor, creativity, engagement and format.
2. Auto reject if written in Q&A format or does not follow tweet conventions.
3. Respond with only 'approved' or 'needs_improvement' and a short feedback for improvement (if applicable).
\n{parser.get_format_instructions()}"""
    response = model.invoke(prompt)
    response = parser.parse(response.content)
    return {'evaluation': response.evaluation,
            'feedback': response.feedback,
            "feedback_history":[response.feedback]}

def optimise_tweet(state: TweetState) -> TweetState:
    prompt = f"""You are a social media expert. Punch up tweets for virality and humor based on this feedback:
\n{state['feedback']}
\nTopic - {state['topic']}
\nOriginal tweet - {state['tweet']}
Re write it as short, viral worthy tweet. Avoid Q&A style and keep under 280 characters.
"""
    response = model.invoke(prompt).content
    iteration = state["iteration"]+1
    return {'tweet': response,
            'iteration': iteration,
            "tweet_history":[response]}

def router(state: TweetState) -> Literal["needs_improvement", "approved"]:
    if state['evaluation'] == "approved" or state['iteration'] >= state['max_iterations']:
        return "approved"
    return "needs_improvement"

graph = StateGraph(TweetState)

graph.add_node("generate_tweet",generate_tweet)
graph.add_node("evaluate_tweet",evaluate_tweet)
graph.add_node("optimise_tweet",optimise_tweet)

graph.add_edge(START, "generate_tweet")
graph.add_edge("generate_tweet", "evaluate_tweet")
graph.add_conditional_edges("evaluate_tweet", router,{"approved": END, "needs_improvement": "optimise_tweet"})
graph.add_edge("optimise_tweet", "evaluate_tweet")

workflow = graph.compile()

initial_state = {
    "topic": "using ai in making songs",
    "iteration": 0,
    "max_iterations": 5
}

if __name__ == "__main__":
    print(workflow.invoke(initial_state))