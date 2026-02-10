from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI 
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.output_parsers import PydanticOutputParser

import operator
import os

load_dotenv()

model = ChatOpenAI(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
    timeout=60
)

class EvaluationSchema(BaseModel):
    feedback: str = Field(description="Feedback for the essay")
    score: int = Field(description="Score for the essay, from 1 to 10")

parser = PydanticOutputParser(pydantic_object=EvaluationSchema)
structured_model = model
#structured_model = model.with_structured_output(EvaluationSchema)

class EssayState(TypedDict):
    essay_text: str
    language_feedback: str
    analysis_feedback : str
    thought_clarity_feedback : str
    summary_feedback : str
    individual_scores : Annotated[list[int], operator.add]
    avg_score : float

def evaluate_language(state: EssayState) -> EssayState:
    essay_text = state['essay_text']
    prompt = f"""Provide feedback on the language used in the following essay and provide a score out of 10:\n
    {essay_text}\n
    {parser.get_format_instructions()}"""
    response = structured_model.invoke(prompt)
    response = parser.parse(response.content)
    return {'language_feedback':response.feedback,
            'individual_scores':[response.score]} 

def evaluate_analysis(state: EssayState) -> EssayState:
    essay_text = state['essay_text']
    prompt = f"""Provide feedback on the analysis in the following essay and provide a score out of 10:\n
    {essay_text}\n
    {parser.get_format_instructions()}"""
    response = structured_model.invoke(prompt)
    response = parser.parse(response.content)
    return {'analysis_feedback':response.feedback,
            'individual_scores':[response.score]} 

def evaluate_thought_clarity(state: EssayState) -> EssayState:
    essay_text = state['essay_text']
    prompt = f"""Provide feedback on the thought clarity in the following essay and provide a score out of 10:\n
    {essay_text}\n
    {parser.get_format_instructions()}"""
    response = structured_model.invoke(prompt)
    response = parser.parse(response.content)
    return {'thought_clarity_feedback':response.feedback,
            'individual_scores':[response.score]} 

def final_evaluation(state: EssayState) -> EssayState:
    combined = f"""
Language Feedback - {state['language_feedback']}\n
Analysis Feedback - {state['analysis_feedback']}\n
Thought Clarity Feedback - {state['thought_clarity_feedback']}\n
"""
    prompt = f"Based on the following feedback create a summarised feedback:\n{combined}"
    summary_feedback = model.invoke(prompt).content
    avg_score = sum(state['individual_scores']) / len(state['individual_scores'])
    return {"summary_feedback": summary_feedback, 
            "avg_score": avg_score}

def join_node(state: EssayState) -> EssayState:
    return {}

graph = StateGraph(EssayState)

graph.add_node('evaluate_language', evaluate_language)
graph.add_node('evaluate_analysis', evaluate_analysis)
graph.add_node('evaluate_thought', evaluate_thought_clarity)
graph.add_node('final_evaluation', final_evaluation)
graph.add_node("join", join_node)

graph.add_edge(START, 'evaluate_language')
graph.add_edge(START, 'evaluate_analysis')
graph.add_edge(START, 'evaluate_thought')

graph.add_edge('evaluate_language', 'join')
graph.add_edge('evaluate_analysis', 'join')
graph.add_edge('evaluate_thought', 'join')

graph.add_edge('join', 'final_evaluation')

graph.add_edge('final_evaluation', END)

workflow = graph.compile()

essay = """**The Rise of Corruption in India**

Corruption, in its various forms, has become a pervasive problem in India. Bribery, nepotism, and crony capitalism are just a few examples of the corrupt practices that have become entrenched in Indian society. According to a report by Transparency International, India ranks 80th out of 180 countries in terms of corruption, with a score of 41 out of 100. The country has witnessed several high-profile scams in recent years, including the 2G spectrum scam and the coal allocation scam, which have highlighted the depth of corruption in India's government and business sectors.

The causes of corruption in India are complex and multifaceted. Weak institutions, lack of transparency and accountability, and a culture of impunity have all contributed to the proliferation of corrupt practices. The country's rapid economic growth has also created new opportunities for corruption, as politicians and business leaders seek to accumulate wealth and power. For example, the real estate sector in India is notorious for its corrupt practices, with many developers and politicians colluding to acquire land and build luxury projects, often at the expense of the poor and marginalized.

**The Growing Class Divide in India**

India's economic growth has also been accompanied by a growing class divide. The country's economic inequality is stark, with the top 10% of the population holding over 70% of the country's wealth. The poor and marginalized, on the other hand, struggle to access basic services such as education, healthcare, and sanitation. The visible manifestations of class division in India are stark, with luxury developments and slums existing side by side in many cities. The wealthy and powerful have access to the best education, healthcare, and infrastructure, while the poor are forced to make do with substandard services.

The social and economic implications of class division in India are far-reaching. Limited access to education and healthcare has resulted in a lack of social mobility, with many people trapped in poverty. Social unrest and protests have become increasingly common, as the poor and marginalized demand better living conditions and access to basic services. For example, the recent protests against the Citizenship Amendment Act (CAA) and the National Register of Citizens (NRC) have highlighted the deep-seated anger and frustration among India's marginalized communities.

**The Intersection of Corruption and Class Division**

Corruption and class division are closely intertwined in India. Corrupt politicians and business leaders accumulate wealth and power at the expense of the poor, who are forced to pay bribes and suffer from substandard services. The poor and marginalized are also more vulnerable to exploitation by corrupt officials, who often deny them basic services and rights. For example, a recent report found that over 60% of India's rural population pays bribes to access basic services such as healthcare and education.

Class division also enables corruption in India, as the wealthy and powerful use their influence to maintain their privilege. The country's elite have a stranglehold on power and wealth, and use their connections to accumulate more wealth and power. This has resulted in a system of crony capitalism, where the wealthy and powerful are able to accumulate wealth and power at the expense of the poor and marginalized.

**Consequences and Implications**

The consequences of rising corruption and class division in India are far-reaching. In the short term, social unrest and protests are likely to increase, as the poor and marginalized demand better living conditions and access to basic services. In the long term, corruption and class division could have serious implications for India's development and growth. The country's brain drain, decreased foreign investment, and reduced economic competitiveness could all be exacerbated by corruption and class division.

The implications for India's democracy are also serious. Corruption and class division could lead to a loss of trust in institutions, and a decline in the legitimacy of the government. This could have serious consequences for India's stability and security, as the country faces numerous external and internal challenges.

**Solutions and Recommendations**

So, what can be done to address corruption and class division in India? Strengthening institutions, increasing transparency and accountability, and implementing policies to reduce inequality are all crucial steps. The government, business, and civil society must work together to address these issues, and ensure that the benefits of economic growth are shared by all.

Some potential solutions include:

* Implementing policies to reduce inequality, such as progressive taxation and social welfare programs
* Strengthening institutions, such as the judiciary and the media, to ensure accountability and transparency
* Increasing access to education and healthcare, particularly for the poor and marginalized
* Implementing anti-corruption laws and regulations, and ensuring that they are enforced effectively

**Conclusion**

In conclusion, corruption and class division are serious challenges facing India today. The country's economic growth and development are threatened by these twin evils, which could have serious consequences for India's stability and security. It is imperative that the government, business, and civil society work together to address these issues, and ensure that the benefits of economic growth are shared by all.

As we look to the future, it is clear that India has a long way to go in addressing corruption and class division. However, with the right policies and initiatives, it is possible to create a more equal and just society, where everyone has access to basic services and opportunities. We must demand change, and work together to build a better future for all Indians. The time to act is now, before it's too late."""

initial_state = {
    'essay_text': essay
}

final_state = workflow.invoke(initial_state)

if __name__ == "__main__":
    print(final_state["summary_feedback"])
    print(final_state["avg_score"])