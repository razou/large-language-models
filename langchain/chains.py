from utils.setup import PrepareEnv
import os
# import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import (
    LLMChain ,
    SimpleSequentialChain,
    SequentialChain 
)

from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain,RouterOutputParser


def multi_chain(question: str):
    
    physics_template = """You are a very smart physics professor. \
    You are great at answering questions about physics in a concise and easy to understand manner. \
    When you don't know the answer to a question you admit that you don't know.

    Here is a question:
    {input}"""


    math_template = """You are a very good mathematician. \
    You are great at answering math questions. \
    You are so good because you are able to break down  hard problems into their component parts, \
    answer the component parts, and then put them together to answer the broader question.

    Here is a question:
    {input}"""

    history_template = """You are a very good historian. \
    You have an excellent knowledge of and understanding of people,\
    events and contexts from a range of historical periods. \
    You have the ability to think, reflect, debate, discuss and \
    evaluate the past. You have a respect for historical evidence\
    and the ability to make use of it to support your explanations and judgements.

    Here is a question:
    {input}"""


    computerscience_template = """ You are a successful computer scientist.\
    You have a passion for creativity, collaboration,\
    forward-thinking, confidence, strong problem-solving capabilities,\
    understanding of theories and algorithms, and excellent communication \
    skills. You are great at answering coding questions. \
    You are so good because you know how to solve a problem by \
    describing the solution in imperative steps \
    that a machine can easily interpret and you know how to \
    choose a solution that has a good balance between \
    time complexity and space complexity. 

    Here is a question:
    {input}"""


    prompt_infos = [
        {
        "name": "physics", 
        "description": "Good for answering questions about physics", 
        "prompt_template": physics_template
        },
        {
            "name": "math", 
            "description": "Good for answering math questions", 
            "prompt_template": math_template
        },
        {
            "name": "History", 
            "description": "Good for answering history questions", 
            "prompt_template": history_template
        },
        {
            "name": "computer science", 
            "description": "Good for answering computer science questions", 
            "prompt_template": computerscience_template
        }
    ]
    
    llm = ChatOpenAI(temperature=0)
    
    
    destination_chains = {}
    for p_info in prompt_infos:
        name = p_info["name"]
        prompt_template = p_info["prompt_template"]
        prompt = ChatPromptTemplate.from_template(template=prompt_template)
        chain = LLMChain(llm=llm, prompt=prompt)
        destination_chains[name] = chain  
        
    destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
    destinations_str = "\n".join(destinations)
    
    default_prompt = ChatPromptTemplate.from_template("{input}")
    default_chain = LLMChain(llm=llm, prompt=default_prompt)
    
    
    
    MULTI_PROMPT_ROUTER_TEMPLATE = """Given a raw text input to a \
    language model select the model prompt best suited for the input. \
    You will be given the names of the available prompts and a \
    description of what the prompt is best suited for. \
    You may also revise the original input if you think that revising\
    it will ultimately lead to a better response from the language model.

    << FORMATTING >>
    Return a markdown code snippet with a JSON object formatted to look like:
    ```json
    {{{{
        "destination": string \ name of the prompt to use or "DEFAULT"
        "next_inputs": string \ a potentially modified version of the original input
    }}}}
    ```

    REMEMBER: "destination" MUST be one of the candidate prompt \
    names specified below OR it can be "DEFAULT" if the input is not\
    well suited for any of the candidate prompts.
    REMEMBER: "next_inputs" can just be the original input \
    if you don't think any modifications are needed.

    << CANDIDATE PROMPTS >>
    {destinations}

    << INPUT >>
    {{input}}

    << OUTPUT (remember to include the ```json)>>"""
    
    router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
    router_prompt = PromptTemplate(template=router_template, input_variables=["input"], output_parser=RouterOutputParser())
    router_chain = LLMRouterChain.from_llm(llm, router_prompt)
    chain = MultiPromptChain(router_chain=router_chain, destination_chains=destination_chains, default_chain=default_chain, verbose=True)
    return chain.run(question)
    
        
    
    


if __name__ == "__main__":
    
    PrepareEnv.set_api_key()
    # print(f"{os.environ.get('OPENAI_API_KEY')}")
    
    # model
    llm = ChatOpenAI(temperature=0.9)

    product = "Queen Size Sheet Set"

    # prompt = ChatPromptTemplate.from_template(
    # "What is the best name to describe \
    # a company that makes {product}?"
    # )
    
    
    # chain = LLMChain(llm=llm, prompt=prompt)
    
    # print(chain.run(product))
    
    
    ### SimpleSequentialChain
    
    # prompt template 1
    first_prompt = ChatPromptTemplate.from_template(
        "What is the best name to describe a company that makes {product}?"
    )

    # Chain 1
    chain_one = LLMChain(llm=llm, prompt=first_prompt)
    
    # prompt template 2
    second_prompt = ChatPromptTemplate.from_template(
        "Write a 20 words description for the following company: {company_name}"
    )
    # chain 2
    chain_two = LLMChain(llm=llm, prompt=second_prompt)
    
    overall_simple_chain = SimpleSequentialChain(chains=[chain_one, chain_two], verbose=True)
    # print(overall_simple_chain.run(product))

    ### SequentialChain
    
    # prompt a: translate to english
    prompt_a = ChatPromptTemplate.from_template(
        "Translate the following review to english: \n\n{Review}"
    )
    # chain a: input= Review and output= English_Review
    chain_a = LLMChain(llm=llm, prompt=prompt_a, output_key="English_Review")

    prompt_b = ChatPromptTemplate.from_template(
        "Can you summarize the following review in 1 sentence: \n\n{English_Review}"
    )
    
    # chain b: input= English_Review and output= summary
    chain_b = LLMChain(llm=llm, prompt=prompt_b, output_key="summary")
    
    
    prompt_c = ChatPromptTemplate.from_template(
        "What language is the following review: \n\n{Review}"
    )
    
    # chain c: input= Review and output= language
    chain_c = LLMChain(llm=llm, prompt=prompt_c, output_key="language")
    
    prompt_d = ChatPromptTemplate.from_template(
        "Write a follow up response to the following "
        "summary in the specified language:"
        "\n\nSummary: {summary}\n\nLanguage: {language}"
    )
    # chain d: input= summary, language and output= followup_message
    chain_d = LLMChain(llm=llm, prompt=prompt_d, output_key="followup_message")
    
    # overall_chain: input= Review and output= English_Review, summary, followup_message
    overall__seq_chain = SequentialChain(
        chains=[chain_a, chain_b, chain_c, chain_d],
        input_variables=["Review"],
        output_variables=["English_Review", "summary", "followup_message"],
        verbose=True
    )
    
    review = "J trouve le goût médiocre. La mousse ne tiens, c'est bizarre. \
        J'achete les mêmes dans le commerce et le goût est bien meilleur ... \n Vieux lot \
            ou cnontrefaçon ?!"
            
    # print(overall__seq_chain(review))
    
    
    ### LLMRouterChain
    print(multi_chain("What is the square root of 2 ?")) # math
    print(multi_chain("What is black body radiation?")) # physics
    print(multi_chain("Why does every cell in our body contain DNA?"))
    
    


    
    