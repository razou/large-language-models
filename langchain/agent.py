import langchain
from langchain.agents.agent_toolkits import create_python_agent
from langchain.agents import (
    load_tools, initialize_agent,
    AgentType,
    tool
)

from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.chat_models import ChatOpenAI
from datetime import date

from utils.setup import PrepareEnv


@tool
def time(text: str) -> str:
    """Returns todays date, use this for any questions related to knowing todays date. \
    The input should always be an empty string, and this function will always return todays \
    date - any date mathmatics should occur outside this function."""
    return str(date.today())



if __name__ == "__main__":
    
    
    prep_env = PrepareEnv
    prep_env.set_api_key()
    
    llm = ChatOpenAI(temperature=0)
    
    tools = load_tools(["llm-math","wikipedia"], llm=llm)
    
    agent= initialize_agent(
        tools=tools, 
        llm=llm, 
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose = True
    )
    
    ### Example 1 (math tool)
    # print(agent("What is the 25% of 300?"))
    
    ### Example 2 (wikipedia tool)
    # question = "Tom M. Mitchell is an American computer scientist \
    #     and the Founders University Professor at Carnegie Mellon University (CMU) what book did he write?"
    # print(agent(question))
    
    ### Example 3
    
    
    
    ### Python Agent
    pyAgent = create_python_agent(
        llm=llm,
        tool=PythonREPLTool(),
        verbose=True
    )
    
    # Example 4
    customer_list = [["Harrison", "Chase"], 
                     ["Lang", "Chain"],
                     ["Dolly", "Too"],
                     ["Elle", "Elem"], 
                     ["Geoff","Fusion"], 
                     ["Trance","Former"],
                     ["Jen","Ayai"]
                    ]
    # response = agent.run(
    #     f"""Sort these customers by \last name and then first name and print the output: {customer_list}"""
    # ) 
    # print(response)
    
    # Example 4 in debuging model
    # langchain.debug=True
    response = agent.run(
        f"""Sort these customers by \last name and then first name and print the output: {customer_list}"""
    )
    # print(response)
    # langchain.debug=False
    
    ### Using custom tool
    # Example 5
    
    agent= initialize_agent(
        tools= tools + [time], 
        llm=llm, 
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose = True
    )
    
    try:
        print(agent("whats the date today?"))
    except: 
        print("exception on external access")
    
 