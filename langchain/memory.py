import os
import openai
from dotenv import load_dotenv, find_dotenv


from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationTokenBufferMemory,
    ConversationSummaryBufferMemory
)



def set_api_key():
    _ = load_dotenv(find_dotenv())
    openai.api_key = os.environ["OPENAI_API_KEY"]


if __name__ == "__main__":
    ##0
    set_api_key()
    
    llm = ChatOpenAI(temperature=0.0)

    # # ConversationBufferMemory
    # print("*** ConversationBufferMemory ***")
    
    # memory = ConversationBufferMemory()
    # conversation = ConversationChain(llm=llm, memory = memory, verbose=False)
    
    # print(conversation.predict(input="Hi, my name is Abder and I'm 20 years old"))
    # print(conversation.predict(input="What is my age?"))
    # print(conversation.predict(input="What is my name?"))
    # print('--'*10)
    # print("memory.buffer (contains the conversation): ")
    # print(memory.buffer)
    # print('-'*10)
    
    # print("load_memory_variables: (everything that Human or AI has said)")
    # print(memory.load_memory_variables({}))
    
    # print("-"*10)
    # print("save a context with an input message of “Hi” and an output message of “What’s up”")
    
    # memory = ConversationBufferMemory()
    # memory.save_context({"input": "Hi"}, {"output": "What's up"})
    # print(memory.buffer)
    # print(memory.load_memory_variables({}))
    
    # memory.save_context({"input": "Not much, just hanging"}, {"output": "Cool"})
    # print(memory.load_memory_variables({}))

    
    # ###ConversationBufferWindowMemory
    # print("*** ConversationBufferWindowMemory ***")
    # """
    # - Allows to keep for example only most recent conversation part (parameter k)
    # """
    # print("*"*20)
    # window_memory = ConversationBufferWindowMemory(k=1)
    
    # window_memory.save_context({"input": "Hi"}, {"output": "What's up"})
    # window_memory.save_context({"input": "Not much, just hanging"}, {"output": "Cool"})
    # print(window_memory.load_memory_variables({}))
    
    # conversation2 = ConversationChain(llm=llm, memory=window_memory,verbose=False)
    # print(conversation2.predict(input="Hi, my name is Abder and 20 years old"))
    # print(conversation2.predict(input="What is my age?"))
    # print(conversation2.predict(input="What is my name?"))
    
    ### ConversationTokenBufferMemory
    # print("*** ConversationTokenBufferMemory ***")
    # for token_limit in [30, 50, 100]:
    #     token_memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=token_limit)
    #     token_memory.save_context({"input": "AI is what?!"}, {"output": "Amazing!"})
    #     token_memory.save_context({"input": "Backpropagation is what?"}, {"output": "Beautiful!"})
    #     token_memory.save_context({"input": "Chatbots are what?"}, {"output": "Charming!"})
    #     print(token_memory.load_memory_variables({}))
    #     print("\n")
    
    ### ConversationSummaryMemory
    print("*** ConversationSummaryMemory ***")
    
    
    # create a long string
schedule = "There is a meeting at 8am with your product team. \
You will need your powerpoint presentation prepared. \
9am-12pm have time to work on your LangChain \
project which will go quickly because Langchain is such a powerful tool. \
At Noon, lunch at the italian resturant with a customer who is driving \
from over an hour away to meet you to understand the latest in AI. \
Be sure to bring your laptop to show the latest LLM demo."

# for max_token_limit in [30, 60, 100]:
#     summary_memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=max_token_limit)
#     summary_memory.save_context({"input": "Hello"}, {"output": "What's up"})
#     summary_memory.save_context({"input": "Not much, just hanging"}, {"output": "Cool"})
#     summary_memory.save_context(
#         {"input": "What is on the schedule today?"}, 
#         {"output": f"{schedule}"}
#         )
#     print(summary_memory.load_memory_variables({}))
#     print("\n")

summary_memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=300)  
conversation3 = ConversationChain(llm=llm, memory = summary_memory,verbose=True)

print(conversation3.predict(input="What would be a good demo to show?"))
print('-'*10)
print(summary_memory.load_memory_variables({}))

