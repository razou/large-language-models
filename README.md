# LLMs

## Set up

- Python version: 3.8.x
- Install dependecies: `pip install requirements.txt`

- Get an OpenAI API key
  - https://platform.openai.com/account/api-keys



## Large Language Models

- How they works
  - *It is build by using supervised learning to repetidly predict the next word (token)*. Andrew NG
- Two types of LLM models
  - Base LLM
    - Predicts next word based on text train data
  - Instruction tuned LLM
    - Try to follow instructions
- From Base LLM to Instruction Tuned LLM
  1. Train a Base LLM model on a lot of data
  2. Further train the model
     - Fine-tune on a smaller set of examples where the output follows an input instruction
     - Obtain Humain-ratings of the quality of different LLM outputs, on criateri such as whether the output is helpful, honest and harmless
     - Tune the LLM to increase probability that it generates the more highly rated outputs (using
    RLHF: Reinforcement Learning from Human Feedback)

## Langchain

### Models

### Memory

- [https://python.langchain.com/en/latest/modules/memory.html](https://python.langchain.com/en/latest/modules/memory.html)
- [https://python.langchain.com/en/latest/modules/memory/how_to_guides.html](https://python.langchain.com/en/latest/modules/memory/how_to_guides.html)
- The ability models (or applications) to remember the previous part of the conversations
- Create an interactive converation
- See following langchain classes
  - `langchain.memory.ConversationBufferMemory`
  - `langchain.memory.ConversationBufferWindowMemory`
  - `ConversationTokenBufferMemory`
  - ``ConversationSummaryMemory`


### Chains

- [https://python.langchain.com/en/latest/modules/chains.html](https://python.langchain.com/en/latest/modules/chains.html)
- A chain in LangChain is made up of links, which can be either primitives like LLMs or other chains. The most core type of chain is an LLMChain, which consists of a PromptTemplate and an LLM
- A chain usually combine toghter an LLM model with a prompt

#### Sequential Chain

- Example of simple sequantial chain (source: deeplearning.ai)

![Simple Sequanial Chain](/images/simpleSequantialChain.png)

- Example of sequantial chain (source: deeplearning.ai)

![Sequantial Chain](/images/sequantialChain.png)

- Example of RouterChain (source: deeplearning.ai)
![Routre Chain](/images/routerChain.png)


## Vector Database 

- [Vector Database for Large Language Models in Production](https://www.youtube.com/watch?v=9VgpXcfJYvw&ab_channel=MLOpsLearners)

