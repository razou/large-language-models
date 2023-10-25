from typing import List
import langchain
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.evaluation.qa import (
    QAEvalChain, 
    QAGenerateChain
)


from utils.setup import PrepareEnv


def example_generation(data: List["Document"], llm: ChatOpenAI , n: int = 5) -> list:
    """Generate question and answers from documents that can be used for evalution

    Args:
        data (List[&quot;Document&quot;]): Docments
        llm (ChatOpenAI): LLM model
        n (int, optional): Number of example to generate. Defaults to 5.

    Returns:
        Sequence: Sequence of examples where each element is a dict with two items:
            {'query': '<QUERY>', 'answer': '<ANSWER>'}
    """
    example_gen_chain = QAGenerateChain.from_llm(llm=llm)
    new_examples = example_gen_chain.apply_and_parse([{"doc": t} for t in data[:n]])
    return new_examples


def createQandAApp(loaders: List["BaseLoader"]):
    index = VectorstoreIndexCreator(
        vectorstore_cls=DocArrayInMemorySearch
    ).from_loaders(loaders=loaders)
    
    # Create QA Retrieval App    
    qa = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=index.vectorstore.as_retriever(), 
        verbose=True,
        chain_type_kwargs = {
            "document_separator": "<<<<>>>>>"
        }
    )
    return qa
    

if __name__ == "__main__":
    
    
    PrepareEnv.set_api_key()
    

    csv_file = 'llm-models/data/OutdoorClothingCatalog_1000.csv'
    csv_loader = CSVLoader(file_path=csv_file, encoding='utf8')
    data = csv_loader.load()
    # print(data[10])

    # BaseLoaders
    loaders = [csv_loader]

    # LLM Model
    llm = ChatOpenAI(temperature=0)

    ## QandA Application
    qa = createQandAApp(loaders=loaders)
    
    # ### Hard-coded examples
    examples = [
        {
            "query": "Do the Cozy Comfort Pullover Set have side pockets?",
            "answer": "Yes"
        },
        {
            "query": "What collection is the Ultra-Lofty 850 Stretch Down Hooded Jacket from?",
            "answer": "The DownTek collection"
        }
    ]

    # # ### LLM-Generated examples
    # llm_based_examples = example_generation(data=data, llm=llm, n=5)
    # print("Example of generated example: ")
    # print(llm_based_examples[0])
    # print("Compare with original document: ")
    # print(data[0])

    # # ### Combine examples
    # examples += llm_based_examples
    
  

    # # # ## Manual Evaluation
    # response = qa.run(examples[0]["query"])
    # print("response: ", response)
    
    # langchain.debug = True
    # qa.run(examples[0]["query"])
    # # Turn off the debug mode
    # langchain.debug = False


    # ## LLM assisted evaluation (QAEvalChain)
    predictions = qa.apply(examples)
    print("predictions sample: ", predictions[:3])
    eval_chain = QAEvalChain.from_llm(llm=llm)
    graded_outputs = eval_chain.evaluate(examples, predictions)
    print("graded_outputs sample: ", graded_outputs[:3])

    for i, _ in enumerate(examples):
        print(f"Example {i}:")
        print("Question: " + predictions[i]['query'])
        print("Real Answer: " + predictions[i]['answer'])
        print("Predicted Answer: " + predictions[i]['result'])
        print("Predicted Grade: " + graded_outputs[i]['text'])
        print()
