from utils.setup import PrepareEnv
import os


from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import OpenAIEmbeddings




if __name__ == "__main__":
    
    # print(os.getcwd())
    # print(os.listdir())
    file = "llm-models/data/user_profiles.csv"
    
    prep_env = PrepareEnv
    prep_env.set_api_key()
    
    csv_loader = CSVLoader(file_path=file)

    index = VectorstoreIndexCreator(vectorstore_cls=DocArrayInMemorySearch).from_loaders([csv_loader])
    
    query ="Please count how many Male and Female are in table, \
        calculate percentage of each of them and return the response a \
        summarized  table"
        
    print("index.query's response: ", index.query(query))
    
    docs = csv_loader.load()
    # print("Sample docs: ",  docs[0])
    embeddings = OpenAIEmbeddings()
    embed = embeddings.embed_query("Hi, I'm computer scientist")
    # print("len(embed): ", len(embed))
    # print("embed content's sample: ", embed[:5])
    
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    query = "Please count number of rows in  table"
    docs = db.similarity_search(query)
    # print("len(docs): ", len(docs))
    # print("Sample docs: ", docs[0])
    
    retriever = db.as_retriever()
    llm = ChatOpenAI(temperature = 0.0)
    
    print("index.query(query, llm=llm) ", index.query(query, llm=llm))

    
    # print("page_content: ", docs[0].page_content) # a doc corresponds to one row
    
    qdocs = "".join([docs[i].page_content for i in range(len(docs))]) 
    
    # response = llm.call_as_llm(f"{qdocs} Question: Please calculate the average and the median age per gender in the whole documents") 
    # print(response)
    
    
    qa_stuff = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, verbose=True)
    query_stuff =  "Please list your top five most common occupations in a table and summarize each one."
    print("qa_stuff.run(query_stuff): ", qa_stuff.run(query_stuff))

    
    # Create new index with embeddings
    index2 = VectorstoreIndexCreator(vectorstore_cls=DocArrayInMemorySearch, embedding=embeddings).from_loaders([csv_loader])
    
    print("index2.query(query, llm=llm) ", index2.query(query, llm=llm))
