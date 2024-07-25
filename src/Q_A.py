from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import SentenceTransformer


#Chain invoke

def question_answer_system(ensemble_retriever,query,model):
        
    template="""
    You are an AI Assistant that can give answer for any query asked based on the documents provided to answer.
    Please be truthful and give direct answers. Please tell 'I don't know' if user query is not in CONTEXT
    \n Question: {query}
    Context: {context}

    """
    

    prompt= ChatPromptTemplate.from_template(template)
    output_parser= StrOutputParser()

    chain= (
        {"context": ensemble_retriever,"query":RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    
    response= chain.invoke(query)
    return response