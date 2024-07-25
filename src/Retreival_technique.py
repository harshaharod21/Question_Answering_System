from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from pymilvus.model.reranker import CrossEncoderRerankFunction
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


def create_vectorstore(all_texts):
    """Creating vectorstore for milvus"""

    embedding_milvus = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Creating Index

    vectorstore = Milvus.from_texts(  
        texts=all_texts,
        embedding= embedding_milvus,
        connection_args={
            "uri": "http://localhost:19530",
        },
        drop_old=True,  # Drop the old Milvus collection if it exists
    )

    # Convert the vector store to a retriever
    retriever = vectorstore.as_retriever()
    return retriever


def ensemble_retrieval_method(all_texts,retriever):

    bm25_retriever = BM25Retriever.from_texts(
        all_texts)
    bm25_retriever.k = 2  #parameter to change

    # initialize the ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, retriever], weights=[0.5, 0.5]
    )

    return ensemble_retriever



def Reranking_algo(ensemble_retriever):

    """
    - query intialisation
    - Re-ranking algorithm cross encoder(open source)
    
    """

    query="What is catterpillar?"

    doc_rel= ensemble_retriever.get_relevant_documents(query)

    doc_texts = [doc.page_content for doc in doc_rel]
    doc_metadata = [doc.metadata for doc in doc_rel]


    # Define the rerank function
    

    model_rerank = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    compressor = CrossEncoderReranker(model=model_rerank, top_n=3)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=ensemble_retriever
    )

    compressed_docs = compression_retriever.invoke(query)

    return compression_retriever








