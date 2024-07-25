from src.data_ingestion import load_data
from src.Text_preprocessing import preprocess_text
from src.Raptor_indexing import recursive_embed_cluster_summarize, collapsed_tree_retrieval,LLM
from src.MILVUS_DB import setting_conn, MILVUS_data_ingestion, insert_data
from src.Retreival_technique import create_vectorstore,Reranking_algo,ensemble_retrieval_method
from src.Q_A import question_answer_system
import os



def main():
    # Step 1: Data Ingestion
    data = load_data()
    
    # Step 2: Text Preprocessing and Creating Vector Embeddings
    preprocessed_text = preprocess_text(data)
        
    # Step 3: RAPTOR Indexing and Implementing Collapsed Tree Retrieval

    model_llm=LLM()
    results = recursive_embed_cluster_summarize(preprocessed_text, level=1, n_levels=3)
    
    raptor_results = collapsed_tree_retrieval(preprocessed_text,results)
    
    # Step 4: Creating MILVUS Database

    text_collection= setting_conn()
    insertion_data=MILVUS_data_ingestion(text_collection,raptor_results,results)
    insert_data(text_collection,insertion_data)

    
    # Step 5: Retrieve Techniques and Reranking
    retriever= create_vectorstore(raptor_results)
    ensemble_retriever=  ensemble_retrieval_method(raptor_results,retriever)
    reranked_results = Reranking_algo(ensemble_retriever)
    
    # Step 6: LLM for Question Answering

         
    question = "Who was Kafka?"
    answer=question_answer_system(reranked_results,question,model_llm)
    print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
