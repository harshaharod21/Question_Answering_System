# Question_Answering_System
This is a question answering system code implementation.

# Question Answering System with LLM, Milvus, and Raptor Indexing

## Prerequisites

Before running the project, ensure you have the following prerequisites installed:

1. **Docker Desktop**: [Download Docker Desktop](https://www.docker.com/products/docker-desktop) and install it on your system.
2. **Python 3.12.4**: Ensure you have Python 3.x installed on your system.
3. **pip**: Ensure you have pip installed to manage Python packages.

## Setup

1. Clone the repository:
    ```sh
    git clone https://github.com/your-repo/question-answering-system.git
    cd question-answering-system
    ```

2. Install the required Python libraries:
    ```sh
    pip install -r requirements.txt
    ```

3. Download Docker Desktop and ensure it is running.

4.Download the script from Milvus documentation :https://milvus.io/docs/install_standalone-docker-compose.md
- Download the script to the directory of your project (provided in Milvus documentation), note :go to docker compose
- Open your terminal
- Now run docker compose up -d  (not to include sudo)
- These will create and start the container

    ```sh
    docker-compose up -d
    ```

## Components

### 1. Data Ingestion
This component loads data from PDF files. It reads the content of the PDFs and splits them into manageable chunks for further processing.

**Link to Books PDF to download:**
- [Book 1] Kafka on the Shore - Haruki Murakami [Worldfreebooks.com].pdf
- [Book 2] The Invisible Life of Addie LaRue By V E Schwab.pdf
- [Book 3] (link-to-pdf-3)

### 2. Text Preprocessing and Creating Vector Embedding
This component preprocesses the ingested text data and creates vector embeddings. It involves tasks such as tokenization, removing stop words, and generating embeddings using models like SentenceTransformers.

### 3. Raptor Indexing and Implementing Collapsed Tree Retrieval
This component performs Raptor indexing on the text data to create hierarchical clusters. It implements collapsed tree retrieval for efficient search and summarization.

### 4. Creating Milvus Database
This component sets up the Milvus vector database to store and manage the text embeddings. It involves creating a connection, defining schemas, and inserting data into the database.

**Steps:**
1. Install Docker Desktop.
2. Download the script from the Milvus documentation and place it in your project directory.
3. Open your terminal and navigate to the project directory.
4. Run the following command to create and start the Milvus container:
    ```sh
    docker-compose up -d
    ```

### 5. Retrieve Techniques and Reranking
Objective: To create a hybrid search that combines sparse retrievers with dense retrievers.
- **Creating Index**: Index the text embeddings stored in Milvus.
- **Creating Vector Store and Retriever**: Create a vector store and a retriever using the indexed data.
- **Using Ensemble Retriever**: Utilize the `ensemble_retriever` from Langchain to perform retrieval combining BM25 and the created retriever.
- **Using Reranking Algorithm**: Apply a reranking algorithm like CrossEncoder to refine the retrieved documents.

### 6. LLM for Question Answering
This component uses a Language Model (LLM) to generate answers to user queries based on the context provided by the retrieved documents. It involves constructing prompts and invoking the LLM to generate accurate and relevant responses.

## Running the Pipeline

To run all the components in order, execute the `pipeline.py` script:
```sh
python pipeline.py

