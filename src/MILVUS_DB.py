#creating MILVUS Database
from pymilvus import connections, db
from pymilvus import FieldSchema, CollectionSchema, DataType,Collection

def setting_conn():
    """Setting up connection and creating schema for the collection"""

    conn = connections.connect(host="127.0.0.1", port=19530)

    database = db.create_database("my_database_11")

    db.using_database("my_database_11")


    #Creating Milvus schema

    #Define schema for the collection


    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="text_all", dtype=DataType.VARCHAR, max_length=5000),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)

    ]

    schema = CollectionSchema(fields, description="Text embeddings collection")

    # Create the collection
    text_collection = Collection(name="text_embeddings", schema=schema)
    return text_collection


def MILVUS_data_ingestion(text_collection,all_texts,results):
    #prepare data for ingestion

    text_embeddings=[]

    for level in sorted(results.keys()):
        # Extract summaries from the current level's DataFrame
        embeddings_text = results[level][0]["embd"].tolist()
        text_embeddings.extend(embeddings_text)
        

    def prepare_data(all_texts,text_embeddings):
        ids= list(range(len(all_texts)))
        return ids, all_texts,text_embeddings

    data_for_insertion= prepare_data(all_texts,text_embeddings)

    return data_for_insertion

    

def insert_data(collection, data):
        """
        insert data to milvus
        Data is in tuple,it has id,summaries and embeddings as separate lists in it
        """
        ids, all_texts, embeddings = data
        entities = [
            ids,
            all_texts,
            embeddings
        ]
        collection.insert(entities)
        collection.flush()







