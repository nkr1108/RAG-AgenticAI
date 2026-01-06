# CROMA DB SIMILARITY SEARCH DEMO CODE

# STEP1 - IMPORTING THE NECESSARY MODULES 
# Importing the necessary modules from the chromadb package:
# chromadb is used to interact with the Chroma DB database,
import chromadb

# embedding_functions is used to define the embedding model
from chromadb.utils import embedding_functions

# STEP2 - DEFINE THE EMBEDDING FUNCTION
# Define the embedding function using SentenceTransformers
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# STEP3 - CREATE A NEW INSTANCE OF CHROMACLIENT TO INTERACT WITH THE CHROMA DB
client = chromadb.Client()

# STEP4 - DEFINE THE NAME FOR THE COLLECTION TO BE CREATED OR RETRIEVED
collection_name = "my_grocery_collection"

# STEP10 - FUNCTION TO PERFORM A SIMILARITY SEARCH IN THE COLLECTION
def perform_similarity_search(collection, all_items):
    try:
        # Place your similarity search code inside this block
        # Define the query term you want to search for in the collection
        # query_term = "apple"
        query_term = "beef"

        # Perform a query to search for the most similar documents to the 'query_term'
        results = collection.query(
            query_texts=[query_term],
            n_results=3  # Retrieve top 3 results
        )
        print(f"\nQuery results for '{query_term}':")
        print(results)

        # Check if no results are returned or if the results array is empty
        if not results or not results['ids'] or len(results['ids'][0]) == 0:
            # Log a message indicating that no similar documents were found for the query term
            print(f'No documents found similar to "{query_term}"')
            return

        print(f'\nTop 3 similar documents to "{query_term}":')
        
        # Access the nested arrays in 'results["ids"]' and 'results["distances"]'
        for i in range(min(3, len(results['ids'][0]))):
            doc_id = results['ids'][0][i]  # Get ID from 'ids' array
            score = results['distances'][0][i]  # Get score from 'distances' array
            # Retrieve text data from the results
            text = results['documents'][0][i]
            if not text:
                print(f' - ID: {doc_id}, Text: "Text not available", Score: {score:.4f}')
            else:
                print(f' - ID: {doc_id}, Text: "{text}", Score: {score:.4f}')

    except Exception as error:
        print(f"Error in similarity search: {error}")

# Define the main function to interact with the Chroma DB
def main():
    try:
        # Create a collection in the Chroma database with a specified name, 
        # distance metric, and embedding function. In this case, we are using 
        # cosine distance
        print(f"Inside Main")
        
        # STEP5 - CREATE A COLLECTION IN THE CHROMA DATABASE WITH A SPECIFIED NAME, 
        # DISTANCE METRIC, AND EMBEDDING FUNCTION. IN THIS CASE, WE ARE USING 
        # COSINE DISTANCE
        
        collection = client.create_collection(
            name=collection_name,
            metadata={"description": "A collection for storing grocery data"},
            configuration={
                "hnsw": {"space": "cosine"},
                "embedding_function": ef
            }
        )
        print(f"Collection created: {collection.name}")

        # STEP6 - GET THE INPUT DATA - ARRAY OF GROCERY-RELATED TEXT ITEMS
        texts = [
            'fresh red apples',
            'organic bananas',
            'ripe mangoes',
            'whole wheat bread',
            'farm-fresh eggs',
            'natural yogurt',
            'frozen vegetables',
            'grass-fed beef',
            'free-range chicken',
            'fresh salmon fillet',
            'aromatic coffee beans',
            'pure honey',
            'golden apple',
            'red fruit'
        ]

        # STEP7 - CREATE A LIST OF UNIQUE IDS FOR EACH TEXT ITEM IN THE 'TEXTS' ARRAY
        # EACH ID FollowS THE FORMAT 'FOOD_<INDEX>', WHERE <INDEX> STARTS FROM 1
        ids = [f"food_{index + 1}" for index, _ in enumerate(texts)]

        # STEP8 - ADD DOCUMENTS AND THEIR CORRESPONDING IDS TO THE COLLECTION
        # THE `ADD` METHOD INSERTS THE DATA INTO THE COLLECTION
        # THE DOCUMENTS ARE THE ACTUAL TEXT ITEMS, AND THE IDS ARE UNIQUE IDENTIFIERS
        # CHROMA DB WILL AUTOMATICALLY GENERATE EMBEDDINGS USING THE CONFIGURED EMBEDDING FUNCTION
        collection.add(
            documents=texts,
            metadatas=[{"source": "grocery_store", "category": "food"} for _ in texts],
            ids=ids
        )

        # STEP9 - RETRIEVE ALL THE ITEMS (DOCUMENTS) STORED IN THE COLLECTION
        # THE `GET` METHOD FETCHES ALL DATA FROM THE COLLECTION
        all_items = collection.get()
        # LOG THE RETRIEVED ITEMS TO THE CONSOLE FOR INSPECTION
        # THIS WILL PRINT OUT ALL THE DOCUMENTS, IDS, AND METADATA STORED IN THE COLLECTION
        print("\nCollection contents:")
        print(f"\nNumber of documents: {len(all_items['documents'])}")

        # STEP9 - PERFORM SIMILARITY SEARCH IN THE COLLECTION
        perform_similarity_search(collection, all_items)


    except Exception as error:  # Catch any errors and log them to the console
        print(f"Error: {error}")

if __name__ == "__main__":
    main()