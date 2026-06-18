import chromadb
import os

# 1. Define your path explicitly to rule out relative pathing errors
# Update this to exactly where you believe the DB is stored
DB_PATH = os.path.abspath(os.path.join(os.getcwd(),"./gemma_chroma_db/")) 

try:
    print(f"Attempting to connect to Chroma at: {DB_PATH}")
    # Use PersistentClient for local directories
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # 2. List all collections to see if the DB is empty or populated
    collections = client.list_collections()
    
    if not collections:
        print("STATUS: Connected, but NO collections found. The DB is empty.")
    else:
        print(f"STATUS: Success! Found {len(collections)} collection(s):")
        for c in collections:
            print(f" - {c.name}")
            
        # 3. Inspect the first collection to view its "schema" and data
        target_collection_name = collections[0].name
        collection = client.get_collection(name=target_collection_name)
        
        print(f"\n--- Inspecting Collection: '{target_collection_name}' ---")
        print(f"Total documents inside: {collection.count()}")
        
        if collection.count() > 0:
            # peek() grabs the first few rows without needing to do a vector search
            sample = collection.peek(limit=1)
            
            print("\n--- Sample 'Schema' (Data Structure) ---")
            print(f"ID Structure:      {sample['ids']}")
            print(f"Metadata Keys:     {list(sample['metadatas'][0].keys()) if sample['metadatas'][0] else 'No metadata'}")
            print(f"Metadata Sample:   {sample['metadatas']}")
            print(f"Document Sample:   {sample['documents']}")
            # Note: sample['embeddings'] is also available, but usually too messy to print

except Exception as e:
    print(f"\nCONNECTION FAILED: {str(e)}")