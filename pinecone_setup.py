from pinecone import Pinecone, ServerlessSpec
import os
import json

def initialize_pinecone():
    return Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

def create_index(pc):
    pc.create_index(
    name=os.getenv('PINECONE_INDEX_NAME'),
    dimension=384, # Replace with your model dimensions
    metric="cosine", # Replace with your model metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
    )

def upsert_data(pc, data):
    index = pc.Index(os.getenv('PINECONE_INDEX_NAME'))
    for idx, faculty in enumerate(data):
        embedding = faculty.get("Research Embedding")
        if embedding is None:
            print(f"Missing embedding for faculty with index {idx} and name {faculty.get('Name')}")
            continue

        metadata = {
            "name": faculty.get("Name"),
            "title": faculty.get("Title"),
            "url": faculty.get("url"),
            "researchSummary": faculty.get("Research Summary"),
        }

        try:
            index.upsert(vectors=[
                {
                    "id": str(idx),
                    "values": embedding,
                    "metadata": metadata
                }
            ])
            print(f"Upserted vector {idx} into Pinecone.")
        except Exception as e:
            print(f"Error upserting vector {idx}: {e}")

def main():
    print("Initializing Pinecone client...")
    pc = initialize_pinecone()

    if not pc.has_index(os.getenv('PINECONE_INDEX_NAME')):
        print("Creating Pinecone index...")
        create_index(pc)

    print("Upserting data...")
    with open('faculty_data.json', 'r') as json_file:
        faculty_data = json.load(json_file)
    upsert_data(pc, faculty_data)

if __name__ == "__main__":
    main()
