#using the chromadb vector database - getting fimiliar with chromadb

import chromadb

client = chromadb.Client()

#collections is like a table where the records can be inserted
collection = client.create_collection(name = 'my_collection')

#to add documents
collection.add(
    documents=[
        'this document is about New York',
        'this document is about Delhi',
        'this document is about Japan'
    ],
    ids = [
        'id1', 'id2', 'id3'
    ]
)
#getting all the documents
all_docs = collection.get()
#print(all_docs)

#the individual documents can be obtained by getting an individual id

#chromadb uses semnatic search to tie a query to the vector database's documents based on 
#the distance (how similar is the documents to query)
results = collection.query(
    query_texts = ['query is about chole bhature'],
    n_results = 3
)
#print(results)

#other queries to try - 
#query is about sushi
#query is about brooklyn bridge
#query is about big apple
#query is about 

#to delete docs
collection.delete(ids=all_docs['ids'])
collection.get()