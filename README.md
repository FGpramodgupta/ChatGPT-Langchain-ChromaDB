# chroma-langchain

A repository to highlight examples of using the Chroma (vector database) with LangChain (framework for developing LLM applications).

## Document Question-Answering

For an example of using Chroma+LangChain to do question answering over documents, see [this notebook](qa.ipynb).
To use a persistent database with Chroma and Langchain, see [this notebook](qa_persistent.ipynb).


Installation
We start off by installing the required packages.


COPY
!pip install chromadb -q
!pip install sentence-transformers -q
For our demonstration, we use a set of text files stored in a folder named "pets". Each file contains information about a different aspect of pet care.

Next, we need to connect to ChromaDB and create a collection. By default, ChromaDB uses the Sentence Transformers all-MiniLM-L6-v2 model to create embeddings.


COPY
import chromadb

client = chromadb.Client()
collection = client.create_collection("yt_demo")
Adding Documents
We add some documents to our collection, along with corresponding metadata and unique IDs.


COPY
collection.add(
    documents=["This is a document about cat", "This is a document about car"],
    metadatas=[{"category": "animal"}, {"category": "vehicle"}],
    ids=["id1", "id2"]
)
Querying
Now, we can query our collection. Let's search for the term "vehicle". The returned result should be the document about the car.


COPY
results = collection.query(
    query_texts=["vehicle"],
    n_results=1
)
print(results)

COPY
{'ids': [['id2']],
 'embeddings': None,
 'documents': [['This is a document about car']],
 'metadatas': [[{'category': 'vehicle'}]],
 'distances': [[0.8069301247596741]]}
Reading Files from a Folder
Our output is as expected, providing the id, document content, metadata, and distance value for the best-matching document.

Now, let's add our pet documents to the collection. We start by reading all the text files from the "pets" folder and storing the data in a list.


COPY
import os

def read_files_from_folder(folder_path):
    file_data = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            with open(os.path.join(folder_path, file_name), 'r') as file:
                content = file.read()
                file_data.append({"file_name": file_name, "content": content})

    return file_data

folder_path = "pets"
file_data = read_files_from_folder(folder_path)
Adding File Contents to ChromaDB
Then, we create separate lists for documents, metadata, and ids, which we add to our collection.


COPY
documents = []
metadatas = []
ids = []

for index, data in enumerate(file_data):
    documents.append(data['content'])
    metadatas.append({'source': data['file_name']})
    ids.append(str(index + 1))

pet_collection = client.create_collection("pet_collection")

pet_collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)
Performing Semantic Searches
Let's now query the collection for the different kinds of pets people commonly own.


COPY
results = pet_collection.query(
    query_texts=["What are the different kinds of pets people commonly own?"],
    n_results=1
)
print(results)

COPY
{'ids': [['1']],
 'embeddings': None,
 'documents': [['Pet animals come in all shapes and sizes, each suited to different lifestyles and home environments. Dogs and cats are the most common, known for their companionship and unique personalities. Small mammals like hamsters, guinea pigs, and rabbits are often chosen for their low maintenance needs. Birds offer beauty and song, and reptiles like turtles and lizards can make intriguing pets. Even fish, with their calming presence, can be wonderful pets.']],
 'metadatas': [[{'source': 'Different Types of Pet Animals.txt'}]],
 'distances': [[0.7325009703636169]]}
Our query successfully retrieves the most relevant document, which talks about different types of pet animals.

Filtering Results
If you want to refine your search further, you can use the where_document parameter to specify a condition that must be met in the document text. For example, if you want to find documents about the emotional benefits of owning a pet that mention reptiles, you could use the following query:


COPY
pet_collection.query(
    query_texts=["What are the emotional benefits of owning a pet?"],
    n_results=1,
    where_document={"$contains":"reptiles"}
)
print(results)
The results show that the document talking about the emotional bond between humans and pets is the most relevant to our query.

Similarly, if you want to use metadata to filter your search results, you can use the where parameter. Let's say you want to find information about the emotional benefits of owning a pet, but you want to retrieve this information specifically from the document about pet training and behaviour. You could do so with the following query:


COPY
results = pet_collection.query(
    query_texts=["What are the emotional benefits of owning a pet?"],
    n_results=1,
    where={"source": "Training and Behaviour of Pets.txt"}
)
print(results)
The results now show the document about the training and behaviour of pets, as we specified in our query.

Using a different model for embedding
While ChromaDB uses the Sentence Transformers all-MiniLM-L6-v2 model by default, you can use any other model for creating embeddings. In this example, we use the 'paraphrase-MiniLM-L3-v2' model from Sentence Transformers.

First, we load the model and create embeddings for our documents.


COPY
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

documents = []
embeddings = []
metadatas = []
ids = []

for index, data in enumerate(file_data):
    documents.append(data['content'])
    embedding = model.encode(data['content']).tolist()
    embeddings.append(embedding)
    metadatas.append({'source': data['file_name']})
    ids.append(str(index + 1))
Then, we create a new collection and add the documents, embeddings, metadata, and ids to it.


COPY
pet_collection_emb = client.create_collection("pet_collection_emb")

pet_collection_emb.add(
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas,
    ids=ids
)
Now, when we perform a query, we need to provide the embedding of the query text instead of the text itself. Let's search again for the different kinds of pets people commonly own.


COPY
query = "What are the different kinds of pets people commonly own?"
input_em = model.encode(query).tolist()

results = pet_collection_emb.query(
    query_embeddings=[input_em],
    n_results=1
)
print(results)
The results are similar to our previous query, with the same document about different types of pet animals being returned.

Finally, let's make a more specific query about what foods are recommended for dogs.


COPY
query = "foods that are recommended for dogs?"
input_em = model.encode(query).tolist()

results = pet_collection_emb.query(
    query_embeddings=[input_em],
    n_results=1
)
print(results)
The result correctly provides the document about the nutrition needs of pet animals.

Conclusion
ChromaDB is a powerful tool that allows us to handle and search through data in a semantically meaningful way. It provides flexibility in terms of the transformer models used to create embeddings and offers efficient ways to narrow down search results. Whether you're managing a small collection of documents or a large database, ChromaDB's ability to handle semantic search can help you find the most relevant information quickly and accurately.