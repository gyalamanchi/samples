import chromadb

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from chromadb.utils import embedding_functions

#llm = OpenAI()


# load the document and split it into chunks
loader = TextLoader("/opt/app/workspace/ai_projects/revature/chromatest/sotu.txt")
documents = loader.load()

# split it into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=25, chunk_overlap=0)
doccument_chunks = text_splitter.split_documents(documents)

# create the open-source embedding function
embedding_function= embedding_functions.SentenceTransformerEmbeddingFunction(model_name="/opt/app/workspace/ai_projects/revature/all-MiniLM-L6-v2")

# load it into Chroma
#client = chromadb.HttpClient(host="localhost", port=8000)
#embedding_function = OpenAIEmbeddings()
#db = Chroma(client=client, collection_name="my_collection", embedding_function=embedding_function)
chroma_client = chromadb.PersistentClient(path="/Users/goldieyalamanchi/workspace/chroma")
collection = chroma_client.get_or_create_collection(name="sotu", embedding_function=embedding_function)
for count, chunk in enumerate(doccument_chunks):
    #print(chunk)
    collection.add(
        documents=chunk.page_content,
        metadatas=[{"source":chunk.metadata['source']}],
        ids=[chunk.metadata['source'] + str(count)]        
    )

results = collection.query(
    query_texts=["Tell me about Putin"],
    n_results=2
)
print(results)

#print(collection.peek())
