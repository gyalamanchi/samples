from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI

llm = OpenAI()


# load the document and split it into chunks
loader = TextLoader("./sotu.txt")
documents = loader.load()

# split it into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# load it into Chroma
#client = chromadb.HttpClient(host="localhost", port=8000)
#embedding_function = OpenAIEmbeddings()
#db = Chroma(client=client, collection_name="my_collection", embedding_function=embedding_function)
db = Chroma.from_documents(docs, embedding_function)
# query it
#query = "What did the president say about Ketanji Brown Jackson"
#docs = db.similarity_search(query)

# print results
#print(docs[0].page_content)

template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, try to look at the extra text, and if still impossible to find a good answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)
#...similar code...

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

"""
llm = LlamaCpp(
    #model_path="/opt/app/workspace/ai_projects/revature/llama-2-7b-chat.Q3_K_M.gguf",
    #model_path="/opt/app/workspace/ai_projects/revature/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    #model_path="/opt/app/workspace/ai_projects/revature/zephyr-7b-beta.Q5_K_M.gguf",
    model_path="/opt/app/workspace/ai_projects/revature/ggml-model-q4_k_m.gguf",
    temperature=0.75,
    max_tokens=2000,
    top_p=1,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)
"""

qa_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=db.as_retriever(),
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

result = qa_chain({"query": "What did the president say about Ketanji Brown Jackson"})

print(result)
