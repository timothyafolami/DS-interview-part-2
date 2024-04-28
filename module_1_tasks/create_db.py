import os
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')

# Text file directory
txt_file = './data/unique_descriptions.txt'

# Loading the text file using the TextLoader
loader =  TextLoader(txt_file)

# Documents
documents = loader.load()
# Splitting the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
# Splitting the documents into chunks
docs = text_splitter.split_documents(documents)

# Creating the embeddings
embeddings = OpenAIEmbeddings()
# Index name
index_name = "productrec"
# Creating the Pinecone vector store
docsearch = PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)