from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

files = ["clinic_hours.md", "cancellation_policy.md", "scheduling_rules.md"]
docs = []

# Load docs
for f in files:
    loader = TextLoader(f)
    docs.extend(loader.load())

# Split docs into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Create embeddings and Chroma vector store
embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(chunks, embeddings, collection_name="policies_rag")

print("Chroma index created successfully!")
