from langchain_community.vectorstores import Milvus
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import GPT4AllEmbeddings

# Load the text from the downloaded file
loader = TextLoader("state_of_the_union.txt")
documents = loader.load()

# Split the text into chunks
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=20)
docs = text_splitter.split_documents(documents)

# Initialize OpenAIEmbeddings
# This will use text-embedding-ada-002 model by default
# embeddings = OpenAIEmbeddings()
embeddings = GPT4AllEmbeddings()

vector_db = Milvus.from_documents(
    docs,
    embeddings,
    collection_name="collection_1",
    connection_args={"host": "0.0.0.0", "port": "19530"},
)

print(vector_db._collection.count())
query = "What is this document about?"

docs = vector_db.similarity_search(query)

print(f"{print(docs[0].page_content)}")
