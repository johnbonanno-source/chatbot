from glob import glob
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from langchain_chroma import Chroma



embeddings = GoogleGenerativeAIEmbeddings(
    model="text-embedding-004",
    google_api_key=os.environ["GEMINI_API_KEY"],
)

pdf_paths = glob("./context/*.pdf")
if not pdf_paths: 
    raise FileNotFoundError("No PDF files found in ./context")

# Accumulate pages from every PDF in every PDF path.
all_pages = []  
for path in pdf_paths:  
    print(f"Loading {path}...") 
    if not os.path.exists(path):
        raise FileNotFoundError(f"PDF file not found: {path}") 
    try: 
        loader = PyPDFLoader(path)
        pages = loader.load()  
        print(f"PDF has been loaded and has {len(pages)} pages")
        all_pages.extend(pages) 
    except Exception as e: 
        print(f"Error loading PDF: {e}") 
        raise 

 # Split documents into overlapping chunks.
text_splitter = RecursiveCharacterTextSplitter( 
    chunk_size=1000,
    chunk_overlap=200
)

# Chunk the combined PDF pages.
pages_split = text_splitter.split_documents(all_pages)  
persist_directory = str(Path(__file__).resolve().parent / "Agents")  # Chroma persistence directory.
collection_name = "stock_market"  # Chroma collection name.

# Ensure the persistence directory exists.
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

try: 
    # Build the chroma vector store of PDF chunks using the embedding model
    vectorstore = Chroma.from_documents(
        documents=pages_split, 
        embedding=embeddings,
        persist_directory=persist_directory, 
        collection_name=collection_name,
    )
    vectorstore.persist() 
    print(f"Created ChromaDB vector store!")

except Exception as e:
    print(f"Error setting up ChromaDB: {str(e)}")
    raise

# Now we create our retriever 
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5} # K is the amount of chunks to return
)

def match_query_with_data(query, retriever):
    docs = retriever.getRelevantDocuments(query)
    if not docs:
        return "I found no relevant information in the pdf Context related to your query."
    
    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")
    
    return "\n\n".join(results)