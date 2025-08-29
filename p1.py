import os
import bs4
from langchain_community.document_loaders import WebBaseLoader
import openai
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain import hub
from langchain_community.vectorstores import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from SmartAITool.core import *
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter

#NOTE remove api before pull
API_KEY = ''


loader = WebBaseLoader(
    web_paths=("https://kbourne.github.io/chapter1.html",),
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(
        class_=("post-content","post-title","post-header")
    ))
)

docs = loader.load()

# 1) Safe pre-split to avoid >8191-token chunks BEFORE any semantic step
pre_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,   # lower if you still hit limits (e.g., 1500)
    chunk_overlap=200
)
docs_small = pre_splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",         
    api_key=API_KEY,
    openai_api_base="https://api.gilas.io/v1/"
)

sem_chunker = SemanticChunker(embeddings)
splits = []
for d in docs_small:
    # Each d is small now, so the per-"sentence" embedding won't blow token limits
    splits.extend(sem_chunker.split_documents([d]))

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings 
)

retriever = vectorstore.as_retriever()

prompt = hub.pull("jclemens24/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


llm = ChatOpenAI(model_name = 'gpt-4o-mini', api_key=API_KEY, openai_api_base= "https://api.gilas.io/v1/" )


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print(rag_chain.invoke("What are the Advantages of using RAG?"))
