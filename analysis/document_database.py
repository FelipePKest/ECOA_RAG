from typing import List
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import os
from database import Database
from dotenv import load_dotenv
load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")

class DocumentDatabase(Database):
  
    def format_docs(self, docs: List[Document]):
        return "\n\n".join(doc.page_content for doc in docs)
    

    def _initialize(self, load=True, file_path="data/pdfs", text_splitter=None, loader=None):
        if os.path.exists("./chroma_db") and load:
        
            self.vectorstore = Chroma(
                persist_directory="./chroma_db",
                embedding_function=OpenAIEmbeddings()
            )
        else:

            print("Loading documents from PDFs")
            documents_paths = []
            splits = []
            
            # load all pdfs in the directory and subdirectories
            for root, dirs, files in os.walk(file_path):
                for file in files:
                    if file.endswith(".pdf"):
                        file_path = os.path.join(root, file)
                        documents_paths.append(file_path)

            print(f"Found {len(documents_paths)} PDFs")
            i = 0
            for document_path in documents_paths:
                print(f"Processing document {i+1}/{len(documents_paths)}")
                loader = PDFPlumberLoader(file_path=document_path)
                docs = loader.load()

                if text_splitter is None:
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

                splits.extend(text_splitter.split_documents(docs))

                i += 1
            self.vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(), persist_directory="./chroma_db")            
    
    def _setup_rag(self, *args, **kwargs):
        retriever = self.vectorstore.as_retriever()
        prompt = hub.pull("rlm/rag-prompt")
        llm = ChatOpenAI(model_name="gpt-4o-mini", api_key=openai_key)

        rag_chain_from_docs = ( RunnablePassthrough.assign(
            context=(lambda x: self.format_docs(x["context"])))
            | prompt
            | llm
            | StrOutputParser()
        )

        rag_chain_with_source = RunnableParallel(
            {"context": retriever, "question": RunnablePassthrough()}
        ).assign(answer=rag_chain_from_docs)

        return rag_chain_with_source

    def ask_rag(self, query,debug=False, *args):
        rag_chain = self._setup_rag()
        llm = ChatOpenAI(model_name="gpt-4o-mini", api_key=openai_key)
        if debug:
            responses = {"query": query, "llm": "LLM ANSWER", "rag": "RAG ANSWER"}
            return responses
        # llm.invoke(query).content - not calling llm
        responses = {"query": query, "llm": "", "rag": rag_chain.invoke(query)}
        return responses
