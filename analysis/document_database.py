from typing import List
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import config
import os
from database import Database
from dotenv import load_dotenv
load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")

class DocumentDatabaseAnalysis(Database):
  
    def format_docs(self, docs: List[Document]):
        return "\n\n".join(doc.page_content for doc in docs)
    

    def _initialize(self, load=True, file_path="data/", text_splitter=None, loader=None):
        self.vectorstore = Chroma(
            persist_directory= config.PERSIST_DIRECTORY,
            embedding_function=OpenAIEmbeddings()
        )
       
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
