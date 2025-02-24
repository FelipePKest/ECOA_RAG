from typing import List
# import promptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import os
from model.database import Database
# from database import Database

from dotenv import load_dotenv
load_dotenv(override=True)

openai_key = os.getenv("OPENAI_API_KEY")

class DocumentDatabase(Database):

    paths = []
    base_path = "data/pdfs"

    def format_docs(self, docs: List[Document]):
        return "\n\n".join(doc.page_content for doc in docs)
    

    def _initialize(self, load=True, file_path="data/", text_splitter=None, loader=None):
        if os.path.exists("./chroma_db") and load:
            print("Loading")
            self.vectorstore = Chroma(
                persist_directory="./chroma_db",
                embedding_function=OpenAIEmbeddings()
            )
            print(len(self.vectorstore.get()))
        else:

            print("Loading documents from PDFs")
            documents_paths = []
            splits = []
            
            # get subdirectories names
            subjects = [f.path for f in os.scandir(file_path) if f.is_dir()]

            # load all pdfs in the directory and subdirectories
            for root, dirnames, files in os.walk(file_path):
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
                for doc in docs:
                    for subject in subjects:
                        if subject in document_path:
                            doc.metadata['subject'] = subject
                    # doc.metadata['subject'] = document_path
                
                # {"source": document_path}


                if text_splitter is None:
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

                splits.extend(text_splitter.split_documents(docs))

                i += 1
            print("Finished loading documents")
            self.vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(), persist_directory="./chroma_db")            
    
    def _setup_rag(self, *args, **kwargs):
        if len(args) > 0:
            chain_params = args[0]
        if "filter_dict" in kwargs.keys():
            print("Filter dict in setup: ", kwargs["filter_dict"])
            filter_dict = kwargs["filter_dict"]
            for i in range(len(filter_dict["filters"])):
                full_filter = self.base_path + "/" + filter_dict["filters"][i]
                filter_dict["filters"][i] = full_filter
            # print("Full Filter dict in setup: ", filter_dict)
            retriever = self.vectorstore.as_retriever(
                
                search_kwargs={
                    "k": chain_params["retriever_k"],
                    "filter": {"subject": {"$in":filter_dict["filters"]}},
                }
            )
        else:
            retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": chain_params["retriever_k"]}
            )
        template = PromptTemplate.from_template('''
            You are IAris, a concise chatbot expert in the social, cultural and envirnoment impact that will advice leaders that want to build social businesses.
            All answers must be related to the domain of positive societal impact.
            ## The client asks you the following question: "{question}"
            ## You have to provide an answer based on the following documents:"{context}"
            Your answer should only be based on the documents provided.
            Be provocative and ask one follow-up inquiry to question the client, making sure that gaps are considered.
        ''')
        # prompt = hub.pull("rlm/rag-prompt")
        
        llm = ChatOpenAI(model_name="gpt-4o-mini", api_key=openai_key)

        rag_chain_from_docs = ( RunnablePassthrough.assign(
            context=(lambda x: self.format_docs(x["context"])))
            | template
            | llm
            | StrOutputParser()
        )

        rag_chain_with_source = RunnableParallel(
            {"context": retriever, "question": RunnablePassthrough()}
        ).assign(answer=rag_chain_from_docs)

        return rag_chain_with_source

    def ask_rag(self, query, debug=False, *args, **kwargs) -> dict:
        print("args = ",args)
        # kwargs = len(args) > 0
        chain_params = {}
        output_format = "string"
        if len(args) > 0:
            chain_params = args[0]
            if len(args) > 1:
                output_format = args[1]
        filter_dict = {}
        if "filter_dict" in kwargs:
            filter_dict = kwargs["filter_dict"]
            # print("Filter dict: ", filter_dict)
            rag_chain = self._setup_rag(chain_params, filter_dict=filter_dict)
        else:
            rag_chain = self._setup_rag(chain_params)
        llm = ChatOpenAI(model_name="gpt-4o-mini", api_key=openai_key)
        if debug:
            fake_docs = [Document(page_content="CONTEXT", metadata={"source":"SOURCE"+str(i)}) for i in range(1, chain_params["retriever_k"]+1)]
            responses = {"query": query, "llm": "LLM ANSWER", "rag": {"answer":"RAG ANSWER", "context": fake_docs}} 
            return responses
        if output_format == "stream":
            # Get sources
            sources = []
            context = rag_chain.invoke(query)["context"]
            if len(context) == 0:
                print("No context found!!!!!!")
            answer_chain = rag_chain.pick("answer")
            for i in range(len(context)):
                print("Context: ",context[i].metadata["source"])
                sources.append(context[i].metadata["source"])
                if "page" in context[i].metadata:
                    sources[i] += "\n\n Pagina " + str(context[i].metadata["page"])

            # llm.stream(query) - not asking llm
            responses = {"query": query, "llm_stream": "", "rag_stream": answer_chain.stream(query), "sources": sources}
        else:
            # llm.invoke(query).content - not asking llm
            responses = {"query": query, "llm": "", "rag": rag_chain.invoke(query)}
        return responses

print("Document Database Loaded")