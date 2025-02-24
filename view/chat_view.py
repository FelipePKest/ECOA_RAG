import os
import streamlit as st
from view.view import View
from streamlit_chat import message
from streamlit_float import float_init
from PIL import Image
from dotenv import load_dotenv
import urllib.parse
import re

# Load environment variables from .env
load_dotenv()

# Get FILE_LOCATIONS from .env
BASE_FILE_PATH = os.getenv("FILES_LOCATION", "")

# from model.database import ask_rag

def _generate_context(prompt, context_data='generated'):
    context = []
    # If any history exists
    if st.session_state['generated']:
        # Add the last three exchanges
        EXCHANGE_LIMIT = 3
        size = len(st.session_state['generated'])
        for i in range(max(size-EXCHANGE_LIMIT, 0), size):
            context.append(
                {'role': 'user', 'content': st.session_state['user_input'][i]})
            context.append(
                {'role': 'assistant', 'content': st.session_state[context_data][i]})
    # Add the latest user prompt
    context.append({'role': 'user', 'content': str(prompt)})
    return context

def _get_text():
    input_text = st.text_input("Ask away", "", key="input")
    return input_text

def format_function(text, wrap_limit=10):
    "Wrap text in newline if it's to big"
    new_text = ""
    line = ""
    for word in text.split():
        line += word + " "
        new_text += word + " "
        if len(line) > wrap_limit:
            new_text += "\n"
            line = ""
    return new_text
# if 'user_input' not in st.session_state:


class ChatView(View):

    def __init__(self):
        logo = st.columns(1)[0]
        with logo:
            st.image("view/images/IARIS.png", width=100)
        
        search_params = st.expander("Parametros de busca", expanded=False)

        with search_params:
            k_filter, docs_filters = st.columns([1, 1])
            with k_filter:
                self.values = st.slider("Buscar em quantos documentos", 1, 10)
            with docs_filters:
                # TODO: Change hardcode strings to dynamic values
                topics = ["Negocios Sociais"]
                self.search_filter = st.multiselect("Filtrar por", topics, default=topics, format_func=format_function)
                # self
        self.retriever_k = 1


        col1, col2 = st.columns([4, 1])

        self.key:int=0

        with col2:
            self.sources_tab = st.empty()
            self.third_placeholder = st.empty()
        with col1:
            self.human_message = st.chat_message("user")
            self.rag_message = st.chat_message("🤖")

        text_container = st.container()
        with text_container:
            self.user_input = st.text_input("", "", key="input")
        
        text_container.float(
            "display:flex; align-items:center;justify-content:center; overflow:hidden visible;flex-direction:column; position:fixed;bottom:15px;"
        )

        st.session_state['user_input'] = []
        st.session_state['generated_stream'] = None
        st.session_state['generated'] = []
        st.session_state['rag_stream'] = None
        st.session_state['rag_generated'] = []
        st.session_state['sources'] = None

    def _generate_context(self, prompt, context_data='generated'):
        context = []
        # If any history exists
        if st.session_state['generated']:
            # Add the last three exchanges
            EXCHANGE_LIMIT = 3
            size = len(st.session_state['generated'])
            for i in range(max(size-EXCHANGE_LIMIT, 0), size):
                context.append(
                    {'role': 'user', 'content': st.session_state['user_input'][i]}
                )
                context.append(
                    {'role': 'assistant', 'content': st.session_state[context_data][i]}
                )
        # Add the latest user prompt
        context.append({'role': 'user', 'content': str(prompt)})
        return context

    def get_text(self):
        self.retriever_k = self.values
        return self.user_input
    
    def get_search_filters(self):
        return self.search_filter

    def display(self, responses:dict=None):
        float_init()
        if self.user_input:
            st.session_state['user_input'].append(self.user_input)
            if 'llm' in responses:
                st.session_state['generated'].append(responses['llm'])
            if 'rag' in responses:
                st.session_state['rag_generated'].append(responses['rag']['answer'])
                sources = []
                for i in range(len(responses['rag']['context'])):
                    print("Context: ",responses['rag']['context'][i].metadata["source"])
                    sources.append(responses['rag']['context'][i].metadata["source"])
                st.session_state['sources'] = sources
            if "llm_stream" in responses:
                st.session_state['generated_stream'] = responses['llm_stream']
            if "rag_stream" in responses:
                st.session_state['rag_stream'] = responses['rag_stream']
                st.session_state['sources'] = responses['sources']

                        
        with self.human_message.container():
            if st.session_state['user_input']:
                self.human_message.markdown(st.session_state['user_input'][-1])
        
        # with self.llm_message.container():
        #     if st.session_state['generated_stream']:
        #         st.write_stream(st.session_state['generated_stream'])
        #     elif st.session_state['generated']:
        #         self.llm_message.markdown(st.session_state['generated'][-1])
            
        with self.rag_message.container():
            if st.session_state['rag_stream']:
                st.write_stream(st.session_state['rag_stream'])
                # for chunk in st.session_state['rag_stream']:
                #     if "answer" in chunk:
                #         self.rag_message.markdown(chunk["answer"])
            elif st.session_state['rag_generated']:
                self.rag_message.markdown(st.session_state['rag_generated'][-1])

        # Generated Cypher statements
        with self.sources_tab.container():
            if st.session_state['sources']:
                for i in range(len(st.session_state['sources'])):
                    answer_position = st.session_state['sources'][i]
                    file_path = re.search(r"^(.+?\.pdf)\b", answer_position) # removes the "Pagina X"
                    if file_path:
                        cleaned_path = file_path.group(1)
                        answer_file_position = os.path.basename(answer_position)  # Extract just the file name
                        file_url = BASE_FILE_PATH + cleaned_path  # Concatenate the base path with the file name
                        file_url = urllib.parse.quote(file_url, safe=":/")  # Encode special characters
                        
                        st.link_button(label=f"{answer_file_position}", url=file_url)
                        st.divider()
                # st.text_area("Source",
                #             st.session_state['source'],key=self.key, height=240)
        
        self.key += 1


# # if __name__ == "__main__":
# view = ChatView()
# view.display()
# #     # setup_view()