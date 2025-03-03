import os
import streamlit as st
from view.view import View
from streamlit_chat import message
from streamlit_float import float_init
from PIL import Image
from dotenv import load_dotenv
import urllib.parse
import re
import json
import config
from langchain.prompts import PromptTemplate

# Load environment variables from .env
load_dotenv()

# Get FILE_LOCATIONS from .env
BASE_FILE_PATH = os.getenv("FILES_LOCATION", "")

# from model.database import ask_rag


class ChatView:
    def __init__(self, file_path: str):

        st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
        # Display Logo
        st.columns(1)[0].image("view/images/iarisLogo.jpeg", width=80)

        # Sidebar for chatbot customization
        with st.sidebar:
            st.title("⚙️ Chatbot Settings")

            # Search Filters (Expander for Search Parameters)
            with st.expander("🔍 Search Parameters", expanded=False):
                self.values = st.slider("Search in how many documents", 1, 10)
            
                topics = self._load_topics_json()
                self.search_filter = st.multiselect("Filter by", topics, default=topics, format_func=self._format_topic)

                
            self.user_prompt = st.text_area(
                "Edit the chatbot's prompt template",
                '''You are IAris, a concise chatbot expert in the social, cultural and envirnoment impact that will advice leaders that want to build social businesses.
                All answers must be related to the domain of positive societal impact.
                ## The client asks you the following question: "{question}"
                ## You have to provide an answer based on the following documents:"{context}"
                Your answer should only be based on the documents provided.
                Be provocative and ask one follow-up inquiry to question the client, making sure that gaps are considered.''', 
                height=400
            )

        # Apply user settings to generate the final prompt
        self.promptTemplate = PromptTemplate.from_template(self.user_prompt)

        
        self.retriever_k = 1
        self.key = 0

       
        self.human_message = st.chat_message("🙋")
        self.rag_message = st.chat_message("🤖")
        self.sources_tab = st.empty()

        text_container = st.container()
        # Text Input Box
        with text_container:
            self.user_input = st.text_area("", "", key="input", placeholder="Pergunte alguma coisa")

        # Initialize Session State
        self._init_session_state()

    def _init_session_state(self):
        """Initialize session state variables."""
        st.session_state['user_input'] = []
        st.session_state['generated_stream'] = None
        st.session_state['generated'] = []
        st.session_state['rag_stream'] = None
        st.session_state['rag_generated'] = []
        st.session_state['sources'] = None

    @staticmethod
    def _format_topic(topic):
        """Format function for topic names."""
        return topic.replace("_", " ").title()

    def get_text(self):
        self.retriever_k = self.values
        return self.user_input

    def get_edited_prompt(self):
        return self.promptTemplate

    def get_search_filters(self):
        return self.search_filter

    def display(self, responses: dict = None):
        """Handles displaying the chat messages and responses."""
        if self.user_input:
            st.session_state['user_input'].append(self.user_input)

            if responses:
                self._handle_responses(responses)

        with self.human_message.container():
            if st.session_state['user_input']:
                self.human_message.markdown(st.session_state['user_input'][-1])

        with self.rag_message.container():
            if st.session_state['rag_stream']:
                st.write_stream(st.session_state['rag_stream'])
            elif st.session_state['rag_generated']:
                self.rag_message.markdown(st.session_state['rag_generated'][-1])

        self._display_sources()
        self.key += 1

    def _handle_responses(self, responses):
        """Handles LLM and RAG responses."""
        if 'llm' in responses:
            st.session_state['generated'].append(responses['llm'])

        if 'rag' in responses:
            st.session_state['rag_generated'].append(responses['rag']['answer'])
            sources = [doc.metadata["source"] for doc in responses['rag']['context']]
            st.session_state['sources'] = sources

        if "llm_stream" in responses:
            st.session_state['generated_stream'] = responses['llm_stream']

        if "rag_stream" in responses:
            st.session_state['rag_stream'] = responses['rag_stream']
            st.session_state['sources'] = responses['sources']

    def _display_sources(self):
        """Displays the document sources as clickable links in a horizontal layout."""
        with self.sources_tab.container():
            if st.session_state['sources']:
                st.markdown("📚 **Fontes**")

                cols = st.columns(len(st.session_state['sources']))  # Create one column per source
                
                for col, source_path in zip(cols, st.session_state['sources']):
                    cleaned_path = re.match(r"^(.+?\.pdf)\b", source_path)
                    if cleaned_path:
                        file_name = os.path.basename(source_path)
                        file_url = urllib.parse.quote(BASE_FILE_PATH + cleaned_path.group(1), safe=":/")
                        with col:
                            st.link_button(label=f"{file_name}", url=file_url)


    def _load_topics_json(self):
        topics_json_path = config.TOPICS_FILE

        if not os.path.exists(topics_json_path):
            print("⚠️ topics.json not found. Trying to fetch from current data dir.")
            topics_dirs = [f.path for f in os.scandir(config.DATA_DIR) if f.is_dir()]
            topics = [re.search(r'[^/]+$', topic).group() for topic in topics_dirs]
            return topics

        with open(topics_json_path, "r", encoding="utf-8") as f:
            topics_clean = json.load(f)

        return topics_clean
    

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
