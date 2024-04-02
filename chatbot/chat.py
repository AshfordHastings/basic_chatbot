from typing import Any
from uuid import UUID
from langchain_core.outputs import ChatGenerationChunk, GenerationChunk
import streamlit as st 

from langchain_openai import ChatOpenAI
from langchain.chains.conversation.base import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler

st.set_page_config(page_title="Chatbot", page_icon="🤖")
st.header("A Basic Chatbot!")
st.write('Interact with an LLM-based chatbot.')

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text
    
    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text)
    

class ChatBot:
    def __init__(self, model="gpt-4", temperature=0.0):
        self.model_type = model
        self.temperature = temperature

    #@st.cache_resource
    @st.cache_resource # Prevent reinitialization upon user message
    def _init_chain(_self):
        memory = ConversationBufferMemory()
        llm = ChatOpenAI(model=_self.model_type, temperature=_self.temperature, streaming=True)
        chain = ConversationChain(llm=llm, memory=memory, verbose=True)
        return chain 
    
    
    def _enable_chat_history(self):
        if "messages" not in st.session_state:
            #st.session_state["messages"] = [{"role": "assistant", "content": "Hello! How can I help you today?"}] 
            st.session_state.messages = []
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    
    def run(self):
        self._enable_chat_history()
        chain = self._init_chain()

        user_input = st.chat_input("Ask me anything!")
        if user_input:
            with st.chat_message("Human"):
                st.markdown(user_input)
                st.session_state.messages.append({"role": "Human", "content": user_input})
            with st.chat_message("AI"):
                stream_cb = StreamHandler(st.empty())
                response = chain.invoke(user_input, config={"callbacks": [stream_cb]})["response"]
                # st.markdown(response)
                st.session_state.messages.append({"role": "AI", "content": response})

chatbot = ChatBot()
chatbot.run()
    