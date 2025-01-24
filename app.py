import os
import streamlit as st
from llama_index import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.llms import Groq
from llama_index.utils import WorkflowExecutor

# Streamlit Page Configuration
st.set_page_config(
    page_title="Sherlock Holmes Chatbot",
    page_icon="🕵️",
    layout="centered",
)

# Title
st.title("Sherlock Holmes Chatbot 🕵️")
st.info("Ask questions about the Sherlock Holmes : ", icon="❓")

# Folder Path for Stories
FOLDER_PATH = r"C:\Users\user\Desktop\test"

# Initialize Session State for Messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello! Ask me anything about Sherlock Holmes."}
    ]

# Function to Load Data and Create Index
@st.cache_resource
def create_index():
    # Load data using SimpleDirectoryReader
    reader = SimpleDirectoryReader(input_dir=FOLDER_PATH)
    documents = reader.load_data()

    # Define Groq LLM and ServiceContext
    llm = Groq(model_name="groq-small", model_kwargs={"temperature": 0.5, "max_length": 256})
    service_context = ServiceContext.from_defaults(llm=llm)

    # Create a VectorStoreIndex
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    return index

# Create or Load Index
index = create_index()
chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

# User Input and Chat
if prompt := st.chat_input("Your question"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            st.session_state["messages"].append({"role": "assistant", "content": response.response})

# Display Chat History
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])
