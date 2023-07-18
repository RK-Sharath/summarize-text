import streamlit as st
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains.question_answering import load_qa_chain
from genai.extensions.langchain import LangChainInterface
from genai.schemas import ModelType, GenerateParams
from genai.model import Credentials
from genai.credentials import Credentials
import os 

st.title("Text Summarization App")
st.caption("This app was developed by Sharath Kumar RK, Ecosystem Engineering Watsonx team")

# Text input
txt_input = st.text_area('Enter your text', '', height=400)

genai_api_key = st.sidebar.text_input("GenAI API Key", type="password")
genai_api_url = st.sidebar.text_input("GenAI API URL", type="default")
chunk_size = st.sidebar.text_input("Select Chunk size", type="default")
chunk_overlap = st.sidebar.text_input("Select Chunk overlap", type="default")
max_new_tokens = st.sidebar.text_input("Select max new tokens", type="default")
min_new_tokens = st.sidebar.text_input("Select min new tokens", type="default")
    
    


def generate_res(text):
    #Define llm
    llm = LangChainInterface(
        model=ModelType.FLAN_T5_11B,
        credentials=Credentials(api_key=genai_api_key),
        params=GenerateParams(
            decoding_method="greedy",
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            repetition_penalty=2,
        ).dict())
    # Split text
   # splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
   # chunked_docs = splitter.split_texts(text)
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_text(txt)
    # Create multiple documents
    docs = [Document(page_content=t) for t in texts]
    # Text summarization
    chain = load_summarize_chain(llm, chain_type='map_reduce')
    return chain.run(chunked_docs)


# Form to accept user's text input for summarization
result = []
with st.form('summarize_form', clear_on_submit=True):
    submitted = st.form_submit_button('Submit')
    if submitted and genai_api_key.startswith('pak-'):
        with st.spinner('Working on it...'):
            response = generate_res(txt_input)
            result.append(response)
            del genai_api_key

if len(result):
    st.info(response)
    
