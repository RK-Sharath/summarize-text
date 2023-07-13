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





def generate_res(text):
    #Define llm
    llm = LangChainInterface(
        model=ModelType.FLAN_T5_11B,
        credentials=Credentials(api_key=genai_api_key),
        params=GenerateParams(
            decoding_method="greedy",
            max_new_tokens=600,
            min_new_tokens=150,
            repetition_penalty=2,
        ).dict())
    # Split text
    splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunked_docs = splitter.create_documents(text)
    # Text summarization
    chain = load_summarize_chain(llm, chain_type='map_reduce')
    return chain.run(chunked_docs)




# Page title
st.set_page_config(page_title='🦜🔗 Text Summarization App using Watsonx')
st.title('🦜🔗 Text Summarization App using Watsonx')



# Text input
txt_input = st.text_area('Enter your text', '', height=200)

# Form to accept user's text input for summarization
result = []
with st.form('summarize_form', clear_on_submit=True):
    api_key = st.text_input('Genai_api_key', disabled=not txt_input)
    submitted = st.form_submit_button('Submit')
    if submitted and api_key.startswith('pak-'):
        with st.spinner('Calculating...'):
            response = generate_response(txt_input)
            result.append(response)
            del api_key

if len(result):
    st.info(response)
    
