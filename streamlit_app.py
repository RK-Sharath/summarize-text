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
import os 


llm = LangChainInterface(
        model=ModelType.FLAN_T5_11B,
        credentials=Credentials(api_key=genai_api_key),
        params=GenerateParams(
            decoding_method="greedy",
            max_new_tokens=1000,
            min_new_tokens=150,
            repetition_penalty=2,
        ).dict()
    )


def generate_response(txt):
    # Split text
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(txt)
    # Create multiple documents
    docs = [Document(page_content=t) for t in texts]
    # Text summarization
    chain = load_summarize_chain(llm, chain_type='map_reduce')
    return chain.run(docs)




# Page title
st.set_page_config(page_title='🦜🔗 Text Summarization App using Watsonx')
st.title('🦜🔗 Text Summarization App using Watsonx')



# Text input
txt_input = st.text_area('Enter your text', '', height=200)

# Form to accept user's text input for summarization
result = []
with st.form('summarize_form', clear_on_submit=True):
    genai_api_key = st.text_input('genai_api_key', disabled=not txt_input)
    submitted = st.form_submit_button('Submit')
    if submitted and genai_api_key.startswith('pak-'):
        with st.spinner('Calculating...'):
            response = generate_response(txt_input)
            result.append(response)
            del genai_api_key

if len(result):
    st.info(response)
    
