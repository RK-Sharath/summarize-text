import streamlit as st
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.question_answering import load_qa_chain
from genai.extensions.langchain import LangChainInterface
from genai.schemas import ModelType, GenerateParams
from genai.credentials import Credentials

st.subheader("Text Summarization App powered by IBM Watsonx")
st.caption("This app was developed by Sharath Kumar RK, IBM Ecosystem Engineering Watsonx team")

# Text input
input_data = st.text_area('Enter your text below:', '', height=400)

genai_api_key = st.sidebar.text_input("GenAI API Key", type="password")
genai_api_url = st.sidebar.text_input("GenAI API URL", type="default")
max_new_tokens = st.sidebar.text_input("Select max new tokens", type="default")
min_new_tokens = st.sidebar.text_input("Select min new tokens", type="default")
     


def generate_res(query):
     
    # Instantiate the LLM model
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
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(query)
     
    # Create multiple documents
    docs = [Document(page_content=t) for t in texts]
     
    # Text summarization
    chain = load_summarize_chain(llm, chain_type='map_reduce')
    return chain.run(docs)


# Capture text input for summarization

result = []
with st.form('summarize_form', clear_on_submit=True):
    submitted = st.form_submit_button('Submit')
    if submitted and genai_api_key.startswith('pak-'):
        with st.spinner('Working on it...'):
            response = generate_res(input_data)
            result.append(response)
            del openai_api_key

if len(result):
    st.info(response)
    