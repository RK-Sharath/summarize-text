import streamlit as st
import os
import tempfile
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.question_answering import load_qa_chain
from genai.extensions.langchain import LangChainInterface
from genai.schemas import GenerateParams
from genai.credentials import Credentials
import PyPDF2
from io import StringIO
from langchain.document_loaders import PyPDFLoader
from langchain import PromptTemplate


st.title("Document Summarization App powered by IBM Watsonx")
st.caption("This app was developed by Sharath Kumar RK, IBM Ecosystem Engineering Watsonx team")

model = st.radio("Select the Watsonx LLM model",('google/flan-t5-xl','google/flan-t5-xxl','google/flan-ul2'))
genai_api_key = st.sidebar.text_input("GenAI API Key", type="password")
genai_api_url = st.sidebar.text_input("GenAI API URL", type="default")
max_new_tokens = st.sidebar.number_input("Select max new tokens", value=600)
min_new_tokens = st.sidebar.number_input("Select min new tokens", value=150)
chunk_size = st.sidebar.number_input("Select chunk size", value=1200)
chunk_overlap = st.sidebar.number_input("Select chunk overlap", value=100)
chain_type = st.sidebar.selectbox("Chain Type", ["map_reduce", "stuff", "refine"])
with st.sidebar:
    decoding_method = st.radio(
        "Select decoding method",
        ('greedy', 'sample')
    )
temperature = st.sidebar.number_input("Temperature (Choose a decimal number between 0 & 2)", value=0.4)
repetition_penalty = st.sidebar.number_input("Repetition penalty (Choose either 1 or 2)", value=2)
num_summaries = st.sidebar.number_input("Number of Summaries", min_value=1, max_value=10, step=1, value=1)


uploaded_file = st.file_uploader("Upload a PDF or TXT Document", type=(['pdf', "txt"]))
temp_file_path = os.getcwd()
#while uploaded_file is None:
    #x = 1
        
if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    temp_dir = tempfile.TemporaryDirectory()
    temp_file_path = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(uploaded_file.read())

    st.write("Full path of the uploaded file:", temp_file_path)



@st.cache_data
def setup_documents(chunk_size, chunk_overlap):
    loader = PyPDFLoader(temp_file_path)
    docs_raw = loader.load()
    docs_raw_text = [doc.page_content for doc in docs_raw]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.create_documents(docs_raw_text)
    return docs


def custom_summary(docs,llm, custom_prompt, chain_type, num_summaries):
    
    custom_prompt = custom_prompt + """:\n\n {text}"""
    COMBINE_PROMPT = PromptTemplate(template=custom_prompt, input_variables=["text"])
    MAP_PROMPT = PromptTemplate(template="Summarize:\n\n{text}", input_variables=["text"])
    
    if chain_type == "map_reduce":
        chain = load_summarize_chain(llm, chain_type=chain_type, 
                                    map_prompt=MAP_PROMPT, combine_prompt=COMBINE_PROMPT)
    else:
        chain = load_summarize_chain(llm, chain_type=chain_type)
        
    summaries = []
    for i in range(num_summaries):
        summary_output = chain({"input_documents": docs}, return_only_outputs=True)["output_text"]
        summaries.append(summary_output)
    
    return summaries


def main():
    user_prompt = st.text_input("Enter the user prompt")
    creds = Credentials(api_key=genai_api_key, api_endpoint=genai_api_url)
    # Define parameters
    params = GenerateParams(decoding_method=decoding_method, temperature=temperature, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens, repetition_penalty=repetition_penalty)
    # Instantiate LLM model
    llm=LangChainInterface(model=model, params=params, credentials=creds)
    if temp_file_path != "":
        docs = setup_documents(chunk_size, chunk_overlap)
    # Display the number of text chunks
    num_chunks = len(docs)
    st.write(f"Number of text chunks: {num_chunks}")
    st.write("Pdf was loaded successfully")
    if st.button("Summarize"):
        with st.spinner('Working on it...'):
            result = custom_summary(docs,llm, user_prompt, chain_type, num_summaries)
            st.write("Summaries:")
            for summary in result:
                st.write(summary)

if __name__ == "__main__":
    main()
