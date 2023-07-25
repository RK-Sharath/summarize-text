import streamlit as st
import os
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


genai_api_key = st.sidebar.text_input("GenAI API Key", type="password")
genai_api_url = st.sidebar.text_input("GenAI API URL", type="default")
max_new_tokens = st.sidebar.number_input("Select max new tokens")
min_new_tokens = st.sidebar.number_input("Select min new tokens")
chunk_size = st.sidebar.number_input("Select chunk size")
chunk_overlap = st.sidebar.number_input("Select chunk overlap")
chain_type = st.sidebar.selectbox("Chain Type", ["map_reduce", "stuff", "refine"])
with st.sidebar:
    decoding_method = st.radio(
        "Select decoding method",
        ('greedy', 'sample')
    )
temperature = st.sidebar.number_input("Temperature (Choose a decimal number between 0 & 2)")

@st.cache_data
def setup_documents(pdf_file_path, chunk_size, chunk_overlap):
    loader = PyPDFLoader(pdf_file_path)
    docs_raw = loader.load()
    docs_raw_text = [doc.page_content for doc in docs_raw]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.create_documents(docs_raw_text)
    return docs


def custom_summary(docs,llm, custom_prompt, chain_type, num_summaries):
    custom_prompt = custom_prompt + """:\n\n {text}"""
    COMBINE_PROMPT = PromptTemplate(template=custom_prompt, input_variables=["text"])
    MAP_PROMPT = PromptTemplate(template="Summarize:\n\n{text}", input_variables=["text"])
    creds = Credentials(api_key=genai_api_key, api_endpoint=genai_api_url)
    # Define parameters
    params = GenerateParams(decoding_method=decoding_method, temperature=temperature, max_new_tokens=max_tokens, min_new_tokens=min_tokens, repetition_penalty=repetition_penalty)
    # Instantiate LLM model
    llm=LangChainInterface(model=model, params=params, credentials=creds)
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
    pdf_file_path = st.text_input("Enter the pdf file path")
    if pdf_file_path != "":
        docs = setup_documents(pdf_file_path, chunk_size, chunk_overlap)
        st.write("Pdf was loaded successfully")
        if st.button("Summarize"):
            result = custom_summary(docs,llm, user_prompt, chain_type, num_summaries)
            st.write("Summaries:")
            for summary in result:
                st.write(summary)

if __name__ == "__main__":
    main()
