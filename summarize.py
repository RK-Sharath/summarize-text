import streamlit as st
import os
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.question_answering import load_qa_chain
from genai.extensions.langchain import LangChainInterface
from genai.schemas import ModelType, GenerateParams
from genai.credentials import Credentials


st.title("Document Summarization App powered by IBM Watsonx")
st.caption("This app was developed by Sharath Kumar RK, IBM Ecosystem Engineering Watsonx team")


#genai_api_key = st.sidebar.text_input("GenAI API Key", type="password")
genai_api_url = st.sidebar.text_input("GenAI API URL", type="default")
max_new_tokens = st.sidebar.number_input("Select max new tokens")
min_new_tokens = st.sidebar.number_input("Select min new tokens")
chunk_size = st.sidebar.number_input("Select chunk size")
chunk_overlap = st.sidebar.number_input("Select chunk overlap")
decoding_method = st.sidebar.text_input("Decoding method (Choose either greedy or sample) ", type="default")


@st.cache_data
def load_docs(files):
    st.info("`Reading doc ...`")
    all_text = ""
    for file_path in files:
        file_extension = os.path.splitext(file_path.name)[1]
        if file_extension == ".pdf":
            pdf_reader = PyPDF2.PdfReader(file_path)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            all_text += text
        elif file_extension == ".txt":
            stringio = StringIO(file_path.getvalue().decode("utf-8"))
            text = stringio.read()
            all_text += text
        else:
            st.warning('Please provide txt or pdf file.', icon="⚠️")
    return all_text
     
     
#@st.cache_resource
def split_texts(text, chunk_size, chunk_overlap, split_method):

    st.info("`Splitting doc ...`")

    split_method = "RecursiveTextSplitter"
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    splits = text_splitter.split_text(text)
    if not splits:
        st.error("Failed to split document")
        st.stop()

    return splits

def generate_res(query):
     
    # Instantiate the LLM model
    llm = LangChainInterface(
    model="google/flan-t5-xxl",
    credentials=Credentials(api_key=st.session_state.genai_api_key),
    params=GenerateParams(
    decoding_method=decoding_method,
    max_new_tokens=max_new_tokens,
    min_new_tokens=min_new_tokens,
    repetition_penalty=2,
    ).dict()) 
     
    # Text summarization
    chain = load_summarize_chain(llm, chain_type='map_reduce')
    return chain.run(query)
    

def main():

    if 'genai_api_key' not in st.session_state:
        genai_api_key = st.text_input(
            'Please enter your GenAI API key', value="", placeholder="Enter the GenAI API key which begins with pak-")
        if genai_api_key:
            st.session_state.genai_api_key = genai_api_key
            os.environ["GENAI_API_KEY"] = genai_api_key
        else:
            return
    else:
        os.environ["GENAI_API_KEY"] = st.session_state.genai_api_key

    uploaded_files = st.file_uploader("Upload a PDF or TXT Document", type=[
                                      "pdf", "txt"], accept_multiple_files=True)

    if uploaded_files:
        # Check if last_uploaded_files is not in session_state or if uploaded_files are different from last_uploaded_files
        if 'last_uploaded_files' not in st.session_state or st.session_state.last_uploaded_files != uploaded_files:
            st.session_state.last_uploaded_files = uploaded_files
                 # Load and process the uploaded PDF or TXT files.
        loaded_text = load_docs(uploaded_files)
        st.write("Documents uploaded and processed.")
        
        # Split the document into chunks
        splitter_type = "RecursiveCharacterTextSplitter"
        splits = split_texts(loaded_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap, split_method=splitter_type)

        # Display the number of text chunks
        num_chunks = len(splits)
        st.write(f"Number of text chunks: {num_chunks}")
        
        # Generate summary
        submitted = st.form_submit_button('Submit')
        if submitted:
            response = generate_res(loaded_text)
            st.write("Answer:", response)
            st.download_button("Download the results", response)
            

if __name__ == "__main__":
    main()
    
