import pandas as pd
import streamlit as st
from pandas.api.types import is_datetime64_any_dtype, is_object_dtype
from pandas.api.types import CategoricalDtype
import tempfile
from openpyxl import Workbook, load_workbook
import subprocess
subprocess.call(['pip', 'install', '-r', 'https://raw.githubusercontent.com/vkim20016/Project-work/main/requirements.txt'])

def filter_dataframe(df: pd.DataFrame, filter_columns: list) -> pd.DataFrame:
    filtered_df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in filtered_df.columns:
        if is_object_dtype(filtered_df[col]):
            try:
                filtered_df[col] = pd.to_datetime(filtered_df[col])
            except Exception:
                pass
        if is_datetime64_any_dtype(filtered_df[col]):
            filtered_df[col] = filtered_df[col].dt.tz_localize(None)

    for column in filter_columns:
        with st.expander(f"Filter by {column}", expanded=False):
            if isinstance(filtered_df[column].dtype, CategoricalDtype) or filtered_df[column].nunique() < 10000:
                unique_values = filtered_df[column].unique()
                selected_values = st.multiselect(
                    f"Values for {column}",
                    unique_values,
                    default=[],
                )
                if len(selected_values) > 0:
                    filtered_df = filtered_df[filtered_df[column].isin(selected_values)]

    return filtered_df

def make_clickable(url):
    return f'<a href="{url}" target="_blank">{url}</a>'

# Set Streamlit page configuration
st.set_page_config(
    page_title="Hematology & Oncology Field Medical Insights Report",
    page_icon=":bar_chart:",
    layout="wide"
)

# Upload Excel file
uploaded_file = st.sidebar.file_uploader("Upload Excel file", type=["xlsx"])

if uploaded_file is not None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file_path = f"{tmp_dir}/uploaded_file.xlsx"
        with open(tmp_file_path, "wb") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())

        try:
            df = pd.read_excel(tmp_file_path)

            filter_columns = st.sidebar.multiselect("Filter dataframe on", df.columns, key="filter_columns")
            if len(filter_columns) > 0:
                df = filter_dataframe(df, filter_columns)

            st.dataframe(df)
        except Exception as e:
            st.error(f"An error occurred: {e}")
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain

# Define the path for generated embeddings
DB_FAISS_PATH = 'vectorstore/db_faiss'

# Load the model of choice
def load_llm():
    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

# Set the title for the Streamlit app
st.title("Llama2 Chat CSV - 🦜🦙")

# Create a file uploader in the sidebar
uploaded_file = st.sidebar.file_uploader("Upload File", type="csv")

# Handle file upload
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # Load CSV data using CSVLoader
    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={'delimiter': ','})
    data = loader.load()

    # Create embeddings using Sentence Transformers
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})

    # Create a FAISS vector store and save embeddings
    db = FAISS.from_documents(data, embeddings)
    db.save_local(DB_FAISS_PATH)

    # Load the language model
    llm = load_llm()

    # Create a conversational chain
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

    # Function for conversational chat
    def conversational_chat(query):
        result = chain({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
        return result["answer"]

    # Initialize chat history
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    # Initialize messages
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello ! Ask me(LLAMA2) about " + uploaded_file.name + " 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey ! 👋"]

    # Create containers for chat history and user input
    response_container = st.container()
    container = st.container()

    # User input form
    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="Talk to csv data 👉 (:", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output = conversational_chat(user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    # Display chat history
    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
# Define the path for generated embeddings
DB_FAISS_PATH = 'vectorstore/db_faiss'

# Load the model of choice
def load_llm():
    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

# Set the title for the Streamlit app
st.title("Llama2 Chat Excel - 🦜🦙")

# Create a file uploader in the sidebar
uploaded_file = st.sidebar.file_uploader("Upload File", type=["csv", "xlsx"])

# Handle file upload
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # Load Excel data using UnstructuredExcelLoader
    loader = UnstructuredExcelLoader(file_path=tmp_file_path, mode="elements")
    docs = loader.load()
    doc = docs[0]  # Access the first document in the list

    # Create embeddings using Sentence Transformers
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})

    # Create a FAISS vector store and save embeddings
    db = FAISS.from_documents([doc], embeddings)  # Pass the document as a list
    db.save_local(DB_FAISS_PATH)

    # Load the language model
    llm = load_llm()

    # Create a conversational chain
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

    # Function for conversational chat
    def conversational_chat(query):
        result = chain({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
        return result["answer"]

    # Initialize chat history
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    # Initialize messages
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me (LLAMA2) about " + uploaded_file.name + " 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

    # Create containers for chat history and user input
    response_container = st.container()
    container = st.container()

    # User input form
    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="Talk to Excel data 👉 (:", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output = conversational_chat(user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    # Display chat history
    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                st.write(st.session_state["past"][i], key=str(i) + '_user')
                st.write(st.session_state["generated"][i], key=str(i))
