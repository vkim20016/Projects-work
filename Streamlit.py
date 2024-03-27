import pandas as pd
import streamlit as st
from pandas.api.types import is_datetime64_any_dtype, is_object_dtype
from pandas.api.types import CategoricalDtype
import tempfile
from openpyxl import Workbook, load_workbook
import subprocess
subprocess.call(['pip', 'install', '-r', 'https://raw.githubusercontent.com/vkim20016/Project-work/main/requirements.txt'])
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
import replicate
import os

# Define the path for generated embeddings
DB_FAISS_PATH = 'vectorstore/db_faiss'

# Load Excel data using UnstructuredExcelLoader
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

            loader = UnstructuredExcelLoader(file_path=tmp_file_path, mode="elements")
            docs = loader.load()
            doc = docs[0]  # Access the first document in the list
        except Exception as e:
            st.error(f"An error occurred: {e}")
  


# Create embeddings using Sentence Transformers
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})

# Create a FAISS vector store and save embeddings
db = FAISS.from_documents(data, embeddings)
db.save_local(DB_FAISS_PATH)

# Refactored from https://github.com/a16z-infra/llama2-chatbot
# Models and parameters
st.subheader('Models and parameters')
selected_model = st.sidebar.selectbox('Choose a Llama2 model', ['Llama2-7B', 'Llama2-13B', 'Llama2-70B'], key='selected_model')
if selected_model == 'Llama2-7B':
    llm = 'a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea'
elif selected_model == 'Llama2-13B':
    llm = 'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5'
else:
    llm = 'replicate/llama70b-v2-chat:e951f18578850b652510200860fc4ea62b3b16fac280f83ff32282f87bbd2e48'

temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
max_length = st.sidebar.slider('max_length', min_value=64, max_value=4096, value=512, step=8)

# Create a conversational chain
chain = ConversationalRetrievalChain.from_llm(llm=llm, temperature=temperature, top_p=top_p, max_length=max_length)
# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Function for generating LLaMA2 response
def generate_llama2_response(prompt_input):
    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    output = replicate.run(llm, 
                           input={"prompt": f"{string_dialogue} {prompt_input} Assistant: ",
                                  "temperature":temperature, "top_p":top_p, "max_length":max_length, "repetition_penalty":1})
    return output

# User-provided prompt
if prompt := st.chat_input(disabled=not replicate_api):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_llama2_response(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
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
# Set the title for the Streamlit app
st.title("Medical Insights Chatbot")
# Upload Excel file
uploaded_file = st.sidebar.file_uploader("Upload Excel file", type=["xlsx"])
# Replicate Credentials
with st.sidebar:
    if 'REPLICATE_API_TOKEN' in st.secrets:
        st.success('API key already provided!', icon='✅')
        replicate_api = st.secrets['REPLICATE_API_TOKEN']
    else:
        replicate_api = st.text_input('Enter Replicate API token:', type='password')
        if not (replicate_api.startswith('r8_') and len(replicate_api)==40):
            st.warning('Please enter your credentials!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')

