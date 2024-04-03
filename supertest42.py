import pandas as pd
import streamlit as st
from pandas.api.types import is_datetime64_any_dtype, is_object_dtype
from pandas.api.types import CategoricalDtype
import tempfile
from openpyxl import Workbook, load_workbook
import subprocess
from datetime import datetime
import tempfile
import os
import pinecone
import sys
from langchain.llms import Replicate
from langchain.vectorstores import Pinecone
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
subprocess.call(['pip', 'install', '-r', 'https://raw.githubusercontent.com/vkim20016/Project-work/main/req1.txt'])
with st.sidebar:
    st.title('🦙💬 Llama 2 Chatbot')
    st.write('This chatbot is created using the open-source Llama 2 LLM model from Meta.')
    if 'REPLICATE_API_TOKEN' in st.secrets:
        st.success('API key already provided!', icon='✅')
        replicate_api = st.secrets['REPLICATE_API_TOKEN']
    else:
        replicate_api = st.text_input('Enter Replicate API token:', type='password')
        if not (replicate_api.startswith('r8_') and len(replicate_api)==40):
            st.warning('Please enter your credentials!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')
    os.environ['REPLICATE_API_TOKEN'] = replicate_api

pinecone.init(api_key='YOUR_API_HERE', environment='YOUR_ENVIRONMENT_HERE')

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

# Upload Excel file
uploaded_file = st.sidebar.file_uploader("Upload Excel file", type=["xlsx"])
if uploaded_file is not None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file_path = f"{tmp_dir}/uploaded_file.xlsx"
        with open(tmp_file_path, "wb") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
        # Transform Excel to CSV
            csv_file_path = f"{tmp_dir}/uploaded_file.csv"
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(uploaded_file.csv)
df = pd.read_excel(tmp_file_path)
df.to_csv(csv_file_path, index=False)    
try:
            df = pd.read_csv(csv_file_path)

            filter_columns = st.sidebar.multiselect("Filter dataframe on", df.columns, key="filter_columns")

            if len(filter_columns) > 0:
                df = filter_dataframe(df, filter_columns)

            st.dataframe(df)
except Exception as e:
            st.error(f"An error occurred: {e}")
            embeddings = HuggingFaceEmbeddings()
            index_name = "YOUR_INDEX_HERE"
index = pinecone.Index(index_name)
vectordb = Pinecone.from_documents(texts, embeddings, index_name=index_name)
llm = Replicate(
    model="a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
    input={"temperature": 0.75, "max_length": 3000}
)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    vectordb.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True
)
chat_history = []
while True:
    query = input('Prompt: ')
    if query == "exit" or query == "quit" or query == "q":
        print('Exiting')
        sys.exit()
    result = qa_chain({'question': query, 'chat_history': chat_history})
    print('Answer: ' + result['answer'] + '\n')
    chat_history.append((query, result['answer']))