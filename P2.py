import requests
import random
import os
import openai
import streamlit as st
import pandas as pd
import openpyxl
import tempfile
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Function to convert Excel file to CSV
def excel_to_csv(excel_file):
    wb = openpyxl.load_workbook(excel_file)
    sheet = wb.active
    df = pd.DataFrame(sheet.values)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        df.to_csv(tmp_file.name, index=False, header=False)
        tmp_file_path = tmp_file.name
    return tmp_file_path

# Function to retrieve endpoints
def get_endpoints(environment, model_name, model_version, num_endpoints_desired=None):
    """
    Return Azure OpenAI endpoints that meet your environment, model, and version inputs.
    If no endpoint is found, return None.
    If num_endpoints_desired exceeds the actual length, return the actual length.
    """
    try:
        endpoints = [e for e in bms_openai_urls[environment][model_name] if model_version in e['model_version']]

        if not endpoints:
            print(f"No results for combination: {environment}, {model_name}, {model_version}")
            return None
        else:
            if not num_endpoints_desired:
                return random.choice(endpoints)['endpoint']
            else:
                n = len(endpoints) if num_endpoints_desired > len(endpoints) else num_endpoints_desired
                ep_return = []
                for i in range(n):
                    ep_return.append(random.choice(endpoints)['endpoint'])
                return ep_return
    except KeyError as e:
        print(f"No deployments for model {e}")
        return None

# Download the latest copy
try:
    response = requests.get('')
    bms_openai_urls = response.json()
except requests.exceptions.RequestException as e:
    print(f"Unable to retrieve file, {e}")

# Initialize session state variables
if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []

# Sidebar input for Azure OpenAI API key and file upload
user_api_key = st.sidebar.text_input(
    label="#### Your Azure OpenAI API key 👇",
    placeholder="Paste your Azure OpenAI API key",
    type="password")

uploaded_file = st.sidebar.file_uploader("Upload", type=["csv", "xlsx"])

# Clear chat history button
clear_history = st.sidebar.button("Clear Chat History")

# Process uploaded file
if uploaded_file:
    # Convert Excel to CSV if uploaded file is Excel
    if uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        tmp_file_path = excel_to_csv(uploaded_file)
    else:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

    try:
        df = pd.read_excel(uploaded_file)
        st.dataframe(df)

        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=100,
            length_function=len,
            add_start_index=True,
        )

        # Load data into LangChain and split into chunks
        loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8")
        data = loader.load()
        chunks = text_splitter.split_documents(data)

        # Initialize Azure OpenAI embeddings and FAISS vectors
        embeddings = AzureOpenAIEmbeddings(api_key=user_api_key)
        vectors = FAISS.from_documents(chunks, embeddings)

        # Initialize ConversationalRetrievalChain for chat
        chain = ConversationalRetrievalChain.from_llm(llm=AzureChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo', api_key=user_api_key), retriever=vectors.as_retriever())
    except Exception as e:
        # Display error if file reading fails
        st.error(f"An error occurred: {e}")

# Clear chat history if button is clicked
if clear_history:
    st.session_state['history'] = []
    st.session_state['generated'] = []
    st.session_state['past'] = []

# Function for conversational chat
def conversational_chat(query):
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    st.session_state['generated'].append(result["answer"])

    return result["answer"]

# Container for the chat history
response_container = st.container()
# Container for the user's text input
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        # Text input for user query
        user_input = st.text_input("Text Box:", placeholder="Ask Here", key='input')
        # Submit button
        submit_button = st.form_submit_button(label='Send')

    # If user submits a query
    if submit_button and user_input:
        output = conversational_chat(user_input)
        # Append user input to chat history
        st.session_state['past'].append(user_input)

# Display chat history and responses
if st.session_state['generated']:
    with response_container:
        for i, response in enumerate(st.session_state['generated']):
            st.write(st.session_state["past"][i], " (User)")
            st.write(response, " (Model)")