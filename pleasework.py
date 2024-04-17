import streamlit as st
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Import the text splitter
import tempfile
from pandas.api.types import is_object_dtype, is_datetime64_any_dtype
from pandas.api.types import CategoricalDtype
import pandas as pd
import openpyxl

# Function to filter DataFrame based on selected columns
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

    # Display expanders for each selected column for filtering
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

# Function to convert Excel file to CSV
def excel_to_csv(excel_file):
    # Load Excel file
    wb = openpyxl.load_workbook(excel_file)
    # Assume only one sheet
    sheet = wb.active
    # Create DataFrame from sheet
    df = pd.DataFrame(sheet.values)
    # Write DataFrame to temporary CSV file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        df.to_csv(tmp_file.name, index=False, header=False)
        tmp_file_path = tmp_file.name
    return tmp_file_path

# Sidebar input for OpenAI API key and file upload
user_api_key = st.sidebar.text_input(
    label="#### Your OpenAI API key 👇",
    placeholder="Paste your openAI API key, sk-",
    type="password")

uploaded_file = st.sidebar.file_uploader("Upload", type=["csv", "xlsx"])

# Clear chat history button
clear_history = st.sidebar.button("Clear Chat History")

# Initialize session state variables
if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

# Clear chat history if button is clicked
if clear_history:
    st.session_state['history'] = []
    st.session_state['generated'] = []
    st.session_state['past'] = []

if uploaded_file:
    if uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        # Convert Excel to CSV if uploaded file is Excel
        tmp_file_path = excel_to_csv(uploaded_file)
    else:
        # Store uploaded CSV file directly
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

    # Read the uploaded file into DataFrame
    try:
        df = pd.read_excel(uploaded_file)

        # Allow user to select columns for filtering
        filter_columns = st.sidebar.multiselect("Filter dataframe on", df.columns, key="filter_columns")
        if len(filter_columns) > 0:
            df = filter_dataframe(df, filter_columns)

        # Display DataFrame
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

        # Initialize OpenAI embeddings and FAISS vectors
        embeddings = OpenAIEmbeddings(openai_api_key=user_api_key)
        vectors = FAISS.from_documents(chunks, embeddings)

        # Initialize ConversationalRetrievalChain for chat
        chain = ConversationalRetrievalChain.from_llm(llm=ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo', openai_api_key=user_api_key),
                                                      retriever=vectors.as_retriever())
    except Exception as e:
        # Display error if file reading fails
        st.error(f"An error occurred: {e}")

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
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
                message(response, key=str(i) + '_response', avatar_style="thumbs")