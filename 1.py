import streamlit as st
import openai
import os
import json
import random
import time
from functools import lru_cache
from datetime import datetime
import requests
import tempfile
import pandas as pd
import openpyxl
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pandas.api.types import is_object_dtype, is_datetime64_any_dtype
from pandas.api.types import CategoricalDtype

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

# Retrieve BMS OpenAI URLs
OPENAI_URLS_CACHE_PATH = "set_your_own_cache_path.json"  # '/local/path/to/saved/openai-urls.json'
OPENAI_URLS_REMOTE_PATH = os.environ.get("OPENAI_URLS_REMOTE_PATH", "https://bms-openai-proxy-eus-prod.azu.bms.com/openai-urls.json")
OPENAI_URLS_REFRESH_DAYS = 7
user_api_key = st.sidebar.text_input(
    label="#### Your Azure OpenAI API key 👇",
    placeholder="Paste your Azure OpenAI API key here",
    type="password"
)


@lru_cache(maxsize=2)
def get_openai_urls(verbose=False, ttl_hash=None):
    """
    Retrieve BMS OpenAI URLs from a remote source if the local cache is outdated or not available.

    This function first checks if a local cache file exists at the specified path (OPENAI_URLS_CACHE_PATH).
    If the cache file exists and is up-to-date, the function reads the URLs from the cache file.
    If the cache file is outdated or doesn't exist, the function downloads the URLs from the remote source
    (OPENAI_URLS_REMOTE_PATH) and updates the local cache file.

    Returns:
        dict: A dictionary containing BMS OpenAI URLs and metadata.
    """
    del ttl_hash  # to prevent linters from getting angry

    # Try to load a cached file
    bms_openai_urls = {}
    if os.path.isfile(OPENAI_URLS_CACHE_PATH):
        try:
            with open(OPENAI_URLS_CACHE_PATH) as file:
                bms_openai_urls = json.loads(file.read())
            print(f"Loaded OpenAI URLs from cache") if verbose else None
        except json.JSONDecodeError as jse:
            print(f"Unable to read existing cache file, will attempt to retrieve new file. {jse}")

    # If cache unavailable or outdated, download a new copy
    last_updated_date = bms_openai_urls.get('meta', {}).get('updated_on', "2024-01-01T00:00:00Z")
    days_since_last_update = (datetime.now() - datetime.strptime(last_updated_date, "%Y-%m-%dT%H:%M:%SZ")).days
    print(f"It's been {days_since_last_update} days since last refresh of OpenAI URLs") if verbose else None
    if not bms_openai_urls or days_since_last_update >= OPENAI_URLS_REFRESH_DAYS:
        try:
            # Download the latest copy
            response = requests.get(bms_openai_urls.get('meta', {}).get('source', OPENAI_URLS_REMOTE_PATH))
            bms_openai_urls = response.json()
            if os.path.dirname(OPENAI_URLS_CACHE_PATH):
                os.makedirs(os.path.dirname(OPENAI_URLS_CACHE_PATH), exist_ok=True)
            with open(OPENAI_URLS_CACHE_PATH, "w") as file:
                json.dump(bms_openai_urls, file, indent=4)
            print(f"Downloaded new copy of OpenAI URLs") if verbose else None
        except requests.exceptions.RequestException as e:
            print(f"Unable to retrieve and cache BMS OpenAI URLs listing file, {e}")

    return bms_openai_urls


def get_ttl_hash(seconds=60 * 60):
    """Returns the same value within `seconds` time period"""
    return round(time.time() / seconds)


def get_endpoint_details(environment, model_name, model_version=None, num_endpoints_desired=None):
    """
    Return Azure OpenAI endpoint details that meet your environment, model, and version inputs.
    If no endpoint is found, return None.
    If no model_version specified, returns all available
    If num_endpoints_desired exceeds the actual length, return all matching endpoints.

    Arguments:
        environment: string
            nonprod, prod
        model_name: string
        model_version: string
        num_endpoints_desired: int, Optional
            Defaults to 1

    Returns:
        dict (or list of dicts) containing keys like 'endpoint' and 'deployment_name'
    """

    try:
        bms_openai_urls = get_openai_urls(ttl_hash=get_ttl_hash())
        endpoints = [e for e in bms_openai_urls[environment][model_name]]
        if model_version is not None:
            endpoints = [e for e in endpoints if model_version in e['model_version']]

        if not endpoints:
            print(f"No results for combination: {environment}, {model_name}, {model_version}")
            return None
        else:
            if not num_endpoints_desired:
                return random.choice(endpoints)
            else:
                n = len(endpoints) if num_endpoints_desired > len(endpoints) else num_endpoints_desired
                ep_return = []
                for i in range(n):
                    ep_return.append(random.choice(endpoints))
                return ep_return
    except KeyError as e:
        print(f"No deployments for model {e}")
        return None


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
    except Exception as e:
        st.write("An error occurred:", e)

# Set your Azure OpenAI config as normal
os.environ["AZURE_OPENAI_API_VERSION"] = "2024-02-01"  # Check here for latest: https://learn.microsoft.com/en-us/azure/ai-services/openai/whats-new
os.environ["AZURE_OPENAI_KEY"] = "YOUR_BMS_API_KEY"

# We prepare our OpenAI payload as usual
messages = [{'role': 'user', 'content': 'Summarize Bristol Myers Squibb in under 10 words.'}]

max_retries = 3  # Keep it reasonable...
for i in range(max_retries):
    try:
        endpoint = get_endpoint_details('nonprod', 'gpt-35-turbo')
        # endpoint = get_endpoint_details('nonprod', 'gpt-4-turbo')
        # endpoint = get_endpoint_details('nonprod', 'gpt-4-turbo', '1106-preview')
        azure_endpoint = endpoint['endpoint']
        break
    except openai.RateLimitError as e:
        # NOTE: You should add exponential backoff or a sleep here...
        print("HTTP 429 received, trying another endpoint.")
        print(e)
        continue
    except openai.APIStatusError as e:
        print(e)
        continue
else:
    print("Azure OpenAI call failed")
