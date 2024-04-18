import streamlit as st
import openai
from openai import AzureOpenAI
import os
import json
import random
import time
from functools import lru_cache
from datetime import datetime
import requests

# Retrieve BMS OpenAI URLs
OPENAI_URLS_CACHE_PATH = "set_your_own_cache_path.json" # '/local/path/to/saved/openai-urls.json'
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


def get_ttl_hash(seconds=60*60):
    """Returns the same value withing `seconds` time period"""
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
                n = len(endpoints) if num_endpoints_desired>len(endpoints) else num_endpoints_desired
                ep_return = []
                for i in range(n):
                    ep_return.append(random.choice(endpoints))
                return ep_return
    except KeyError as e:
        print(f"No deployments for model {e}")
        return None