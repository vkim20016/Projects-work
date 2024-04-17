import streamlit as st
import openai
from openai import AzureOpenAI

# Sidebar input for Azure OpenAI API key
user_api_key = st.sidebar.text_input(
    label="#### Your Azure OpenAI API key 👇",
    placeholder="Paste your Azure OpenAI API key here",
    type="password"
)

# We prepare our OpenAI payload as usual
messages = [{'role': 'user', 'content': 'Summarize Bristol Myers Squibb in under 10 words.'}]

max_retries = 3 # Keep it reasonable...
for i in range(max_retries):
    endpoint = get_endpoint_details('nonprod', 'gpt-4-turbo')
    # endpoint = get_endpoint_details('nonprod', 'gpt-4-turbo')
    # endpoint = get_endpoint_details('nonprod', 'gpt-4-turbo', '1106-preview')
    azure_endpoint = endpoint['endpoint']
    deployment_name = endpoint['deployment_name']
    client = AzureOpenAI(
        api_version="2024-02-01", 
        api_key=user_api_key,  # Use the retrieved API key
        azure_endpoint=azure_endpoint
    )

    try:
        response = client.chat.completions.create(
            model=deployment_name,
            max_tokens=15,
            messages=messages,
            temperature=0
        )
        print(response.choices[0].message.content)
        print(f"Tokens Used: {response.usage.total_tokens}")

        break
    except openai.APIConnectionError as e:
        print("The server could not be reached")
        print(e.__cause__)  # an underlying Exception, likely raised within httpx.
        continue
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