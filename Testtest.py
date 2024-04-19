import requests
import random
import os
import openai
from openai import AzureOpenAI
# Download the latest copy
try:
    response = requests.get('https://bms-openai-proxy-eus-prod.azu.bms.com/openai-urls.json')
    bms_openai_urls = response.json()
    
except requests.exceptions.RequestException as e:
    print(f"Unable to retrieve file, {e}")

# Example if you've already saved it locally:
# with open('/local/path/to/saved/openai-urls.json') as file:
# 	bms_openai_urls = json.loads(file.read())print(f"URL={bms_openai_urls['meta']['source']}")
print(f"URL={bms_openai_urls['meta']['source']}")
print(f"Updated={bms_openai_urls['meta']['updated_on']}")
model_deployments = []
for model in bms_openai_urls['nonprod']:
    for deployment in bms_openai_urls['nonprod'][model]:
        if deployment['deployment_name'] not in model_deployments:
            model_deployments.append(deployment['deployment_name'])

print(model_deployments)
for model in bms_openai_urls['prod']:
    print(model)
    path = bms_openai_urls['nonprod']['gpt-35-turbo']
for p in path:
    print(f"Endpoint={p['endpoint']}, Version={p['model_version']}")
    try:
    endpoint = random.choice(bms_openai_urls['nonprod']['gpt-35-turbo'])['endpoint']  # Use this in your Azure OpenAI call...
    print(endpoint)
except KeyError as e:
    print(f"No deployments for model {e}")
environment = 'nonprod'
model_name = 'gpt-35-turbo'
model_version = '0125'

try:
    endpoints = [e for e in bms_openai_urls[environment][model_name] if model_version in e['model_version']]
    
    if not endpoints:
        print(f"No results for combination: {environment}, {model_name}, {model_version}")
    else:
        endpoint = random.choice(endpoints)['endpoint'] # Use this in your Azure OpenAI call...
        print(endpoint)
except KeyError as e:
    print(f"No deployments for model {e}")
    # Example: Function to find prod gpt-4 with model version 0613
# Useful if your app/framework/code accepts a list of endpoints as an argument

# NOTE: This assumes you've already loaded bms_openai_urls into memory...
def get_endpoints(environment, model_name, model_version, num_endpoints_desired=None):
    """
    Return Azure OpenAI endpoints that meet your environment, model, and version inputs.
    If no endpoint is found, return None.
    If num_endpoints_desired exceeds the actual length, return the actual length.

    Parameters
    ------------
    environment: string
        nonprod, prod
    model_name: string
    model-version: string
    num_endpoints_desired: int, Optional
        Defaults to 1
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
                n = len(endpoints) if num_endpoints_desired>len(endpoints) else num_endpoints_desired
                ep_return = []
                for i in range(n):
                    ep_return.append(random.choice(endpoints)['endpoint'])
                return ep_return
    except KeyError as e:
        print(f"No deployments for model {e}")
        return None
# Get 5 endpoints back
endpoints = get_endpoints('prod', 'gpt-4', '0613', 5)
print(endpoints)
# The best way to avoid 429s is to ensure you use the right max_tokens and implement exponential backoff

# Set your Azure OpenAI config as normal
os.environ["AZURE_OPENAI_API_VERSION"] = "2024-02-01"
os.environ["AZURE_OPENAI_KEY"] = "YOUR_BMS_API_KEY"

# We prepare our OpenAI payload as usual
messages = [{'role': 'user', 'content': 'Summarize Bristol Myers Squibb in under 10 words.'}]

max_retries = 3 # Keep it reasonable...
for i in range(max_retries):
	endpoint = get_endpoints('nonprod', 'gpt-35-turbo', '0613')

	client = AzureOpenAI(
		api_version = os.getenv("AZURE_OPENAI_API_VERSION"), 
		api_key = os.getenv("AZURE_OPENAI_KEY"),
		azure_endpoint = endpoint
	)

	try:
		response = client.chat.completions.create(
			model='gpt-35-turbo',  # make sure to use the 'deployment_name' from the endpoint file 
			max_tokens= 15,
			messages= messages,
			temperature= 0
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