import openai
from openai import AzureOpenAI

# Set your Azure OpenAI config as normal
os.environ["AZURE_OPENAI_API_VERSION"] = "2024-02-01"  # Check here for latest: https://learn.microsoft.com/en-us/azure/ai-services/openai/whats-new
os.environ["AZURE_OPENAI_KEY"] = "API_KEY"

# We prepare our OpenAI payload as usual
messages = [{'role': 'user', 'content': 'Summarize Bristol Myers Squibb in under 10 words.'}]

max_retries = 3 # Keep it reasonable...
for i in range(max_retries):
	endpoint = get_endpoint_details('nonprod', 'gpt-35-turbo')
	# endpoint = get_endpoint_details('nonprod', 'gpt-4-turbo')
	# endpoint = get_endpoint_details('nonprod', 'gpt-4-turbo', '1106-preview')
	azure_endpoint = endpoint['endpoint']
	deployment_name = endpoint['deployment_name']

	client = AzureOpenAI(
		api_version = os.getenv("AZURE_OPENAI_API_VERSION"), 
		api_key = os.getenv("AZURE_OPENAI_KEY"),
		azure_endpoint = azure_endpoint
	)
	try:
		response = client.chat.completions.create(
			model = deployment_name,
			max_tokens = 15,
			messages = messages,
			temperature = 0
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