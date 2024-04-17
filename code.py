Here is the streamlit app code:
import json
import os
import re
from threading import Thread
import uuid
from datetime import date, datetime
from collections import defaultdict
from opensearchpy import OpenSearch
import openai
import tiktoken
import logging
import time
 
 
# Get the current value of LD_LIBRARY_PATH
current_ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
# Update LD_LIBRARY_PATH with the /opt/conda/lib
new_ld_library_path = '/opt/conda/lib'
os.environ['LD_LIBRARY_PATH'] = new_ld_library_path + ':' + current_ld_library_path
 
from opensearchpy import OpenSearch, RequestsHttpConnection
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer
import streamlit as st
from streamlit.web.server.websocket_headers import _get_websocket_headers
import streamlit.components.v1 as components
from urllib3.util import Retry
 
 
INDICATION = "config/indications.txt"
TA = "config/therapeutic_area.txt"
PRODUCT = "config/products.txt"
COUNTRY = "config/country.txt"
KIT = "config/kits.txt"
DOC_TYPE = "config/document_type.txt"
CAUSAL_LM = './synthia_7b_med'
EMB_LM = 'BAAI/bge-base-en-v1.5'
default_model = "gpt-4-32k"
MODELS = {
          "GPT-3.5": "gpt-35-turbo-16k",
          "GPT-4": "gpt-4-32k",
          "BMS Mistral": "bms_mistral"}
MODELS_DROPDOWN = ["GPT-4",
                   "GPT-3.5"]
 
DOMINO_DATASET = "/domino/datasets/local/MEG/application_feedback/"
 
MAX_TOKENS = 10000
MAX_NEW_TOKENS = 2000
DEFAULT_SYSTEM = "Elaborate on the topic using the context provided. Always answer without hesitation. Use chain-of-thought reasoning, although don't mention the concept of chain-of-thought reasoning in your responses. Include source citations when possible. Be concise and direct."
 
# ---------- Access Credentials for OpenAI ----------
# [OpenAI Setup]
openai.api_key = os.environ['OPENAI_API_KEY']
openai.api_type = os.environ['OPENAI_API_TYPE']
openai.api_base = os.environ['OPENAI_API_BASE']
openai.api_version = os.environ['OPENAI_API_VERSION']
 
http_headers = _get_websocket_headers()
domino_user = http_headers.get("domino-username")
if not domino_user:
    domino_user = http_headers.get("Domino-Username")
if not domino_user:
    domino_user = "**unknown_user**"
 
 
#def filter_dataframe_by_kits(dataframe, selected_kits):
#    """
#    Filters the DataFrame based on the selected KITs.
#    Args:
#        dataframe: The DataFrame to be filtered.
#        selected_kits: A list of KITs to filter by.
#    Returns:
#        A filtered DataFrame containing only the rows that match the selected KITs.
#    """
#    kits_column = 'key_insight_topic'
#    filtered_data = dataframe[dataframe[kits_column].apply(lambda x: any(kit in x.split(',') for kit in selected_kits))]
#    return filtered_data
 
def _calculate_retry_delay(exception, attempt):
    """
    Calculate the delay before retrying the API call.
    Args:
        exception: The exception raised during the API call.
        attempt: The current retry attempt number.
 
    Returns:
        int: The delay time in seconds.
    """
    default_delay = 25  # default delay in seconds
    exponential_backoff_factor = 2  # to increase delay time exponentially
    max_delay = 300  # maximum delay time in seconds
 
    # Use the retry-after header from the response, if available
    if hasattr(exception, 'response') and 'Retry-After' in exception.response.headers:
        return int(exception.response.headers['Retry-After'])
    else:
        return min(default_delay * (exponential_backoff_factor ** attempt), max_delay)
 
def get_completion(prompt, model="gpt-4-32k", max_tokens=10000, max_retries=100):
    """
    Get completion from OpenAI API with retry logic for handling rate limits.
 
    Args:
        prompt (str): The prompt to send to the API.
        model (str): The model to use for the completion.
        max_tokens (int): The maximum number of tokens to generate.
        max_retries (int): The maximum number of retry attempts.
 
    Returns:
        str: The completion response from the API or None in case of failure.
    """
    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                engine=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=max_tokens
            )
            return response.choices[0].message["content"]
        except openai.error.RateLimitError as e:
            retry_delay = _calculate_retry_delay(e, attempt)
            logging.warning(f"Rate limit exceeded for prompt '{prompt[:30]}...'. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
        except openai.error.OpenAIError as e:
            logging.error(f"OpenAI API call failed for prompt '{prompt[:30]}...': {e}")
            if attempt >= max_retries - 1:
                logging.error("Max retries reached. API call failed.")
                return None
 
def count_tokens(text, model="cl100k_base"):
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(text))
    return num_tokens
 
def ChangeButtonColour(widget_label, font_color, background_color='transparent'):
    """
    Change the font color and background color of a button with a specific label.
 
    This function generates a JavaScript code snippet that is used to change the font color and background color
    of a button with a specific label in a web page. The JavaScript code is then used to inject the script
    into the web page.
 
    Args:
       widget_label: The label of the button to change.
       font_color: The new font color for the button.
       background_color: The new background color for the button. Default is 'transparent'.
 
    Returns:
       None
    """
    htmlstr = f"""
<script>
            var elements = window.parent.document.querySelectorAll('button');
            for (var i = 0; i < elements.length; ++i) {{ 
                if (elements[i].innerText == '{widget_label}') {{ 
                    elements[i].style.color ='{font_color}';
                    elements[i].style.background = '{background_color}';
                    elements[i].style.border = '{background_color}';
                }}
            }}
</script>
        """
    components.html(f"{htmlstr}", height=0, width=0)

 
def read_configuration_file(filename):
    """
    Reads a configuration file and returns its content as a list of unique, stripped strings.
 
    Reads a file specified by 'filename' parameter, reads all lines in the file, removes duplicates by 
    converting the lines to a set, and then converts back to a list of strings. Then, it strips 
    leading and trailing whitespaces or newline characters from each line in the list.
 
    Args:
       filename: The name of the file to read.
 
    Returns:
       A list of unique, stripped strings read from the file.
    """
    with open(filename, 'r') as fr:
        data = fr.readlines()
    data = list(set(data))
    for i in range(len(data)):
        data[i] = data[i].strip()
    data.sort(key=str.lower)
    return data
 
# Set Streamlit page configuration with sidebar initially hidden
st.set_page_config(
   page_title="Medical Evidence Generation Proof of Concept",
   page_icon="⚕️",
   layout="wide",
   initial_sidebar_state="collapsed"  # Ensures sidebar is collapsed by default
)
 
 
@st.cache_resource()
def load_models():
    """Function to load the models into the streamlit application"""
    emb_tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-base-en-v1.5')
    emb_model = AutoModel.from_pretrained('BAAI/bge-base-en-v1.5')
    emb_model.to('cpu')
    oai_encoding = tiktoken.encoding_for_model("gpt-4")
 
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,)
 
    causal_model = AutoModelForCausalLM.from_pretrained(
        CAUSAL_LM,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",)
    causal_tokenizer = AutoTokenizer.from_pretrained(CAUSAL_LM)
    return causal_model, causal_tokenizer, emb_model, emb_tokenizer, oai_encoding
 
 
causal_model, causal_tokenizer, emb_model, emb_tokenizer, oai_encoding = load_models()
 
 
def generate_embeddings(text, model, tokenizer):
    """Method to embed a text document.
 
    Args:
        text (str): String value of document to be embedded
        tokenizer (obj): HF Tokenizer object
        model(obj): HF Model object
 
    Returns:
        list: 512 Dimensional Torch Tensor object as a list
    """
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt').to("cpu")
    with torch.no_grad():
        model_output = model(**encoded_input)
        # Perform pooling. In this case, cls pooling.
        sentence_embeddings = model_output[0][:, 0]
        # normalize embeddings
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings.tolist()[0]
 
 
def generate_prompt(instruction, context=None, system=DEFAULT_SYSTEM):
    """ Function to create the appropriate prompt template."""
    if context:
        return f"SYSTEM: {system}\nUSER: {instruction} CONTEXT: {context}\nASSISTANT: "
    else:
        return f"SYSTEM: {system}\nUSER: {instruction} \nASSISTANT: "
 
 
def generate_response(prompt, model_choice, system_prompt=DEFAULT_SYSTEM, max_new_tokens=MAX_NEW_TOKENS):
    """ Generate a stream of text based on a prompt.
 
    This function generates a stream of text based on a given prompt. It uses a pre-trained model and tokenizer to generate the text. The generated text is then streamed to the user in real-time.
 
    Args: 
         prompt (str): The prompt for the generation. 
         max_new_tokens (int): The maximum number of new tokens to generate. Default is 512.
 
    Returns: 
           str: The generated text. """
    model_selection = MODELS[model_choice]
    if model_selection == "bms_mistral":
        prompt = create_prompt(prompt, st.session_state.context, causal_tokenizer, max_tokens=MAX_TOKENS, generation_len=MAX_NEW_TOKENS, system_prompt=system_prompt)
        inputs = causal_tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        streamer = TextIteratorStreamer(causal_tokenizer, skip_prompt=True)
        generation_kwargs = dict(inputs=inputs,
                             streamer=streamer,
                             max_new_tokens=max_new_tokens,
                             eos_token_id=causal_tokenizer.eos_token_id,
                             early_stopping=True,
                             repetition_penalty=1.5
                            )
        thread = Thread(target=causal_model.generate, kwargs=generation_kwargs)
        thread.start()
        generated_text = ""
        with st.empty():
            for idx, new_text in enumerate(streamer):
                generated_text += new_text
                generated_text = re.sub("<\s>", "", generated_text)
                st.write(generated_text)
        return generated_text
    else:
        prompt = create_prompt(prompt, st.session_state.context, oai_encoding, max_tokens=MAX_TOKENS, generation_len=max_new_tokens, system_prompt=system_prompt)
        generated_text = get_completion(prompt, model=model_selection, max_tokens=max_new_tokens)
        st.write(generated_text)
        return generated_text

 
def extract_source_to_df(response):
    """
    Extracts the source from the search response and converts it to a Pandas DataFrame.
    Args:
        response (dict): The search response from OpenSearch.
    Returns:
        pd.DataFrame: The Pandas DataFrame of the extracted source.
    """
    hits = response['hits']['hits']
    records = []
    cols = [
            "product_name",
            "insight_date",
            "indication",
            "country",
            "therapeutic_area",
            "insight_content_text",
            "insight_about_text",
#            "keywords",
#            "topic",
            "insight_id",
            "insight_text",
            "document_type",
            "file_name"
            ]                
    for i in hits:
        i['_source']['score'] = i["_score"]
        records.append(i['_source'])
    df = pd.DataFrame(records, columns=cols)
    return df
 
 
def normalize_bm25(score, max_score):
    """Helper function to normalize BM25 scores to compare with KNN scores."""
    return score / max_score

 
def merge(dict1, dict2):
    """Function to merge two dictionaries"""
    return (dict2.update(dict1))

 
def get_min_score(common_elements, elements_dictionary, min_score=0.01):
    if len(common_elements):
        return min([min(v) for v in elements_dictionary.values()])
    else:
        # No common results - assign arbitrary minimum score value
        return 0.01
 
 
def combine_knn_bm25(knn_docs, bm25_docs):
    """
    Combines KNN and BM25 documents based on their IDs.
 
    This function takes two parameters: knn_docs and bm25_docs, which are
    dictionaries expected to have "hits" key pointing to another
    dictionary with "hits" and "max_score". This function processes
    these documents to calculate a combined score for each common
    document in both knn_docs and bm25_docs.
 
    It starts by extracting all the document IDs from both knn_docs
    and bm25_docs. Then, it finds the common documents and processes
    them by calculating their combined score. If a document is only
    present in either knn_docs or bm25_docs, it will be given a
    score of min_value, which is the minimum score from the common
    documents.
 
    Args:
       knn_docs (Dict): A dictionary with KNN documents.
       bm25_docs (Dict): A dictionary with BM25 documents.
 
    Returns:
       Dict: A dictionary with the combined scores for each document.
    """
    bm25_ids = []
    knn_ids = []
    results_dict = defaultdict(list)
    knn_hits = knn_docs["hits"]["hits"]
    bm25_hits = bm25_docs["hits"]["hits"]
    bm25_max_score = bm25_docs["hits"]["max_score"]
    for knn_hit in knn_hits:
        knn_ids.append(knn_hit["_id"])
    for bm25_hit in bm25_hits:
        bm25_ids.append(bm25_hit["_id"])
    common_documents = set(bm25_ids) & set(knn_ids)
    for common_doc in common_documents:
        for idx, knn_hit in enumerate(knn_hits):
            if knn_hit["_id"] == common_doc:
                results_dict[common_doc].append(knn_hit["_score"])
        for idx, bm25_hit in enumerate(bm25_hits):
            if bm25_hit["_id"] == common_doc:
                results_dict[common_doc].append(normalize_bm25(bm25_hit["_score"], bm25_max_score))
    min_value = get_min_score(common_documents, results_dict)
    for knn_hit in knn_hits:
        if knn_hit["_id"] not in common_documents:
            new_scored_id = knn_hit["_id"]
            results_dict[new_scored_id] = [min_value]
    for bm25_hit in bm25_hits:
        if bm25_hit["_id"] not in common_documents:
            new_scored_id = bm25_hit["_id"]
            results_dict[new_scored_id] = [min_value]
    return results_dict

 
def apply_boosting(results_dict, vector_boost, bm25_boost):
    """Helper function to apply vector and keyword boosting
    Args:
        results_dict (dict): results dictionary with id as key and scores as a list value to each key
        vector_boost (float): desired boost amount for knn
        bm25_boost (float): desired boost amount for bm25
    Returns:
        list: sorted results
    """
    def boost(list_values):
        if len(list_values) == 1:
            scalar = list_values[0] * (vector_boost + bm25_boost)
        else:
            scalar = list_values[0] * vector_boost + list_values[1] * bm25_boost
        return scalar
    results_dict = {k: boost(v) for k,v in results_dict.items()}   
    sorted_results = [k for k, v in sorted(results_dict.items(), key=lambda item: item[1], reverse=True)]
    return sorted_results
 
def filter_recall(search_query, product, therapeutic_area, country, indication, document_type, date, emb_model, emb_tokenizer, search_size=25, index='meg', vector_boost=1.0, bm25_boost=1.0):
    """
    Perform filter-based recall.
    Args:
        query (str): The query to be searched.
        index_definition (str): The index definition.
        cross_encoder (object): The cross-encoder model.
        search_size (int, optional): The number of documents to be retrieved. Defaults to 25.
        index (str, optional): The index to be searched. Defaults to "earnings".
    Returns:
        pd.DataFrame: The dataframe of retrieved documents.
    """
 
    # Configure urllib3 Retry
    retries = Retry(
        total=10,  # Total number of retries
        backoff_factor=1,  # Backoff factor for subsequent retries
        status_forcelist=[500, 502, 503, 504]  # HTTP status codes to trigger a retry
    )
 
    os_username = "username"
    os_password = "password"
    host = "host"
    port = 443
    auth = (os_username, os_password)
    client = OpenSearch(
        hosts=[{'host': host, 'port': port}],
        http_auth=auth,
        http_compress = True,
        use_ssl = True,
        timeout=60,
        connection_class=RequestsHttpConnection,
        max_retries=10
    )
    search_query = f"Represent this sentence for searching relevant passages: {search_query}"
    query_embeddings = generate_embeddings(search_query, emb_model, emb_tokenizer)
    # Initialize the query with a match_all clause
    bm25_query = {
        "size": search_size,
        "query": {
            "bool": {
                "should": [
                    {"multi_match": {
                        "query":search_query,
                        "fields": [
                            "insight_text",
                            ],
                    }
                }
                ],
                "filter": []
            }
        }
    }
    knn_query = {
        "size": search_size,
        "query": {
            "bool": {
                "should": [
                    {"script_score": {
                        "query": {
                            "match_all": {}
                        },
                        "script": {
                            "source": "knn_score",
                            "lang": "knn",
                            "params": {
                                "field": "text_vector",
                                "query_value": query_embeddings,
                                "space_type": "cosinesimil"
                            }
                        }
                    }
                }
                ],
                "filter": []
            }
        }
    }
 
    def non_empty_values(values):
        return [value for value in values if value]
    def filter_date(date_range):
        """
        Filters the date range to ensure that both dates are valid and that the start date is 
        not greater than the end date. Returns a formatted date range or None if invalid.
 
        Args:
            date_range (tuple): Tuple containing the start and end dates.
 
        Returns:
            dict: Dictionary with a valid date range for the OpenSearch query, or None.
        """
        if date_range and len(date_range) == 2:
            start_date, end_date = date_range
            # Check if both dates are not None and that start_date is less than or equal to end_date
            if start_date and end_date and start_date <= end_date:
                start_date_str = start_date.strftime('%Y-%m-%d')
                end_date_str = end_date.strftime('%Y-%m-%d')
                return {
                    'range': {
                        'insight_date': {
                            'gte': start_date_str, # gte: greater than or equal to
                            'lte': end_date_str    # lte: less than or equal to
                        }
                    }
                }
        # Return None if the date_range is invalid
        return None
    # Modified the following `if` blocks to check for non-empty lists
    if indication:
        indications = non_empty_values(indication)
        if indications:
            bm25_query['query']['bool']['filter'].append({'terms': {'indication.keyword': [ind for ind in indication if ind]}})
            knn_query['query']['bool']['filter'].append({'terms': {'indication.keyword': [ind for ind in indication if ind]}})
    if country:
        countries = non_empty_values(country)
        if countries:
            bm25_query['query']['bool']['filter'].append({'terms': {'country.keyword': [c for c in country if c]}})
            knn_query['query']['bool']['filter'].append({'terms': {'country.keyword': [c for c in country if c]}})
    if therapeutic_area:
        tas = non_empty_values(therapeutic_area)
        if tas:
            bm25_query['query']['bool']['filter'].append({'terms': {'therapeutic_area.keyword': [ta for ta in therapeutic_area if ta]}})
            knn_query['query']['bool']['filter'].append({'terms': {'therapeutic_area.keyword': [ta for ta in therapeutic_area if ta]}})
    if product:
        products = non_empty_values(product)
        if products:
            bm25_query['query']['bool']['filter'].append({'terms': {'product_name.keyword': [prd for prd in product if prd]}})
            knn_query['query']['bool']['filter'].append({'terms': {'product_name.keyword': [prd for prd in product if prd]}})
    if document_type:
        dtypes = non_empty_values(document_type)
        if dtypes:
            bm25_query['query']['bool']['filter'].append({'terms': {'document_type.keyword': [dtyp for dtyp in document_type if dtyp]}})
            knn_query['query']['bool']['filter'].append({'terms': {'document_type.keyword': [dtyp for dtyp in document_type if dtyp]}})
    date_filter = filter_date(date)
    if date_filter:
        bm25_query['query']['bool']['filter'].append(date_filter)
        knn_query['query']['bool']['filter'].append(date_filter)

    if search_query.strip() and (indication or country or therapeutic_area or product or document_type or (date and date[0] and date[1])):
        bm25_docs = client.search(request_timeout=60, index=index, body=bm25_query)
        knn_docs = client.search(request_timeout=60, index=index, body=knn_query)
        results_dict = combine_knn_bm25(knn_docs, bm25_docs)
        sorted_results = apply_boosting(results_dict, vector_boost, bm25_boost)
        knn_hits = knn_docs["hits"]["hits"]
        bm25_hits = bm25_docs["hits"]["hits"]
        merged_hits = defaultdict(list)
        for hit in knn_hits:
            doc_id = hit["_id"]
            source = hit["_source"]
            merged_hits[doc_id] = source
        for hit in bm25_hits:
            doc_id = hit["_id"]
            source = hit["_source"]
            merged_hits[doc_id] = source
        sorted_values = []
        sorted_context = []
        for element in sorted_results:
            sorted_values.append(merged_hits[element])
            sorted_context.append(merged_hits[element]["insight_text"])
    else:
        raise ValueError(#"Invalid search query and no valid search criteria provided"
                        "Thanks for your query"
                        )
    return sorted_values, sorted_context
 
# Helper for development inside Domino Workspace
project_username = os.environ['DOMINO_PROJECT_OWNER']  # or is it DOMINO_STARTING_USERNAME?
project_name = os.environ['DOMINO_PROJECT_NAME']
workspace_run_id = os.environ['DOMINO_RUN_ID']
streamlit_port = 8501
streamlit_app_url = f"""https://domino.web.bms.com/{project_username}/{project_name}/notebookSession/{workspace_run_id}/proxy/{streamlit_port}/"""
print(f"\n\nStreamlit app starting. If running in a Domino Workspace, access it via:\n{streamlit_app_url}")
 
 
def count_tokens(text, tokenizer):
    """Function to count total number of tokens with the tokenizer"""
    return len(tokenizer.encode(text))
 
 
def create_prompt(query, dfContext, tokenizer, max_tokens=MAX_TOKENS, generation_len=MAX_NEW_TOKENS, system_prompt=DEFAULT_SYSTEM, assess=False):
    used_tokens = count_tokens(f"SYSTEM: {system_prompt}\nUSER: Based on the information below enclosed in <> answer this question: {query}  <>\nASSISTANT: ", tokenizer)
    tokens_left = max_tokens - generation_len - used_tokens
    current_context_count = 0
    context = []
    if isinstance(dfContext, list):
        prompt = format_prompt(query, None, system=system_prompt)
        return prompt
 
    for index, row in dfContext.iterrows():
        proposed_context = f"{row['insight_text']}\n"
        content_len = count_tokens(proposed_context, tokenizer)
        current_context_count += content_len
        if current_context_count <= tokens_left:
            context.append(proposed_context)
        else:
            break
    context = list(set(context))
    context = ''.join(context)
    st.session_state.context_list = context
    prompt = format_prompt(query, context, system=system_prompt)
    return prompt
 
def format_prompt(instruction, context=None, system=DEFAULT_SYSTEM):
    if context:
        return f"SYSTEM: {system}\nUSER: {instruction}\nCONTEXT:\n{context}\nASSISTANT: "
    else:
        return f"SYSTEM: {system}\nUSER: {instruction}\nASSISTANT: "
 
    
def assess_query(instruction, dfContext, system=DEFAULT_SYSTEM):
    previous_message = st.session_state.messages[-2]['content']
    tokenizer = oai_encoding
    used_tokens = count_tokens(f"SYSTEM: {system} \nUSER:Determine if the information enclosed in <> relates to the provided context or previous message if one exists. <{instruction}>\nCONTEXT: \n\nPREVIOUS MESSAGE:{previous_message}\n Respond 'Yes' or 'No'" , tokenizer)
    tokens_left = MAX_TOKENS - MAX_NEW_TOKENS - used_tokens
    current_context_count = 0
    context = []
    if isinstance(dfContext, list):
        # No context, not relevant
        return False
    for index, row in dfContext.iterrows():
        # Format context
        proposed_context = f"{row['insight_text']}\n"
        content_len = count_tokens(proposed_context, tokenizer)
        current_context_count += content_len
        if current_context_count <= tokens_left:
            context.append(proposed_context)
        else:
            break
    context = list(set(context))
    context = ''.join(context)
    prompt = f"SYSTEM: {system} \nUSER:Determine if the information enclosed in <> relates to the provided context or previous message if one exists. <{instruction}>\nCONTEXT: \n{context}\nPREVIOUS MESSAGE:{previous_message}\n Respond 'Yes' or 'No'" 
    assessment =  get_completion(prompt, model=MODELS.get("GPT-4", "default_model"), max_tokens=512)
    if assessment == 'Yes': 
        st.warning("Continuing with the conversation")
        return True
    else: 
        st.warning("Researching the new topic")
        return False
 
    
def perform_content_search(query, size, prod, ta, country, indic, doc_type, date#, 
                           #keywords, topic
                          ):
    if not query:
        st.warning("Please enter query text before continuing.")
        return
 
    if size < 100:
        size = 100
    sorted_values, sorted_context = filter_recall(query,
                                                  prod,
                                                  ta,
                                                  country,
                                                  indic,
                                                  doc_type,
                                                  date,
#                                                  keywords,
#                                                  topic,
                                                  emb_model,
                                                  emb_tokenizer,
                                                  search_size=size)
    df = pd.DataFrame(sorted_values)
    df = df.loc[df.astype(str).drop_duplicates().index]
    st.session_state.context = df
    return
 
def save_prompt(username, prompt, context, response, location=DOMINO_DATASET):
    temp_dict = {"user": username,
                "prompt": prompt,
                "context": context,
                "response": response}
    filename = f"{DOMINO_DATASET}{str(uuid.uuid4())}.json"
    with open(filename, 'w') as f:
        json.dump(temp_dict, f)
 
 
# -- Application Interface --
 
indications = read_configuration_file(INDICATION)
therapeutic_area = read_configuration_file(TA)
products = read_configuration_file(PRODUCT)
countries = read_configuration_file(COUNTRY)
document_type = read_configuration_file(DOC_TYPE)
#key_insight_topic = read_configuration_file(KIT)
 
search_results = st.expander("Search Results")
chat_interface = st.container()
 
# Create context search
size = st.sidebar.number_input("Total Desired Visible Search Results", value=25, min_value=1, max_value=150)
prod = st.sidebar.multiselect("Product", options=products)
ta = st.sidebar.multiselect("Therapeutic Area", options=therapeutic_area)
indic = st.sidebar.multiselect("Indication", options=indications)
country = st.sidebar.multiselect("Country", options=countries)
doc_type = st.sidebar.multiselect("Document Type", options=document_type)
#kit = st.sidebar.multiselecty("Key Insight Topic", options=key_insight_topic)
 
scol1, scol2 = st.sidebar.columns([5, 7])
with scol1: 
    date1 = st.date_input("First date", None, format="MM/DD/YYYY")
with scol2:
    date2 = st.date_input("Second date", None, format="MM/DD/YYYY")
if not date1 and date2:
    date = None
date = [date1, date2]
 
#filtered_kits_df = filter_dataframe_by_kits(df, kit)                                                                                                  
model = st.sidebar.selectbox("Model", options=MODELS_DROPDOWN)
show_sys_prompt = st.sidebar.checkbox("Show System Prompt", value=False)
if show_sys_prompt:
    system_prompt = st.sidebar.text_area("System Prompt", value=DEFAULT_SYSTEM, placeholder=DEFAULT_SYSTEM)
else:
    system_prompt = DEFAULT_SYSTEM
 
st.sidebar.markdown("Please reach out to Sam Meyer (samuel.meyer@bms.com) with any requests, feedback, or suggestions.")
 
ChangeButtonColour("Search", background_color='#be2bbb', font_color='#ffffff')
 
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you today?"}]
if "context" not in st.session_state.keys():
    st.session_state.context = []
 
for message in st.session_state.messages:
    with chat_interface.chat_message(message["role"]):
        st.write(message["content"])
 
if prompt := chat_interface.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with chat_interface.chat_message("user"):
        st.write(prompt)
 
# Question asked. One of two things: respond with context or respond with generated message
if st.session_state.messages[-1]["role"] != "assistant":
    #check for relevance
    query = st.session_state.messages[-1]["content"]
    relevance = assess_query(query, st.session_state.context) 
    # Respond with context if new text is not relevant
    if relevance: 
    # Generating response if new text is relevant to conversation
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                response = generate_response(prompt, model, system_prompt)
                response = re.sub("</s>", "", response) 

        save_prompt(username=domino_user,
                            prompt=prompt,
                            context=st.session_state.context_list,
                            response=response,
                            location=DOMINO_DATASET)
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)
    else: 
        #Respond with context if new text is not relevant
        perform_content_search(query, size, prod, ta, country, indic, date, doc_type#, 
                               #keywords, topic
                              )
 
        if not isinstance(st.session_state.context, list):
            counter = 0
            #if st.session_state.context.empty: 
            #    st.warning("No results found for this combination.")
                # Clear messages before looking for new context
            for index, row in st.session_state.context.iterrows():
                if counter >= size:
                    break
                search_results.write(f'{row["product_name"]} | {row["insight_id"]} | {row["insight_date"]}')
                search_results.write(row["insight_content_text"])
                counter += 1
            with st.chat_message("assistant"):
                with st.spinner("Generating response..."):
                    response = generate_response(prompt, model, system_prompt)
                    response = re.sub("</s>", "", response) 
            save_prompt(username=domino_user,
                            prompt=prompt,
                            context=st.session_state.context_list,
                            response=response,
                            location=DOMINO_DATASET)
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message)