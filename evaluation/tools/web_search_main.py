import json
import sys
# sys.path.append("..")
from urllib.parse import urljoin
import time 
from argparse import Namespace
from tools.bing_search import bing_web_search
from tools.bing_search import extract_relevant_info
import re

def deep_search(search_query, top_k=10, use_jina=False, jina_api_key="empty", bing_subscription_key="xxxxx", bing_endpoint="xxxxx/search"):
    args = Namespace(
        dataset_name='qa',
        split='test',
        subset_num=-1,
        max_search_limit=15,
        top_k=top_k,  
        use_jina=use_jina,  
        jina_api_key=jina_api_key,  
        temperature=0.7,
        top_p=0.8,
        min_p=0.05,
        top_k_sampling=20,
        repetition_penalty=1.05,
        max_tokens=4096,
        bing_subscription_key=bing_subscription_key,  
        bing_endpoint=bing_endpoint,  
        eval=False,
        seed=1742208600,
        api_base_url='xxxxx',  
        model_name='search-agent',
        concurrent_limit=200
    )
    
    search_cache = {}

    question = search_query

    try:
        results = bing_web_search(question, args.bing_subscription_key, args.bing_endpoint) 
        search_cache[question] = results
    except Exception as e:
        print(f"Error during search query '{question}': {e}")
        results = {}
    
 
    relevant_info = extract_relevant_info(results)[:args.top_k]
    print("--------------------------------get bing search result--------------------------------")

    result = ""
    for info in relevant_info:
        snippet = info['snippet']
        clean_snippet = re.sub('<[^<]+?>', '', snippet)  
        result+=clean_snippet

    extracted_info = result

    return extracted_info


if __name__ == "__main__":

    extracted_info = ''
    
    
    question = "What is the capital of France?"

    result = deep_search(
        question
    )
    print('-------------------------------------')
    print(result)
    print('-------------------------------------')
    
    