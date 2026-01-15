import json
import sys
# sys.path.append("..")
from urllib.parse import urljoin
import time 
from argparse import Namespace
from web_search.bing_search import bing_web_search
from web_search.bing_search import extract_relevant_info
from web_search.webpage_utils import extract_text_from_urls, extract_snippet_with_context, error_indicators
import re
import asyncio
from openai import OpenAI
# from deep_search_dgt import deep_search_dgt
from transformers import AutoTokenizer
import requests



def deep_search_snippet(search_query, top_k=10, use_jina=False, jina_api_key="empty", bing_subscription_key="your bing api key", bing_endpoint="https://api.bing.microsoft.com/v7.0/search"):
    # 根据函数参数构建 args
    args = Namespace(
        dataset_name='qa',
        split='test',
        subset_num=-1,
        max_search_limit=15,
        top_k=top_k,  # 使用函数参数
        use_jina=use_jina,  # 使用函数参数
        jina_api_key=jina_api_key,  # 使用函数参数
        temperature=0.7,
        top_p=0.8,
        min_p=0.05,
        top_k_sampling=20,
        repetition_penalty=1.05,
        max_tokens=4096,
        bing_subscription_key=bing_subscription_key,  # 使用函数参数
        bing_endpoint=bing_endpoint,  # 使用函数参数
        eval=False,
        seed=1742208600,
        concurrent_limit=200
    )
    # print(args)
    
    search_cache = {}
    url_cache = {}

    question = search_query

    try:
        # 调用必应搜索API
        results = bing_web_search(question, args.bing_subscription_key, args.bing_endpoint) 
        search_cache[question] = results
    except Exception as e:
        print(f"Error during search query '{question}': {e}")
        results = {}
    
    # 提取相关信息并限制结果数量
 
    relevant_info = extract_relevant_info(results)[:args.top_k]
    print("--------------------------------Search Bing Result--------------------------------")

    result = ""
    for info in relevant_info:
        # info['snippet'] = formatted_documents
        snippet = info['snippet']
        clean_snippet = re.sub('<[^<]+?>', '', snippet)  # Removes HTML tags
        result+=clean_snippet

    extracted_info = result

    return extracted_info

def deep_search_browser(search_query, top_k=10, use_jina=False, jina_api_key="empty", bing_subscription_key="your bing api key", bing_endpoint="https://api.bing.microsoft.com/v7.0/search"):
    # workflow: search + 获取所有网页信息 + 根据snippet提取
    # 根据函数参数构建 args
    args = Namespace(
        dataset_name='qa',
        split='test',
        subset_num=-1,
        max_search_limit=10,
        top_k=top_k,  # 使用函数参数
        use_jina=use_jina,  # 使用函数参数
        jina_api_key=jina_api_key,  # 使用函数参数
        temperature=0.7,
        top_p=0.8,
        min_p=0.05,
        top_k_sampling=20,
        repetition_penalty=1.05,
        max_tokens=4096,
        bing_subscription_key=bing_subscription_key,  # 使用函数参数
        bing_endpoint=bing_endpoint,  # 使用函数参数
        eval=False,
        seed=1742208600,
        concurrent_limit=200
    )
    # print(args)
    
    search_cache = {}
    url_cache = {}

    question = search_query

    try:
        # 调用必应搜索API
        results = bing_web_search(question, args.bing_subscription_key, args.bing_endpoint) 
        search_cache[question] = results
    except Exception as e:
        print(f"Error during search query '{question}': {e}")
        results = {}
    
    # 提取相关信息并限制结果数量
 
    relevant_info = extract_relevant_info(results)[:args.top_k]
    print("--------------------------------Search Bing Result--------------------------------")

    # Process documents
    urls_to_fetch = []
    for doc_info in relevant_info:
        url = doc_info['url']
        if url not in url_cache:
            urls_to_fetch.append(url)

    print(f"--------------------------------Search Bing Result finish--------------------------------")

    try:
        contents = extract_text_from_urls(urls_to_fetch)
        for url, content in zip(urls_to_fetch, contents):
            # Only cache content if it doesn't contain error indicators
            has_error = (any(indicator.lower() in content.lower() for indicator in error_indicators) and len(content.split()) < 64) or len(content) < 50 or len(content.split()) < 20
            if not has_error:
                url_cache[url] = content
    except Exception as e:
        print(f"Error fetching URLs: {e}")


    # Get web page information for each result
    if len(relevant_info) == 0:
        print("No relevant information found in url fetching.")
    formatted_documents = ""
    for i, doc_info in enumerate(relevant_info):
        url = doc_info['url']
        snippet = doc_info.get('snippet', "")
        raw_context = url_cache.get(url, "")
        if raw_context == "":
            print(f"Naive-RAG Agent warning: No relevant information found in {url}.")
        success, context = extract_snippet_with_context(raw_context, snippet, context_chars=6000)
        if success:
            context = context
        else:
            context = raw_context[:2 * 3000]

        # Clean snippet from HTML tags if any
        clean_snippet = re.sub('<[^<]+?>', '', snippet)  # Removes HTML tags

        formatted_documents += f"**Document {i + 1}:**\n"
        formatted_documents += f"**Title:** {doc_info.get('title', '').replace('<b>', '').replace('</b>', '')}\n"
        formatted_documents += f"**URL:** {url}\n"
        formatted_documents += f"**Snippet:** {clean_snippet}\n"
        formatted_documents += f"**Content:** {context}\n\n"

    # Construct the instruction with documents and question
    if formatted_documents == '':
        formatted_documents = 'No relevant information found.'

    return formatted_documents

def deep_search_browser_summarize(search_query, top_k=10, use_jina=False, jina_api_key="empty", bing_subscription_key="your bing api key", bing_endpoint="https://api.bing.microsoft.com/v7.0/search"):
    # workflow: search + 获取所有网页信息 + 根据snippet提取
    # 根据函数参数构建 args
    args = Namespace(
        dataset_name='qa',
        split='test',
        subset_num=-1,
        max_search_limit=15,
        top_k=top_k,  # 使用函数参数
        use_jina=use_jina,  # 使用函数参数
        jina_api_key=jina_api_key,  # 使用函数参数
        temperature=0.7,
        top_p=0.8,
        min_p=0.05,
        top_k_sampling=20,
        repetition_penalty=1.05,
        max_tokens=4096,
        bing_subscription_key=bing_subscription_key,  # 使用函数参数
        bing_endpoint=bing_endpoint,  # 使用函数参数
        eval=False,
        seed=1742208600,
        concurrent_limit=200
    )
    # print(args)
    
    search_cache = {}
    url_cache = {}

    question = search_query

    try:
        # 调用必应搜索API
        results = bing_web_search(question, args.bing_subscription_key, args.bing_endpoint) 
        search_cache[question] = results
    except Exception as e:
        print(f"Error during search query '{question}': {e}")
        results = {}
    
    # 提取相关信息并限制结果数量
 
    relevant_info = extract_relevant_info(results)[:args.top_k]
    print("--------------------------------Search Bing Result--------------------------------")

    # Process documents
    urls_to_fetch = []
    # print(json.dumps(relevant_info, indent=4, ensure_ascii=False))
    for doc_info in relevant_info:
        url = doc_info['url']
        if url not in url_cache:
            urls_to_fetch.append(url)

    try:
        contents = extract_text_from_urls(urls_to_fetch)
        for url, content in zip(urls_to_fetch, contents):
            # Only cache content if it doesn't contain error indicators
            has_error = (any(indicator.lower() in content.lower() for indicator in error_indicators) and len(content.split()) < 64) or len(content) < 50 or len(content.split()) < 20
            if not has_error:
                url_cache[url] = content
    except Exception as e:
        print(f"Error fetching URLs: {e}")

    if len(relevant_info) == 0:
        print("No relevant information found in url fetching.")
    formatted_documents = ""
    for i, doc_info in enumerate(relevant_info):
        url = doc_info['url']
        snippet = doc_info.get('snippet', "")
        raw_context = url_cache.get(url, "")
        if raw_context == "":
            print(f"Naive-RAG Agent warning: No relevant information found in {url}.")
    
        #     print(f'--------------------raw_context------------------')
        success, context = extract_snippet_with_context(raw_context, snippet, context_chars=6000)
        if success:
            context = context
        else:
            context = raw_context[:2 * 3000]


        # Clean snippet from HTML tags if any
        clean_snippet = re.sub('<[^<]+?>', '', snippet)  # Removes HTML tags

        formatted_documents += f"**Document {i + 1}:**\n"
        formatted_documents += f"**Title:** {doc_info.get('title', '').replace('<b>', '').replace('</b>', '')}\n"
        formatted_documents += f"**URL:** {url}\n"
        formatted_documents += f"**Snippet:** {clean_snippet}\n"
        formatted_documents += f"**Content:** {context}\n\n"


    # Construct the instruction with documents and question
    if formatted_documents == '':
        formatted_documents = 'No relevant information found.'
    summarize_result = summarize_text(question, formatted_documents)

    return summarize_result



def summarize_text(search_query, search_result):

    API_BASE_URL = "your sumarize model api"
    MODEL_NAME = "Qwen2.5-3B-Instruct" # change your summarize model name
    client = OpenAI(
        api_key="empty",
        base_url=API_BASE_URL,
    )


    prompt = f"""You are a web explorer. Your task is to find relevant information for the search query.

    Based on the search results:
    - If the information answers the query, output it directly
    - If more information is needed:
    1. Search again: <|begin_search_query|>query<|end_search_query|>
    2. Access webpage content with information seeking intent using: <|begin_click_link|>URL<|end_click_link|> <|begin_click_intent|>intent<|end_click_intent|>
    - If you can't find any helpful information, output "No helpful information found."

    **Inputs:**

    **Search Query:**
    {search_query}

    **Search Results:**
    {search_result}

    Final Output Format:

    **Final Information:**

    [Factual information to the search query] or [No helpful information found.]

    Now you should analyze each web page and find helpful information for the search query "{search_query}"."""

    def sanitize_input(text):
        if isinstance(text, str):
            # 将单个反斜杠替换为双反斜杠
            text = text.replace('\\', '\\\\')
            # 转义大括号
            text = text.replace('{', '{{').replace('}', '}}')
        return text
    
    search_query = sanitize_input(search_query)
    search_result = sanitize_input(search_result)

    prompt = prompt.format(search_query=search_query, search_result=search_result)
    
    
    chat_response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )
    # print(chat_response.choices[0].message.content)
    return chat_response.choices[0].message.content






if __name__ == "__main__":

    extracted_info = ''
    
    
    question = "三角形的勾股定理是什么"


    result = deep_search_snippet(question)
    print('-------------------------------------')
    print(result)
    print('-------------------------------------')

   