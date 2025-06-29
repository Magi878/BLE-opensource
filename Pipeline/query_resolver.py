# 查询理解模块

import time
import asyncio
import aiohttp
import requests
from utils import get_embedding
from prompt import REWRITE_DYNAMICS_FEW_SHOT_SYS_PROMPT ,INTENT_RECOGNITION_ITEMNAME_CATEGORY_SYS_PROMPT

def search_rewrite_faiss(query, k=3, server_url='http://localhost:5001'):
    
    query_vectors = get_embedding(query)
    
    response = requests.post(f"{server_url}/search", json={
        'vectors': query_vectors,
        'k': k
    })
    
    if response.status_code != 200:
        print(f"错误: {response.json().get('error')}")
        return None
    
    results = response.json()['results']
    positive_examples = []
    for result in results:
        query = result.get("Query", "")
        response = result.get("Response", "")
        
        # 直接构建示例，不尝试解析JSON
        example = f"输入：{query}\n输出：{response}"
        positive_examples.append(example)
    
    # 拼接成字符串
    return "\n\n".join(positive_examples)

# 槽位抽取和改写
async def query_rewrite_LLM_FewShot(query,session):

    few_shots_examples = search_rewrite_faiss(query)
    escaped_few_shots = few_shots_examples.replace("{", "{{").replace("}", "}}")
    REWRITE_SYETEM_PROMPT = REWRITE_DYNAMICS_FEW_SHOT_SYS_PROMPT.format(positive_examples=escaped_few_shots)
    
    ollama_url = "http://localhost:11434/api/chat"
    
    payload = {
        "model": "qwen3:4b",   #    "qwen3:30b-a3b"
        "messages": [
            {
                "role": "system",
                "content": REWRITE_SYETEM_PROMPT 
            },
            {
                "role": "user",
                "content": f"用户输入{query},请输出JSON: /no_think"
            }
        ],
        "stream": False,
        "temperature": 0
    }
    
    start_time = time.time()
    try:
        async with session.post(ollama_url, json=payload) as response:
            response.raise_for_status()
            print(response)
            result = await response.json(content_type=None)
            elapsed = (time.time() - start_time) * 1000
            print(f"查询改写的识别时间: {elapsed:.2f}ms")
            print(result)
            print(result.get('message', {}).get('content'))
            return result.get('message', {}).get('content')
        
    except aiohttp.ClientError as e:
        print(f"Error in async_llm_rerank: {e}")
        elapsed = (time.time() - start_time) * 1000
        print(f"⏱️ LLM Rerank API call failed after: {elapsed:.2f}ms")
        return None
    
    except asyncio.TimeoutError:
        print("Timeout in async_llm_intent_recognition") # 更正函数名打印
        elapsed = (time.time() - start_time) * 1000
        print(f"⏱️ LLM Intent Recog")
        


async def async_llm_intent(query,session): # 添加 session 参数
    
    ollama_url = "http://localhost:11434/api/chat"
    payload = {
        "model": "qwen3:4b", # qwen3:30b-a3b
        "messages": [
            {
                "role": "system",
                "content":INTENT_RECOGNITION_ITEMNAME_CATEGORY_SYS_PROMPT,
            },
            {
                "role": "user",
                "content": f"当前用户输入：{query},请输出JSON:/no_think"
            }
        ],
        "stream": False,
        "temperature": 0
    }
    
    start_time = time.time()
    try:
        async with session.post(ollama_url, json=payload) as response:
            response.raise_for_status()
            result = await response.json(content_type=None)
            elapsed = (time.time() - start_time) * 1000
            print(f"意图命名实体识别时间: {elapsed:.2f}ms")
            return result.get('message', {}).get('content')
        
    except aiohttp.ClientError as e:
        print(f"Error in async_llm_rerank: {e}")
        elapsed = (time.time() - start_time) * 1000
        print(f"⏱️ LLM Rerank API call failed after: {elapsed:.2f}ms")
        return None
    
    except asyncio.TimeoutError:
        print("Timeout in async_llm_intent_recognition") # 更正函数名打印
        elapsed = (time.time() - start_time) * 1000
        print(f"⏱️ LLM Intent Recognition API call timed out after: {elapsed:.2f}ms") # 更正打印信息
        return None