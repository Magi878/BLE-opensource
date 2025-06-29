import asyncio
import aiohttp

async def async_llm_rerank(user_prompt, session):
    """异步调用FastAPI LLM重排序服务"""
    fastapi_url = "http://localhost:10000/rerank"
    
    payload = {
        "user_instruction": user_prompt
    }
    
    try:
        async with session.post(fastapi_url, json=payload) as response:
            response.raise_for_status()
            result = await response.json()
            
            if result.get('status') == 'success':
                content = result.get('generated_text', '')
                
                # 清理响应内容
                if '<think>' in content and '</think>' in content:
                    content = content.split('</think>')[-1].strip()
                    
                content = content.replace('\n', '').strip()
                
                return content
            else:
                print(f"FastAPI返回错误状态: {result}")
                return None
                
    except Exception as e:
        print(f"Error in async_llm_rerank: {e}")
        return None




async def async_rerank(query, candidates, session):
  
    url = "http://localhost:8000/rerank"
    
    data = {
        "query": query,
        "results": candidates
    }
    
    try:
        async with session.post(url, json=data) as response:
            response.raise_for_status()
            results = await response.json(content_type=None)
            return results
        
    except aiohttp.ClientError as e:
        return None
    
    except asyncio.TimeoutError:
        return None