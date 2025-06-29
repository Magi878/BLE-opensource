from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from FlagEmbedding import FlagReranker



"""
    uvicorn reranker_service:app --host 0.0.0.0 --port 8000
"""
app = FastAPI()

# 全局变量保存模型
global_model = None
device = 'cuda:0'  # 使用指定的 GPU

# 请求模型
class RerankRequest(BaseModel):
    query: str
    results: List[Dict]  # 修改为接收包含 ItemName 的字典列表

# 响应模型
class RerankResponse(BaseModel):
    ranked_results: List[Dict]  # 返回排序后的完整结果

@app.on_event("startup")
def load_model():
    global global_model
    model_path = "/data/ganshushen/Projects/BLE_CODE_OPEN/model/bge-reranker-large"  
    global_model = FlagReranker(model_path, use_fp16=False, devices=device)

@app.post("/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    try:
        # 准备待打分的文本对
        pairs = []
        items = []
        for item in request.results:
            text_pair = [request.query, f"{item['DishName']}"]
            # text_pair = [request.query, item]
            pairs.append(text_pair)
            items.append(item)
        
        # 使用 FlagReranker 计算相关性得分
        scores = global_model.compute_score(pairs, normalize=True)
        
        # 将 items 和 scores 组合并排序
        ranked_results = list(zip(items, scores))
        ranked_results.sort(key=lambda x: x[1], reverse=True)
        
        # 返回排序后的结果
        return RerankResponse(
            ranked_results=[{**item, 'score': score} for item, score in ranked_results]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))