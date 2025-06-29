import torch
from typing import List
from pydantic import BaseModel
from FlagEmbedding import FlagModel
from fastapi import FastAPI, HTTPException

app = FastAPI()
'''
嵌入模型服务
'''

# 全局变量保存模型
global_model = None
device = torch.device("cuda:0")

# 请求模型
class EmbeddingRequest(BaseModel):
    texts: List[str]
    instruction: str = "为这个句子生成表示以用于检索相关文章："

# 响应模型
class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    elapsed_time: float

@app.on_event("startup")
async def load_model():
    global global_model
    model_path = "model/bge-large-zh-v1.5"
    global_model = FlagModel(
        model_path,
        use_fp16=False,
        query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
        devices='cuda:0'
    )

@app.post("/embed", response_model=EmbeddingResponse)
async def generate_embeddings(request: EmbeddingRequest):
    try:
        import time
        start_time = time.time()
        
        # 生成嵌入向量
        embeddings = global_model.encode(
            request.texts,
        )
        
        # 计算耗时
        elapsed_time = (time.time() - start_time) * 1000  # 转换为毫秒
        
        # 确保输出是普通Python列表
        embeddings_list = embeddings.tolist()
        
        return EmbeddingResponse(
            embeddings=embeddings_list,
            elapsed_time=elapsed_time
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)