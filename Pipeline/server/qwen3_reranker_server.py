import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from typing import List, Dict
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException


from transformers import AutoModelForCausalLM, AutoTokenizer

from prompt import RECOMMENDATION_SYS_PROMPT 



'''
方式一：
python qwen3_reranker_server.py
方式二：
uvicorn qwen3_reranker_server:app --reload --host 0.0.0.0 --port 10000
'''

app = FastAPI()

# 初始化模型和tokenizer
device = "cuda:1" if torch.cuda.is_available() else "cpu"
model_name_or_path = "model/Qwen3_4B_Lora"

# 加载模型
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {str(e)}")

class InferenceRequest(BaseModel):
    user_instruction: str
    # 可以添加其他参数如max_new_tokens等

class InferenceResponse(BaseModel):
    generated_text: str
    status: str

@app.post("/rerank", response_model=InferenceResponse)
async def rerank_dishes(request: InferenceRequest):
    try:
        # 构造输入
        prompt = f"用户指令：{request.user_instruction}"
        print(prompt)
        
        messages = [
            {"role": "system", "content": RECOMMENDATION_SYS_PROMPT },
            {"role": "user", "content": prompt}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        # Tokenizer输入也要放到 GPU
        model_inputs = tokenizer(text, return_tensors="pt", return_attention_mask=True).to(device)

        # 推理
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.0,       # 核心参数：设置温度为0
                top_p=0.0,             # 当temperature=0时，建议同时设置top_p=0（避免冲突）
                top_k=1                # 只考虑概率最高的1个token
            )

        # 解码结果
        generated_text = tokenizer.decode(
            generated_ids[0][model_inputs.input_ids.shape[-1]:], 
            skip_special_tokens=True
        )

        return InferenceResponse(
            generated_text=generated_text,
            status="success"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)