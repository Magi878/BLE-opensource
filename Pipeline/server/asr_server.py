import os
import uuid
import asyncio
from pathlib import Path
from typing import List, Optional
from fastapi.responses import JSONResponse
from fastapi import FastAPI, UploadFile, File

from inference import ASRInference # 推理脚本

app = FastAPI()
'''
语音识别服务
'''

# 初始化模型（只初始化一次）, configuration.json中修改要使用的权重，这里使用model.pt.avg10
model_path = "model/ASR_Paraformer/weight"
asr_output = "Pipeline/server/asr_logs"
asr_infer = ASRInference(model_name_or_path=model_path, modes=None, output_dir=asr_output)

# 配置设置
class Config:
    def __init__(self):
        self.temp_dir = "/data/ganshushen/Projects/BLE_CODE_OPEN/Pipeline/server/temp_wavs"  # 默认临时目录
        
config = Config()

def set_temp_dir(temp_dir: str):
    """设置临时文件存储目录"""
    os.makedirs(temp_dir, exist_ok=True)
    config.temp_dir = temp_dir
    print(f"临时文件目录设置为: {config.temp_dir}")

async def process_file(file: UploadFile):
    """处理单个音频文件"""
    temp_file_path = None
    try:
        # 确保临时目录存在
        os.makedirs(config.temp_dir, exist_ok=True)
        
        # 根据原始文件名确定后缀
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ['.wav', '.ogg']:
            return {"filename": file.filename, "error": "只支持.wav和.ogg格式文件"}
        
        # 生成唯一文件名
        temp_file_name = f"{uuid.uuid4().hex}{file_ext}"
        temp_file_path = os.path.join(config.temp_dir, temp_file_name)
        
        # 保存上传文件
        with open(temp_file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        print(f"处理文件: {file.filename} -> 临时存储: {temp_file_path}")
        
        # 执行ASR推理
        result_text = asr_infer.single_inference(temp_file_path)
        return {"filename": file.filename, "text": result_text}
        
    except Exception as e:
        print(f"处理文件 {file.filename} 出错: {str(e)}")
        return {"filename": file.filename, "error": str(e)}
    finally:
        # 清理临时文件
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as e:
                print(f"删除临时文件 {temp_file_path} 失败: {str(e)}")

@app.post("/infer/")
async def infer(files: List[UploadFile] = File(...)):
    """批量处理音频文件"""
    tasks = [process_file(file) for file in files]
    results = await asyncio.gather(*tasks)
    return JSONResponse(content={"results": results})

@app.post("/set_temp_dir/")
async def set_temp_directory(path: str):
    """设置临时文件存储路径"""
    try:
        set_temp_dir(path)
        return {"status": "success", "temp_dir": config.temp_dir}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)