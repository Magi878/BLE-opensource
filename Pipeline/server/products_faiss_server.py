import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import faiss
import pickle
import numpy as np
from typing import List, Dict
from flask import Flask, request, jsonify

app = Flask(__name__)
'''
向量数据库服务
'''

class FAISSServer:
    def __init__(self, dim: int = 1024):
        # 创建GPU资源
        self.res = faiss.StandardGpuResources()
        # 创建FAISS索引，使用L2距离的平面索引，并转移到GPU
        cpu_index = faiss.IndexFlatL2(dim)
        self.index = faiss.index_cpu_to_gpu(self.res, 0, cpu_index)  # 使用第4个GPU (索引从0开始)
        self.metadata: List[Dict] = []
        print("GPU FAISS服务器初始化完成")
        
    def load(self, path: str):
        # 加载FAISS索引到CPU
        print(f"正在加载索引: {path}")
        cpu_index = faiss.read_index(f"{path}/vector.index")
        # 将索引转移到GPU
        self.index = faiss.index_cpu_to_gpu(self.res, 0, cpu_index)
        # 加载元数据
        with open(f"{path}/metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)
        print(f"索引加载完成，包含 {len(self.metadata)} 条数据")
    
    def search(self, query_vectors, k=10):
        # 确保向量格式正确
        if len(query_vectors.shape) == 1:
            query_vectors = query_vectors.reshape(1, -1)
        
        query_vectors = query_vectors.astype('float32')
        
        # 批量检索
        distances, indices = self.index.search(query_vectors, k)
        
        # 返回检索结果
        results = []
        for i in range(len(indices)):
            batch_results = []
            for j, (idx, distance) in enumerate(zip(indices[i], distances[i])):
                if idx < len(self.metadata):  # 确保索引有效
                    item = self.metadata[idx]
                    item['score'] = float(1/(1+distance))  # 添加分数
                    batch_results.append(item)
            results.append(batch_results)
        
        return results[0] if len(results) == 1 else results

# 创建全局FAISS服务实例
faiss_service = FAISSServer()

@app.route('/load', methods=['POST'])
def load_index():
    data = request.json
    path = data.get('path')
    if not path:
        return jsonify({"error": "Missing 'path' parameter"}), 400
    
    try:
        faiss_service.load(path)
        return jsonify({"status": "success", "message": f"Index loaded from {path}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    vectors = data.get('vectors')
    k = data.get('k', 10)
    
    if not vectors:
        return jsonify({"error": "Missing 'vectors' parameter"}), 400
    
    try:
        vectors_np = np.array(vectors, dtype=np.float32)
        start_time = time.time()
        results = faiss_service.search(vectors_np, k)
        elapsed = (time.time() - start_time) * 1000
        
        return jsonify({
            "results": results,
            "time_ms": elapsed
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # 可以在启动时预加载索引
    # import sys
    # if len(sys.argv) > 1:
    #     faiss_service.load(sys.argv[1])
    faiss_service.load("/data/ganshushen/Projects/BLE_CODE_OPEN/Faiss/database/New_Products_Faiss_Database")
    # 启动服务器
    app.run(host='0.0.0.0', port=5006)