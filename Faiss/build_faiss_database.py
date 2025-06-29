import faiss
import numpy as np
import json
import pickle
from typing import Dict, List

class FAISSDatabase:
    def __init__(self, dim: int = 512):
        # 创建GPU资源
        self.res = faiss.StandardGpuResources()
        
        # 使用简单的FlatL2索引，避免IVF索引训练问题
        cpu_index = faiss.IndexFlatL2(dim)
        
        # 将索引转移到GPU
        self.index = faiss.index_cpu_to_gpu(self.res, 3, cpu_index)
        
        # 存储额外的元数据
        self.metadata: List[Dict] = []
        print("GPU FAISS数据库初始化完成 - 使用FlatL2索引")
        
    def save(self, path: str):
        # 将GPU索引转回CPU进行保存
        cpu_index = faiss.index_gpu_to_cpu(self.index)
        # 保存FAISS索引
        faiss.write_index(cpu_index, f"{path}/vector.index")
        # 保存元数据
        with open(f"{path}/metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)
            
    def load(self, path: str):
        # 加载FAISS索引到CPU
        cpu_index = faiss.read_index(f"{path}/vector.index")
        # 将索引转移到GPU
        self.index = faiss.index_cpu_to_gpu(self.res, 3, cpu_index)
        # 加载元数据
        with open(f"{path}/metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)

def insert_data_from_jsonl(db: FAISSDatabase, jsonl_file_path: str, embedding_model) -> None:
    # 增大批处理大小以提高效率
    batch_vectors = []
    batch_metadata = []
    batch_size = 1000  # 增大批处理大小
    
    print("收集向量并添加到索引...")
    with open(jsonl_file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % 1000 == 0:
                print(f"已处理 {i} 条数据...")
                
            # 解析JSON行
            data = json.loads(line.strip())
            
            # 提取字段
            item_name = data.get("item_name", "")
            category_name = data.get("category_name", "")
            
            # 生成向量
            item_name_vector = embedding_model.encode(item_name)
            batch_vectors.append(item_name_vector)
            
            # 构造元数据
            metadata = {
                "ItemName": item_name,
                "CategoryName": category_name,
            }
            batch_metadata.append(metadata)
            
            # 如果达到批处理大小，则添加到索引
            if len(batch_vectors) >= batch_size:
                vectors = np.array(batch_vectors).astype('float32')
                db.index.add(vectors)
                db.metadata.extend(batch_metadata)
                print(f"已添加 {len(batch_vectors)} 个向量，总计 {len(db.metadata)} 个")
                batch_vectors = []
                batch_metadata = []
    
    # 处理剩余的数据
    if batch_vectors:
        vectors = np.array(batch_vectors).astype('float32')
        db.index.add(vectors)
        db.metadata.extend(batch_metadata)
        print(f"已添加 {len(batch_vectors)} 个向量，总计 {len(db.metadata)} 个")


def Get_Embedding_Model(model_name:str):
    if model_name == "bge":
        from FlagEmbedding import FlagModel
        embedding_model = FlagModel('/data/ele/Projects/gss/bge-large-zh-v1.5', 
                        query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                        use_fp16=False,
                        devices='cuda:3',
                        )
 
    return embedding_model

if __name__ == "__main__":
    # 创建FAISS数据库实例
    db = FAISSDatabase(dim=1024)
    
    # 获取embedding模型
    embedding_model = Get_Embedding_Model("bge")
    
    # 插入数据
    insert_data_from_jsonl(db, "/data/ele/Projects/gss/V511/Catelogry/new_products.jsonl", embedding_model)
    
    # 保存数据库
    db.save("/data/ele/Projects/gss/Faiss/database/GPU_Large_DataBase")
    
    print("索引构建完成，已保存到指定路径")