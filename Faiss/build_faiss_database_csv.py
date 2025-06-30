import faiss
import numpy as np
import pickle
import pandas as pd
from typing import Dict, List


class FAISSDatabase:
    def __init__(self, dim: int = 512):
        # 创建 GPU 资源
        self.res = faiss.StandardGpuResources()

        # 使用简单的 FlatL2 索引，避免 IVF 索引训练问题
        cpu_index = faiss.IndexFlatL2(dim)

        # 将索引转移到 GPU
        self.index = faiss.index_cpu_to_gpu(self.res, 1, cpu_index)

        # 存储额外的元数据
        self.metadata: List[Dict] = []
        print("GPU FAISS 数据库初始化完成 - 使用 FlatL2 索引")

    def save(self, path: str):
        # 将 GPU 索引转回 CPU 进行保存
        cpu_index = faiss.index_gpu_to_cpu(self.index)
        # 保存 FAISS 索引
        faiss.write_index(cpu_index, f"{path}/vector.index")
        # 保存元数据
        with open(f"{path}/metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self, path: str):
        # 加载 FAISS 索引到 CPU
        cpu_index = faiss.read_index(f"{path}/vector.index")
        # 将索引转移到 GPU
        self.index = faiss.index_cpu_to_gpu(self.res, 3, cpu_index)
        # 加载元数据
        with open(f"{path}/metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)


def insert_data_from_csv(db: FAISSDatabase, csv_file_path: str, embedding_model) -> None:
    # 增大批处理大小以提高效率
    batch_vectors = []
    batch_metadata = []
    batch_size = 1000  # 增大批处理大小

    print("收集向量并添加到索引...")
    df = pd.read_csv(csv_file_path)

    for i in range(len(df)):
        if i % 1000 == 0:
            print(f"已处理 {i} 条数据...")

        # 提取字段
        query = df.loc[i, '查询'] if '查询' in df.columns else ""
        response = df.loc[i, '原始响应'] if '原始响应' in df.columns else ""

        # 生成向量
        item_name_vector = embedding_model.encode(query)
        batch_vectors.append(item_name_vector)

        # 构造元数据
        metadata = {
            "Query": query,
            "Response": response ,
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


def Get_Embedding_Model(model_name: str):
    if model_name == "bge":
        from FlagEmbedding import FlagModel
        embedding_model = FlagModel('model/bge-large-zh-v1.5',
                                    query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                                    use_fp16=False,
                                    devices='cuda:1',
                                    )

    return embedding_model


if __name__ == "__main__":
    # 创建 FAISS 数据库实例
    db = FAISSDatabase(dim=1024)

    # 获取 embedding 模型
    embedding_model = Get_Embedding_Model("bge")

    # 插入数据，这里使用你生成的 CSV 文件路径
    csv_file_path = "/data/ganshushen/Projects/MainBranch/Faiss/rewrite_results_20250608_130228_32B_FP16_REWRITE_extract .csv"
    insert_data_from_csv(db, csv_file_path, embedding_model)

    # 保存数据库
    db.save("Faiss/database/REWRITE_DATABASE")

    print("索引构建完成，已保存到指定路径")