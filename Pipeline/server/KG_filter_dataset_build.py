import os
import time
import json
import pickle
import faiss
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm
from FlagEmbedding import FlagModel
'''
测试KG_filter_server效果
'''

class FAISSDatabase:
    def __init__(self, dim: int = 1024):  # bge-large-zh-v1.5的维度是1024
        # 创建GPU资源
        self.res = faiss.StandardGpuResources()
        
        # 使用简单的FlatL2索引，避免IVF索引训练问题
        cpu_index = faiss.IndexFlatL2(dim)
        
        # 将索引转移到GPU
        self.index = faiss.index_cpu_to_gpu(self.res, 0, cpu_index)
        
        # 存储额外的元数据
        self.metadata: List[Dict] = []
        # 添加名称到ID的映射
        self.name_to_id = {}
        print("GPU FAISS数据库初始化完成 - 使用FlatL2索引")
        
    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict]):
        """添加向量和元数据到数据库"""
        if len(vectors) != len(metadata):
            raise ValueError("向量数量和元数据数量不匹配")
        
        start_id = self.index.ntotal
        self.index.add(vectors)
        self.metadata.extend(metadata)
        
        # 更新名称到ID的映射
        for idx, data in enumerate(metadata, start=start_id):
            if 'dish_name' in data:  # 菜品数据库
                self.name_to_id[data['dish_name']] = idx
            elif 'id' in data:  # 用户数据库
                self.name_to_id[str(data['id'])] = idx
                
    def get_vector_by_name(self, name: str) -> np.ndarray:
        """根据名称获取向量"""
        if name not in self.name_to_id:
            raise ValueError(f"名称 '{name}' 不在数据库中")
        vec_id = self.name_to_id[name]
        return self.index.reconstruct(vec_id).reshape(1, -1)
    
    def get_metadata_by_name(self, name: str) -> Dict:
        """根据名称获取元数据"""
        if name not in self.name_to_id:
            raise ValueError(f"名称 '{name}' 不在数据库中")
        vec_id = self.name_to_id[name]
        return self.metadata[vec_id]
        
    def save(self, path: str):
        # 将GPU索引转回CPU进行保存
        cpu_index = faiss.index_gpu_to_cpu(self.index)
        # 保存FAISS索引
        faiss.write_index(cpu_index, f"{path}/vector.index")
        # 保存元数据
        with open(f"{path}/metadata.pkl", "wb") as f:
            pickle.dump({
                'metadata': self.metadata,
                'name_to_id': self.name_to_id
            }, f)
            
    def load(self, path: str):
        # 加载FAISS索引到CPU
        cpu_index = faiss.read_index(f"{path}/vector.index")
        # 将索引转移到GPU
        self.index = faiss.index_cpu_to_gpu(self.res, 0, cpu_index)
        # 加载元数据
        with open(f"{path}/metadata.pkl", "rb") as f:
            data = pickle.load(f)
            self.metadata = data['metadata']
            self.name_to_id = data['name_to_id']
    
    def search(self, query_vector: np.ndarray, k: int = 20) -> List[Dict]:
        """搜索最相似的k个向量"""
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            dist = distances[0][i]
            results.append({
                "metadata": self.metadata[idx],
                "distance": dist,
                "similarity": 1 - dist  # 添加相似度分数
            })
        
        return results

class VectorDatabaseBuilder:
    def __init__(self, model_name: str = "bge"):
        self.embedding_model = self._get_embedding_model(model_name)
    
    def _get_embedding_model(self, model_name: str):
        if model_name == "bge":
            model = FlagModel('/data/ganshushen/Projects/BLE_CODE_OPEN/Embedding_model/bge-large-zh-v1.5', 
                            query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                            use_fp16=False,
                            devices='cuda:0')
            return model
        else:
            raise ValueError(f"不支持的模型: {model_name}")
    
    def build_dish_database(self, jsonl_path: str, output_path: str, batch_size: int = 64):
        """构建菜品向量库"""
        print("开始构建菜品向量库...")
        
        # 1. 收集所有菜品数据
        all_dishes = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="收集菜品数据"):
                data = json.loads(line)
                all_dishes.append(data)
        
        # 2. 批量编码所有菜品名称
        dish_names = [dish['dish_name'] for dish in all_dishes]
        dish_embeddings = []
        
        print("开始编码菜品名称...")
        for i in tqdm(range(0, len(dish_names), batch_size), desc="编码菜品"):
            batch = dish_names[i:i+batch_size]
            embeddings = self.embedding_model.encode(batch)
            dish_embeddings.append(embeddings)
        
        dish_embeddings = np.concatenate(dish_embeddings, axis=0)
        
        # 3. 创建并填充数据库
        db = FAISSDatabase()
        db.add_vectors(dish_embeddings, all_dishes)
        
        # 4. 保存数据库
        db.save(output_path)
        print(f"菜品向量库已保存到: {output_path}")
        print(f"共处理 {len(all_dishes)} 个菜品")
    
    def build_user_database(self, jsonl_path: str, output_path: str, batch_size: int = 32):
        """构建用户向量库"""
        print("开始构建用户向量库...")
        
        # 1. 收集所有用户数据
        all_users = []
        all_dish_lists = []
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="收集用户数据"):
                user_data = json.loads(line)
                
                # 收集所有历史菜品
                all_dishes = []
                for order in user_data['order_history']:
                    all_dishes.extend(order['dishes'])
                
                if not all_dishes:
                    continue  # 如果没有历史菜品，跳过该用户
                
                # 准备用户元数据（不包含历史订单，因为已经编码到向量中）
                user_metadata = {
                    'id': user_data['id'],
                    'name': user_data['name'],
                    'gender': user_data['gender'],
                    'age_range': user_data['age_range'],
                    'region': user_data['region'],
                    'health_conditions': user_data['health_conditions'],
                    'taste_preferences': user_data['taste_preferences'],
                    'texture_preferences': user_data['texture_preferences'],
                    'query': user_data['query']
                }
                
                all_users.append(user_metadata)
                all_dish_lists.append(all_dishes)
        
        # 2. 批量编码所有菜品列表
        print("开始编码用户历史菜品...")
        user_embeddings = []
        
        for dish_list in tqdm(all_dish_lists, desc="处理用户历史"):
            # 编码用户的所有历史菜品
            embeddings = self.embedding_model.encode(dish_list)
            # 计算平均向量作为用户表征
            avg_embedding = np.mean(embeddings, axis=0).reshape(1, -1)
            user_embeddings.append(avg_embedding)
        
        user_embeddings = np.concatenate(user_embeddings, axis=0)
        
        # 3. 创建并填充数据库
        db = FAISSDatabase()
        db.add_vectors(user_embeddings, all_users)
        
        # 4. 保存数据库
        db.save(output_path)
        print(f"用户向量库已保存到: {output_path}")
        print(f"共处理 {len(all_users)} 个用户")


class DishRecommender:
    def __init__(self, dish_db_path: str, user_db_path: str):
        # 加载菜品数据库
        self.dish_db = FAISSDatabase()
        self.dish_db.load(dish_db_path)
        
        # 加载用户数据库
        self.user_db = FAISSDatabase()
        self.user_db.load(user_db_path)
    
    def recommend_dishes(self, user_id: int, dish_names: List[str], top_k: int = 20) -> List[Dict]:
        """
        为指定用户推荐菜品
        
        参数:
            user_id: 用户ID
            dish_names: 待推荐的菜品名称列表(必须在菜品库中存在)
            top_k: 返回的推荐数量
            
        返回:
            包含推荐菜品和相似度得分的列表
        """
        # 1. 从用户库中获取用户向量
        try:
            user_vector = self.user_db.get_vector_by_name(str(user_id))
            user_metadata = self.user_db.get_metadata_by_name(str(user_id))
            print(f"User {user_id} metadata: {user_metadata}")
        except ValueError as e:
            raise ValueError(f"无法找到用户: {e}")
        
        # 2. 从菜品库中获取所有待推荐菜品的向量
        dish_vectors = []
        valid_dish_names = []
        
        for name in dish_names:
            try:
                vec = self.dish_db.get_vector_by_name(name)
                dish_vectors.append(vec)
                valid_dish_names.append(name)
            except ValueError:
                print(f"警告: 菜品 '{name}' 不在菜品库中，已跳过")
        
        if not valid_dish_names:
            raise ValueError("没有有效的菜品可供推荐")
        
        dish_vectors = np.concatenate(dish_vectors, axis=0)
        
        # 3. 创建临时FAISS索引
        temp_index = faiss.IndexFlatL2(1024)
        temp_index.add(dish_vectors)
        
        # 4. 搜索与用户向量最相似的菜品
        distances, indices = temp_index.search(user_vector, min(top_k, len(valid_dish_names)))
        
        # 5. 准备结果
        recommendations = []
        for i in range(len(indices[0])):
            dish_idx = indices[0][i]
            dish_name = valid_dish_names[dish_idx]
            distance = distances[0][i]
            
            # 获取菜品的完整信息
            dish_info = self.dish_db.get_metadata_by_name(dish_name)
            
            recommendations.append({
                'dish': dish_info,
                'similarity_score': 1 - distance,  # 转换为相似度分数
                'user_info': user_metadata
            })
        
        # 按相似度降序排序
        recommendations.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return recommendations[:top_k]


def find_light_taste_dishes(file_path, query_taste):
    """
    读取JSONL文件，返回所有taste包含"清淡"的菜品名称列表
    
    参数:
        file_path (str): JSONL文件路径
        
    返回:
        list: 包含所有符合条件的菜品名称的列表
    """
    light_taste_dishes = []
    line_count = 0
    error_count = 0
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line_count += 1
            line = line.strip()
            if not line:  # 跳过空行
                continue
                
            try:
                dish_data = json.loads(line)
                
                # 确保dish_data是字典类型
                if not isinstance(dish_data, dict):
                    print(f"警告: 第{line_count}行不是字典格式: {line[:100]}...")
                    error_count += 1
                    continue
                    
                # 检查数据结构是否完整
                analysis = dish_data.get("analysis")
                if not analysis or not isinstance(analysis, dict):
                    print(f"警告: 第{line_count}行缺少analysis字段或格式不正确")
                    error_count += 1
                    continue
                    
                # 检查taste字段是否存在且是列表
                taste = analysis.get("taste")
                if taste is None:
                    print(f"警告: 第{line_count}行缺少taste字段")
                    error_count += 1
                    continue
                    
                if not isinstance(taste, list):
                    print(f"警告: 第{line_count}行taste字段不是列表格式: {taste}")
                    error_count += 1
                    continue
                
                # 检查是否包含"清淡"
                if query_taste in taste:
                    dish_name = dish_data.get("dish_name")
                    if dish_name:  # 确保dish_name存在
                        light_taste_dishes.append(dish_name)
                    else:
                        print(f"警告: 第{line_count}行缺少dish_name字段")
                        
            except json.JSONDecodeError as e:
                print(f"JSON解析错误(第{line_count}行): {e}\n行内容: {line[:100]}...")
                error_count += 1
                continue
            except Exception as e:
                print(f"未知错误(第{line_count}行): {e}\n行内容: {line[:100]}...")
                error_count += 1
                continue
                
    print(f"处理完成。总行数: {line_count}, 错误行数: {error_count}, 找到符合条件的菜品: {len(light_taste_dishes)}")
    return light_taste_dishes


if __name__ == '__main__':
    # 初始化构建器
    # builder = VectorDatabaseBuilder(model_name="bge")
    dish_database = "/data/ganshushen/Projects/BLE_CODE_OPEN/Faiss/database/dish_database"
    user_database = "/data/ganshushen/Projects/BLE_CODE_OPEN/Faiss/database/user_database"
    os.makedirs(dish_database, exist_ok=True)
    os.makedirs(user_database, exist_ok=True)

    # # 构建菜品向量库
    # builder.build_dish_database("/data/ganshushen/Projects/MainBranch_Git/Integrate/testTime/DualTowerTrain/dish_analysis_results_standardized_category.jsonl", 
    #                             dish_database)

    # # 构建用户向量库
    # builder.build_user_database("/data/ganshushen/Projects/MainBranch_Git/Integrate/testTime/DualTowerTrain/1000_user_processed_with_query_infos.jsonl", 
    #                             user_database)

    # 初始化推荐器
    recommender = DishRecommender(
        dish_db_path = dish_database,
        user_db_path = user_database,
    )

    # 示例使用
    user_id = 6  # 用户ID
    
    dish_list = ["一碗好米饭", "清蒸瘦肉汤", "西兰花小炒小份", "番茄榨菜鸡蛋汤"]  # 待推荐的菜品列表
    recall_dish_list = find_light_taste_dishes("/data/ganshushen/Projects/MainBranch_Git/Integrate/testTime/DualTowerTrain/dish_analysis_results_standardized_category.jsonl",
                                               query_taste="清淡")
    print(f"Recall dish nums: {len(recall_dish_list)}")

    # 获取推荐
    recommend_start_time = time.time()
    recommendations = recommender.recommend_dishes(user_id, recall_dish_list, top_k=30)
    recommend_total_time = (time.time() - recommend_start_time) * 1000
    print(f"recommend_total_time: {recommend_total_time}ms")

    # 打印结果
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. 菜品: {rec['dish']['dish_name']}")
        print(f"   相似度: {rec['similarity_score']:.4f}")
        # print(f"   适合人群: {rec['user_info']['age_range']} {rec['user_info']['health_conditions']}")
        # print(f"   菜品`分析: {rec['dish']['analysis']}\n")    