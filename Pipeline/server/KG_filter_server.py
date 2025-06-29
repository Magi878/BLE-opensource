from flask import Flask, request, jsonify
import os
import time
import json
import pickle
import faiss
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm
from FlagEmbedding import FlagModel

from KG_filter_dataset_build import DishRecommender

app = Flask(__name__)
'''
该服务使用用户历史菜品数据，对知识图谱召回的菜品做筛选
'''

# 初始化推荐器
dish_database = "/data/ganshushen/Projects/BLE_CODE_OPEN/Faiss/database/dish_database"
user_database = "/data/ganshushen/Projects/BLE_CODE_OPEN/Faiss/database/user_database"

recommender = DishRecommender(
    dish_db_path = dish_database,
    user_db_path = user_database,
)

@app.route('/search', methods=['POST'])
def search():
    try:
        # 获取请求参数
        data = request.json
        user_id = data.get('user_id')
        candidate_dishes = data.get('candidate_dishes', [])
        top_k = data.get('top_k', 20)
        
        if not user_id:
            return jsonify({"error": "user_id is required"}), 400
        
        # 获取推荐结果
        recommend_start = time.time()
        recommendations = recommender.recommend_dishes(
            user_id=user_id,
            dish_names=candidate_dishes,
            top_k=top_k
        )
        recommend_total_time = (time.time() - recommend_start) * 1000
        
        # 简化返回结果
        simplified_results = []
        for rec in recommendations:
            simplified_results.append({
                "dish_name": rec['dish']['dish_name'],
                "similarity_score": rec['similarity_score'],
                "analysis": rec['dish']['analysis'],
                "category": rec['dish']['category']
            })
        
        return jsonify({
            "user_id": user_id,
            "recommendations": simplified_results,
            "count": len(simplified_results),
            "time_ms": recommend_total_time
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # 检查数据库是否存在
    if not os.path.exists(dish_database) or not os.path.exists(user_database):
        print("Error: Database directories not found. Please build databases first.")
        exit(1)
    
    app.run(host='0.0.0.0', port=6666, debug=True)