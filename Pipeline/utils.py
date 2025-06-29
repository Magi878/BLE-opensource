import time
import json
import requests
from neo4j import GraphDatabase

def neo4j_search(cypher_query):
    
    uri = "bolt://localhost:7687"
    username = "neo4j"
    password = "xxxxxx"
    
    # 初始化records变量
    records = []
    
    try:
        # 创建驱动程序
       
        driver = GraphDatabase.driver(uri, auth=(username, password))
        
        # 执行查询
        with driver.session() as session:
            result = session.run(cypher_query)
            records = [record.data() for record in result]
                
    except Exception as e:
        print(f"数据库查询出错: {str(e)}")
        
    finally:
        # 关闭数据库连接
        if 'driver' in locals():
            driver.close()
    
    return records


def get_embedding(query):
    """获取查询的向量嵌入"""
    
    url = "http://localhost:8001/embed"
    data = {
        "texts": [query] if isinstance(query, str) else query,  # 确保输入是列表
        "instruction": "为这个句子生成表示以用于检索相关文章："
    }

    response = requests.post(url, json=data)
    result = response.json()

    return result["embeddings"]

def get_meal():
    """根据当前时间判断用餐时段"""
    current_hour = time.localtime().tm_hour

    # 判断时间段对应的用餐类型
    if 5 <= current_hour < 10:
        return "早餐"
    elif 10 <= current_hour < 15:
        return "午餐" 
    elif 15 <= current_hour < 22:
        return "晚餐"
    else:
        return "夜宵"

def get_user_info(idx):
    """读取指定行的老人信息"""
    info_path = r"Pipeline/data/UserData/1000_user_processed_with_query_infos.jsonl"
    
    with open(info_path, 'r', encoding='utf-8') as f:
        # 逐行读取直到指定行
        for i, line in enumerate(f):
            if i == idx:
                # 解析JSON行数据
                user_data = json.loads(line.strip())
                return user_data
    return None

def search_faiss(query, k=50, server_url='http://localhost:5006'):
    """通过FAISS服务器搜索"""
 
    query_vectors = get_embedding(query)
 
    response = requests.post(f"{server_url}/search", json={
        'vectors': query_vectors,
        'k': k
    })
    
    if response.status_code != 200:
        print(f"错误: {response.json().get('error')}")
        return None
    
    results = response.json()['results']

    return results

def extract_info_from_web(user_info, user_info_web):
    """
    从web数据中提取用户信息，并检查空字段

    Args:
        user_info (dict): 要更新的用户信息字典
        user_info_web (dict): 从web获取的原始用户信息

    Returns:
        dict: 更新后的用户信息字典
    """
    # 定义所有要提取的字段
    fields_to_extract = [
        "gender",
        "age_range",
        "region",
        "health_conditions",
        "taste_preferences",
        "texture_preferences",
    ]

    # 记录缺失的字段
    missing_fields = []

    for field in fields_to_extract:
        value = user_info_web.get(field, "")
        user_info[field] = value

        # 检查是否为空值
        if not value:  # 空字符串、None等都会被认为是空值
            missing_fields.append(field)

    # 如果有缺失字段，打印提示信息
    if missing_fields:
        print(f"警告: 以下字段为空: {', '.join(missing_fields)}")
        print("建议: 请检查数据源或提示用户补充这些信息")

def extract_json_from_response(data):
    """
    快速从输入字典中提取并解析intent_results和rewrite_results中的JSON数据
    返回包含两个字典的元组 (intent_dict, rewrite_dict)

    异常处理包括：
    1. 输入数据非字典类型
    2. 缺少必要键
    3. JSON格式不正确
    4. 其他解析错误

    如果发生错误，返回的字典中将包含错误信息
    """
    # 初始化默认返回字典
    default_result = {"error": "Failed to parse JSON"}
    intent_dict = rewrite_dict = default_result

    try:
        # 检查输入是否为字典
        if not isinstance(data, dict):
            raise ValueError("Input data must be a dictionary")

        # 检查必要键是否存在
        required_keys = ["intent_results", "rewrite_results"]
        for key in required_keys:
            if key not in data:
                raise KeyError(f"Missing required key: {key}")

        try:
            # 提取intent_results中的JSON部分
            intent_parts = data["intent_results"].split("\n\n", 2)
            if len(intent_parts) < 3:
                raise ValueError("Intent results format incorrect")
            intent_json = intent_parts[-1]
            intent_dict = json.loads(intent_json)
        except Exception as e:
            intent_dict = {
                "error": f"Failed to parse intent_results: {str(e)}",
                "original_data": data.get("intent_results"),
            }

        try:
            # 提取rewrite_results中的JSON部分
            rewrite_parts = data["rewrite_results"].split("\n\n", 2)
            if len(rewrite_parts) < 3:
                raise ValueError("Rewrite results format incorrect")
            rewrite_json = rewrite_parts[-1]
            rewrite_dict = json.loads(rewrite_json)
        except Exception as e:
            rewrite_dict = {
                "error": f"Failed to parse rewrite_results: {str(e)}",
                "original_data": data.get("rewrite_results"),
            }

    except Exception as e:
        # 处理整体错误
        error_dict = {"error": f"Processing failed: {str(e)}"}
        intent_dict = rewrite_dict = error_dict

    return intent_dict, rewrite_dict

def hasHallucination_rerank(recommend,products_list)->bool:
    if recommend not in products_list:
        print("LLM出现了幻觉，使用BGE重排结果")
        return True
    return False

def hasHallucination_rewrite(rewrite_result:str)->bool:
    pass

def hasHallucination_intent(intent_result:str)->bool:
    pass
