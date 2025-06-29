import json
import requests
from neo4j import GraphDatabase
from cypher_templates import get_cypher_template

def fetch_dishes_from_KG(
    taste=None, 
    texture=None, 
    function=None, 
    user_info=None, 
    meal=None
):
    """
    根据用户偏好（口味、口感、功效等）从 Neo4j 获取匹配的菜品
    """
    # 用用户信息自动补全缺失条件（仅当无功效约束时补全口味/口感）
    cypher_health_constraint = ""
    if user_info:
        
        user_taste = user_info.get("taste_preferences")
        user_texture = user_info.get("texture_preferences")
        user_health = user_info.get("health_conditions")
        user_health_constraint = user_info.get("health_constraint") # Health_Level_Agent的判断结果
        
        # 在检索知识图谱的时候，过滤掉对当前健康状况不利的食品
        if  user_health_constraint:
            # cypher_health_constraint = f"AND NOT EXISTS {{ (d:Dish)-[:UNSUITABLE_FOR]->(n:illness_unsuitable) WHERE n.name IN {json.dumps(user_health)} }}"
            illness_list = ", ".join([f"'{illness}'" for illness in user_health])
            cypher_health_constraint = f"AND NOT EXISTS {{ (d:Dish)-[:UNSUITABLE_FOR]->(n:illness_unsuitable) WHERE n.name IN [{illness_list}] }}"
            
        if not function:
            taste = taste or user_taste 
            texture = texture or user_texture
    
    # 确定使用的 Cypher 模板类型
    template_key = _get_template_key(taste, texture, function)
    
    # 构建查询参数（条件格式化）
    query_params = _prepare_query_params(taste, texture, function, meal, cypher_health_constraint)
    
    # 生成并执行查询
    cypher_query = get_cypher_template(template_key).format(**query_params)
    print(f"Cypher Query: {cypher_query}")
    
    results = execute_neo4j_query(cypher_query)
    return [record['d.name'] for record in results]



def _get_template_key(taste, texture, function):
    """根据条件组合，返回对应的 Cypher 模板键"""
    if function:
        return "only_function"
    elif taste and not texture and not function:
        return "only_taste"
    elif texture and not taste and not function:
        return "only_texture"
    elif taste and texture and not function:
        return "taste_and_texture"
    else:
        return "only_taste"  # 默认逻辑（可根据实际需求调整）


def _prepare_query_params(taste, texture, function, meal,health_constraint=None):
    """格式化查询条件，生成模板所需的参数字典"""
    params = {}
    
    # 处理多类型条件（口味、口感、功效）
    for cond_type, cond_value in [("taste", taste), ("texture", texture), ("function", function)]:
        if cond_value:
            condition = (
                f"IN {json.dumps(cond_value, ensure_ascii=False)}" 
                if isinstance(cond_value, list) 
                else f"= '{cond_value}'"
            )
            params[f"{cond_type}_condition"] = condition
    
    # 处理餐次条件
    if meal:
        meal_condition = f"= '{meal}'"
    else:
        meal_condition = "IS NULL"
    params["meal_condition"] = meal_condition
    
    # 处理健康约束
    if health_constraint:
        params["health_constraint"] = health_constraint
    
    return params

def execute_neo4j_query(cypher_query):
    
    uri = "bolt://localhost:7687"
    username = "neo4j"
    password = "Susan712"
    
    records = []
    
    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        
        with driver.session() as session:
            result = session.run(cypher_query)
            records = [record.data() for record in result]
                
    except Exception as e:
        print(f"数据库查询出错: {str(e)}")
        
    finally:
        if 'driver' in locals():
            driver.close()
    
    return records


def search_dish_recommendation(
    user_id, candidates, k=20, server_url="http://localhost:6666"
):
    """通过菜品推荐服务器搜索推荐菜品"""

    if len(candidates) == 0:
        return []

    response = requests.post(
        f"{server_url}/search",
        json={"user_id": user_id, "candidate_dishes": candidates, "top_k": k},
    )

    if response.status_code != 200:
        error_msg = response.json().get("error", "Unknown error")
        print(f"错误: {error_msg}")
        return None

    results = response.json()

    # 提取推荐结果
    recommendations = results.get("recommendations", [])

    # 格式化返回结果，包含菜品名和相似度分数
    formatted_results = []
    for rec in recommendations:
        formatted_results.append(rec["dish_name"])

    return formatted_results


# ------------------------ 连通性测试函数 ------------------------ 
def test_neo4j_connectivity():
    """测试 Neo4j 数据库连通性（修复字段名错误）"""
    # ✅ 修复：将 `version` 改为 `versions`
    test_query = "CALL dbms.components() YIELD name, versions RETURN name, versions LIMIT 10000"
    
    try:
        results = execute_neo4j_query(test_query)
        if results:
            print(f"✅ Neo4j 连通性测试通过！")
            print(f"数据库核心组件：{results[0]}")
        else:
            print("⚠️ 查询返回空结果，可能数据库未初始化或无权限访问。")
    except Exception as e:
        print(f"❌ Neo4j 连通性测试失败：{str(e)}")


# ------------------------ 脚本入口（测试触发） ------------------------ 
if __name__ == "__main__":
    # 1. 测试数据库连通性
    test_neo4j_connectivity()
    
    # 2. 测试实际业务逻辑（示例调用）
    sample_user_info = {
        "taste_preferences": "清淡",
        "texture_preferences": "软烂",
        "region": "西南地区",
        "health_conditions": ["糖尿病"],
        "health_constraint":True
    }
    try:
        dishes = fetch_dishes_from_KG(
            taste="清淡", 
            user_info=sample_user_info, 
            meal="晚餐"
        )
        print(f"\n业务逻辑测试 - 匹配的菜品：{dishes}")
    except Exception as e:
        print(f"\n业务逻辑测试失败：{str(e)}")
