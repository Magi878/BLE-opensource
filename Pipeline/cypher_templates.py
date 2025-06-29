

CYPHER_TEMPLATES = {
    
    # 单一条件模板
    "only_taste": """
    MATCH (d:Dish)-[:HAS_TASTE]->(t:Taste) 
    WHERE t.name {condition} 
    AND EXISTS {{
        MATCH (d)-[:SUITABLE_FOR_MEAL]->(m_check:Meal) 
        WHERE m_check.name {meal_condition}
    }}
    {health_constraint}
    RETURN d.name, collect(t.name) as taste LIMIT 10000
    """,
    
    "only_texture": """
    MATCH (d:Dish)-[:HAS_TEXTURE]->(t:Texture) 
    WHERE t.name {condition} 
    AND EXISTS {{
        MATCH (d)-[:SUITABLE_FOR_MEAL]->(m_check:Meal) 
        WHERE m_check.name {meal_condition}
    }}
    {health_constraint}
    RETURN d.name, collect(t.name) as texture LIMIT 10000
    """,
    
    "only_function": """
    MATCH (d:Dish)-[:HAS_NUTRITION]->(f:Nutrition) 
    WHERE f.name {condition} 
    AND EXISTS {{
        MATCH (d)-[:SUITABLE_FOR_MEAL]->(m_check:Meal) 
        WHERE m_check.name {meal_condition}
    }}
    {health_constraint}
    RETURN d.name, collect(f.name) as function LIMIT 10000
    """,
    
    # 双条件模板
    
    "taste_and_texture": """
    MATCH (d:Dish)
    WHERE EXISTS {{
        MATCH (d)-[:HAS_TASTE]->(t_check:Taste)
        WHERE t_check.name {taste_condition}
    }}
    AND EXISTS {{
        MATCH (d)-[:HAS_TEXTURE]->(tx_check:Texture)
        WHERE tx_check.name {texture_condition}
    }}
    AND EXISTS {{
     MATCH (d)-[:SUITABLE_FOR_MEAL]->(m_check:Meal) 
     WHERE m_check.name {meal_condition}
    }}
    {health_constraint}
    WITH d
    OPTIONAL MATCH (d)-[:HAS_TASTE]->(t:Taste)
    OPTIONAL MATCH (d)-[:HAS_TEXTURE]->(tx:Texture)
    OPTIONAL MATCH (d)-[:HAS_NUTRITION]->(f:Nutrition)
    OPTIONAL MATCH (d)-[:SUITABLE_FOR_MEAL]->(m:Meal)
    RETURN d.name, 
           collect(DISTINCT t.name) as tastes,
           collect(DISTINCT tx.name) as textures,
           collect(DISTINCT m.name) as meal_times
    LIMIT 10000
    """,
    
    # 口味和功效
    "taste_and_function": """
    MATCH (d:Dish)
    WHERE EXISTS {{
        MATCH (d)-[:HAS_TASTE]->(t_check:Taste)
        WHERE t_check.name {taste_condition}
    }}
    AND EXISTS {{
        MATCH (d)-[:HAS_NUTRITION]->(f_check:Nutrition)
        WHERE f_check.name {function_condition}
    }}
    AND EXISTS {{
     MATCH (d)-[:SUITABLE_FOR_MEAL]->(m_check:Meal) 
     WHERE m_check.name {meal_condition}
    }}
    {health_constraint}
    WITH d
    OPTIONAL MATCH (d)-[:HAS_TASTE]->(t:Taste)
    OPTIONAL MATCH (d)-[:HAS_NUTRITION]->(f:Nutrition)
    OPTIONAL MATCH (d)-[:SUITABLE_FOR_MEAL]->(m:Meal)
    RETURN d.name,
            collect(DISTINCT t.name) as tastes,
            collect(DISTINCT f.name) as functions,
            collect(DISTINCT m.name) as meal_times
    LIMIT 10000
    """,
    
    # 口感和功效
    "texture_and_function": """
    MATCH (d:Dish)
    WHERE EXISTS {{
        MATCH (d)-[:HAS_TEXTURE]->(tx_check:Texture)
        WHERE tx_check.name {texture_condition}
    }}
    AND EXISTS {{
        MATCH (d)-[:HAS_NUTRITION]->(f_check:Nutrition)
        WHERE f_check.name {function_condition}
    }}
    AND EXISTS {{
     MATCH (d)-[:SUITABLE_FOR_MEAL]->(m_check:Meal) 
     WHERE m_check.name {meal_condition}
    }}
    {health_constraint}
    WITH d
    OPTIONAL MATCH (d)-[:HAS_TEXTURE]->(tx:Texture)
    OPTIONAL MATCH (d)-[:HAS_FUNCTION]->(f:Function)
    OPTIONAL MATCH (d)-[:SUITABLE_FOR_MEAL]->(m:Meal)
    RETURN d.name,
           collect(DISTINCT tx.name) as textures,
           collect(DISTINCT f.name) as functions,
           collect(DISTINCT m.name) as meal_times
    LIMIT 10000
    """,
}

def get_cypher_template(template_key):
    """根据模板键获取对应的Cypher查询模板"""
    return CYPHER_TEMPLATES.get(template_key, "")