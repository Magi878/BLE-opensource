import os
import json
import asyncio
from tqdm import tqdm
from openai import OpenAI
from loguru import logger
import concurrent.futures
from prompt import HEALTH_LEVEL_AGENT_SYS_PROMPT  # 导入提示词

# 输入和输出文件路径
input_file = "Pipeline/data/UserData/1000_user_processed_with_query_infos.jsonl"
output_file = "Pipeline/data/UserData/1000_user_processed_with_health_constraints.jsonl"

# QWEN-MAX API配置
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY = "sk-xxx"

def format_user_info(user_data):
    """将用户数据格式化为提示词中需要的用户信息格式"""
    user_info = f"""用户基本信息：
- 用户ID：{user_data.get('user_id', '未知')}
- 姓名：{user_data.get('name', '未知')}
- 性别：{user_data.get('gender', '未知')}
- 年龄范围：{user_data.get('age_range', '未知')}
- 地区：{user_data.get('region', '未知')}
- 健康状况：{', '.join(user_data.get('health_conditions', []))}
- 口味偏好：{', '.join(user_data.get('taste_preferences', []))}
- 质地偏好：{', '.join(user_data.get('texture_preferences', []))}
- 当前查询：{user_data.get('query', '未知')}

用户订单历史（最近30天）：
{format_order_history(user_data.get('order_history', []))}"""
    return user_info

def format_order_history(order_history):
    """将订单历史格式化为易读的文本"""
    if not order_history:
        return "无订单历史记录"
    
    formatted = ""
    for order in order_history:
        formatted += f"第{order['date']}天: {', '.join(order['dishes'])}\n"
    return formatted

# 使用线程池执行同步API调用
async def call_api(client, model, messages, temperature, max_tokens):
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(
            pool,
            lambda: client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
        )
    return result

async def analyze_user_health_async(user_data, temperature=0.7, max_retries=3):
    """异步使用QWEN-MAX API分析用户健康状况并返回饮食限制建议"""
    # 格式化用户信息
    user_info = format_user_info(user_data)
    
    # 使用prompt.py中的提示词模板
    system_prompt = HEALTH_LEVEL_AGENT_SYS_PROMPT.format(user_info=user_info)
    
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            client = OpenAI(
                api_key=API_KEY,
                base_url=API_URL,
            )
            
            # 使用线程池执行同步API调用
            completion = await call_api(
                client,
                model="qwen-max",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": "请根据以上用户信息判断是否需要健康约束。"
                    }
                ],
                temperature=temperature,
                max_tokens=100  # 减少token数量，因为只需要返回True/False
            )
            
            response_content = completion.choices[0].message.content.strip()
            print(response_content)
            
            logger.info(f"API返回内容: {response_content}")
            
            # 解析返回结果，判断是否需要约束
            needs_restriction = False
            if "True" in response_content or "true" in response_content:
                needs_restriction = True
            elif "False" in response_content or "false" in response_content:
                needs_restriction = False
            
            return needs_restriction
            
        except Exception as e:
            logger.error(f"API调用出错: {str(e)}")
            retry_count += 1
            temperature += 0.1
            await asyncio.sleep(1)  # 短暂等待后重试
    
    # 如果所有重试都失败，返回默认结果
    return {
        "needs_restriction": False,
        "raw_response": "API调用失败，使用默认值"
    }

async def process_batch_async(batch_data, output_file_path, batch_size=10):
    """异步处理一批用户数据"""
    tasks = []
    results = []
    
    # 创建异步任务
    for user_data in batch_data:
        task = asyncio.create_task(analyze_user_health_async(user_data))
        tasks.append((user_data, task))
    
    # 等待所有任务完成
    for user_data, task in tasks:
        try:
            health_constraint = await task
            # 添加health_constraint字段
            user_data["health_constraint"] = health_constraint
            results.append(user_data)
        except Exception as e:
            logger.error(f"等待任务完成时出错: {str(e)}")
            # 添加默认的health_constraint字段
            user_data["health_constraint"] = {
                "needs_restriction": False,
                "raw_response": "处理失败，使用默认值"
            }
            results.append(user_data)
    
    # 写入结果
    with open(output_file_path, 'a', encoding='utf-8') as output_file:
        for user_data in results:
            output_file.write(json.dumps(user_data, ensure_ascii=False) + '\n')
    
    return len(results)

async def process_jsonl_file_async(limit=None):
    """异步处理JSONL文件并添加health_constraint字段"""
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 清空输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        pass
    
    # 读取输入文件
    user_data_list = []
    with open(input_file, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            user_data = json.loads(line.strip())
            user_data_list.append(user_data)
            if limit and len(user_data_list) >= limit:
                break
    
    total_users = len(user_data_list)
    print(f"共读取 {total_users} 条用户数据")
    
    # 批处理大小
    batch_size = 10
    
    # 使用tqdm创建进度条
    with tqdm(total=total_users, desc="处理用户数据", ncols=100) as pbar:
        for i in range(0, total_users, batch_size):
            batch_data = user_data_list[i:i+batch_size]
            
            # 异步处理批次
            processed = await process_batch_async(batch_data, output_file, batch_size)
            pbar.update(processed)

async def main_async():
    print(f"开始处理文件: {input_file}")
    # 只处理前100条数据
    await process_jsonl_file_async(limit=None)
    print(f"处理完成，结果已保存至: {output_file}")

if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main_async())