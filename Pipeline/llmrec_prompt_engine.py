import json
import random
import asyncio
import aiofiles
from typing import List, Dict, Union, Optional
from concurrent.futures import ThreadPoolExecutor


class FastPromptBuilder:
    """异步LLM重排序提示词生成器
    
    使用异步IO提高文件加载和数据处理性能。
    """
    
    def __init__(self, 
                 user_jsonl_path: str = "Pipeline/data/UserData/1000_user_processed_with.jsonl",
                 dish_jsonl_path: str = "Pipeline/data/DishData/dish_infos.jsonl",
                 weather_jsonl_path: str = "Pipeline/data/UserData/weather_info.jsonl",
                 max_workers: int = 3):
        """初始化异步提示词生成器
        
        Args:
            user_jsonl_path: 用户信息文件路径
            dish_jsonl_path: 菜品信息文件路径
            weather_jsonl_path: 天气信息文件路径
            max_workers: 线程池最大工作线程数
        """
        self.user_jsonl_path = user_jsonl_path
        self.dish_jsonl_path = dish_jsonl_path
        self.weather_jsonl_path = weather_jsonl_path
        self.max_workers = max_workers
        
        # 数据缓存
        self.users_cache: Dict[int, Dict] = {}
        self.dishes_cache: Dict[str, Dict] = {}
        self.weather_cache: List[Dict] = []
        
        # 初始化状态
        self._initialized = False
        self._initialization_lock = asyncio.Lock()
        
        # 线程池用于CPU密集型操作
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出"""
        await self.close()
    
    async def _load_jsonl_async(self, path: str) -> List[Dict]:
        """异步加载JSONL文件"""
        try:
            async with aiofiles.open(path, 'r', encoding='utf-8') as f:
                content = await f.read()
                
            # 在线程池中执行JSON解析（CPU密集型操作）
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.executor,
                self._parse_jsonl_content,
                content
            )
        except FileNotFoundError:
            print(f"警告：文件不存在 {path}")
            return []
        except Exception as e:
            print(f"错误：加载文件 {path} 失败: {e}")
            return []
    
    def _parse_jsonl_content(self, content: str) -> List[Dict]:
        """在线程池中解析JSONL内容"""
        lines = content.strip().split('\n')
        result = []
        for line in lines:
            line = line.strip()
            if line:
                try:
                    result.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"警告：跳过无效JSON行: {line[:50]}... 错误: {e}")
                    continue
        return result
    
    async def initialize(self):
        """异步初始化，加载所有必要的数据文件"""
        if self._initialized:
            return
        
        async with self._initialization_lock:
            if self._initialized:
                return
            
            print("开始异步加载数据文件...")
            start_time = asyncio.get_event_loop().time()
            
            # 并发加载所有文件
            tasks = [
                self._load_jsonl_async(self.user_jsonl_path),
                self._load_jsonl_async(self.dish_jsonl_path),
                self._load_jsonl_async(self.weather_jsonl_path)
            ]
            
            users_data, dishes_data, weather_data = await asyncio.gather(*tasks)
            
            # 在线程池中构建索引（CPU密集型操作）
            loop = asyncio.get_event_loop()
            self.users_cache, self.dishes_cache = await asyncio.gather(
                loop.run_in_executor(self.executor, self._build_users_index, users_data),
                loop.run_in_executor(self.executor, self._build_dishes_index, dishes_data)
            )
            
            self.weather_cache = weather_data
            
            load_time = asyncio.get_event_loop().time() - start_time
            print(f"数据加载完成，耗时: {load_time*1000:.2f} ms")
            print(f"已加载 {len(self.users_cache)} 个用户，{len(self.dishes_cache)} 个菜品，{len(self.weather_cache)} 条天气信息")
            
            self._initialized = True
    
    def _build_users_index(self, users_data: List[Dict]) -> Dict[int, Dict]:
        """构建用户索引"""
        return {u["id"]: u for u in users_data if "id" in u}
    
    def _build_dishes_index(self, dishes_data: List[Dict]) -> Dict[str, Dict]:
        """构建菜品索引"""
        return {d["dish_name"]: d for d in dishes_data if "dish_name" in d}
    
    async def close(self):
        """关闭资源"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
    
    async def _extract_dish_names_async(self, dish_list: Union[List[str], List[Dict]]) -> List[str]:
        """异步提取菜品名称"""
        if not dish_list:
            return []
        
        # 对于大量数据，在线程池中处理
        if len(dish_list) > 100:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.executor,
                self._extract_dish_names_sync,
                dish_list
            )
        else:
            return self._extract_dish_names_sync(dish_list)
    
    def _extract_dish_names_sync(self, dish_list: Union[List[str], List[Dict]]) -> List[str]:
        """同步提取菜品名称（用于线程池）"""
        if not dish_list:
            return []
        
        # 检查第一个元素的类型来判断整个列表的格式
        if isinstance(dish_list[0], str):
            # 原格式：直接是菜品名称列表
            return dish_list
        elif isinstance(dish_list[0], dict):
            # 新格式：字典列表，需要提取DishName字段
            dish_names = []
            for dish_dict in dish_list:
                if isinstance(dish_dict, dict) and "DishName" in dish_dict:
                    dish_names.append(dish_dict["DishName"])
                else:
                    print(f"警告：跳过无效的菜品项: {dish_dict}")
            return dish_names
        else:
            print(f"警告：不支持的菜品列表格式: {type(dish_list[0])}")
            return []
    
    async def generate_prompt(self, 
                             query: str, 
                             user_id: int, 
                             dish_list: Union[List[str], List[Dict]]) -> str:
        """异步生成LLM重排序提示词
        
        Args:
            query: 用户查询
            user_id: 用户ID
            dish_list: 候选菜品列表，支持字符串列表或字典列表格式
            
        Returns:
            构造好的提示词字符串
            
        Raises:
            ValueError: 当找不到指定用户ID时
            RuntimeError: 当未初始化时
        """
        # 确保已初始化
        if not self._initialized:
            await self.initialize()
        
        # 检查用户是否存在
        if user_id not in self.users_cache:
            raise ValueError(f"未找到 id 为 {user_id} 的用户信息")
        
        # 异步提取菜品名称
        dish_names = await self._extract_dish_names_async(dish_list)
        print(f"提取到 {len(dish_names)} 个菜品名称")
        
        # 获取用户信息（深拷贝）
        user = self.users_cache[user_id].copy()
        
        # 在线程池中处理菜品匹配（可能是CPU密集型操作）
        loop = asyncio.get_event_loop()
        simplified_dishes = await loop.run_in_executor(
            self.executor,
            self._match_dishes_sync,
            dish_names
        )
        
        # 添加菜品和随机天气信息到用户数据
        user["dishes"] = simplified_dishes
        user["weather"] = random.choice(self.weather_cache) if self.weather_cache else {}
        
        # 在线程池中生成提示词（文本处理）
        prompt = await loop.run_in_executor(
            self.executor,
            self._build_prompt_sync,
            query, user
        )
        
        return prompt
    
    def _match_dishes_sync(self, dish_names: List[str]) -> List[Dict]:
        """同步匹配菜品信息（用于线程池）"""
        simplified_dishes = []
        for name in dish_names:
            matched = self.dishes_cache.get(name)
            if matched:
                simplified_dishes.append({
                    "dish_name": matched.get("dish_name"),
                    "category": matched.get("category"),
                    "nutrition": matched.get("nutrition"),
                    "meal": matched.get("meal"),
                    "region": matched.get("region")
                })
            else:
                print(f"警告：未找到菜品信息: {name}")
        return simplified_dishes
    
    def _build_prompt_sync(self, query: str, user: Dict) -> str:
        """同步构建提示词（用于线程池）"""
        user_info = self._parse_user_info(user)
        dish_info = self._parse_dishes(user)
        return f"用户指令：{query}{dish_info}{user_info}"
    
    async def generate_batch_prompts(self, 
                                   requests: List[Dict]) -> List[str]:
        """批量异步生成提示词
        
        Args:
            requests: 请求列表，每个请求包含 {"query": str, "user_id": int, "dish_list": List}
            
        Returns:
            提示词列表
        """
        if not self._initialized:
            await self.initialize()
        
        tasks = []
        for req in requests:
            task = self.generate_prompt(
                req["query"],
                req["user_id"], 
                req["dish_list"]
            )
            tasks.append(task)
        
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def _parse_user_info(self, user_json: Dict) -> str:
        """从用户JSON中提取信息并格式化"""
        user = user_json
        parts = ["\n用户信息：性别"]
        
        if "gender" in user:
            parts.append(user["gender"])
        if "age_range" in user:
            parts.append(f'年龄{user["age_range"]}')
        if "region" in user:
            parts.append(f'地区：{user["region"]}')
        if "health_conditions" in user and user["health_conditions"]:
            parts.append("健康状况：" + "、".join(user["health_conditions"]))
        if "taste_preferences" in user and user["taste_preferences"]:
            parts.append("口味偏好：" + "、".join(user["taste_preferences"]))
        if "texture_preferences" in user and user["texture_preferences"]:
            parts.append("口感偏好：" + "、".join(user["texture_preferences"]))
        if "meal" in user:
            parts.append(f'当前用餐：{user["meal"]}')
        
        # 处理天气信息
        if "weather" in user:
            weather = user["weather"]
            weather_str = []
            parts.append(f"\n天气情况：")
            if "weather" in weather:
                weather_str.append(weather["weather"])
            if "temperature" in weather:
                weather_str.append(f'温度{weather["temperature"]}℃')
            if "humidity" in weather:
                weather_str.append(f'湿度{weather["humidity"]}%')
            if "season" in weather:
                weather_str.append(f'季节{weather["season"]}')
            if weather_str:
                parts.append("天气：" + "，".join(weather_str))
        
        return "".join(parts) + "。"
    
    def _parse_dishes(self, user_dict: Dict) -> str:
        """从用户信息中提取菜品并生成描述文本"""
        if not isinstance(user_dict, dict) or "dishes" not in user_dict:
            return "无候选菜品信息"

        dishes = user_dict["dishes"]
        if not isinstance(dishes, list):
            return "候选菜品格式错误（非列表）"

        descriptions = []
        for dish in dishes:
            if isinstance(dish, str):
                try:
                    dish = json.loads(dish)
                except Exception:
                    continue

            if not isinstance(dish, dict):
                continue

            parts = []
            name = dish.get("dish_name") or dish.get("DishName")
            if not name:
                continue
            parts.append(name)

            # 提取菜品属性
            for attr, prefix in [
                ("category", "分类："),
                ("meal", "适合"),
                ("season", "适合"),
                ("region", "适合"),
                ("ill_adapt", "适合"),
                ("nutrition", "营养价值：")
            ]:
                value = dish.get(attr) or dish.get(attr.capitalize())
                if value:
                    if isinstance(value, list):
                        value_str = "、".join(value)
                    else:
                        value_str = str(value)
                    parts.append(f"{prefix}{value_str}")

            # 格式化菜品描述
            if len(parts) > 1:
                main_name = parts[0]
                details = "，".join(parts[1:])
                descriptions.append(f'[{main_name}({details})]')
            else:
                descriptions.append(parts[0])

        if not descriptions:
            return "无有效候选菜品信息"

        return "\n候选菜品：" + "；".join(descriptions)
    
    async def get_user_info(self, user_id: int) -> Optional[Dict]:
        """异步获取指定用户的信息"""
        if not self._initialized:
            await self.initialize()
        return self.users_cache.get(user_id)
    
    async def get_dish_info(self, dish_name: str) -> Optional[Dict]:
        """异步获取指定菜品的信息"""
        if not self._initialized:
            await self.initialize()
        return self.dishes_cache.get(dish_name)
    
    async def get_stats(self) -> Dict[str, int]:
        """异步获取数据统计信息"""
        if not self._initialized:
            await self.initialize()
        return {
            "users_count": len(self.users_cache),
            "dishes_count": len(self.dishes_cache),
            "weather_count": len(self.weather_cache)
        }

async def main():
    """异步主函数示例"""
    
    # 方式1：使用异步上下文管理器（推荐）
    print("=== 使用异步上下文管理器 ===")
    async with FastPromptBuilder() as generator:
        stats = await generator.get_stats()
        print(f"数据统计: {stats}")
        
        # 单个请求测试
        user_id = 2
        query = "今天吃点什么"
        dish_names = ['红烧土豆饭', '红烧豆腐饭', '小碗海带炖豆腐']
        
        start_time = asyncio.get_event_loop().time()
        prompt = await generator.generate_prompt(query, user_id, dish_names)
        generation_time = asyncio.get_event_loop().time() - start_time
        
        print(f"单个请求生成耗时: {generation_time*1000:.2f} ms")
        print(f"生成的提示词:\n{prompt[:200]}...")
        
        # 批量请求测试
        print("\n=== 批量请求测试 ===")
        batch_requests = [
            {"query": f"请求{i}", "user_id": 2, "dish_list": dish_names[:3]}
            for i in range(5)
        ]
        
        start_time = asyncio.get_event_loop().time()
        batch_results = await generator.generate_batch_prompts(batch_requests)
        batch_time = asyncio.get_event_loop().time() - start_time
        
        print(f"批量处理 {len(batch_requests)} 个请求耗时: {batch_time*1000:.2f} ms")
        print(f"平均每个请求: {batch_time*1000/len(batch_requests):.2f} ms")
    
    # 方式2：手动管理生命周期
    print("\n=== 手动管理生命周期 ===")
    generator = FastPromptBuilder()
    
    try:
        await generator.initialize()
        
        # 测试新格式（字典列表）
        dish_list_new = [
            {"DishName": "红烧土豆饭", "score": 0.95},
            {"DishName": "红烧豆腐饭", "score": 0.88},
            {"DishName": "小碗海带炖豆腐", "score": 0.82}
        ]
        
        start_time = asyncio.get_event_loop().time()
        prompt_new = await generator.generate_prompt("今天想吃什么", 2, dish_list_new)
        generation_time = asyncio.get_event_loop().time() - start_time
        
        print(f"新格式生成耗时: {generation_time*1000:.2f} ms")
        print(f"生成的提示词:\n{prompt_new[:200]}...")
        
    finally:
        await generator.close()


if __name__ == "__main__":
    # 运行异步示例
    asyncio.run(main())