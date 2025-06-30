import asyncio
import aiohttp


from ast import Dict
from typing import Dict, Any, List 

from llmrec_prompt_engine import FastPromptBuilder

from query_resolver import async_llm_intent,query_rewrite_LLM_FewShot

from utils import (
    get_meal,
    get_user_info,
    hasHallucination_rerank,
    search_faiss,
    extract_json_from_response
)

from llmrec_engine import async_llm_rerank, async_rerank
from kg_retriever import fetch_dishes_from_KG,search_dish_recommendation
from multi_path_retriever import BM25_Retrieval,rrf_fusion,Multi_Path_Search
from ASR_Paraformer import ASR



async def parallel_rerank(builder, user_id, query, candidates: List[Dict], session):
    """并行执行BGE和LLM重排序"""
    timeout_seconds = 30.0

    print("🚀 Starting parallel rerank tasks...")

    bge_task = asyncio.create_task(
        asyncio.wait_for(
            async_rerank(query, candidates, session), timeout=timeout_seconds
        )
    )

    llm_user_prompt = await builder.generate_prompt(query, user_id, candidates)
    print("llm_user_prompt",llm_user_prompt)
    llm_task = asyncio.create_task(
        asyncio.wait_for(
            async_llm_rerank(llm_user_prompt, session), timeout=timeout_seconds
        )
    )

    bge_results = None
    llm_results = None
  
    try:
        bge_results = await bge_task
    except Exception as e:
        print(f"🔴 Error in BGE rerank task: {e}")
    
    try:
        llm_results = await llm_task
    except Exception as e:
        print(f"🔴 Error in LLM rerank task: {e}")

    print({
        "bge_results": bge_results,
        "llm_results": llm_results,
    })
    return {
        "bge_results": bge_results,
        "llm_results": llm_results,
    }

async def Parrallel_Multi_Path_Search(
    query: str, k: int, itemname: str = None
) -> List[dict]:
    # Define all the async tasks we might need
    async def bm25_task():
        candidates, _ = BM25_Retrieval(query, k=k)
        return candidates

    async def faiss_task(queries):
        return search_faiss(queries, k=k)

    # Run the appropriate tasks in parallel based on whether itemname exists
    if itemname is not None:
        # Start all tasks concurrently
        bm25_task_obj = (
            bm25_task() if False else asyncio.create_task(asyncio.sleep(0))
        )  # bm25_candidates is empty list in original
        faiss_task_obj = faiss_task([query, itemname])

        # Wait for all tasks to complete
        _, vector_candidates = await asyncio.gather(bm25_task_obj, faiss_task_obj)

        # Process results
        bm25_candidates = []  # As per original code
        query_candidates = vector_candidates[0]
        item_candidates = vector_candidates[1]

        rrf_result = rrf_fusion(
            [query_candidates, item_candidates, bm25_candidates], k=k
        )
    else:
        # Start all tasks concurrently
        bm25_task_obj = bm25_task()
        faiss_task_obj = faiss_task([query])

        # Wait for all tasks to complete
        bm25_candidates, vector_candidates = await asyncio.gather(
            bm25_task_obj, faiss_task_obj
        )

        # Process results
        query_candidates = vector_candidates[0]
        rrf_result = rrf_fusion([query_candidates, bm25_candidates], k=k)

    return rrf_result


async def parallel_search_combined(
    query: str,
    k: int = 100,
    itemname: str = None,
    taste: str = None,
    texture: str = None,
    function: str = None,
    user_info: Dict[str, Any] = None,
    meal: str = None,
) -> Dict[str, List[str]]:
    """
    并行执行Cypher查询和多路召回搜索，保留各自原始结果

    返回:
        {
            "cypher_results": List[str],  # Cypher查询结果
            "multi_path_results": List[str],  # 多路召回结果
        }
    """
    if itemname is None:
        itemname = query

    # 获取事件循环
    loop = asyncio.get_event_loop()

    # 并行执行两个任务
    cypher_task = loop.run_in_executor(
        None, fetch_dishes_from_KG, taste, texture, function, user_info, meal
    )

    multi_path_task = Parrallel_Multi_Path_Search(query, k, itemname)
    # multi_path_task = Multi_Path_Search(query, k, itemname)

    # 等待两个任务完成
    cypher_results, multi_path_results = await asyncio.gather(
        cypher_task, multi_path_task
    )

    return {
        "cypher_results": cypher_results,
        "multi_path_results": multi_path_results,
    }


async def run_parallel_rerank_async_v2(
    builder, user_id, query, candidates, session, k, itemname=None
):
    """
    异步版本的并行重排序函数
    加入用户信息（包括天气）+ 菜品属性的重排模型
    """

    # 首先进行检索召回

    if not candidates:
        return None

    """
        暂时加一段处理candidates的逻辑代码
    """
    
    # 都处理为只有菜品的列表
    processed_candidates = []
    for candidate in candidates:
        if isinstance(candidate, dict) and "DishName" in candidate:  ##接受字典
            processed_candidates = candidates
            break
        else:
            processed_candidates.append({"DishName": candidate})  ##接受列表

    print(f"处理后的候选项数量: {len(processed_candidates)}")

    # 执行并行重排序
    # results = await parallel_rerank(query,candidates, session)
    results = await parallel_rerank(
        builder, user_id, query, processed_candidates, session
    )

    return results
    
async def parallel_intent_and_rewrite(query, session):
  
    intent_task = asyncio.create_task(async_llm_intent(query, session))
    rewrite_task = asyncio.create_task(query_rewrite_LLM_FewShot(query, session))
    
    intent_results = None
    rewrite_results = None
    
    try:
        intent_results = await intent_task
    except Exception as e:
        print(f"🔴 Error in intent recognition task: {e}")
    
    # 执行查询改写
    try:
        rewrite_results = await rewrite_task
    except Exception as e:
        print(f"🔴 Error in query rewrite task: {e}")
    
    return {
        "intent_results": intent_results,
        "rewrite_results": rewrite_results,
    }

async def main(save_wav_path, builder, user_info_web=None):
    """
    save_wav_path: ogg或wav音频，list格式
    builder：构建llm reranker prompt
    user_info_web：网页传入的用户信息
    """

    meal = get_meal()
    print("当前用餐时间：", meal)

    # 用于答辩展示的前端
    if user_info_web is not None:
        print("✅ 使用网页端的用户信息")
        # extract_info_from_web(user_info, user_info_web)
        user_info = user_info_web
        user_id = user_info["id"]
    
    # 后端
    else:
        user_id = 105
        print(f"✅ 本地用户{user_id}信息")
        user_info = get_user_info(user_id)

    if "全国地区" not in user_info["region"]:
        user_info["region"] = [user_info["region"], "全国地区"]

    # 原始菜品库(用于判断大模型是否有幻觉)
    product_list = []
    product_list_path = r"Pipeline/data/DishData/dim_ai_exam_food_category_filter_out.txt"

    with open(product_list_path, "r", encoding="utf-8") as db_file:
        for line in db_file:
            product = line.strip().split("\t")[0]
            product_list.append(product)

    k = 30

    # Step 1: ASR语音识别
    print("\n📢 Step 1: ASR语音识别")
    # query = ASR(save_wav_path)
    query = "来点清淡的吧"
    print(f"ASR识别结果：{query}")

    async with aiohttp.ClientSession() as session:  
        # Step 2: 意图识别
        print("\n🧠 Step 2: 意图识别")

        intent_rewrite_result = await parallel_intent_and_rewrite(query, session)

        print("intent_rewrite_result:\n", intent_rewrite_result)
        intent_result, rewrite_result = extract_json_from_response(
            intent_rewrite_result
        )

        # 菜名
        itemname = intent_result.get("ItemName", "")
        # 提取口味 口感 功效
        taste = rewrite_result.get("taste", "")
        texture = rewrite_result.get("texture", "")
        function = rewrite_result.get("function", "")
        print(f"taste:{taste},texture:{texture},function:{function}")

        # 构建cypher查询语句
        # 判断是否为点餐意图
        if intent_result.get("Intent") == "0":
            print("\n❌ 非点餐意图,结束当前流程")
            return None, query

        # 如果是点餐意图
        elif intent_result.get("Intent") == "1":
            # 目标菜品明确的点菜意图--> 精准推荐 + 健康建议
            print("用户语音有点餐意图✅")
            if itemname != "":
                print("\n🚩 点餐菜品✅，进入多路召回")

                # 使用 原始查询 + 菜品名 召回
                fusion_candidates = Multi_Path_Search(query, k, itemname)

            # 模糊意义
            elif itemname == "":
              
                if taste or texture or function:
                  
                   
                    parallel_search_results = await parallel_search_combined(
                        query, k, itemname, taste, texture, function, user_info, meal
                    )
                    cypher_results, backup_candidates = (
                        parallel_search_results["cypher_results"],
                        parallel_search_results["multi_path_results"],
                    )
                    print(f"知识图谱原始召回菜品长度：{len(cypher_results)}")

                    # 如果知识图谱召回太多，使用用户历史信息进行筛选
                    if len(cypher_results) > 30:
                        cypher_results = search_dish_recommendation(
                            user_info["id"], cypher_results, k
                        )

                    print(
                        f"知识图谱召回结果(菜品列表，无属性，已根据历史记录筛选)；长度:{len(cypher_results)}, 菜品：{cypher_results[:3]}... \n三路召回结果(菜品列表，有属性)：；长度:{len(backup_candidates)}, 菜品：{backup_candidates[:3]}..."
                    )

                    # 知识图谱召回数量大于5
                    if len(cypher_results) >= 5:
                        print("采用知识图谱召回结果✅")
                        fusion_candidates = cypher_results

                    # 知识图谱小于5,多路并行检索兜底
                    else:
                        print("知识图谱无召回结果，多路召回兜底✅")
                        fusion_candidates = backup_candidates

                # 没有明确的菜名,也没有对口味,口感,功效方面的要求,那么直接走偏好推荐.
                else:
                    # 后续应该加上用户信息.
                    print(
                        "\n🚩 点餐菜品❌，菜品类别❌，口味要求❌，填入用户的信息进行知识图谱召回✅✅"
                    )
                  
                    print(user_info)
                    cypher_results = fetch_dishes_from_KG(
                        taste, texture, function, user_info, meal
                    )
                    print(f"知识图谱原始召回菜品长度：{len(cypher_results)}")

                    if len(cypher_results) > 30:
                        cypher_results = search_dish_recommendation(
                            user_info["id"], cypher_results, k
                        )

                    print(
                        f"知识图谱召回结果(菜品列表，无属性，已根据历史记录筛选)；长度:{len(cypher_results)}, 菜品：{cypher_results[:3]}..."
                    )
                    fusion_candidates = cypher_results

            # 根据用户信息,对所有召回结果做重排序
            print(
                f"融合之后的结果， 菜品列表长度：{len(fusion_candidates)}， 菜品：{fusion_candidates[:2]}..."
            )
            print(f"🚩 开始并行重排(bge-reranker + llm reranker): ")

            results = await run_parallel_rerank_async_v2(
                builder, user_id, query, fusion_candidates, session, k
            )

            # 幻觉护栏
            if not hasHallucination_rerank(results["llm_results"], product_list):
                final_result = results["llm_results"]  # 没有幻觉，直接使用LLM结果
            else:
                final_result = results["bge_results"]["ranked_results"][0][
                    "DishName"
                ]  # LLM发生了幻觉，使用BGE结果作为兜底

            print(
                f"用户信息：{user_info} \n原始query：{query} \n改写query:{rewrite_result}"
            )
            print("幻觉护栏之后的最终推荐结果final_result", final_result)

            return final_result, query


if __name__ == "__main__":
    save_wav_path = [
        "/data/ganshushen/Projects/MainBranch/Integrate/testTime/ASR_data/00a4c749-f568-4c49-aabf-082855200ea4.wav"
    ]

    builder = FastPromptBuilder()
    asyncio.run(main(save_wav_path, builder)) 
