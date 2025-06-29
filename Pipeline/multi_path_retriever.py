# 多路召回

import pickle
import numpy as np
from rank_bm25 import BM25Okapi
from collections import defaultdict
from utils import search_faiss
from typing import List 


async def BM25_Retrieval(query, k=100):
    file_path = "/data/ganshushen/Projects/MainBranch/Faiss/database/GPU_Large_DataBase_INFO/metadata.pkl"
    with open(file_path, "rb") as f:
        db_metadata = pickle.load(f)

    corpus = [item["DishName"] for item in db_metadata]
    tokenized_corpus = [list(doc) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = list(query)
    scores = bm25.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:k]
    results = [db_metadata[i] for i in top_indices]

    return results, scores

def rrf_fusion(candidate_lists, k=100, rrf_k=60):
    rrf_scores = defaultdict(float)
    item_map = {}
    # print(candidate_lists)
    for cand_list in candidate_lists:
        for rank, item in enumerate(cand_list):
            # print("item",item)
            key = (item["DishName"], item.get("Category", ""))
            rrf_scores[key] += 1.0 / (rrf_k + rank)
            if key not in item_map:
                item_map[key] = item

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return [item_map[key] for key, _ in sorted_items]


def Multi_Path_Search(query, k, itemname=None) -> List[dict]:
    if itemname is not None:
        # bm25_candidates, _ = BM25_Retrieval(query,k=k)
        bm25_candidates = []
        
        vector_candidates = search_faiss(
            [query, itemname], k=k
        ) 
        query_candidates = vector_candidates[0]
        item_candidates = vector_candidates[1]

        rrf_result = rrf_fusion(
            [query_candidates, item_candidates, bm25_candidates], k=k
        )
        
    else:
        bm25_candidates, _ = BM25_Retrieval(query, k=k)
        vector_candidates = search_faiss([query], k=k) 
        query_candidates = vector_candidates[0]
        rrf_result = rrf_fusion([query_candidates, bm25_candidates], k=k)

    return rrf_result