#!/bin/bash

# 快速启动脚本 - 简化版
SERVER_DIR="/data/ganshushen/Projects/BLE_CODE_OPEN/Pipeline/server"

cd "$SERVER_DIR"

echo "🚀 启动所有服务..."

# 1.ASR服务已在4服务器上启动
echo "启动 ASR 服务 (端口 8003)..."
nohup uvicorn asr_server:app --host 0.0.0.0 --port 8003 > logs/asr.log 2>&1 &

# 2.启动 Embedding 服务
echo "启动 Embedding 服务 (端口 8001)..."
nohup uvicorn embedding_service:app --host 0.0.0.0 --port 8001 > logs/embedding.log 2>&1 &

# 3.启动 Reranker 服务
echo "启动 Reranker 服务 (端口 8000)..."
nohup uvicorn reranker_service:app --host 0.0.0.0 --port 8000 > logs/reranker.log 2>&1 &

# 4.启动 Rewrite FAISS 服务
echo "启动 Rewrite FAISS 服务 (端口 5001)..."
nohup python rewrite_faiss_server.py > logs/rewrite_faiss.log 2>&1 &

# 5.启动 new_products_faiss_server 服务
echo "启动 new_products_faiss_server 服务 (端口 5006)..."
nohup python new_products_faiss_server.py > logs/new_products_faiss_servers.log 2>&1 &

# 6.启动 llm reranker 服务
# echo "启动 vllm_fastapi_server 服务 (端口 10000)..."
# nohup python vllm_fastapi_server.py > logs/vllm_fastapi_server.log 2>&1 &

# 6.启动 KG_filter 服务
echo "启动 KG_filter_server 服务 (端口 6666)..."
nohup python KG_filter_server.py > logs/KG_filter_server.log 2>&1 &


echo "✅ 所有服务启动完成！"
echo ""
echo "服务端口信息:"
echo "- 1.ASR 服务: http://localhost:8003"
echo "- 2.Embedding 服务: http://localhost:8001"
echo "- 3.Reranker 服务: http://localhost:8000"
echo "- 4.Rewrite FAISS 服务: http://localhost:5001"
echo "- 5.New_products_faiss_server 服务: http://localhost:5006"
echo "- 6.KG_filter_server 服务: http://localhost:6666"
echo "- 7.qwen3_reranker_server 因为不在一个环境中，服务需要单独启动: python ./BLE_CODE_OPEN/Pipeline/server/qwen3_reranker_server.py"
echo ""
echo "查看日志: tail -f logs/服务名.log"
echo "停止所有服务: pkill -f 'uvicorn|python.*server.py'"