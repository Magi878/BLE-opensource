# 2025年“饿了么”AI大赛 赛道一：智慧养老 冠军 饱了吗队方案

一个基于大语言模型和知识图谱的智能菜品推荐系统，专为老年人群体设计，支持语音交互和个性化推荐。

## 🎯 项目简介

本项目是一个智能菜品推荐系统，结合了自动语音识别(ASR)、大语言模型(LLM)、知识图谱(KG)和向量检索技术，为用户提供个性化的菜品推荐服务。系统特别针对老年人群体的健康需求和饮食偏好进行了优化。

## 🏗️ 项目架构

### 核心技术栈
- **语音识别**: Paraformer ASR模型
- **向量检索**: FAISS + BGE嵌入模型
- **知识图谱**: Neo4j数据库
- **重排序**: BGE重排器 + LLM重排器
- **Web服务**: Flask + SocketIO
- **多路召回**: BM25 + 向量检索 + 知识图谱检索

## 📁 文件结构

```
BLE_CODE_OPEN/
├── .gitignore                    # Git忽略文件配置
├── Faiss/                        # FAISS向量数据库相关
│   ├── build_faiss_database.py   # 构建FAISS数据库脚本
│   └── build_faiss_database_csv.py # 从CSV构建FAISS数据库
├── model/                        # 模型文件目录(需要下载相应模型)
└── Pipeline/                     # 主要业务逻辑
    ├── main.py                   # 主要推荐流程入口
    ├── _tianmao_server.py        # 天猫精灵服务器(Web界面)
    ├── ASR_Paraformer.py         # 语音识别模块
    ├── kg_retriever.py           # 知识图谱检索器
    ├── multi_path_retriever.py   # 多路召回检索器
    ├── llmrec_engine.py          # LLM推荐引擎
    ├── llmrec_prompt_engine.py   # LLM提示词构建器
    ├── query_resolver.py         # 查询解析和重写
    ├── health_level_agent.py     # 健康约束代理
    ├── cypher_templates.py       # Cypher查询模板
    ├── prompt.py                 # 提示词模板
    ├── utils.py                  # 工具函数
    ├── data/                     # 数据文件
    │   ├── DishData/            # 菜品数据
    │   │   └── dim_ai_exam_food_category_filter_out.txt
    │   ├── UserData/            # 用户数据
    │   │   └── 1000_user_processed_with_health_constraints.jsonl
    │   └── WavData/             # 音频文件
    │       └── *.wav            # 语音文件
    └── server/                   # 微服务模块
        ├── quick_start_all_sever.sh      # 一键启动所有服务脚本
        ├── asr_server.py                 # ASR服务 
        ├── embedding_service.py          # 嵌入服务 
        ├── reranker_service.py           # 重排序服务 
        ├── rewrite_faiss_server.py       # 查询重写FAISS服务 
        ├── products_faiss_server.py      # 产品FAISS服务 
        ├── KG_filter_server.py           # 知识图谱过滤服务 
        ├── KG_filter_dataset_build.py    # KG过滤数据集构建
        ├── qwen3_reranker_server.py      # Qwen3个性化推荐服务
        ├── inference.py                  # 推理服务
        └── temp_wavs/                    # 临时音频文件
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 确保Neo4j数据库运行
# 确保相关模型文件已下载到model/目录
```

### 2. 启动服务

```bash
# 一键启动所有微服务
cd Pipeline/server
bash quick_start_all_sever.sh
```

### 3. 启动Web界面

```bash
# 启动天猫精灵服务器
cd Pipeline
python _tianmao_server.py

# 访问 http://localhost:9999 进行交互
```

## 🔧 服务端口说明

| 服务名称 | 端口 | 功能描述 |
|---------|------|----------|
| ASR服务 | 8003 | 语音识别 |
| Embedding服务 | 8001 | 文本嵌入 |
| Reranker服务 | 8000 | BGE重排序 |
| Rewrite FAISS服务 | 5001 | 查询重写检索 |
| Products FAISS服务 | 5006 | 产品向量检索 |
| KG Filter服务 | 6666 | 知识图谱过滤 |
| 天猫精灵服务 | 9999 | Web交互界面 |

## 💡 核心功能

### 1. 多路召回机制
- **向量检索**: 基于FAISS的语义相似度检索
- **BM25检索**: 基于关键词的传统检索
- **知识图谱检索**: 基于用户偏好和健康约束的结构化检索
- **RRF融合**: 多路召回结果的融合排序

### 2. 智能重排序
- **BGE重排序**: 基于BGE模型的语义重排序
- **LLM重排序**: 基于大语言模型的智能重排序
- **并行处理**: 两种重排序方式并行执行，提高效率

### 3. 健康约束
- **健康状况识别**: 自动识别用户健康状况
- **食品过滤**: 过滤不适合用户健康状况的食品
- **个性化推荐**: 基于用户偏好和健康约束的个性化推荐

### 4. 语音交互
- **语音识别**: 支持语音输入
- **意图理解**: 智能理解用户意图
- **查询重写**: 优化用户查询表达

## 🔍 使用示例

1. **Web界面交互**: 访问 http://localhost:9999，设置用户信息，进行语音或文本交互
2. **API调用**: 直接调用各个微服务的API接口
3. **批量处理**: 使用main.py进行批量推荐处理

## 📊 数据说明

- **用户数据**: 包含用户基本信息、健康状况、饮食偏好等
- **菜品数据**: 包含菜品名称、分类、营养信息等
- **知识图谱**: 存储菜品、食材、营养、健康状况之间的关系

## 🛠️ 开发说明

### 添加新的检索路径
1. 在 `multi_path_retriever.py` 中添加新的检索函数
2. 在 `Multi_Path_Search` 函数中集成新的检索路径
3. 更新RRF融合逻辑

### 添加新的重排序器
1. 在 `llmrec_engine.py` 中添加新的重排序函数
2. 在 `parallel_rerank` 函数中集成新的重排序器
3. 更新结果融合逻辑

## 📝 注意事项

1. 确保Neo4j数据库正常运行并包含相关数据
2. 模型文件需要单独下载并放置在model/目录下
3. FAISS数据库需要预先构建
4. 各个服务之间存在依赖关系，建议按照启动脚本的顺序启动

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进项目。在提交代码前，请确保：
1. 代码符合项目的编码规范
2. 添加必要的测试用例
3. 更新相关文档

## 📄 许可证

本项目采用开源许可证，具体信息请查看LICENSE文件。
        
