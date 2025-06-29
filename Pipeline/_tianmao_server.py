##该脚本用于：接收天猫精灵传回ogg文件，返回推荐结果
# 运行该脚本之前要 运行 ./BLE_CODE/Integrate/testTime/server/quick_start_all_sever.sh 中的所有服务
# 运行之后可以打开网页 http://172.21.141.97:9999：选择基础用户信息；查看返回asr结果和推荐结果

import os
import time
import json
import asyncio
import requests
import urllib.parse
from typing import List
from datetime import datetime
from pydub import AudioSegment
from flask_socketio import SocketIO, emit
from flask import Flask, request, Response, render_template_string, jsonify

from OldManInfo_Neo4j_Vector_Pipeline import main # 语音识别 + 意图识别 + 菜品推荐（多路召回+知识图谱召回  BGE重排+大模型重排）
from prompt_llm_reranker import FastPromptBuilder # 快速构建 LLM 重排模型的prompt

# 语音传输日志
base_save_path = '/data/ganshushen/Projects/BLE_CODE_OPEN/Pipeline/_tianmao_server_logs'

# 五个基础的用户信息
default_users_file = '/data/ganshushen/Projects/BLE_CODE_OPEN/Pipeline/user_web/default_users_infos.jsonl'

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # 设置密钥用于SocketIO
socketio = SocketIO(app, cors_allowed_origins="*")  # 添加SocketIO支持

# 全局变量存储用户配置
current_user_info = {
    'id': 1,
    'name': '默认用户',
    'gender': 'Male',
    'age_range': '30-50',
    'region': '华南地区',
    'health_conditions': ['健康'],
    'taste_preferences': '清淡',
    'texture_preferences': '软糯',
    'query': ''
}

builder = FastPromptBuilder()

@app.route('/', methods=['GET'])
def index():
    # 返回用户配置页面
    with open('/data/ganshushen/Projects/BLE_CODE_OPEN/Pipeline/user_web/user_info_web_page.html', 'r', encoding='utf-8') as f:
        return f.read()

@app.route('/hello', methods=['GET'])
def hello():
    return f"============================\n" \
           f"ELE2025 SZU-饱了吗\n" \
           f"============================"

@app.route('/get_user_config', methods=['GET'])
def get_user_config():
    """获取当前用户配置"""
    try:
        return jsonify({
            "code": 200,
            "message": "success",
            "data": current_user_info
        })
    except Exception as e:
        return jsonify({
            "code": 500,
            "message": "获取配置失败: " + str(e)
        })

@app.route('/set_user_config', methods=['POST'])
def set_user_config():
    """设置用户配置"""
    global current_user_info
    try:
        new_config = request.get_json()
        print(f"接受到的用户信息： \n{new_config}")
        
        # 验证必填字段
        required_fields = ['id', 'name', 'gender', 'age_range', 'region', 'taste_preferences', 'texture_preferences']
        for field in required_fields:
            if field not in new_config or not new_config[field]:
                return jsonify({
                    "code": 400,
                    "message": f"缺少必填字段: {field}"
                })
        
        # 更新用户配置（保留query字段）
        current_query = current_user_info.get('query', '')
        current_user_info.update(new_config)
        current_user_info['query'] = current_query
        
        # 保存配置到文件（可选）
        # config_file = os.path.join(base_save_path, 'user_config.json')
        config_file = '/data/ganshushen/Projects/BLE_CODE_OPEN/Pipeline/user_web/user_config.json'
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(current_user_info, f, ensure_ascii=False, indent=2)
        
        return jsonify({
            "code": 200,
            "message": "配置保存成功",
            "data": current_user_info
        })
        
    except Exception as e:
        return jsonify({
            "code": 500,
            "message": "保存配置失败: " + str(e)
        })

@app.route('/get_default_users', methods=['GET'])
def get_default_users():
    """获取默认用户列表"""
    try:
        users = []
        if os.path.exists(default_users_file):
            with open(default_users_file, 'r', encoding='utf-8') as f:
                for line in f:
                    users.append(json.loads(line))
        
        return jsonify({
            "code": 200,
            "message": "success",
            "data": users
        })
    except Exception as e:
        return jsonify({
            "code": 500,
            "message": "获取默认用户失败: " + str(e)
        })

@app.route('/asr/api/v1', methods=['POST'])
def asr():
    global current_user_info
    try:
        # ---- 开始计时 ----
        # start_time = time.time()

        # 获取参数
        params = request.get_json()
        print(f"Params: {params}")
        audioUrl = params.get('audioUrl', '')
        # audioUrl = urllib.parse.unquote(audioUrl)
        # print(audioUrl)

        # 提取文件名
        audio_filename = audioUrl.split('?')[0].split('/')[-1]
        if not audio_filename.lower().endswith('.ogg'):
            return Response(json.dumps({
                "code": 1002,
                "errorMsg": "Unsupported audio format, only .ogg Audio: speex, 16000 Hz is supported"
            }, ensure_ascii=False))

        # 构建存储路径：按年月日_时分创建子文件夹
        now = datetime.now().strftime("%Y%m%d_%H%M")
        save_dir = os.path.join(base_save_path, now)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 下载音频文件
        audio_save_path = os.path.join(save_dir, audio_filename)
        log_file_path = os.path.join(save_dir, 'info.log')
        response = requests.get(audioUrl, stream=True)
        if response.status_code != 200:
            return Response(json.dumps({
                "code": 1001,
                "errorMsg": f"Failed to download file. Status code: {response.status_code}"
            }, ensure_ascii=False))
        with open(audio_save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        # 从ASR结果中提取query并更新用户信息
        audio_save_path = [audio_save_path]
        print(audio_save_path)
        
        # 准备用户信息，包含从音频识别得到的查询内容
        user_info_for_main = current_user_info.copy()
        
        # 调用ASR模型进行识别，传入用户信息
        # 注意：这里假设您的main函数已经修改为接受user_info参数
        # 如果main函数还没有修改，需要先修改main函数的签名
        start_time = time.time()
        Recommended_dish, Asr_result = asyncio.run(main(audio_save_path, builder, user_info_for_main))
        
        # 更新用户信息中的query字段
        current_user_info['query'] = Asr_result

        # 推荐菜品

        if Recommended_dish:
            result = f"给我点{Recommended_dish}"
        else:
            result = Asr_result

        print(f"返回天猫服务端结果： {result}")

        # ---- 计算耗时 ----
        end_time = time.time()
        processing_time_ms = int((end_time - start_time) * 1000)

        # 写入日志文件
        mode = 'a' if os.path.exists(log_file_path) else 'w'
        with open(log_file_path, mode, encoding='utf-8') as log:
            if mode == 'a':
                log.write("\n" + "-" * 40 + "\n\n")  # 分隔不同请求的日志
            log.write(f"Request Time: {datetime.now().isoformat()}\n")
            log.write(f"Audio URL: {audioUrl}\n")
            log.write(f"Saved File: {audio_filename}\n")
            log.write(f"Request Params: {json.dumps(params, ensure_ascii=False, indent=2)}\n")
            log.write(f"User Info: {json.dumps(user_info_for_main, ensure_ascii=False, indent=2)}\n")
            log.write(f"Asr Results: {Asr_result}\n")
            log.write(f"Recommended_dish: {Recommended_dish}\n")
            log.write(f"Expected reply: {result}\n")
            log.write(f"Processing Time: {processing_time_ms}ms\n\n")

        # 通过WebSocket实时发送结果到前端
        result_data = {
            'asr_result': Asr_result,
            'recommended_dish': Recommended_dish,
            'processing_time': processing_time_ms,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'user_info': user_info_for_main
        }
        
        # 发送到所有连接的客户端
        socketio.emit('recognition_result', result_data)
        
        # 模拟识别结果（后续替换为实际逻辑）
        # result = "帮我点一杯霸王茶姬的伯牙绝弦，中杯半糖。"

        # ASR结果识别异常
        if not result:
            return Response(json.dumps({
                "code": 1003,
                "errorMsg": "No result returned from the model"
            }, ensure_ascii=False))

        # 返回响应
        return Response(json.dumps({
            "code": 200,
            "message": "success",
            "data": {
                "result": result,
                "ext": {
                    "user_info": user_info_for_main,
                    "asr_result": Asr_result,
                    "recommended_dish": Recommended_dish,
                    "processing_time": processing_time_ms,
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            }
        }, ensure_ascii=False))

    except Exception as e:
        return Response(json.dumps({
            "code": 500,
            "message": "system error, exception: " + str(e)
        }, ensure_ascii=False))

# WebSocket事件处理
@socketio.on('connect')
def handle_connect():
    print('客户端已连接')
    emit('status', {'msg': '已连接到服务器'})

@socketio.on('disconnect')
def handle_disconnect():
    print('客户端已断开连接')

def load_user_config():
    """启动时加载用户配置"""
    global current_user_info
    config_file = os.path.join(base_save_path, 'user_config666.json')
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                saved_config = json.load(f)
                current_user_info.update(saved_config)
                print(f"已加载用户配置: {current_user_info}")
        except Exception as e:
            print(f"加载用户配置失败: {e}")
    
    # 如果没有配置文件，加载第一个默认用户
    if not os.path.exists(config_file) and os.path.exists(default_users_file):
        try: 
            with open(default_users_file, 'r', encoding='utf-8') as f:
                first_user = json.loads(f.readline())
                current_user_info.update(first_user)
                print(f"已加载默认用户配置: {current_user_info}")
        except Exception as e:
            print(f"加载默认用户配置失败: {e}") 

if __name__ == "__main__":
    app.config['JSON_AS_ASCII'] = False
    
    # 启动时加载用户配置
    load_user_config()
    
    # 使用SocketIO运行应用
    socketio.run(app, host='0.0.0.0', port=9999, debug=True)