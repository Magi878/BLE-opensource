import os
import requests
from typing import List

def ASR(
    files: List[str], API_URL="http://localhost:8003/infer/"
):  # http://172.21.141.4:8003/infer/
    multiple_files = [
        ("files", (os.path.basename(f), open(f, "rb"), "audio/ogg")) for f in files
    ]
    response = requests.post(API_URL, files=multiple_files)
    try:
        response.raise_for_status()
        result = response.json().get("results")
        return result[0]["text"]

    except Exception as e:
        print("请求失败：", e)
        print("响应内容：", response.text)