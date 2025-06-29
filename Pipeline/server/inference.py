import os
import glob
import json
import re
import soundfile
from funasr import AutoModel

class ASRInference:
    def __init__(self, model_name_or_path, modes, output_dir):
        '''
        model_name_or_path: ASR模型权重文件夹，要保证文件夹下有configuration.json、config.yaml文件，
        可在configuration.json更改要推理的权重名称
        modes: 要推理的数据集，可选 ['train', 'val', 'test', 'a_test', 'b_test']
        output_dir: txt文件输出文件夹
        '''
        self.model = AutoModel(model=model_name_or_path, device="cuda:0")
        self.modes  = modes
        self.output_dir = output_dir 
    
    def batch_inference(self, epoch_num='best'):
        '''
        epoch_num: 输出txt文件的epoch编号
        '''
        for mode in self.modes:
            inputs = []
            jsonl_path = f"/data/kyy/Project/ASR/asr_dataset/{mode}.jsonl"  # 修改成你的 JSONL 文件路径
            inputs = []
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        wav_path = item.get("source", "")
                        if os.path.isfile(wav_path) and wav_path.endswith(".wav"):
                            inputs.append(wav_path)
                    except Exception as e:
                        print(f"解析失败：{line.strip()} -> {e}")

            print(f"共找到 {len(inputs)} 个音频文件")
            print(inputs[0])

            # 输出目录和目标文本路径
            os.makedirs(self.output_dir, exist_ok=True)
            output_txt_path = os.path.join(self.output_dir, f"paraformer_{epoch_num}e_{mode}.txt")
            # 打开文件，边识别边写
            with open(output_txt_path, "w", encoding="utf-8") as f:
                for path in inputs:
                    try:
                        res = self.model.generate(input=path)[0] # 获取识别结果（第一个）
                        # res = model.generate(input=wav_file, batch_size_s=300, batch_size_threshold_s=60, hotword='魔搭')
                        # print(res)
                        text = res["text"]
                        # cleaned_text = emoji.replace_emoji(text, replace='')
                        # cleaned_text = re.sub(r"[。]", "", cleaned_text)  # 删除中英文句号
                        line = f"{res['key']} {text}\n"
                        f.write(line)

                        # print(line.strip())  # 输出当前结果
                    except Exception as e:
                        print(f"识别失败: {path} -> {e}")

    def single_inference(self, wav_path):
        '''
        wav_path: 音频文件路径
        '''
        path = wav_path
        try:
            res = self.model.generate(input=path)[0] # 获取识别结果（第一个） #'猫 猫 声 灵 烟 花 的 声 音 烟 花 的 声 音'
            print(res)
            text = res["text"]
            # cleaned_text = emoji.replace_emoji(text, replace='')
            # cleaned_text = re.sub(r"[。]", "", cleaned_text)  # 删除中英文句号
            print(f"{path} \n识别结果: {text}")
            return text
        except Exception as e:
            print(f"识别失败: {path} -> {e}")
    
    def inference_stream(self, wav_path):
        chunk_size = [0, 10, 5]  # [0, 10, 5] 600ms, [0, 8, 4] 480ms
        encoder_chunk_look_back = 4  # number of chunks to lookback for encoder self-attention
        decoder_chunk_look_back = 1  # number of encoder chunks to lookback for decoder cross-attention

        speech, sample_rate = soundfile.read(wav_path)

        chunk_stride = chunk_size[1] * 960  # 600ms、480ms

        cache = {}
        total_chunk_num = int(len((speech) - 1) / chunk_stride + 1)
        for i in range(total_chunk_num):
            speech_chunk = speech[i * chunk_stride : (i + 1) * chunk_stride]
            is_final = i == total_chunk_num - 1
            res = self.model.generate(
                input=speech_chunk,
                cache=cache,
                is_final=is_final,
                chunk_size=chunk_size,
                encoder_chunk_look_back=encoder_chunk_look_back,
                decoder_chunk_look_back=decoder_chunk_look_back,
            )
            print(res)

if __name__ == '__main__':
    models = {
        'paraformer': "/data/ganshushen/Projects/MainBranch/Integrate/testTime/weight",
        'seaco_paraformer': "/data/kyy/Project/ASR/FunASR/examples/industrial_data_pretraining/seaco_paraformer/outputs",
        'seaco_paraformer_1e-4': "/data/kyy/Project/ASR/FunASR/examples/industrial_data_pretraining/seaco_paraformer/outputs_1e-4",
    }
    infer = ASRInference(model_name_or_path = models['paraformer'],
                        #  modes = ['train', 'val', 'test'], 
                         modes = ['val', 'test', 'a_test', 'b_test'], 
                         output_dir = "/data/kyy/Project/ASR/asr_inference/output")
    infer.single_inference("/data/ganshushen/Projects/MainBranch/Integrate/testTime/API_logs/20250604_1556/213fabb517490238146433415d13d2_1749023817443_b4d1456c614f471086481c4f2f46d9af.ogg")
    # infer.batch_inference("avg10")
    # infer.inference_stream("/data/kyy/Dataset/ELE/dataset_b/478f0ed6-eea7-403d-96b8-09ffa316dbff.wav")