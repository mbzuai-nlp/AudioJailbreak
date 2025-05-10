import json
import os
from transformers import AutoModel
import librosa
import shutil
import torch
import gc
import signal
from contextlib import contextmanager
import time
import argparse

class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

# 全局模型加载，只执行一次
print("正在加载DiVA模型...")
load_start_time = time.time()
model = AutoModel.from_pretrained(
    "WillHeld/DiVA-llama-3-v0-8b",
    trust_remote_code=True
)
print(f"DiVA模型加载完成，耗时: {time.time() - load_start_time:.2f}秒")

def infer_audio(speech_path):
    """
    对音频文件进行推理
    Args:
        speech_path: 音频文件路径
    Returns:
        str: 模型的回答
    """
    # 注意：模型现在是全局加载的，不再需要在函数内部加载
    try:
        # 加载音频
        speech_data, _ = librosa.load(speech_path, sr=16_000)
        
        # 生成响应，设置超时时间为60秒
        try:
            with time_limit(60):
                with torch.no_grad():
                    response = model.generate([speech_data])[0]
            
            # 清理GPU内存
            torch.cuda.empty_cache()
            gc.collect()
            
            return response
            
        except TimeoutException:
            print(f"Timeout processing {speech_path}")
            return "Error: Processing timeout"
            
    except Exception as e:
        print(f"Error processing {speech_path}: {str(e)}")
        return f"Error: {str(e)}"

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='DiVA模型推理')
    parser.add_argument('--start_index', type=int, default=0, help='从第几行jsonl开始处理')
    args = parser.parse_args()

    # 创建输出目录
    output_dir = "./inference/Diva_response_jsonl"
    os.makedirs(output_dir, exist_ok=True)

    # 复制输入文件到输出目录
    input_jsonl = "./convert/wav_combined_output.jsonl"
    output_jsonl = os.path.join(output_dir, "wav_combined_output.jsonl")
    shutil.copy2(input_jsonl, output_jsonl)

    # 读取并处理数据
    data = []
    with open(output_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    # 创建进度记录文件
    progress_file = os.path.join(output_dir, "progress.txt")

    # 如果存在进度文件且未指定起始索引，则从上次中断处继续
    if os.path.exists(progress_file) and args.start_index == 0:
        with open(progress_file, 'r') as f:
            last_index = int(f.read().strip())
            print(f"从上次中断的位置继续: 第 {last_index} 行")
            start_index = last_index
    else:
        start_index = args.start_index
        print(f"从指定位置开始: 第 {start_index} 行")

    # 处理每个样本
    for i, item in enumerate(data[start_index:], start=start_index):
        speech_path = item['speech_path']
        print(f"Processing: {speech_path} (第 {i} 行)")
        
        # 记录当前处理的行号
        with open(progress_file, 'w') as f:
            f.write(str(i))
        
        # 调用推理函数
        response = infer_audio(speech_path)
        
        # 更新response字段
        print(response)
        item['response'] = response
        print(f"Response generated for {speech_path}")
        
        # 每处理完一个文件就保存一次
        with open(output_jsonl, 'w', encoding='utf-8') as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + '\n')
        
        # 记录错误发生的行号
        if response.startswith("Error"):
            with open(os.path.join(output_dir, "error_log.txt"), 'a') as f:
                f.write(f"Error at line {i}: {response}\n")

    # 处理完成后更新进度文件
    with open(progress_file, 'w') as f:
        f.write(str(len(data)))
        
    print("Processing completed!")
