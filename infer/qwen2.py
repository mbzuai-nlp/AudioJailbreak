import json
import os
from io import BytesIO
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import torch
import gc

# 修改为用户目录下的路径
MODEL_CACHE_DIR = os.path.expanduser("~/qian_jiang/models/Qwen2-Audio-7B-Instruct")
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# 设置模型
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-Audio-7B-Instruct",
    cache_dir=MODEL_CACHE_DIR,
)

model = Qwen2AudioForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-Audio-7B-Instruct",
    cache_dir=MODEL_CACHE_DIR,
    device_map="auto",
    torch_dtype=torch.float16  # 使用半精度
)

def infer_audio(audio_path: str) -> str:
    """
    对单个音频文件进行推理
    
    Args:
        audio_path (str): 音频文件路径
        
    Returns:
        str: 模型输出的文本响应
    """
    
    conversation = [
    {"role": "user", "content": [
        {"type": "audio", "audio_url": audio_path},
    ]},
    ]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios = []
    for message in conversation:
        if isinstance(message["content"], list):
            for ele in message["content"]:
                if ele["type"] == "audio":
                    audios.append(librosa.load(
                        ele['audio_url'],
                        sr=processor.feature_extractor.sampling_rate
                    )[0])

    # 处理输入
    inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)
    # 将所有输入移到 GPU
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # 生成响应
    generate_ids = model.generate(
        **inputs,
        max_new_tokens=256,
    )
    generate_ids = generate_ids[:, inputs['input_ids'].size(1):]

    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(response)
    return response
 


# 示例：处理JSONL文件中的所有音频（保持原有功能）
def process_jsonl(jsonl_path: str, output_dir: str, start_index: int = 0):
    """
    处理JSONL文件中的所有音频并保存结果
    
    Args:
        jsonl_path (str): 输入JSONL文件路径
        output_dir (str): 输出目录
        start_index (int): 从第几行jsonl开始处理，默认为0
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 复制输入文件到输出目录
    output_jsonl = os.path.join(output_dir, 'sorted_combined_output.jsonl')
    if not os.path.exists(output_jsonl):
        import shutil
        shutil.copy2(jsonl_path, output_jsonl)

    # 读取并处理数据
    data = []
    with open(output_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    # 创建进度记录文件
    progress_file = os.path.join(output_dir, "progress.txt")
    
    # 如果存在进度文件且未指定起始索引，则从上次中断处继续
    if os.path.exists(progress_file) and start_index == 0:
        with open(progress_file, 'r') as f:
            last_index = int(f.read().strip())
            print(f"从上次中断的位置继续: 第 {last_index} 行")
            start_index = last_index
    else:
        print(f"从指定位置开始: 第 {start_index} 行")

    # 处理每个样本
    for i, item in enumerate(data[start_index:], start=start_index):
        speech_path = item['speech_path']
        print(f"Processing: {speech_path} (第 {i} 行)")
        
        # 记录当前处理的行号
        with open(progress_file, 'w') as f:
            f.write(str(i))
        
        # 使用封装的推理函数
        response = infer_audio(speech_path)
        item['response'] = response
        print(f"Response generated for {speech_path}")

        # 每处理10个样本保存一次结果
        if i % 10 == 0 and i > 0:
            with open(output_jsonl, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')

        # 每处理N个样本清理一次GPU内存
        if i % 20 == 0 and i > 0:
            torch.cuda.empty_cache()
            gc.collect()

def process_batch(batch_items, batch_size=4):
    batch_audios = []
    batch_texts = []
    
    for item in batch_items:
        conversation = [{"role": "user", "content": [{"type": "audio", "audio_url": item['speech_path']}]}]
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audio = librosa.load(item['speech_path'], sr=processor.feature_extractor.sampling_rate)[0]
        
        batch_audios.append(audio)
        batch_texts.append(text)
    
    # 批量处理输入
    inputs = processor(text=batch_texts, audios=batch_audios, return_tensors="pt", padding=True)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    # 批量生成响应
    generate_ids = model.generate(**inputs, max_new_tokens=256)
    generate_ids = generate_ids[:, inputs['input_ids'].size(1):]
    
    responses = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    return responses

# 使用示例
if __name__ == "__main__":
    # 示例1：处理单个音频文件
    # audio_path = "path/to/your/audio.wav"
    # response = infer_audio(audio_path)
    # print(f"Response: {response}")
    
    # 示例2：处理JSONL文件（原有功能）
    jsonl_path = "./convert/sorted_combined_output.jsonl"
    output_dir = "./inference/qwen2_response_jsonl"
    start_index = 1  # 从第几行开始处理，可以根据需要修改
    process_jsonl(jsonl_path, output_dir, start_index)
