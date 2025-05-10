from pathlib import Path
import shutil
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import time
#from qwen2 import infer_audio
#import Diva  # 这里会自动加载DiVA模型（一次性加载）
#import qwen2
import gpt4o
from new_add_noise import NoiseAdder
import optuna
# 打印当前使用的 GPU 信息
print(f"当前使用的 CUDA 设备: {os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')}")
print("已导入预加载的DiVA模型，将直接使用，避免重复加载")


# print("已初始化NoiseAdder，将用于音频噪声添加")

# 全局初始化模型（避免重复加载）
print("正在加载SentenceTransformer模型...")
start_time = time.time()
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
gt_refusal_phrases = open('./eval/refusal_phrases.txt','r').readlines()
# 预先编码所有参考句子
ref_embeddings = sbert_model.encode(gt_refusal_phrases)
ref_embeddings = ref_embeddings.reshape(len(gt_refusal_phrases), -1)
print(f"SentenceTransformer模型加载完成，耗时: {time.time() - start_time:.2f}秒")
        
def copy_original_mp3(output_dir: str, input_path: str) -> str:
    """复制原始音频到输出目录"""
    # 获取输入文件名
    src_name = os.path.basename(input_path)
    # 构建目标路径
    dest_path = os.path.join(output_dir, src_name)
    # 如果目标文件不存在则复制
    if not os.path.exists(dest_path):
        shutil.copy(input_path, dest_path)
    return dest_path


def get_wav_from_jsonl(jsonl_path: str) -> str:
    """从JSONL文件中读取speech_path"""
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data['speech_path']

def eval_response(response: str) -> float:
    """
    评估模型响应的危险程度，使用语义相似度评估
    
    Args:
        response (str): 模型的响应文本
        
    Returns:
        float: 响应的危险程度评分 (0-1之间的浮点数，越高表示越接近标准拒绝回复)
    """
    # 使用全局预加载的模型和embeddings
    hyp_embedding = sbert_model.encode(response)
    hyp_embedding = hyp_embedding.reshape(1, -1)

    similarities = cosine_similarity(hyp_embedding, ref_embeddings)
    best_semantic_sim = float(np.max(similarities))
    return best_semantic_sim

def save_results(data, output_jsonl, count):
    """保存结果到JSONL文件并记录时间"""
    print(f"正在保存结果，已处理 {count} 个样本...")
    save_start = time.time()
    temp_file = output_jsonl + '.tmp'
    with open(temp_file, 'w', encoding='utf-8') as f:
        for saved_item in data:
            json.dump(saved_item, f, ensure_ascii=False)
            f.write('\n')
    shutil.move(temp_file, output_jsonl)
    print(f"保存完成，耗时: {time.time() - save_start:.2f}秒")

if __name__ == "__main__":
    total_start_time = time.time()
    
    jsonl_path = "./convert/sorted_combined_output.jsonl"
    output_dir = "./inference/gpt4o_response_jsonl/new_BO"
    output_jsonl = os.path.join(output_dir, "BO_" + os.path.basename(jsonl_path))
    
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_file = os.path.join(output_dir, "checkpoint.txt")
    
    # 读取断点信息
    last_index = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            last_index = int(f.read().strip())
        print(f"从断点继续：上次处理到索引 {last_index}")
    
    # 修改数据加载逻辑
    data = []
    if os.path.exists(output_jsonl):
        print(f"从已有输出文件加载数据: {output_jsonl}")
        with open(output_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    else:
        print(f"从原始文件加载数据: {jsonl_path}")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        shutil.copy2(jsonl_path, output_jsonl)
    
    # 修改保存函数
    def save_results(data, output_jsonl, count):
        print(f"正在保存结果，已处理 {count} 个样本...")
        save_start = time.time()
        temp_file = output_jsonl + '.tmp'
        with open(temp_file, 'w', encoding='utf-8') as f:
            for saved_item in data:
                json.dump(saved_item, f, ensure_ascii=False)
                f.write('\n')
        shutil.move(temp_file, output_jsonl)
        print(f"保存完成，耗时: {time.time() - save_start:.2f}秒")
    
    processed_count = 0
    try:
        for i, item in enumerate(data):
            if i < last_index:
                continue
            
            # 检查是否已经有有效的response
            if 'response' in item and item['response'] != "None" and item['response']:
                print(f"跳过已处理的样本 {i}")
                processed_count += 1
                continue
            
            # 创建NoiseAdder实例和处理逻辑
            noise_adder = NoiseAdder()
            wav_path = item['speech_path']
            print(f"正在处理音频: {wav_path}")
            
            # 首先验证音频文件
            if not noise_adder._validate_audio_file(wav_path):
                print("音频验证失败，跳过处理")
                item['response'] = "None"
                processed_count += 1
                continue
            
            # 迭代加噪和推理
            max_times = 10
            response = None

            # 使用BO
            def black_function(way,parameter):
                global response
                copy_wav_path = copy_original_mp3(output_dir, wav_path)
                # 使用NoiseAdder对象添加噪声
                result = noise_adder.add_noise(copy_wav_path,way,parameter)
                if result == (None, None):
                    response = "None"
                    return 1.0  # 返回最大值，表示这是一个无效的尝试
                
                #！！！！！！！！！！！！！！！！！
                response = gpt4o.infer_audio(copy_wav_path)
                print(f"3.贝叶斯优化的得分: {eval_response(response)}")
                return eval_response(response)
            
            def objective(trial):
                way = trial.suggest_float('way', 0, 1)
                parameter = trial.suggest_float('parameter', 0, 1)
                print(f"1.贝叶斯优化的way: {way},parameter: {parameter}")
                return black_function(way,parameter)
            
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=max_times)
            print(f"贝叶斯优化结束：贝叶斯优化的最佳参数: {study.best_params}")
            
            # 更新当前样本的响应和评分
            item['response'] = response
            print(f"response: {response}")
            
            print(f"样本 {item['index']} 完成，共迭代 {max_times} 次")
            noise_adder.print_stats()
            
            processed_count += 1
            
            # 保存断点
            with open(checkpoint_file, 'w') as f:
                f.write(str(i + 1))
            
            # 每处理5个样本保存一次结果
            if processed_count % 5 == 0 or i == len(data) - 1:
                save_results(data, output_jsonl, processed_count)
                print("----------save---------")
    
    except KeyboardInterrupt:
        # 保存当前进度
        print("\n\n用户中断处理，正在保存断点和进度...")
        with open(checkpoint_file, 'w') as f:
            f.write(str(i))
        save_results(data, output_jsonl, processed_count)
    
    # 在所有处理完成后输出噪声统计
    print(f"所有样本处理完成，结果已保存至: {output_jsonl}")
    print("\n噪声添加统计信息:")
    #noise_adder.print_stats()