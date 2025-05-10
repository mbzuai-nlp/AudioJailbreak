import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


#这个后面细看一下 不知道为什么跑得这么慢

# 设置模型路径
MODEL_CACHE_DIR = "/mnt/data/huggingface/transformers/models"
MODEL_NAME = "Qwen/Qwen2-7B-Instruct"  # 使用文本模态的模型

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    cache_dir=MODEL_CACHE_DIR,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    cache_dir=MODEL_CACHE_DIR,
    device_map="auto",
    trust_remote_code=True
)

def infer_text(prompt: str) -> str:
    """
    对单个文本进行推理
    
    Args:
        prompt (str): 文本提示
        
    Returns:
        str: 模型输出的文本响应
    """
    # 构建对话格式
    conversation = [
        {"role": "user", "content": prompt}
    ]
    
    # 应用对话模板
    text = tokenizer.apply_chat_template(conversation, tokenize=False)
    
    # 编码输入
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # 生成响应
    with torch.no_grad():
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=256,
        )
    
    # 解码响应
    response = tokenizer.batch_decode(generate_ids[:, inputs['input_ids'].size(1):], 
                                      skip_special_tokens=True, 
                                      clean_up_tokenization_spaces=False)[0]
    
    return response

# 处理JSONL文件
def process_jsonl(input_file: str, output_file: str):
    """
    处理JSONL文件中的所有文本并保存结果
    
    Args:
        input_file (str): 输入JSONL文件路径
        output_file (str): 输出JSONL文件路径
    """
    # 创建输出目录
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 读取并处理数据
    with open(input_file, 'r') as f:
        lines = f.readlines()
        updated_lines = []
        i = 1
        for line in lines:
            data = json.loads(line)
            prompt = data.get('prompt', '')  # 获取文本提示
            
            # 调用模型进行推理
            response_text = infer_text(prompt)
            
            print(f"处理第 {i} 个，共 {len(lines)} 个")
            print(response_text)
            i += 1
            
            # 更新数据并添加到输出列表
            data['response'] = response_text
            updated_lines.append(json.dumps(data) + '\n')
    
    # 写入输出文件
            with open(output_file, 'w') as f:
                f.writelines(updated_lines)

if __name__ == "__main__":
    # 设置输入和输出文件路径
    input_file = '/home/xiuying.chen/qian_jiang/AudioJailbreak/convert/combined_output.jsonl'
    output_dir = '/home/xiuying.chen/qian_jiang/AudioJailbreak/inference/text_Qwen2_response_jsonl'
    output_jsonl = os.path.join(output_dir, 'combined_output.jsonl')
    
    # 处理文件
    process_jsonl(input_file, output_jsonl)