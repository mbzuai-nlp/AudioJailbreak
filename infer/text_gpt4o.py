import os
import json
from openai import OpenAI

# 初始化 OpenAI 客户端
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 下面3个都需要改
input_file = '/home/xiuying.chen/qian_jiang/AudioJailbreak/convert/combined_output.jsonl'
output_file = '/home/xiuying.chen/qian_jiang/AudioJailbreak/inference/text_GPT4o_response_jsonl'
output_jsonl = os.path.join(output_file, 'combined_output.jsonl')

os.makedirs(output_file, exist_ok=True)

with open(input_file, 'r') as f:
    lines = f.readlines()
    updated_lines = []
    i = 1
    for line in lines:
        data = json.loads(line)
        prompt = data.get('prompt', '')  
        
        # 调用 GPT-4o 模型
        completion = client.chat.completions.create(
            model="gpt-4o-mini",  # 使用 gpt-4o 模型
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        
        response_text = completion.choices[0].message.content
        
        print(f"处理第 {i} 个，共 {len(lines)} 个")
        print(response_text)
        i += 1
        
        # 更新数据并添加到输出列表
        data['response'] = response_text
        updated_lines.append(json.dumps(data) + '\n')

# 写入输出文件
with open(output_jsonl, 'w') as f:
    f.writelines(updated_lines)