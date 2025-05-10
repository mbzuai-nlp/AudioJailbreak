import os
import json
from google import genai

# 初始化 Gemini 客户端
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))  # 使用环境变量获取 API 密钥

# 下面3个都需要改
input_file = '/home/xiuying.chen/qian_jiang/AudioJailbreak/convert/combined_output.jsonl'
output_file = '/home/xiuying.chen/qian_jiang/AudioJailbreak/inference/text_Gemini2.0_flash_response_jsonl'
output_jsonl = os.path.join(output_file, 'combined_output.jsonl')

os.makedirs(output_file, exist_ok=True)

with open(input_file, 'r') as f:
    lines = f.readlines()
    updated_lines = []
    i = 1
    for line in lines:
        data = json.loads(line)
        prompt = data.get('prompt', '')  # 获取文本提示
        
        # 调用 Gemini 2.0 Flash 模型处理文本
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=[prompt]
        )
        
        response_text = response.text
        
        print(f"处理第 {i} 个，共 {len(lines)} 个")
        print(response_text)
        i += 1
        
        # 更新数据并添加到输出列表
        data['response'] = response_text
        updated_lines.append(json.dumps(data) + '\n')

# 写入输出文件
        with open(output_jsonl, 'w') as f:
            f.writelines(updated_lines)
