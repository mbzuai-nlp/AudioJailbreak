import json
import os

# 定义输入和输出文件路径
input_jsonl = '/home/xiuying.chen/qian_jiang/AudioJailbreak/convert/sorted_combined_output.jsonl'
output_jsonl = '/home/xiuying.chen/qian_jiang/AudioJailbreak/convert/wav_sorted_combined_output.jsonl'

# 存储转换后的数据
converted_data = []

# 统计信息
total_processed = 0
modified_paths = 0

# 读取输入jsonl文件
with open(input_jsonl, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip():  # 跳过空行
            continue
            
        total_processed += 1
        entry = json.loads(line)
        
        # 获取当前speech_path
        if 'speech_path' in entry and entry['speech_path']:
            current_path = entry['speech_path']
            
            # 解析路径组件
            parts = current_path.split('/')
            
            # 处理文件名和扩展名
            filename = parts[-1]
            if filename.lower().endswith('.mp3'):
                # 替换扩展名
                filename = filename[:-4] + '.wav'
                
                # 构建新路径：添加"../."+保留"audio"+替换一级目录为"total_wav"+文件名
                # 例如："./audio/jailbreak_llms/jailbreak_llms_prompt_1230.mp3" 
                # 变为 "../../audio/total_wav/jailbreak_llms_prompt_1230.wav"
                new_path = "../../audio/total_wav/" + filename
                
                # 更新entry
                entry['speech_path'] = new_path
                modified_paths += 1
        
        # 添加到结果列表
        converted_data.append(entry)

# 写入输出jsonl文件
with open(output_jsonl, 'w', encoding='utf-8') as f:
    for entry in converted_data:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

print(f"转换完成! 总处理: {total_processed} 条记录, 修改了 {modified_paths} 个路径")
print(f"输出文件: {output_jsonl}")
