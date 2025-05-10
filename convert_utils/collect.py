import json
import librosa
import os

def get_audio_duration(audio_path):
    """获取音频文件的时长"""
    try:
        duration = librosa.get_duration(path=audio_path)
        return duration
    except Exception as e:
        print(f"Warning: Could not get duration for {audio_path}: {e}")
        return float('inf')  # 返回无穷大，这样错误的文件会排到最后

# 读取原始JSONL文件
input_path = "./convert/combined_output.jsonl"
output_path = "./convert/sorted_combined_output.jsonl"

# 读取所有数据
data = []
with open(input_path, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))

# 获取每个音频文件的时长并创建排序信息
audio_info = []
for item in data:
    # 检查音频路径是否为空
    if not item.get('speech_path'):
        print(f"跳过空音频路径的数据项: {item}")
        continue
        
    duration = get_audio_duration(item['speech_path'])
    audio_info.append({
        'original_index': item['index'],
        'duration': duration,
        'data': item
    })

# 根据时长排序
audio_info.sort(key=lambda x: x['duration'])

# 只取前500个并重新编号保存
with open(output_path, 'w', encoding='utf-8') as f:
    for new_index, info in enumerate(audio_info[:500]):
        # 创建新的数据项，保持原有内容不变，只更新index
        output_item = info['data'].copy()
        output_item['index'] = new_index
        
        # 写入JSONL文件
        json.dump(output_item, f, ensure_ascii=False)
        f.write('\n')

print(f"排序完成，输出文件保存在: {output_path}")
print(f"共处理并输出前500个音频文件")
