import json
import os
import re

def extract_filename_core(path):
    """提取文件路径中最后一个/和.之间的部分"""
    # 使用正则表达式提取
    match = re.search(r'/([^/]+)\.[^.]+$', path)
    if match:
        return match.group(1)
    return None

def extract_matching_entries():
    # 定义文件路径
    combined_path = "/home/xiuying.chen/qian_jiang/AudioJailbreak/inference/SALMONN_response_jsonl/combined_output.jsonl"
    sorted_path = "/home/xiuying.chen/qian_jiang/AudioJailbreak/convert/sorted_combined_output.jsonl"
    
    # 确定输出文件路径
    output_dir = os.path.dirname(combined_path)
    output_path = os.path.join(output_dir, "sellect_combined_output.jsonl")
    
    print(f"开始处理...")
    print(f"从 {sorted_path} 提取文件名核心部分")
    print(f"匹配 {combined_path} 中的记录")
    print(f"输出到 {output_path}")
    
    # 从sorted_combined_output.jsonl提取文件名核心集合
    filename_cores = set()
    with open(sorted_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if 'speech_path' in data:
                    core = extract_filename_core(data['speech_path'])
                    if core:
                        filename_cores.add(core)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line from sorted file: {line[:50]}...")
    
    print(f"提取到 {len(filename_cores)} 个唯一的文件名核心")
    
    # 从combined_output.jsonl中筛选匹配的条目
    matched_entries = []
    with open(combined_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                if 'speech_path' in data:
                    core = extract_filename_core(data['speech_path'])
                    if core and core in filename_cores:
                        matched_entries.append(data)
                        # 从集合中移除已匹配的文件名核心，提高后续匹配效率
                        filename_cores.remove(core)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line {line_num} from combined file: {line[:50]}...")
    
    print(f"找到 {len(matched_entries)} 个匹配的条目")
    
    # 检查是否有未匹配的文件名核心
    if filename_cores:
        print(f"Warning: {len(filename_cores)} 个文件名核心未找到匹配的条目")
    
    # 将匹配的条目写入输出文件
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in matched_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"处理完成。匹配的条目已写入 {output_path}")

if __name__ == "__main__":
    extract_matching_entries()
