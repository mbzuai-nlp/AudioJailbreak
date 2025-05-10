import json
import os

def change_paths(input_file, output_file):
    """
    读取JSONL文件，修改speech_path字段，只保留文件名，并添加新的路径前缀
    """
    new_prefix = "/home/xiuying.chen/qian_jiang/AudioJailbreak/inference/gpt4o_response_jsonl/new_bon/"
    
    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            data = json.loads(line.strip())
            
            if 'speech_path' in data and data['speech_path']:
                # 获取原路径中的文件名
                filename = os.path.basename(data['speech_path'])
                # 创建新路径
                data['speech_path'] = new_prefix + filename
            
            # 写入修改后的数据
            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    input_jsonl = "/home/xiuying.chen/qian_jiang/AudioJailbreak/convert/bon_sorted_combined_output.jsonl"
    output_jsonl = input_jsonl  # 输出到同一个文件
    
    # 先创建一个临时文件
    temp_output = input_jsonl + ".temp"
    change_paths(input_jsonl, temp_output)
    
    # 替换原文件
    import shutil
    shutil.move(temp_output, output_jsonl)
    
    print(f"文件处理完成，路径已更新: {output_jsonl}")
