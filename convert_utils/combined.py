import os
import json
import glob

def combine_jsonl_files():
    """
    将除了total.jsonl之外的所有JSONL文件合并成一个新的JSONL文件，
    并确保"index"键按顺序从1开始递增，同时删除所有记录中的"response"键的值
    """
    # 获取当前目录下所有的jsonl文件
    jsonl_files = glob.glob("*.jsonl")
    
    # 排除total.jsonl
    jsonl_files = [f for f in jsonl_files if f != "total.jsonl"]
    jsonl_files = [f for f in jsonl_files if f != "wav_jailbreakbench.jsonl"]
    jsonl_files = [f for f in jsonl_files if f != "combined_output.jsonl"]
    
    if not jsonl_files:
        print("❌ 没有找到JSONL文件")
        return
    
    print(f"🔍 找到以下JSONL文件: {jsonl_files}")
    
    # 读取所有文件的内容
    all_records = []
    for file_path in jsonl_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        # 将response键的值设为空，而不是删除键
                        if 'response' in record:
                            record['response'] = ""
                        all_records.append(record)
                    except json.JSONDecodeError:
                        print(f"⚠️ 跳过无效的JSON行: {line[:50]}...")
        except Exception as e:
            print(f"❌ 读取文件 {file_path} 时出错: {str(e)}")
    
    print(f"✅ 总共读取了 {len(all_records)} 条记录")
    
    # 重新分配index
    for i, record in enumerate(all_records, 1):
        record["index"] = i
    
    # 写入新文件
    output_file = "combined_output.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"🎉 合并完成! 已将 {len(all_records)} 条记录写入 {output_file}")

if __name__ == "__main__":
    # 切换到脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    combine_jsonl_files()
