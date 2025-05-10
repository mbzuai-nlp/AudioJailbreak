#!/usr/bin/env python
import json
import os

def calculate_average_policy_violation(file_path):
    # 读取JSON文件
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # 提取所有policy_violation值
    pv_values = [data[category]["policy_violation"] for category in data]
    
    # 计算平均值
    average_pv = sum(pv_values) / len(pv_values) if pv_values else 0
    
    # 添加平均值到原数据
    data["average_policy_violation"] = average_pv
    
    # 写回文件
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    
    return average_pv

if __name__ == "__main__":
    file_path = "/home/xiuying.chen/qian_jiang/AudioJailbreak/eval/text_qwen2/policy_violation_output.jsonl"
    avg = calculate_average_policy_violation(file_path)
    print(f"Average policy violation: {avg}")
