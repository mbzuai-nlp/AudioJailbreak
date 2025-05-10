import json
import os
from collections import defaultdict

def calculate_average():
    # 定义文件路径
    input_path = "/home/xiuying.chen/qian_jiang/AudioJailbreak/eval/SALMONN/sellect_sorted_output.jsonl"
    output_dir = os.path.dirname(input_path)
    output_path = os.path.join(output_dir, "sellect_aver.json")
    
    print(f"开始处理文件: {input_path}")
    
    try:
        # 读取输入文件
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取文件失败: {e}")
        return
    
    # 统计数量
    category_count = len(data)
    print(f"找到 {category_count} 个类别")
    
    if category_count == 0:
        print("没有找到有效数据，处理终止")
        return
    
    # 用于存储所有指标的总和
    metric_sums = defaultdict(float)
    gpt_metric_sums = defaultdict(float)
    metrics_count = defaultdict(int)
    
    # 遍历所有类别并累加各指标值
    for category, metrics in data.items():
        # 处理一级指标
        for metric_name, metric_value in metrics.items():
            if metric_name != "gpt_score" and isinstance(metric_value, (int, float)):
                metric_sums[metric_name] += metric_value
                metrics_count[metric_name] += 1
                
        # 处理gpt_score中的指标
        if "gpt_score" in metrics and isinstance(metrics["gpt_score"], dict):
            for gpt_metric, gpt_value in metrics["gpt_score"].items():
                if isinstance(gpt_value, (int, float)):
                    gpt_metric_sums[gpt_metric] += gpt_value
                    metrics_count[f"gpt_{gpt_metric}"] += 1
    
    # 计算平均值
    averages = {}
    
    # 计算一级指标的平均值
    for metric_name, total in metric_sums.items():
        count = metrics_count[metric_name]
        if count > 0:
            averages[metric_name] = total / count
    
    # 计算gpt_score指标的平均值
    gpt_averages = {}
    for gpt_metric, total in gpt_metric_sums.items():
        count = metrics_count[f"gpt_{gpt_metric}"]
        if count > 0:
            gpt_averages[gpt_metric] = total / count
    
    # 如果有gpt_score相关的指标，将其添加到结果中
    if gpt_averages:
        averages["gpt_score"] = gpt_averages
    
    # 输出结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(averages, f, indent=4)
    
    print(f"处理完成，平均值已保存至: {output_path}")
    
    # 打印结果预览
    print("\n计算结果预览:")
    print(json.dumps(averages, indent=2))

if __name__ == "__main__":
    calculate_average()
