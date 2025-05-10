import json
from collections import Counter

# 文件路径
input_file = "/home/xiuying.chen/qian_jiang/AudioJailbreak/convert/sorted_combined_output.jsonl"
output_file = "/home/xiuying.chen/qian_jiang/AudioJailbreak/convert/category_statistics.txt"

# 统计各个category的数量
def count_categories(file_path):
    categories = []
    
    # 读取JSONL文件，提取category字段
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if 'category' in data:
                    categories.append(data['category'])
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line: {line[:50]}...")
                continue
    
    # 使用Counter统计各个category的数量
    category_counts = Counter(categories)
    return category_counts

# 将结果写入文本文件
def write_statistics(counts, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("类别统计结果:\n")
        f.write("=" * 30 + "\n")
        
        # 按数量从大到小排序
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        
        # 写入各个类别及其数量
        for category, count in sorted_counts:
            f.write(f"{category}: {count}\n")
        
        # 写入总计
        f.write("=" * 30 + "\n")
        f.write(f"总计: {sum(counts.values())}\n")

# 主函数
def main():
    print("开始统计category...")
    category_counts = count_categories(input_file)
    
    print("写入统计结果到文件...")
    write_statistics(category_counts, output_file)
    
    print(f"统计完成，结果已保存到 {output_file}")

if __name__ == "__main__":
    main()
