import os
import json
import shutil
from pydub import AudioSegment
import re

def convert_mp3_to_wav(source_dirs, target_dir):
    """
    将指定目录中的MP3文件转换为WAV格式并保存到目标目录
    
    Args:
        source_dirs: 包含MP3文件的源目录列表
        target_dir: 保存WAV文件的目标目录
    """
    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)
    
    # 记录转换的文件数量
    converted_count = 0
    
    # 遍历每个源目录
    for source_dir in source_dirs:
        print(f"处理目录: {source_dir}")
        
        # 遍历目录中的所有文件
        for root, _, files in os.walk(source_dir):
            for file in files:
                if file.lower().endswith('.mp3'):
                    # 构建源文件和目标文件的完整路径
                    mp3_path = os.path.join(root, file)
                    
                    # 提取文件名（不包含扩展名）
                    filename_without_ext = os.path.splitext(file)[0]
                    
                    # 构建目标WAV文件路径
                    wav_filename = f"{filename_without_ext}.wav"
                    wav_path = os.path.join(target_dir, wav_filename)
                    
                    try:
                        # 使用pydub加载MP3文件并转换为WAV
                        audio = AudioSegment.from_mp3(mp3_path)
                        
                        # 设置为16kHz采样率和单声道（适合大多数语音模型）
                        audio = audio.set_frame_rate(16000).set_channels(1)
                        
                        # 导出为WAV格式
                        audio.export(wav_path, format="wav")
                        
                        converted_count += 1
                        if converted_count % 10 == 0:
                            print(f"已转换 {converted_count} 个文件...")
                            
                    except Exception as e:
                        print(f"转换文件 {mp3_path} 时出错: {str(e)}")
    
    print(f"转换完成! 共转换了 {converted_count} 个文件。")

def update_jsonl_paths(source_jsonl, target_jsonl):
    """
    复制JSONL文件并更新其中的音频路径
    
    Args:
        source_jsonl: 源JSONL文件路径
        target_jsonl: 目标JSONL文件路径
    """
    print(f"更新JSONL文件: {source_jsonl} -> {target_jsonl}")
    
    # 读取源JSONL文件
    with open(source_jsonl, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    updated_lines = []
    for line in lines:
        data = json.loads(line)
        
        # 更新speech_path字段
        if 'speech_path' in data:
            # 使用正则表达式替换路径
            # 将 ./audio/xxx/filename.mp3 替换为 ./audio/total_wav/filename.wav
            old_path = data['speech_path']
            new_path = re.sub(
                r'./audio/[^/]+/([^/]+)\.mp3',
                r'./audio/total_wav/\1.wav',
                old_path
            )
            data['speech_path'] = new_path
        
        # 将更新后的数据添加到列表
        updated_lines.append(json.dumps(data, ensure_ascii=False) + '\n')
    
    # 写入目标JSONL文件
    with open(target_jsonl, 'w', encoding='utf-8') as f:
        f.writelines(updated_lines)
    
    print(f"JSONL文件更新完成，共处理了 {len(updated_lines)} 条记录。")

if __name__ == "__main__":
    # 定义源目录和目标目录
    base_dir = "/home/xiuying.chen/qian_jiang/AudioJailbreak"
    source_dirs = [
        f"{base_dir}/audio/Do_Not_Answer",
        f"{base_dir}/audio/jailbreak_llms",
        f"{base_dir}/audio/jailbreakbench"
    ]
    target_dir = f"{base_dir}/audio/total_wav"
    
    # 定义源JSONL和目标JSONL
    source_jsonl = f"{base_dir}/convert/combined_output.jsonl"
    target_jsonl = f"{base_dir}/convert/wav_combined_output.jsonl"
    
    # 执行MP3到WAV的转换
    convert_mp3_to_wav(source_dirs, target_dir)
    
    # 更新JSONL文件中的路径
    update_jsonl_paths(source_jsonl, target_jsonl)
    
    print("所有任务完成!")
