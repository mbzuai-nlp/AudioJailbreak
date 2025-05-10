import os
import shutil

def copy_audio_files(source_dir, dest_dir, start_num=1, end_num=10):
    """
    将指定范围的MP3文件从源目录复制到目标目录。

    参数:
        source_dir: 源目录路径
        dest_dir: 目标目录路径
        start_num: 起始文件编号 (默认1)
        end_num: 结束文件编号 (默认10)
    """

    # 确保目标目录存在
    os.makedirs(dest_dir, exist_ok=True)

    # 复制指定范围的文件
    for num in range(start_num, end_num + 1):
        file_name = f"jailbreakbench_prompt_{num}.mp3"
        source_path = os.path.join(source_dir, file_name)
        dest_path = os.path.join(dest_dir, file_name)
        
        if os.path.exists(source_path):
            try:
                shutil.copy2(source_path, dest_path)  # 使用copy2保留元数据
                print(f"✅ 已复制: {file_name} 到 {dest_dir}")
            except Exception as e:
                print(f"❌ 复制 {file_name} 时出错: {str(e)}")
        else:
            print(f"⚠️ 文件不存在: {source_path}")

    print(f"🎉 成功复制 jailbreakbench_prompt_{start_num} 到 jailbreakbench_prompt_{end_num} 的文件到 {dest_dir}")

if __name__ == "__main__":
    source_directory = "audio/jailbreakbench"
    destination_directory = "experiment/origin_mp3"
    copy_audio_files(source_directory, destination_directory)
