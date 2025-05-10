import os
from pydub import AudioSegment

def convert_mp3_to_wav():
    # 设置输入和输出目录
    input_dir = "./audio/jailbreakbench"
    output_dir = "./audio/wav_jailbreakbench"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有mp3文件
    mp3_files = [f for f in os.listdir(input_dir) if f.endswith('.mp3')]
    
    # 转换每个文件
    for mp3_file in mp3_files:
        # 构建完整的输入输出路径
        mp3_path = os.path.join(input_dir, mp3_file)
        wav_file = mp3_file.rsplit('.', 1)[0] + '.wav'  # 只改变后缀
        wav_path = os.path.join(output_dir, wav_file)
        
        # 如果wav文件不存在，则进行转换
        if not os.path.exists(wav_path):
            print(f"Converting {mp3_file} to WAV...")
            try:
                # 读取MP3
                audio = AudioSegment.from_mp3(mp3_path)
                # 设置采样率为16kHz
                audio = audio.set_frame_rate(16000)
                # 导出为WAV
                audio.export(wav_path, format='wav')
                print(f"Successfully converted {mp3_file}")
            except Exception as e:
                print(f"Error converting {mp3_file}: {str(e)}")
        else:
            print(f"Skipping {mp3_file} - WAV file already exists")

if __name__ == "__main__":
    convert_mp3_to_wav()
