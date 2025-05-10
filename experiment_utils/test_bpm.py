import librosa
import numpy as np
from pydub import AudioSegment

# 测试BPM的函数
def estimate_bpm(y, sr):
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return float(tempo)


# 保存音频函数
def save_audio(y, sr, audio_path):
    MAX_INT16 = 32767
    y_int16 = (y * MAX_INT16).astype(np.int16)
    to_audio = AudioSegment(
        data=y_int16.tobytes(),
        sample_width=y_int16.dtype.itemsize,
        frame_rate=sr,
        channels=1
    )
    to_audio.export(audio_path, format="mp3")

# 主流程
def process_audio(audio_file, output_audio_file):
    # 加载音频
    y, sr = librosa.load(audio_file, sr=None)

    # 测试原始BPM值
    original_bpm = estimate_bpm(y, sr)
    print(f"原始音频BPM: {original_bpm:.2f}")

    # 时长扩大为两倍（播放速度减半，rate=0.5）
    y_slow = librosa.effects.time_stretch(y, rate=0.5)

    # 保存处理后的音频
    save_audio(y_slow, sr, output_audio_file)

    # 测试处理后的BPM值
    processed_bpm = estimate_bpm(y_slow, sr)
    print(f"处理后音频BPM: {processed_bpm:.2f}")


# 使用示例
audio_path = "/home/xiuying.chen/qian_jiang/AudioJailbreak/experiment/jailbreakbench_prompt_1.mp3"
output_audio_path = "/home/xiuying.chen/qian_jiang/AudioJailbreak/experiment/jailbreakbench_prompt_1_slow.mp3"

process_audio(audio_path, output_audio_path)
