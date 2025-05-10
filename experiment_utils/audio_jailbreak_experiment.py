import os
import json
import shutil
import numpy as np
import librosa
import whisper
import re
from pydub import AudioSegment
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize

# 下载必要的NLTK资源
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# 加载句子转换模型用于计算语义相似度
sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# 加载Whisper模型
whisper_model = whisper.load_model("base")

def setup_directories():
    """
    创建必要的目录结构，并将原始音频复制到实验目录
    """
    # 确保目录存在
    os.makedirs("experiment/origin_mp3", exist_ok=True)
    os.makedirs("experiment/experiment_mp3", exist_ok=True)
    os.makedirs("experiment/jailbreak_results", exist_ok=True)
    
    # 复制原始音频到实验目录
    for file_name in os.listdir("experiment/origin_mp3"):
        if file_name.endswith((".mp3", ".wav")):
            source_path = os.path.join("experiment/origin_mp3", file_name)
            dest_path = os.path.join("experiment/experiment_mp3", file_name)
            shutil.copy2(source_path, dest_path)
    
    print(f"✅ 已将原始音频复制到实验目录")

def extract_number_from_filename(filename):
    """
    从文件名中提取数字
    例如：从 'jailbreakbench_prompt_5.mp3' 提取 5
    
    参数:
        filename: 文件名
        
    返回:
        提取的数字，如果没有找到则返回None
    """
    match = re.search(r'jailbreakbench_prompt_(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

def load_text_prompts():
    """
    从JSONL文件中加载文本提示
    返回: 包含所有提示文本的字典，键为索引
    """
    prompts_dict = {}
    try:
        with open("/home/xiuying.chen/qian_jiang/AudioJailbreak/experiment/jailbreakbench.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                # 使用index作为键，prompt作为值
                if "index" in data and "prompt" in data:
                    prompts_dict[data["index"]] = data["prompt"]
        print(f"✅ 已加载 {len(prompts_dict)} 条文本提示")
        return prompts_dict
    except Exception as e:
        print(f"❌ 加载文本提示失败: {str(e)}")
        return {}

def add_noise(audio_path, method_type):
    """
    对音频添加噪声或调整参数
    
    参数:
        audio_path: 音频文件路径
        method_type: 调整方法 (1-6)
            1 - 音量升高
            2 - 音量降低
            3 - 音调升高
            4 - 音调降低
            5 - 音速升高
            6 - 音速降低
    """
    try:
        # 读取音频
        x = AudioSegment.from_file(audio_path)
        y, sr = librosa.load(audio_path, mono=True)
        
        # 根据方法类型调整音频
        if method_type == 1:  # 音量升高
            #x = x + 2  # 升高2dB
            x = x + 15  
            x.export(audio_path, format="mp3")
            
        elif method_type == 2:  # 音量降低
            #x = x - 2  # 降低2dB
            x = x - 15
            x.export(audio_path, format="mp3")
            
        elif method_type == 3:  # 音调升高
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=1)
            # 转换为AudioSegment并保存
            MAX_INT16 = 32767
            y_int16 = (y * MAX_INT16).astype(np.int16)
            to_audio = AudioSegment(
                data=y_int16.tobytes(),
                sample_width=y_int16.dtype.itemsize,
                frame_rate=sr,
                channels=1
            )
            to_audio.export(audio_path, format="mp3")
            
        elif method_type == 4:  # 音调降低
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=-1)
            # 转换为AudioSegment并保存
            MAX_INT16 = 32767
            y_int16 = (y * MAX_INT16).astype(np.int16)
            to_audio = AudioSegment(
                data=y_int16.tobytes(),
                sample_width=y_int16.dtype.itemsize,
                frame_rate=sr,
                channels=1
            )
            to_audio.export(audio_path, format="mp3")
            
        elif method_type == 5:  # 音速升高
            y = librosa.effects.time_stretch(y, rate=1.1)
            # 转换为AudioSegment并保存
            MAX_INT16 = 32767
            y_int16 = (y * MAX_INT16).astype(np.int16)
            to_audio = AudioSegment(
                data=y_int16.tobytes(),
                sample_width=y_int16.dtype.itemsize,
                frame_rate=sr,
                channels=1
            )
            to_audio.export(audio_path, format="mp3")
            
        elif method_type == 6:  # 音速降低
            y = librosa.effects.time_stretch(y, rate=0.9)
            # 转换为AudioSegment并保存
            MAX_INT16 = 32767
            y_int16 = (y * MAX_INT16).astype(np.int16)
            to_audio = AudioSegment(
                data=y_int16.tobytes(),
                sample_width=y_int16.dtype.itemsize,
                frame_rate=sr,
                channels=1
            )
            to_audio.export(audio_path, format="mp3")
            
        return True
    except Exception as e:
        print(f"❌ 处理音频 {audio_path} 时出错: {str(e)}")
        return False

def analyze_audio(audio_path):
    """
    分析音频的属性
    
    参数:
        audio_path: 音频文件路径
        
    返回:
        包含音频属性的字典
    """
    try:
        # 加载音频
        y, sr = librosa.load(audio_path)
        
        # 1. 音量分析
        rms = librosa.feature.rms(y=y)[0]
        db = librosa.amplitude_to_db(rms)
        
        # 2. 音调分析
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr
        )
        valid_f0 = f0[~np.isnan(f0)]
        
        # 3. 节奏/速度分析
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        
        # 返回分析结果
        return {
            "音量": {
                "RMS均值": float(np.mean(rms)),
                "分贝均值": float(np.mean(db)),
            },
            "音调": {
                "平均基频": float(np.nanmean(f0)) if len(valid_f0) > 0 else 0,
                "音高变化率": float(np.mean(np.abs(np.diff(valid_f0)))) if len(valid_f0) > 1 else 0
            },
            "速度": {
                "估计节奏(BPM)": float(np.asarray(tempo).item()),  # 避免NumPy警告
                "频谱变化率": float(np.mean(np.abs(np.diff(np.abs(librosa.stft(y)), axis=1))))
            }
        }
    except Exception as e:
        print(f"❌ 分析音频 {audio_path} 时出错: {str(e)}")
        return {
            "音量": {"RMS均值": 0, "分贝均值": 0},
            "音调": {"平均基频": 0, "音高变化率": 0},
            "速度": {"估计节奏(BPM)": 0, "频谱变化率": 0}
        }

def transcribe_audio(audio_path):
    """
    使用Whisper将音频转录为文本
    
    参数:
        audio_path: 音频文件路径
        
    返回:
        转录的文本
    """
    try:
        result = whisper_model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        print(f"❌ 转录音频 {audio_path} 时出错: {str(e)}")
        return ""

def calculate_similarity(text1, text2):
    """
    计算两段文本的相似度
    
    参数:
        text1: 第一段文本
        text2: 第二段文本
        
    返回:
        相似度分数 (0-1)
    """
    # 如果任一文本为空，则相似度为0
    if not text1 or not text2:
        return 0.0
        
    # 1. 使用句子转换模型计算语义相似度
    try:
        embedding1 = sentence_model.encode([text1])[0]
        embedding2 = sentence_model.encode([text2])[0]
        semantic_sim = cosine_similarity([embedding1], [embedding2])[0][0]
    except Exception as e:
        print(f"❌ 计算语义相似度时出错: {str(e)}")
        semantic_sim = 0
    
    # 2. 使用BLEU分数计算词级别相似度
    try:
        reference = [word_tokenize(text1.lower())]
        candidate = word_tokenize(text2.lower())
        # 使用平滑函数避免警告
        from nltk.translate.bleu_score import SmoothingFunction
        smooth = SmoothingFunction().method1
        bleu_score = sentence_bleu(reference, candidate, smoothing_function=smooth)
    except Exception as e:
        print(f"❌ 计算BLEU分数时出错: {str(e)}")
        bleu_score = 0
    
    # 综合两种相似度
    similarity = (semantic_sim * 0.7) + (bleu_score * 0.3)
    return float(similarity)

def restore_audio():
    """
    将所有音频恢复到原始状态
    """
    # 删除实验目录中的所有文件
    for file_name in os.listdir("/home/xiuying.chen/qian_jiang/AudioJailbreak/experiment/experiment_mp3"):
        file_path = os.path.join("/home/xiuying.chen/qian_jiang/AudioJailbreak/experiment/experiment_mp3", file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
    
    # 从原始目录复制文件到实验目录
    for file_name in os.listdir("/home/xiuying.chen/qian_jiang/AudioJailbreak/experiment/origin_mp3"):
        if file_name.endswith((".mp3", ".wav")):
            source_path = os.path.join("/home/xiuying.chen/qian_jiang/AudioJailbreak/experiment/origin_mp3", file_name)
            dest_path = os.path.join("/home/xiuying.chen/qian_jiang/AudioJailbreak/experiment/experiment_mp3", file_name)
            shutil.copy2(source_path, dest_path)

def main():
    """
    主函数，执行完整的实验流程
    """
    # 设置目录并复制音频文件
    #setup_directories()
    # 恢复音频到原始状态
    restore_audio()
    # 加载文本提示，现在是一个字典，键为索引
    original_prompts_dict = load_text_prompts()
    if not original_prompts_dict:
        print("❌ 无法加载文本提示，程序退出")
        return
    
    # 获取实验目录中的所有音频文件
    audio_files = [f for f in os.listdir("/home/xiuying.chen/qian_jiang/AudioJailbreak/experiment/experiment_mp3") 
                  if f.endswith((".mp3", ".wav"))]
    
    if not audio_files:
        print("❌ 实验目录中没有音频文件，程序退出")
        return
    
    # 用户选择调整方法
    print("🔹 请选择要调整的属性：")
    print("   1 - 音量升高")
    print("   2 - 音量降低")
    print("   3 - 音调升高")
    print("   4 - 音调降低")
    print("   5 - 音速升高")
    print("   6 - 音速降低")
    
    try:
        method_type = int(input("请输入对应的数字 (1-6)：").strip())
        if method_type < 1 or method_type > 6:
            raise ValueError("选项必须在1-6之间")
    except ValueError as e:
        print(f"❌ 输入错误: {str(e)}")
        return
    
    # 方法名称映射
    method_names = {
        1: "volume_up",
        2: "volume_down",
        3: "pitch_up",
        4: "pitch_down",
        5: "speed_up",
        6: "speed_down"
    }
    
    method_name = method_names[method_type]
    print(f"✅ 你选择了: {method_name}")
    
    # 准备结果存储
    all_results = []
    
    # 执行30轮实验
    for round_idx in range(1,6):
        print(f"\n🚀 开始第 {round_idx}/6 轮实验")
        
        
        # 存储本轮结果
        round_results = {
            "轮次": round_idx,
            "调整方法": method_name,
            "音频属性": [],
            "相似度": [],
            "原始提示": [],
            "转录文本": []
        }
        
        # 处理每个音频文件
        for audio_file in audio_files:
            # 从文件名提取数字作为索引
            file_index = extract_number_from_filename(audio_file)
            audio_path = os.path.join("/home/xiuying.chen/qian_jiang/AudioJailbreak/experiment/experiment_mp3", audio_file)
            
            print(f"  🎵 处理音频: {audio_file} (索引: {file_index})")
            
            # 1. 添加噪声
            success = add_noise(audio_path, method_type)
            if not success:
                continue
            
            # 2. 分析音频属性
            audio_features = analyze_audio(audio_path)
            
            # 3. 转录音频
            transcribed_text = transcribe_audio(audio_path)
            
            # 4. 获取对应的原始提示并计算相似度
            original_prompt = ""
            if file_index is not None and file_index in original_prompts_dict:
                original_prompt = original_prompts_dict[file_index]
                print(f"    ✓ 匹配到原始提示 (索引: {file_index})")
            else:
                print(f"    ⚠️ 未找到索引 {file_index} 对应的原始提示")
                
            similarity = calculate_similarity(original_prompt, transcribed_text)
            
            # 存储结果
            round_results["音频属性"].append({
                "文件名": audio_file,
                "索引": file_index,
                "特征": audio_features
            })
            round_results["相似度"].append({
                "文件名": audio_file,
                "索引": file_index,
                "分数": similarity
            })
            round_results["原始提示"].append({
                "文件名": audio_file,
                "索引": file_index,
                "文本": original_prompt
            })
            round_results["转录文本"].append({
                "文件名": audio_file,
                "索引": file_index,
                "文本": transcribed_text
            })
            
            print(f"    📊 相似度: {similarity:.4f}")
        
        # 计算平均值
        if round_results["相似度"]:
            avg_similarity = sum(item["分数"] for item in round_results["相似度"]) / len(round_results["相似度"])
            round_results["平均相似度"] = avg_similarity
            print(f"  📈 本轮平均相似度: {avg_similarity:.4f}")
        
        # 计算音频属性平均值
        if round_results["音频属性"]:
            avg_features = {
                "音量": {"RMS均值": 0, "分贝均值": 0},
                "音调": {"平均基频": 0, "音高变化率": 0},
                "速度": {"估计节奏(BPM)": 0, "频谱变化率": 0}
            }
            
            count = len(round_results["音频属性"])
            for item in round_results["音频属性"]:
                features = item["特征"]
                avg_features["音量"]["RMS均值"] += features["音量"]["RMS均值"] / count
                avg_features["音量"]["分贝均值"] += features["音量"]["分贝均值"] / count
                avg_features["音调"]["平均基频"] += features["音调"]["平均基频"] / count
                avg_features["音调"]["音高变化率"] += features["音调"]["音高变化率"] / count
                avg_features["速度"]["估计节奏(BPM)"] += features["速度"]["估计节奏(BPM)"] / count
                avg_features["速度"]["频谱变化率"] += features["速度"]["频谱变化率"] / count
            
            round_results["平均音频属性"] = avg_features
        
        # 添加到总结果
        all_results.append(round_results)
        
        # 保存当前结果
        result_file = f"/home/xiuying.chen/qian_jiang/AudioJailbreak/experiment/results/result_{method_name}_round_{round_idx}.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(round_results, f, ensure_ascii=False, indent=2)
        
        print(f"  💾 已保存本轮结果到: {result_file}")
    
    # 保存所有结果
    final_result_file = f"/home/xiuying.chen/qian_jiang/AudioJailbreak/experiment/results/all_results_{method_name}.json"
    with open(final_result_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n🎉 所有实验完成！总结果已保存到: {final_result_file}")

if __name__ == "__main__":
    main()
