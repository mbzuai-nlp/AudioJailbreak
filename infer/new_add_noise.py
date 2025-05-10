import numpy as np
from pydub import AudioSegment
import librosa
import os

class NoiseAdder:
    def __init__(self):
        # 初始化统计计数器
        self.noise_stats = {
            'pitch_shift': 0,     # 调音调
            'time_stretch': 0,    # 调音速
            'frequency_noise': 0, # 高低频扰动
            'dropout': 0,         # 丢帧
            'rain_noise': 0,      # 雨声
            'fade': 0,            # 淡化
            'baby_cry': 0,        # 婴儿哭声
            'car_horn': 0,        # 车喇叭
            'music': 0,           # 背景音乐
            'volume': 0           # 调音高/音量
        }
        
        # 统计每种噪声方式的具体参数使用情况
        self.detailed_stats = {
            'pitch_shift': {'up': 0, 'down': 0},
            'time_stretch': {'faster': 0, 'slower': 0},
            'frequency_noise': {'high': 0, 'low': 0},
            'rain_noise': {'heavy': 0, 'light': 0},
            'baby_cry': {'continuous': 0, 'intermittent': 0},
            'car_horn': {'single': 0, 'multiple': 0},
            'music': {'low_freq': 0, 'mid_freq': 0, 'high_freq': 0},
            'volume': {'up': 0, 'down': 0}
        }
        
        # 加载噪声文件
        self._load_noise_files()
        
    def _load_noise_files(self):
        """预加载噪声文件，提高效率"""
        # 初始化背景音
        try:
            self.rain = AudioSegment.from_mp3("./noise_test/noise/rain.mp3") - 7
            self.baby_laugh = AudioSegment.from_mp3("./noise_test/noise/baby_laugh.mp3") - 7
            self.baby_cry = AudioSegment.from_mp3("./noise_test/noise/baby_cry.mp3") - 9
            self.car = AudioSegment.from_mp3("./noise_test/noise/car.mp3") - 7
            self.music = AudioSegment.from_mp3("./noise_test/noise/music.mp3") - 10
            self.noise_files_loaded = True
        except Exception as e:
            print(f"警告: 无法加载噪声文件: {e}")
            self.noise_files_loaded = False

    def get_stats(self):
        """返回噪声使用统计信息"""
        return {
            'noise_counts': self.noise_stats,
            'detailed_stats': self.detailed_stats,
            'total_noise_added': sum(self.noise_stats.values())
        }
        
    def print_stats(self):
        """打印噪声使用统计信息"""
        stats = self.get_stats()
        print("\n噪声添加统计信息:")
        print(f"总添加噪声次数: {stats['total_noise_added']}")
        print("\n各类型噪声使用次数:")
        for noise_type, count in stats['noise_counts'].items():
            print(f"- {noise_type}: {count}次")
        
        print("\n详细噪声参数统计:")
        for noise_type, params in stats['detailed_stats'].items():
            if sum(params.values()) > 0:
                print(f"- {noise_type}: ", end="")
                details = [f"{param}={count}" for param, count in params.items() if count > 0]
                print(", ".join(details))

    def _validate_audio_file(self, file_path):
        """验证音频文件格式和有效性"""
        if not os.path.exists(file_path):
            print(f"错误：文件不存在 - {file_path}")
            return False
            
        try:
            audio = AudioSegment.from_file(file_path)
            # 检查音频时长是否超过5分钟
            if len(audio) > 5 * 60 * 1000:  # 5分钟 = 5 * 60 * 1000毫秒
                print("错误：音频长度超过5分钟")
                return False
            # 检查音频是否为空或损坏
            if len(audio) == 0:
                print("错误：音频文件为空")
                return False
            return True
        except Exception as e:
            print(f"错误：无效的音频文件格式或文件已损坏 - {e}")
            return False

    def add_noise(self, input_path, way=None, parameter=None):
        """
        向音频添加噪声
        
        Args:
            input_path: 音频文件路径
            way: 噪声类型参数 (0-1之间)，为空则随机生成
            parameter: 噪声参数 (0-1之间)，为空则随机生成
            
        Returns:
            tuple: (way, parameter) 如果成功，(None, None) 如果验证失败
        """
        # 验证输入文件
        if not self._validate_audio_file(input_path):
            return None, None
            
        # 初始化音频
        try:
            audio = AudioSegment.from_file(input_path)
        except Exception as e:
            print(f"错误：无法加载音频文件 - {e}")
            return None, None
            
        try:
            other_audio, sr = librosa.load(input_path, mono=True)  # mono确保是单声道
        except Exception as e:
            print(f"错误：librosa无法加载音频 - {e}")
            return None, None

        # 如果提供了way和parameter参数，则使用它们，否则随机生成
        if way is not None and parameter is not None:
            random_float = round(way, 3)
            random_float2 = round(parameter, 3)
        else:
            # 原来的随机生成方式
            random_float = round(np.random.uniform(0.0, 1.0), 3)
            random_float2 = round(np.random.uniform(0.0, 1.0), 3)

        print(f"2.add_noise的随机数1: {random_float}, 随机数2: {random_float2}")

        # 调音调
        if 0 < random_float < 0.1:
            self.noise_stats['pitch_shift'] += 1
            if random_float2 >= 0 and random_float2 < 0.5 and self.detailed_stats['pitch_shift']['up']<3:
                other_audio = librosa.effects.pitch_shift(other_audio, sr=sr, n_steps=random_float2 * 1)
                self.detailed_stats['pitch_shift']['up'] += 1
            elif self.detailed_stats['pitch_shift']['down']<4:
                other_audio = librosa.effects.pitch_shift(other_audio, sr=sr, n_steps=-random_float2 * 1)
                self.detailed_stats['pitch_shift']['down'] += 1
            
        # 调音速
        elif 0.1 <= random_float < 0.2:
            self.noise_stats['time_stretch'] += 1
            # 确保rate参数为正数
            if random_float2 > 0 and random_float2 < 0.5 and self.detailed_stats['time_stretch']['faster']<3:
                rate = max(1.1, 1+random_float2 * 0.1)  # 确保最大值为1.1
                other_audio = librosa.effects.time_stretch(other_audio, rate=rate)
                self.detailed_stats['time_stretch']['faster'] += 1
            elif self.detailed_stats['time_stretch']['slower']<6:
                rate = max(0.9, 1-random_float2 * 0.1)  # 确保最小值为0.1
                other_audio = librosa.effects.time_stretch(other_audio, rate=rate)
                self.detailed_stats['time_stretch']['slower'] += 1
            
        # 高低频扰动
        elif 0.2 <= random_float < 0.3 and self.noise_stats['frequency_noise']<3:
            self.noise_stats['frequency_noise'] += 1
            duration = len(other_audio) / sr  # 音频时长
            t = np.linspace(0, duration, len(other_audio), endpoint=False)
            if random_float2 > 0.5:
                # 高频扰动 (20000Hz)
                tone = 0.1 * np.sin(2 * np.pi * 20000*(1+random_float2) * t)
                self.detailed_stats['frequency_noise']['high'] += 1
            else:
                # 低频扰动 (20Hz) 
                tone = 0.1 * np.sin(2 * np.pi * 20*(1-random_float2) * t)
                self.detailed_stats['frequency_noise']['low'] += 1

            # 将扰动添加到音频中
            other_audio = other_audio + tone

        # drop_out 丢帧
        elif 0.3 <= random_float < 0.4 and self.noise_stats['dropout']<3:
            self.noise_stats['dropout'] += 1
            begin_time = random_float2 * len(audio)
            end_time = begin_time + random_float2 * 100
            left_audio = audio[:begin_time]
            right_audio = audio[end_time:]
            audio = left_audio + right_audio

        # 添加雨声
        elif 0.4 <= random_float < 0.5 and self.noise_stats['rain_noise']<1:
            self.noise_stats['rain_noise'] += 1
            rain = self.rain[:len(audio)]
            rain_volume = random_float2 * 0.9  # 雨声音量在0-90%之间变化
            rain_adjusted = rain - (8 * (1 - rain_volume))  # 调整雨声音量
            # 随机决定是大雨还是小雨
            if random_float2 > 0.7:
                # 大雨 - 更低的频率
                rain_adjusted = rain_adjusted.low_pass_filter(2000)
                self.detailed_stats['rain_noise']['heavy'] += 1
            else:
                # 小雨 - 更高的频率
                rain_adjusted = rain_adjusted.high_pass_filter(1000)
                self.detailed_stats['rain_noise']['light'] += 1
            audio = audio.overlay(rain_adjusted)

        # 淡化
        elif 0.5 <= random_float < 0.6 and self.noise_stats['fade']<1:
            try:
                if len(audio) > 4000:  # 防止过短
                    fade_duration = int(2000 * random_float2)
                    audio = audio.fade_in(fade_duration).fade_out(fade_duration)
                    self.noise_stats['fade'] += 1
            except Exception as e:
                print(f"淡化效果添加失败，跳过: {e}")
                # 继续处理下一个效果

        # 添加哭声
        elif 0.6 <= random_float < 0.7 and self.noise_stats['baby_cry']<1:
            self.noise_stats['baby_cry'] += 1
            baby_cry = self.baby_cry[:len(audio)]
            # 调整哭声音量，模拟距离远近
            cry_volume = random_float2 * 0.8  # 哭声音量在0-80%之间变化
            baby_cry_adjusted = baby_cry - (10 * (1 - cry_volume))
            # 随机决定是持续哭声还是间歇哭声
            if random_float2 > 0.6:
                # 间歇哭声 - 只在随机位置添加
                start_pos = int(len(audio) * random_float2 * 0.5)
                end_pos = min(start_pos + int(len(baby_cry_adjusted) * 0.7), len(audio))
                segment = audio[start_pos:end_pos]
                segment = segment.overlay(baby_cry_adjusted[:end_pos-start_pos])
                audio = audio[:start_pos] + segment + audio[end_pos:]
                self.detailed_stats['baby_cry']['intermittent'] += 1
            else:
                # 持续哭声 - 全程添加
                audio = audio.overlay(baby_cry_adjusted)
                self.detailed_stats['baby_cry']['continuous'] += 1

        # 添加车喇叭声
        elif 0.7 <= random_float < 0.8 and self.noise_stats['car_horn']<1:
            self.noise_stats['car_horn'] += 1
            car = self.car[:len(audio)]
            # 调整喇叭声音量，模拟距离远近
            horn_volume = random_float2 * 0.7  # 喇叭声音量在0-70%之间变化
            car_adjusted = car - (12 * (1 - horn_volume))
            # 随机决定是单次喇叭还是多次喇叭
            if random_float2 > 0.5:
                # 多次喇叭声 - 在不同位置添加2-3次
                num_horns = 2 + int(random_float2 * 2)
                for i in range(num_horns):
                    start_pos = int(len(audio) * (i + random_float2) / (num_horns + 1))
                    end_pos = min(start_pos + int(len(car) * 0.3), len(audio))
                    segment = audio[start_pos:end_pos]
                    segment = segment.overlay(car_adjusted[:end_pos-start_pos])
                    audio = audio[:start_pos] + segment + audio[end_pos:]
                self.detailed_stats['car_horn']['multiple'] += 1
            else:
                # 单次喇叭声 - 在随机位置添加
                start_pos = int(len(audio) * random_float2)
                audio = audio.overlay(car_adjusted, position=start_pos)
                self.detailed_stats['car_horn']['single'] += 1

        # 添加纯音乐
        elif 0.8 <= random_float < 0.9 and self.noise_stats['music']<1:
            self.noise_stats['music'] += 1
            music = self.music[:len(audio)]
            # 调整音乐音量，作为背景音
            music_volume = random_float2 * 0.6  # 音乐音量在0-60%之间变化
            music_adjusted = music - (15 * (1 - music_volume))
            # 随机决定音乐类型特性
            if random_float2 > 0.7:
                # 低频音乐 - 模拟低音较重的音乐
                music_adjusted = music_adjusted.low_pass_filter(1200)
                self.detailed_stats['music']['low_freq'] += 1
            elif random_float2 > 0.4:
                # 中频音乐 - 保持原样
                self.detailed_stats['music']['mid_freq'] += 1
            else:
                # 高频音乐 - 模拟高音较多的音乐
                music_adjusted = music_adjusted.high_pass_filter(800)
                self.detailed_stats['music']['high_freq'] += 1
            audio = audio.overlay(music_adjusted)

        # 调音高/音量
        elif 0.9 <= random_float <= 1:
            self.noise_stats['volume'] += 1
            if random_float2 > 0 and random_float2 < 0.5 and self.detailed_stats['volume']['up']<6:
                audio = audio + 1 * random_float
                self.detailed_stats['volume']['up'] += 1
            elif self.detailed_stats['volume']['down']<3:
                audio = audio - 0.8 * random_float
                self.detailed_stats['volume']['down'] += 1

        # 准备输出
        try:
            if 0 <= random_float < 0.3:
                # 这样处理是为了防止失真
                MAX_INT16 = 32767
                other_audio_int16 = (other_audio * MAX_INT16).astype(np.int16)
                to_audio = AudioSegment(
                    data=other_audio_int16.tobytes(),
                    sample_width=other_audio_int16.dtype.itemsize,
                    frame_rate=sr,
                    channels=1
                )
                to_audio.export(input_path, format="mp3")
            else:
                audio.export(input_path, format="mp3")
        except Exception as e:
            print(f"错误：导出音频文件失败 - {e}")
            return None, None
            
        return random_float, random_float2  # 返回使用的噪声参数，方便记录

# 为了兼容旧代码，提供一个全局实例和函数
noise_adder = NoiseAdder()

def add_noise(input_path, way=None, parameter=None):
    """兼容性函数，供旧代码调用"""
    return noise_adder.add_noise(input_path, way, parameter)