#text to speech
from google.cloud import texttospeech
import os
from dotenv import load_dotenv
import random
import logging
from datetime import datetime

class TTSService:
    def __init__(self, audio_save_path=None, audio_prefix=''):
        load_dotenv()
        self.client = texttospeech.TextToSpeechClient.from_service_account_file(
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
        )
        self.voice_cache = {}
        self.initialize_voice_cache()
        self.audio_save_path = audio_save_path or os.path.join(os.getcwd(), 'audio_outputs')
        self.audio_prefix = audio_prefix
        # 确保音频保存目录存在
        os.makedirs(self.audio_save_path, exist_ok=True)

    def initialize_voice_cache(self):
        """初始化语音配置缓存"""
        voices = self.client.list_voices().voices
        for voice in voices:
            for language_code in voice.language_codes:
                if language_code not in self.voice_cache:
                    self.voice_cache[language_code] = []
                self.voice_cache[language_code].append({
                    'name': voice.name,
                    'gender': voice.ssml_gender
                })

    def get_random_voice(self, language_code):
        """从缓存中获取指定语言的随机语音配置"""
        if language_code not in self.voice_cache:
            raise ValueError(f"No voices found for language {language_code}")
        
        return random.choice(self.voice_cache[language_code])

    def text_to_speech(self, text, output_file, language='en-US', voice_name=None, gender=None):
        """
        将文本转换为语音
        :param text: 要转换的文本
        :param output_file: 输出音频文件路径
        :param language: 语言代码，默认为英语
        :param voice_name: 可选的指定语音名称
        :param gender: 可选的指定语音性别
        """
        if voice_name is None or gender is None:
            random_voice = self.get_random_voice(language)
            voice_name = voice_name or random_voice['name']
            gender = gender or random_voice['gender']

        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code=language,
            name=voice_name,
            ssml_gender=gender
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

        response = self.client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        with open(output_file, "wb") as out:
            out.write(response.audio_content)
            print(f'Audio content written to file "{output_file}"')

    def batch_process_jsonl(self, input_file, output_file):
        """
        批量处理jsonl文件中的prompt并生成语音
        :param input_file: 输入的jsonl文件路径
        :param output_file: 输出的jsonl文件路径
        """
        import json
        import os

        # 设置日志
        log_dir = os.path.join(os.path.dirname(output_file), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'tts_errors_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        logging.basicConfig(
            filename=log_file,
            level=logging.ERROR,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        # 使用实例的音频保存路径
        output_dir = self.audio_save_path
        os.makedirs(output_dir, exist_ok=True)

        processed_records = []
        checkpoint_file = output_file + '.checkpoint'
        
        # 检查断点文件
        last_processed_index = -1
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r') as f:
                    last_processed_index = int(f.read().strip())
                print(f"从断点继续处理，上次处理到索引: {last_processed_index}")
            except Exception as e:
                logging.error(f"读取断点文件失败: {str(e)}")
        
        try:
            # 读取并处理jsonl文件
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        
                        # 检查是否已处理
                        if record['index'] <= last_processed_index:
                            processed_records.append(record)
                            continue

                        # 生成音频文件名
                        audio_filename = f"{self.audio_prefix}_prompt_{record['index']}.mp3"
                        audio_path = os.path.join(output_dir, audio_filename)
                        
                        # 转换文本到语音
                        if record['prompt']:
                            self.text_to_speech(
                                text=record['prompt'],
                                output_file=audio_path,
                                language="en-US"
                            )
                            record['speech_path'] = audio_path
                        
                        processed_records.append(record)
                        
                        # 更新断点
                        with open(checkpoint_file, 'w') as f:
                            f.write(str(record['index']))
                        
                    except Exception as e:
                        error_msg = f"处理记录 {record.get('index', '未知')} 时出错: {str(e)}"
                        logging.error(error_msg)
                        print(error_msg)
                        continue
            
            # 写入新的jsonl文件
            with open(output_file, 'w', encoding='utf-8') as f:
                for record in processed_records:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
                
            # 处理完成后删除断点文件
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
            
        except Exception as e:
            error_msg = f"批处理过程发生错误: {str(e)}"
            logging.error(error_msg)
            print(error_msg)
            raise

# 使用示例
if __name__ == "__main__":
    tts_service = TTSService()
    
    # 测试中文
    tts_service.text_to_speech(
        text="你好，世界！",
        output_file="chinese_output.mp3",
        language="cmn-CN"
    )

    # 测试英文
    tts_service.text_to_speech(
        text="Hello, World!",
        output_file="english_output.mp3",
        language="en-US"
    )

    # 批量处理示例
    tts_service.batch_process_jsonl(
        input_file="path/to/input.jsonl",
        output_file="path/to/output.jsonl"
    )