import whisper
import json
import jsonlines

def transcribe_audio_files():
    # 加载 Whisper 模型
    model = whisper.load_model("base")
    
    # 读取 jsonl 文件
    with jsonlines.open('data.jsonl', 'r') as reader:
        # 将所有数据读入列表以便稍后更新
        data = list(reader)
    
    # 处理每条记录
    for item in data:
        if 'response_audio_path' in item:
            # 转录音频文件
            result = model.transcribe(item['response_audio_path'])
            # 将转录文本添加到数据中
            item['transcribed_text'] = result["text"]
    
    # 将更新后的数据写回 jsonl 文件
    with jsonlines.open('data.jsonl', 'w') as writer:
        writer.write_all(data)

if __name__ == "__main__":
    transcribe_audio_files()