import base64
import os
import json
import shutil
from openai import OpenAI
from dotenv import load_dotenv
import datetime

def init_openai_client():
    """初始化OpenAI客户端"""
    #load_dotenv()
    load_dotenv("/home/xiuying.chen/qian_jiang/AudioJailbreak/.env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("警告：无法从.env文件获取OPENAI_API_KEY，尝试从系统环境变量获取")
    
    if api_key:
        print(f"API密钥已加载，前10个字符: {api_key[:30]}...")
        return OpenAI(api_key=api_key)
    else:
        print("错误：无法获取OPENAI_API_KEY，请检查环境变量或.env文件")
        exit(1)

def infer_audio(audio_path: str) -> str:
    """
    使用GPT-4对音频进行推理
    Args:
        audio_path: 音频文件路径
    Returns:
        str: 模型的回答文本
    """
    # 初始化客户端
    client = init_openai_client()
    
    # 读取音频文件
    with open(audio_path, "rb") as audio_file:
        wav_data = audio_file.read()
    encoded_string = base64.b64encode(wav_data).decode('utf-8')
    
    # 调用GPT-4o API
    completion = client.chat.completions.create(
        model="gpt-4o-mini-audio-preview",
        modalities=["text"],
        messages=[
            {
                "role": "user", 
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": encoded_string,
                            "format": "mp3"
                        }
                    }
                ]
            }
        ]
    )
    
    return completion.choices[0].message.content

def save_response_data(response, output_dir, input_audio_path):
    """
    保存GPT-4响应的音频和文本数据
    Args:
        response: GPT-4返回的响应对象
        output_dir: 输出目录路径
        input_audio_path: 输入音频文件路径
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        base_filename = os.path.splitext(os.path.basename(input_audio_path))[0]
        
        if response.audio and response.audio.transcript:
            return None, None, response.audio.transcript
        else:
            print("响应中没有文本数据")
            return None, None, None
            
    except Exception as e:
        print(f"处理响应数据时出错: {str(e)}")
        return None, None, None

if __name__ == "__main__":
    # 设置输入和输出路径
    input_jsonl = "/home/xiuying.chen/qian_jiang/AudioJailbreak/convert/combined_output.jsonl"
    output_dir = "/home/xiuying.chen/qian_jiang/AudioJailbreak/inference/gpt4o_response_jsonl"
    output_jsonl = os.path.join(output_dir, "combined_output.jsonl")
    audio_output_dir = os.path.join(output_dir, "response_audio")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(audio_output_dir, exist_ok=True)

    # 复制输入文件到输出目录
    if not os.path.exists(output_jsonl):
        shutil.copy2(input_jsonl, output_jsonl)
        print(f"已复制 {input_jsonl} 到 {output_jsonl}")

    # 读取jsonl文件
    data = []
    with open(output_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    # 限制处理文件数量
    max_files = 1000
    data = data[:max_files]

    # 指定从第几行开始处理
    start_index = 610

    # 处理每个样本
    for i, item in enumerate(data[start_index:], start=start_index):
        speech_path = item.get('speech_path')
        print(f"处理文件 {i+1}/{len(data)}: {speech_path}")
        
        try:
            # 调用推理函数
            transcript = infer_audio(speech_path)
            
            # 更新jsonl中的响应
            print(transcript)
            item['response'] = transcript
            
            # 每处理完一个文件就保存一次
            with open(output_jsonl, 'w', encoding='utf-8') as f:
                for d in data:
                    f.write(json.dumps(d, ensure_ascii=False) + '\n')
            
            print(f"文件 {speech_path} 处理完成！")
            
        except Exception as e:
            print(f"处理文件 {speech_path} 时出错: {str(e)}")
            item['response'] = f"Error: {str(e)}"
            
            # 保存当前进度
            with open(output_jsonl, 'w', encoding='utf-8') as f:
                for d in data:
                    f.write(json.dumps(d, ensure_ascii=False) + '\n')

    print("所有文件处理完成！")
    print("所有文件处理完成！")