import os
from google import genai
import json
import argparse

def infer_audio(audio_path):
    """
    使用Gemini2.0 Flash模型对音频进行推理
    Args:
        audio_path: 音频文件路径
    Returns:
        str: 推理结果文本
    """
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    try:
        myfile = client.files.upload(file=audio_path)
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=[
                'Answer the content of the audio',
                myfile,
            ]
        )
        return response.text
    except Exception as e:
        print(f"推理过程发生错误: {e}")
        raise e

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Gemini2.0 Flash 模型推理')
    parser.add_argument('--start_index', type=int, default=0, help='从第几行jsonl开始处理')
    args = parser.parse_args()

    input_file = '/home/xiuying.chen/qian_jiang/AudioJailbreak/convert/combined_output.jsonl'
    output_file = '/home/xiuying.chen/qian_jiang/AudioJailbreak/inference/Gemini2.0_flash_response_jsonl'
    output_jsonl = os.path.join(output_file, 'combined_output.jsonl')
    progress_file = os.path.join(output_file, 'progress.txt')

    os.makedirs(output_file, exist_ok=True)

    # 检查是否存在进度文件，如果存在则读取上次处理的行数
    start_index = args.start_index
    if os.path.exists(progress_file) and args.start_index == 0:
        with open(progress_file, 'r') as f:
            try:
                start_index = int(f.read().strip())
                print(f"从上次中断的位置继续: 第 {start_index + 1} 行")
            except ValueError:
                print("进度文件内容无效，从头开始处理。")
    else:
        print("从头开始处理。")

    with open(input_file, 'r') as f:
        lines = f.readlines()
        total_lines = len(lines)

        for i, line in enumerate(lines):
            if i < start_index:
                continue

            data = json.loads(line)
            audio_path = data['speech_path']

            try:
                response_text = infer_audio(audio_path)
                print(f"Processing {i + 1} of {total_lines}")
                print(response_text)

                data['response'] = response_text
                updated_line = json.dumps(data) + '\n'

                # 立即写入 response 到文件
                with open(output_jsonl, 'a') as outfile:
                    outfile.write(updated_line)

                # 写入进度到文件
                with open(progress_file, 'w') as pf:
                    pf.write(str(i + 1) + '\n')  # 记录已处理的行数

            except Exception as e:
                print(f"处理第 {i + 1} 行时发生错误: {e}")
                print("保存当前进度并退出。")
                exit()

        print("处理完成！")
        # 清空进度文件
        open(progress_file, 'w').close()