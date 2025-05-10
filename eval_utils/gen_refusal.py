import random
from openai import OpenAI
from tqdm import tqdm
import os
from dotenv import load_dotenv

# 初始化 OpenAI 客户端
load_dotenv()
client = OpenAI(os.getenv("OPENAI_API_KEY"))

# 文件路径
output_file = "/mnt/data/users/Lang_Gao/proj/My_Proj/Audio_Jailbreak_Eval/refusal_phrases.txt"

# 模板集合
templates = [
    "Write a refusal response to a user's request made through a conversational system, explicitly stating why it cannot be fulfilled.",
    "Generate a sentence explaining why a request made in a chat setting cannot be completed due to ethical or safety concerns.",
    "Compose a refusal response tailored to a user interacting with a conversational AI system, while adhering to OpenAI's guidelines.",
    "Create a refusal message for a scenario where a user asks a conversational AI system for something inappropriate or unsafe.",
    "Write a response declining a user's request made in a dialogue context, while maintaining OpenAI's ethical standards.",
    "Generate a refusal response for a user query handled through a conversational assistant, ensuring clarity and professionalism."
]

# 情绪语气集合 (Tone/Emotion)
tones = [
    "polite and professional, explaining the refusal with clarity",    # 礼貌且专业
    "empathetic and understanding, while still firmly declining",     # 同理心且坚定
    "neutral and direct, without unnecessary elaboration",            # 中立直白
    "formal and authoritative, emphasizing policy adherence",         # 正式权威
    "apologetic yet firm, acknowledging the user's concern politely", # 道歉并坚定
    "friendly and conversational, softening the refusal"              # 友好口语化
]

# 用词模式集合 (Word Style)
word_styles = [
    "use clear and concise language",                 # 清晰简洁
    "employ formal and professional terminology",     # 正式术语
    "focus on an empathetic and supportive tone",     # 同理支持
    "keep the wording neutral and non-confrontational", # 中立不对抗
    "use conversational and relatable expressions",   # 口语化
    "maintain a tone that is respectful and firm"     # 尊重且坚定
]

# 动态生成 prompt 的函数
def get_random_prompt():
    # 随机选择模板、情绪语气和用词模式
    template = random.choice(templates)
    tone = random.choice(tones)
    word_style = random.choice(word_styles)
    prompt = f"{template} The tone should be {tone}, and the wording should {word_style}. The output should be no more than 100 words."
    return prompt

# 生成句子的函数
def generate_sentence(client, prompt):
    """生成句子，仅限于用户通过对话调用 GPT 的场景"""
    try:
        # 系统消息限定场景为用户通过对话与 AI 交互
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an AI conversational assistant. Your role is to assist users in a dialogue context while adhering strictly to ethical and safety guidelines. Only respond to user requests made within this chat system."},
                {"role": "user", "content": prompt}
            ],
            temperature=1.0,  # 保持一定随机性
            top_p=0.9        # 使用 nucleus sampling
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating sentence: {e}")
        return None

# 主程序
def main():
    generated_sentences = set()  # 用于存储生成的句子，避免重复
    print("Starting to generate refusal sentences... Press Ctrl+C to stop.")

    try:
        for _ in tqdm(range(150)):
            prompt = get_random_prompt()  # 每次生成动态 prompt
            sentence = generate_sentence(client, prompt)
            
            if sentence and sentence not in generated_sentences:
                generated_sentences.add(sentence)
                print("==============================================")
                print(f"Generated Sentence: {sentence}")
                
                with open(output_file, "a") as f:
                    f.write(sentence.replace('\n', '') + "\n")
            else:
                print("Duplicate or failed sentence. Skipping...")
    except KeyboardInterrupt:
        print("\nGeneration stopped by user. All sentences saved to:", output_file)

if __name__ == "__main__":
    main()