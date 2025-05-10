#链接：https://github.com/0nutation/SpeechGPT，具体部署见该项目README.md


#./inference/SpeechGPT/speechgpt/src/infer/cli_infer.py 用这个
#inference/SpeechGPT/speechgpt/src/infer/cli_infer.py定义了相对路径

#运行

# #python3 speechgpt/src/infer/cli_infer.py \
# --model-name-or-path "path/to/SpeechGPT-7B-cm" \
# --lora-weights "path/to/SpeechGPT-7B-com" \
# --s2u-dir "${s2u_dir}" \
# --vocoder-dir "${vocoder_dir} \
# --output-dir "output" 

#SpeechGPT/speechgpt/src/infer/cli_infer.py修改后的代码如下

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import torch
import torch.nn as nn
from fairseq.models.text_to_speech.vocoder import CodeHiFiGANVocoder
import soundfile as sf
from typing import List
import argparse
import logging
import json
from tqdm import tqdm
import re
import traceback
from peft import PeftModel
from speechgpt.utils.speech2unit.speech2unit import Speech2Unit
import transformers
from transformers import AutoConfig, LlamaForCausalLM, LlamaTokenizer, GenerationConfig


logging.basicConfig()
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



NAME="SpeechGPT"
META_INSTRUCTION="You are an AI assistant whose name is SpeechGPT.\n- SpeechGPT is a intrinsic cross-modal conversational language model that is developed by Fudan University.  SpeechGPT can understand and communicate fluently with human through speech or text chosen by the user.\n- It can perceive cross-modal inputs and generate cross-modal outputs.\n"
DEFAULT_GEN_PARAMS = {
        "max_new_tokens": 1024,
        "min_new_tokens": 10,
        "temperature": 0.8,
        "do_sample": True, 
        "top_k": 60,
        "top_p": 0.8,
        }  
device = torch.device('cuda')


def extract_text_between_tags(text, tag1='[SpeechGPT] :', tag2='<eoa>'):
    pattern = f'{re.escape(tag1)}(.*?){re.escape(tag2)}'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        response = match.group(1)
    else:
        response = ""
    return response



class SpeechGPTInference:
    def __init__(
        self, 
        model_name_or_path: str,
        lora_weights: str=None,
        s2u_dir: str="speechgpt/utils/speech2unit/",
        vocoder_dir: str="speechgpt/utils/vocoder/", 
        output_dir="speechgpt/output/"
        ):
        
        # 检查CUDA是否可用
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This program requires GPU to run.")
        
        self.meta_instruction = META_INSTRUCTION
        self.template= "[Human]: {question} <eoh>. [SpeechGPT]: "


        #speech2unit
        self.s2u = Speech2Unit(ckpt_dir=s2u_dir)
        
        #model
        self.model = LlamaForCausalLM.from_pretrained(
            model_name_or_path,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map="auto",
            )

        if lora_weights is not None:
            self.model = PeftModel.from_pretrained(
                self.model,
                lora_weights,
                torch_dtype=torch.float16,
                device_map="auto",
            )

        self.model.half()  

        self.model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)

        #tokenizer
        self.tokenizer = LlamaTokenizer.from_pretrained(
            model_name_or_path)
        self.tokenizer.pad_token_id = (0)
        self.tokenizer.padding_side = "left" 


        #generation
        self.generate_kwargs = DEFAULT_GEN_PARAMS


        #vocoder
        vocoder = os.path.join(vocoder_dir, "vocoder.pt")
        vocoder_cfg = os.path.join(vocoder_dir, "config.json")
        with open(vocoder_cfg) as f:
            vocoder_cfg = json.load(f)
        self.vocoder = CodeHiFiGANVocoder(vocoder, vocoder_cfg).to(device)

        self.output_dir = output_dir


    def preprocess(
        self,
        raw_text: str,
    ):
        processed_parts = []
        for part in raw_text.split("is input:"):
            if os.path.isfile(part.strip()) and os.path.splitext(part.strip())[-1] in [".wav", ".flac", ".mp4"]:
                processed_parts.append(self.s2u(part.strip(), merged=True))
            else:
                processed_parts.append(part)
        processed_text = "is input:".join(processed_parts)

        prompt_seq = self.meta_instruction + self.template.format(question=processed_text)
        return prompt_seq


    def postprocess(
        self,
        response: str,
    ):

        question = extract_text_between_tags(response, tag1="[Human]", tag2="<eoh>")
        answer = extract_text_between_tags(response + '<eoa>', tag1=f"[SpeechGPT] :", tag2="<eoa>")
        tq = extract_text_between_tags(response, tag1="[SpeechGPT] :", tag2="; [ta]") if "[ta]" in response else ''
        ta = extract_text_between_tags(response, tag1="[ta]", tag2="; [ua]") if "[ta]" in response else ''
        ua = extract_text_between_tags(response + '<eoa>', tag1="[ua]", tag2="<eoa>") if "[ua]" in response else ''

        return {"question":question, "answer":answer, "textQuestion":tq, "textAnswer":ta, "unitAnswer":ua}


    def forward(
        self, 
        prompts: List[str]
    ):
        with torch.no_grad():
            #preprocess
            preprocessed_prompts = []
            for prompt in prompts:
                preprocessed_prompts.append(self.preprocess(prompt))

            input_ids = self.tokenizer(preprocessed_prompts, return_tensors="pt", padding=True, 
                                     truncation=True, max_length=512).input_ids
            for input_id in input_ids:
                if input_id[-1] == 2:
                    input_id = input_id[:, :-1]

            input_ids = input_ids.to(device)

            #generate
            generation_config = GenerationConfig(
                temperature=0.7,
                top_p=0.8,
                top_k=50,
                do_sample=True,
                max_new_tokens=512,
                min_new_tokens=10,
                )

            generated_ids = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
            )
            generated_ids = generated_ids.sequences
            responses = self.tokenizer.batch_decode(generated_ids.cpu(), skip_special_tokens=True)

            #postprocess
            responses = [self.postprocess(x) for x in responses]

            # 只返回textAnswer部分
            text_responses = []
            for r in responses:
                if isinstance(r, dict):  # 确保r是字典
                    if r.get("textAnswer"):
                        text_responses.append(r["textAnswer"])
                    elif r.get("answer"):
                        text_responses.append(r["answer"])
                else:
                    text_responses.append(str(r))  # 如果不是字典，直接转为字符串
            
            return text_responses[0] if len(text_responses) == 1 else text_responses

    def process_jsonl(self, input_jsonl, output_jsonl):
        """处理jsonl文件中的音频文件并保存响应"""
        # 清空输出文件
        #open(output_jsonl, 'w').close()
        
        # 先计算总行数
        with open(input_jsonl, 'r') as f:
            total_lines = sum(1 for _ in f)
        
        # 逐行读取和处理
        with open(input_jsonl, 'r') as f:
            for i, line in enumerate(f, 1):
                print(f"Processing file {i}/{total_lines}...")
                data = json.loads(line)
                
                # 处理每一行
                audio_path = data.get('speech_path')
                audio_path = '../.' + audio_path
                if audio_path and os.path.exists(audio_path):
                    try:
                        print(f"Processing audio: {audio_path}")
                        # 处理音频文件
                        prompt = f"this is input:{audio_path}"
                        response = self.forward([prompt])
                        
                        # 更新response字段
                        if isinstance(response, str):
                            data['response'] = response
                        elif isinstance(response, list):
                            data['response'] = response[0] if response else ""
                        print("Response generated successfully")
                        print(f"Generated response: {data['response'][:100]}...")  # 打印部分响应内容
                        
                    except Exception as e:
                        logger.error(f"Error processing {audio_path}: {str(e)}")
                        data['response'] = f"Error: {str(e)}"
                
                # 保存结果
                with open(output_jsonl, 'a') as f_out:
                    f_out.write(json.dumps(data) + '\n')
                    f_out.flush()

    def dump_wav(self, sample_id, pred_wav, prefix):
        sf.write(
            f"{self.output_dir}/wav/{prefix}_{sample_id}.wav",
            pred_wav.detach().cpu().numpy(),
            16000,
        )
        
    def __call__(self, input):
        return self.forward(input)

    
    def interact(self):
        prompt = str(input(f"Please talk with {NAME}:\n"))
        while prompt != "quit":
            try:
                self.forward([prompt])
            except Exception as e:
                traceback.print_exc()
                print(e)

            prompt = str(input(f"Please input prompts for {NAME}:\n"))
            

            
if __name__=='__main__':
    model_name_or_path = "/mnt/data/qianjiang/SpeechGPT-7B-cm"
    lora_weights = "/mnt/data/qianjiang/SpeechGPT-7B-com"
    s2u_dir = "/mnt/data/qianjiang"
    vocoder_dir = "/home/xiuying.chen/qian_jiang/AudioJailbreak/inference/SpeechGPT/speechgpt/utils/vocoder"
    output_dir = "/home/xiuying.chen/qian_jiang/AudioJailbreak/inference/SpeechGPT_response_jsonl"
    mode = "batch"
    
    os.makedirs(output_dir, exist_ok=True)

    infer = SpeechGPTInference(
        model_name_or_path,
        lora_weights,
        s2u_dir,
        vocoder_dir,
        output_dir
    )

    if mode == 'interact':
        infer.interact()
    else:
        # 处理jsonl文件
        input_jsonl = "/home/xiuying.chen/qian_jiang/AudioJailbreak/convert/wav_combined_output.jsonl"
        output_jsonl = os.path.join(output_dir, "wav_combined_output.jsonl")
        
        # 复制输入文件到输出目录
        import shutil
        shutil.copy2(input_jsonl, output_jsonl)
        
        # 处理文件
        infer.process_jsonl(input_jsonl, output_jsonl)








