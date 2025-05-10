#链接：https://github.com/ictnlp/LLaMA-Omni，具体部署见该项目README.md


#'_flash_supports_window_size' is not defined 这个错误是_flash_attention_版本的问题
#https://github.com/ictnlp/LLaMA-Omni/issues/32这个issue 里面有解决方法

#解决方法（这里需要根据自己的torch,cuda,python,cxx11abi版本选择对应的wheel文件）
# 查看cxx11abi版本
# (import torch
# print(torch._C._GLIBCXX_USE_CXX11_ABI)
# )
# ## 1. 下载 wheel 文件
# wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.1.post1/flash_attn-2.7.1.post1+cu12torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# # 2. 安装下载的 wheel 文件
# pip install flash_attn-2.7.1.post1+cu12torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

#运行命令 在LLaMA-Omni目录下
#bash omni_speech/infer/run.sh omni_speech/infer/examples /mnt/data/huggingface/transformers/models/Llama-3.1-8B-Omni

#主要改动在inference/LLaMA-Omni/omni_speech/infer/examples目录下
#运行to_json.py 将jsonl文件转换为模型需要的json文件 名字是question.json
#运行上面的命令推理 #在run.sh中可以看到读入和读出的文件名字是什么
#运行to_result.py 将answer.json文件转换为最后的jsonl文件 