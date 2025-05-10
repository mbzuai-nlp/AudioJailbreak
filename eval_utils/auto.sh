#!/bin/bash

# 定义输入文件和输出目录的数组
input_output_pairs=(
    # "./inference/Diva_response_jsonl/BO/BO_sorted_combined_output.jsonl ./eval/Diva/BO"
    # "./inference/Diva_response_jsonl/bon/bon_sorted_combined_output.jsonl ./eval/Diva/bon"
    # "./inference/Gemini2.0_flash_response_jsonl/BO_sorted_combined_output.jsonl ./eval/Gemini2.0_flash/BO"
    # "./inference/Gemini2.0_flash_response_jsonl/bon_sorted_combined_output.jsonl ./eval/Gemini2.0_flash/bon"
    # "./inference/LLama_Omni_response_jsonl/BO/BO_wav_sorted_combined_output.jsonl ./eval/LLama_Omni/BO"
    # "./inference/LLama_Omni_response_jsonl/bon/bon_wav_sorted_combined_output.jsonl ./eval/LLama_Omni/bon"
    # "./inference/SALMONN_response_jsonl/BO/BO_wav_sorted_combined_output.jsonl ./eval/SALMONN/BO"
    # "./inference/Diva_response_jsonl/sellect_combined_output.jsonl ./eval/Diva"
    # "./inference/Gemini2.0_flash_response_jsonl/sellect_combined_output.jsonl ./eval/Gemini2.0_flash"
    #"./inference/Gemini2.0_flash_response_jsonl/new_BO/BO_sorted_combined_output.jsonl ./eval/Gemini2.0_flash/BO"
    #"./inference/LLama_Omni_response_jsonl/sellect_combined_output.jsonl ./eval/LLama_Omni"
   # "./inference/SALMONN_response_jsonl/sellect_combined_output.jsonl ./eval/SALMONN"
   #"./inference/LLama_Omni_response_jsonl/new_bon/new_bon_sorted_combined_output.jsonl ./eval/LLama_Omni/bon"
   #"./inference/LLama_Omni_response_jsonl/new_BO/new_BO_sorted_combined_output.jsonl ./eval/LLama_Omni/BO"
   #"./inference/LLama_Omni_response_jsonl/new_bon/nnew_bon_sorted_combined_output.jsonl ./eval/LLama_Omni/bon"
   "./inference/SALMONN_response_jsonl/bon/new_bon_sorted_combined_output.jsonl ./eval/SALMONN/bon"
   "./inference/SALMONN_response_jsonl/BO/new_BO_sorted_combined_output.jsonl ./eval/SALMONN/BO"
)

# 循环处理每个文件
for pair in "${input_output_pairs[@]}"; do
    # 分割输入文件和输出目录
    IFS=' ' read -r input_file output_dir <<< "$pair"

    # 运行 eval.py 脚本
    echo "Running eval.py with input: $input_file and output: $output_dir"
    python /home/xiuying.chen/qian_jiang/AudioJailbreak/eval/eval.py --input "$input_file" --output "$output_dir"
done

echo "All evaluations completed."
