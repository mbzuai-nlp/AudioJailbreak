#!/bin/bash

# 定义输入文件和输出目录的数组
input_output_pairs=(
    "./inference/Diva_response_jsonl/new_bon/bon_sorted_combined_output.jsonl ./eval/Diva/bon"
    "./inference/Gemini2.0_flash_response_jsonl/new_bon/bon_sorted_combined_output.jsonl ./eval/Gemini2.0_flash/bon"
    "./inference/gpt4o_response_jsonl/new_bon/bon_sorted_combined_output.jsonl ./eval/gpt4o/bon"
    "./inference/LLama_Omni_response_jsonl/new_bon/new_bon_sorted_combined_output.jsonl ./eval/LLama_Omni/bon" #!
    "./inference/qwen2_response_jsonl/new_bon/bon_sorted_combined_output.jsonl ./eval/qwen2/bon"
    "./inference/SALMONN_response_jsonl/bon/new_bon_sorted_combined_output.jsonl ./eval/SALMONN/bon" #!

    "./inference/Diva_response_jsonl/new_BO/BO_sorted_combined_output.jsonl ./eval/Diva/BO"
    "./inference/Gemini2.0_flash_response_jsonl/new_BO/BO_sorted_combined_output.jsonl ./eval/Gemini2.0_flash/BO"
    "./inference/gpt4o_response_jsonl/new_BO/BO_sorted_combined_output.jsonl ./eval/gpt4o/BO"
    "./inference/LLama_Omni_response_jsonl/new_BO/new_BO_sorted_combined_output.jsonl ./eval/LLama_Omni/BO" #!
    "./inference/qwen2_response_jsonl/new_BO/BO_sorted_combined_output.jsonl ./eval/qwen2/BO"
    "./inference/SALMONN_response_jsonl/BO/new_BO_sorted_combined_output.jsonl ./eval/SALMONN/BO"  #!
)

# 循环处理每个文件
for pair in "${input_output_pairs[@]}"; do
    # 分割输入文件和输出目录
    IFS=' ' read -r input_file output_dir <<< "$pair"

    # 运行 pv_eval.py 脚本
    echo "Running pv_eval.py with input: $input_file and output: $output_dir"
    python /home/xiuying.chen/qian_jiang/AudioJailbreak/eval/pv_eval.py --input "$input_file" --output "$output_dir"
done

echo "All policy violation evaluations completed."
