#!/usr/bin/env bash

python Attack.py \
  --model_path /root/autodl-tmp/MODEL/Qwen2.5-7B \
  --result_path Result_qwen25_7B.json

python Attack.py \
  --model_path /root/autodl-tmp/MODEL/Qwen2.5-3B \
  --result_path Result_qwen25_3B.json

python Attack.py \
  --model_path /root/autodl-tmp/MODEL/Llama-2-7b-chat-hf \
  --result_path Result_llama2_7B.json
  
python Attack.py \
  --model_path /root/autodl-tmp/MODEL/Qwen1.5-1.8B-Chat\
  --result_path Result_qwen15_1_8B.json
  
python Attack.py \
  --model_path /root/autodl-tmp/MODEL/Qwen2.5-1.5B \
  --result_path Result_qwen25_1_5B.json
