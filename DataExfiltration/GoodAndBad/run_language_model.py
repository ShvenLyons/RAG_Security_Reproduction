import warnings
import json
import os
import fire
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

"""
This file is to run large language model using HuggingFace local models ONLY.
The running instructions have been generated in file f'{experiment name}.sh'
Please run the following command:
nohup bash {experiment name}.sh > output_name.out
or: bash {experiment name}.sh
"""

def main(
        ckpt_dir: str = r"H:\Models\Llama-3.2-1B",   # 默认模型路径
        path: str = None,           # input and output place
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_seq_len: int = 4096,
        max_gen_len: int = 256,
        max_batch_size: int = 1,
):
    """
    Entry point of the program for generating text using a pretrained HuggingFace model
    :param
        ckpt_dir: 本地 HuggingFace 模型路径（如Llama或Qwen）
        path: 输入输出目录
        temperature, top_p, max_seq_len, max_gen_len, max_batch_size: 生成参数
    """
    print(f"Using model: {ckpt_dir}")
    print(f"Prompt dir: {path}")

    # 1. 加载模型和tokenizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(ckpt_dir, trust_remote_code=True)
    model = model.eval().to(device)

    # 2. 读取prompts
    with open(f"./Inputs&Outputs/{path}/prompts.json", 'r', encoding='utf-8') as f:
        all_prompts = json.loads(f.read())

    # 3. 批量生成
    answer = []
    for i in range(len(all_prompts)):
        print("\nThe number of ask:")
        print(str(i))
        prompt = all_prompts[i]
        # tokenize and truncate
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_seq_len)
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)

        # 推理
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_gen_len,
                temperature=temperature,
                top_p=top_p,
                do_sample=True
            )
        # 解码输出，去掉输入prompt部分，只保留生成内容
        output_text = tokenizer.decode(output[0], skip_special_tokens=True)
        # 如需仅输出新增部分，可裁剪
        # output_text = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)
        answer.append(output_text)

    # 4. 保存
    save_path = f"./Inputs&Outputs/{path}/outputs-{os.path.basename(ckpt_dir)}-{temperature}-{top_p}-{max_seq_len}-{max_gen_len}.json"
    with open(save_path, 'w', encoding='utf-8') as file:
        file.write(json.dumps(answer, ensure_ascii=False, indent=2))
    print(f"All results saved to {save_path}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    fire.Fire(main)
