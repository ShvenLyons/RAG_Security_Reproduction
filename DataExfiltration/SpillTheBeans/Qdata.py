# ========== 代理越狱 ==========
import os
import subprocess
import openai
result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
output = result.stdout
for line in output.splitlines():
    if '=' in line:
        var, value = line.split('=', 1)
        os.environ[var] = value
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
openai.api_key = "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"
# ========== 代理越狱 ==========
from datasets import load_dataset
import json

# 1. 加载数据集
ds = load_dataset("microsoft/wiki_qa", split="train")

# 2. 提取所有问题
questions = set(example['question'] for example in ds)

# 3. 保留250个问题
top_250_questions = sorted(questions, key=len, reverse=True)[:250]

# 4. 构造询问对象
result_items = []
for q in top_250_questions:
    item = {
        "query": q,
        "context": [],
        "prompt": [],
        "input": [],
        "output": []
    }
    result_items.append(item)

# 5. 保存
with open('Result_fp16.json', 'w', encoding='utf-8') as f:
    json.dump(result_items, f, ensure_ascii=False, indent=2)

print("Write into Result_fp16.json，the number of query：", len(result_items))
