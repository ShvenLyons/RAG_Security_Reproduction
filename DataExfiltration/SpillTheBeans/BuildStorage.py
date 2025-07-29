import os
import pickle
import json
import subprocess
from transformers import AutoTokenizer
from rank_bm25 import BM25Okapi

def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def chunk_tokens(tokens, chunk_size=256, stride=128):
    chunks = []
    for i in range(0, len(tokens), stride):
        chunk = tokens[i:i + chunk_size]
        if len(chunk) < chunk_size:
            break
        chunks.append(chunk)
    return chunks

def build_bm25_index(text_path, tokenizer_path, output_dir):
    print(f" Loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    print(f" Reading text from {text_path}")
    text = load_text(text_path)
    tokens = tokenizer.tokenize(text)
    print(f" Tokenized into {len(tokens)} tokens")

    print(" Splitting into chunks...")
    chunks = chunk_tokens(tokens, chunk_size=256, stride=128)
    texts = [tokenizer.convert_tokens_to_string(chunk) for chunk in chunks]

    print(" Tokenizing chunks for BM25...")
    tokenized_corpus = [doc.split() for doc in texts]

    print(f" Building BM25 index for {len(tokenized_corpus)} chunks...")
    bm25 = BM25Okapi(tokenized_corpus)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "bm25_index.pkl"), "wb") as f:
        pickle.dump((bm25, texts), f)
    print(f" BM25 index saved to {output_dir}/bm25_index.pkl")

def build_pyserini_index(text_path, tokenizer_path, output_dir):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    print(f" Reading text from {text_path}")
    text = load_text(text_path)
    tokens = tokenizer.tokenize(text)
    print(f" Tokenized into {len(tokens)} tokens")

    print(" Splitting into chunks...")
    chunks = chunk_tokens(tokens, chunk_size=256, stride=128)
    texts = [tokenizer.convert_tokens_to_string(chunk) for chunk in chunks]

    json_dir = os.path.join(output_dir, "json_docs")
    os.makedirs(json_dir, exist_ok=True)

    print(f" Writing {len(texts)} JSON documents to {json_dir}")
    for i, doc in enumerate(texts):
        doc_json = {
            "id": str(i),
            "contents": doc
        }
        with open(os.path.join(json_dir, f"{i}.json"), "w", encoding='utf-8') as f:
            json.dump(doc_json, f, ensure_ascii=False)

    print(f" Building Pyserini index in {output_dir}/lucene_index")
    command = f"""python -m pyserini.index.lucene \
                     --collection JsonCollection \
                     --input {json_dir} \
                     --index {output_dir}/lucene_index \
                     --generator DefaultLuceneDocumentGenerator \
                     --storeContents \
                     --storeDocvectors \
                     --storePositions \
                     --threads 1"""

    ret_code = subprocess.run(command, shell=True)
    if ret_code.returncode != 0:
        print(" Failed to build the index")
        exit()
    else:
        print(" Successfully built the index")

if __name__ == "__main__":
    # ==== configuration ====
    dataset_configs = [
        {
            "name": "Wikipedia-News",
            "text_path": "./Data/wiki/wiki_newest.txt",
            "output_dir": "./Storage/Pyserini/WIKI"
        },
        {
            "name": "Harry Potter",
            "text_path": "./Data/HP/Harry_Potter_all_books_preprocessed.txt",
            "output_dir": "./Storage/Pyserini/HP"
        }
    ]

    tokenizer_path = "/root/autodl-tmp/MODEL/Qwen3-14B"

    for config in dataset_configs:
        print(f"\n===  Building index for: {config['name']} ===")
        # build_bm25_index(config["text_path"], tokenizer_path, config["output_dir"])
        build_pyserini_index(config["text_path"], tokenizer_path, config["output_dir"])
