"""
构建 BEIR HotpotQA 数据集的持久化向量索引（FAISS FlatIP）。
"""
"""
    python build_persistent_index.py \
        --data_path ./datasets/hotpotqa \
        --model_path /root/autodl-tmp/MODEL/contriever \
        --output_dir ./Storage/FAISS/HotpotQA
"""

import os
import json
import pathlib
import argparse
from typing import Dict
import faiss
import numpy as np
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer

# 数据加载
def load_data(data_path: str) -> Dict[str, Dict[str, str]]:
    """
    data_path: str
        包含 corpus.jsonl + queries.jsonl + qrels/* 的目录路径。
    corpus: dict
        {doc_id: {"title": str, "text": str}}
    """
    corpus_file = pathlib.Path(data_path) / "corpus.jsonl"
    if not corpus_file.exists():
        raise FileNotFoundError(f"Cannot find {corpus_file}")

    corpus: Dict[str, Dict[str, str]] = {}
    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            doc_id = obj.get("_id", obj.get("id"))
            corpus[doc_id] = {
                "title": obj.get("title", ""),
                "text": obj["text"],
            }
    return corpus
# Chunk 切分
def chunk_tokens(text: str, tokenizer, max_tokens: int = 512, stride: int = 0):
    """BEIR 语料已足够短，不做切分。"""
    return [text]
# 向量化 + FAISS 数据库保存
def build_index(data_path: str, model_path: str, output_dir: str, *, batch_size: int = 256):
    """建立持久 FAISS 索引
    data_path : str
        路径下需含 corpus.jsonl
    model_path : str
        SentenceTransformer/Contriever 模型权重目录 / HuggingFace 名称
    output_dir : str
        输出目录，生成 `faiss_index.bin` 与 `docid2meta.jsonl`。
    batch_size : int, optional
        编码批大小，默认为 256。
    """
    os.makedirs(output_dir, exist_ok=True)
    # 1) 读 corpus
    corpus = load_data(data_path)
    doc_ids = list(corpus.keys())
    print(f"Loaded {len(doc_ids)} documents from {data_path}")
    # 2) 初始化编码器
    model = SentenceTransformer(model_path)
    device = "cuda"
    print(f"Encoding on device: {device}")
    # 3) 批量向量化
    vectors = []
    for start in tqdm(range(0, len(doc_ids), batch_size), desc="Encoding"):
        batch_ids = doc_ids[start:start + batch_size]
        batch_texts = [corpus[pid]["text"] for pid in batch_ids]
        emb = model.encode(
            batch_texts,
            convert_to_numpy=True,
            batch_size=batch_size,
            normalize_embeddings=False,
            show_progress_bar=False,
            dtype=torch.float16
        ).astype("float32")
        vectors.append(emb)
    matrix = np.vstack(vectors)
    dim = matrix.shape[1]
    # 4) 构建 FAISS FlatIP 索引
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)
    # 5) 保存数据库
    faiss.write_index(index, f"{output_dir}/faiss_index.bin")
    with open(f"{output_dir}/docid2meta.jsonl", "w", encoding="utf-8") as fw:
        for pid in doc_ids:
            meta = {"id": pid, **corpus[pid]}
            fw.write(json.dumps(meta, ensure_ascii=False) + "\n")
    print(f"[✓] FAISS index + metadata written to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./poison/hotpotqa", help="Path to BEIR dataset folder (contains corpus.jsonl)")
    parser.add_argument("--model_path", default="/root/autodl-tmp/MODEL/contriever", help="SentenceTransformer model path or name")
    parser.add_argument("--output_dir", default="./Storage/Poison/contriever/hotpotqa", help="Directory to write FAISS index & metadata")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    build_index(args.data_path, args.model_path, args.output_dir, batch_size=args.batch_size)
