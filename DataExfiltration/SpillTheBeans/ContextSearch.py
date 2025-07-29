import json
import pickle
from transformers import AutoTokenizer
from pyserini.search.lucene import LuceneSearcher

def load_bm25_index(index_path):
    with open(index_path, 'rb') as f:
        bm25, docs = pickle.load(f)
    return bm25, docs

def load_result_json(path="./Result_fp16.json"):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_result_json(data, path="./Result_fp16.json"):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def update_context_with_bm25(result_data, bm25, docs, tokenizer, k=1):
    for item in result_data:
        query = item.get("query", "")
        if not query:
            continue
        # 使用BM25建库一致的 tokenize
        tokenized_query = tokenizer.tokenize(query)
        scores = bm25.get_scores(tokenized_query)
        top_k_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        item["context"] = [docs[i] for i in top_k_idx]
    return result_data

def update_context_with_pyserini(result_data, index_dir, tokenizer, k=1):
    """
    使用 Pyserini LuceneSearcher 进行检索，
    用 top-k 检索结果填入 context 字段
    """
    searcher = LuceneSearcher(index_dir)
    for item in result_data:
        query = item.get("query", "")
        if not query:
            continue
        tokenized_query = tokenizer.tokenize(query)
        query_str = " ".join(tokenized_query)
        hits = searcher.search(query_str, k)
        item["context"] = [hit.lucene_document.get("contents") for hit in hits]

    return result_data

if __name__ == "__main__":
    # ===== configuration =====
    bm25_index_path = "./Storage/BM/WIKI/bm25_index.pkl"
    pyserini_index_path = "./Storage/Pyserini/WIKI/lucene_index"
    result_json_path = "Result_fp16.json"
    tokenizer_path = "/root/autodl-tmp/MODEL/Qwen3-14B"
    top_k = 1

    # ===== searching =====
    print(" Loading BM25 Index and Tokenizer...")
    bm25, docs = load_bm25_index(bm25_index_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    print(" Loading Result_fp16.json and Searching Context...")
    result_data = load_result_json(result_json_path)
    updated_result = update_context_with_bm25(result_data, bm25, docs, tokenizer, k=top_k)
    # updated_result = update_context_with_pyserini(result_data, pyserini_index_path, tokenizer, k=top_k)
    save_result_json(updated_result, result_json_path)

    print(f" Finished. Each Query has {top_k}  chunk(s) as context")
