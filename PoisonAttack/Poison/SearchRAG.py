import os, json, hashlib, argparse, faiss, torch, numpy as np
from typing import Dict, List
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

def build_id_map(meta_path: str) -> Dict[int, Dict[str, str]]:
    """
    把 doc_id ↔ int_id ↔ meta 建立映射:
      int_id = int(md5(doc_id)[:8], 16)
    返回: {int_id: {"id": doc_id, "title": ..., "text": ...}}
    """
    id_map = {}
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            did = obj["id"]
            int_id = int(hashlib.md5(did.encode()).hexdigest()[:8], 16)
            id_map[int_id] = obj
    return id_map

def search(index, id_map, embedder, query, k):
    """向量检索 → 返回 (doc_ids, contexts)"""
    emb = embedder.encode([query], convert_to_numpy=True).astype("float32")
    _, idx = index.search(emb, k)
    doc_ids, ctx = [], []
    for int_id in idx[0]:
        m = id_map.get(int_id)
        if m:
            doc_ids.append(m["id"])
            ctx.append(m["text"])
    return doc_ids, ctx

def answer_llm(tok, llm, query, contexts):
    prompt = (
        "Answer the following question strictly based on the provided contexts. "
        "Give a concise answer only.\n\n"
        f"Question: {query}\n\n"
        "Contexts:\n" + "\n".join(contexts)
    )
    ids = tok(prompt, return_tensors="pt").to(llm.device)
    with torch.no_grad():
        out = llm.generate(**ids, max_new_tokens=MAX_TOK_GEN)
    return tok.decode(out[0], skip_special_tokens=True).strip()

if __name__ == "__main__":
    # ========= 配 置 =========
    CLEAN_DIR = "./Storage/contriever/hotpotqa"
    POISON_DIR = "./Storage/Poison/contriever/hotpotqa"
    QUERIES_PATH = "./datasets/hotpotqa/queries.jsonl"
    OUT_JSON = "./validation_hotpotqa_clean_vs_poison.json"

    MODEL_EMB = "/root/autodl-tmp/MODEL/contriever"
    MODEL_LLM = "/root/autodl-tmp/MODEL/Qwen2.5-7B"
    TOP_K = 5
    MAX_TOK_GEN = 64

    # 1) 模型加载
    embedder = SentenceTransformer(MODEL_EMB, device="cuda")
    tok = AutoTokenizer.from_pretrained(MODEL_LLM, trust_remote_code=True)
    llm = AutoModelForCausalLM.from_pretrained(MODEL_LLM, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True).eval()

    # 2) 索引 + 元数据加载
    idx_c   = faiss.read_index(os.path.join(CLEAN_DIR,  "faiss_index.bin"))
    meta_c  = build_id_map(os.path.join(CLEAN_DIR,  "docid2meta.jsonl"))
    idx_p   = faiss.read_index(os.path.join(POISON_DIR, "faiss_index.bin"))
    meta_p  = build_id_map(os.path.join(POISON_DIR, "docid2meta.jsonl"))

    # 3) 读取 queries
    results = []
    with open(QUERIES_PATH, encoding="utf-8") as f:
        for line in tqdm(f, desc="Processing queries"):
            obj = json.loads(line)
            qid  = obj.get("_id", obj.get("id"))
            query= obj["text"]
            # clean 检索生成
            c_ids, c_ctx = search(idx_c, meta_c, embedder, query)
            c_ans = answer_llm(tok, llm, query, c_ctx)
            # poisoned 检索生成
            p_ids, p_ctx = search(idx_p, meta_p, embedder, query)
            p_ans = answer_llm(tok, llm, query, p_ctx)
            results.append({
                "query":     query,
                "c_id":      c_ids,
                "p_id":      p_ids,
                "c_context": c_ctx,
                "p_context": p_ctx,
                "c_result":  c_ans,
                "p_result":  p_ans
            })

    # 4) 保存 JSON
    with open(OUT_JSON, "w", encoding="utf-8") as fw:
        json.dump(results, fw, ensure_ascii=False, indent=2)
    print(f"\n[✓] Saved validation output → {OUT_JSON}")
