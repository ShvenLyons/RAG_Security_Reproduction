import os, json, hashlib, argparse, faiss, torch, numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

def build_id_map(meta_path: str) -> Dict[int, Dict[str, str]]:
    id_map = {}
    with open(meta_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            obj = json.loads(line)
            id_map[line_no] = obj        
    return id_map

def faiss_search(index: faiss.Index, id_map: Dict[int, Dict[str, str]],
                 embedder: SentenceTransformer, query: str, k: int) -> Tuple[List[str], List[str]]:
    """向量检索，最多 k 条可用结果"""
    emb = embedder.encode([query], convert_to_numpy=True).astype("float32")
    _, idx = index.search(emb, k)
    doc_ids, contexts = [], []
    for int_id in idx[0]:
        meta = id_map.get(int_id)
        if meta:                             
            doc_ids.append(meta["id"])
            contexts.append(meta["text"])
    return doc_ids, contexts

def answer_llm(tok: AutoTokenizer, llm: AutoModelForCausalLM,
               query: str, contexts: List[str], max_toks: int) -> str:
    """ Prompt 拼接 → 生成 concise answer"""
    prompt = (
        "Contexts (use them strictly):\n"
        + "\n".join(contexts)
        + "\n\n"
        "Now answer the following question concisely based ONLY on the contexts above.\n"
        f"Question: {query}\n"
        "Answer:"                  
    )
    inputs = tok(prompt, return_tensors="pt").to(llm.device)
    prompt_len = inputs["input_ids"].shape[-1] 
    with torch.no_grad():
        out_ids = llm.generate(
            **inputs,
            max_new_tokens=max_toks,
            pad_token_id=tok.eos_token_id or tok.pad_token_id
        )[0]
    gen_ids = out_ids[prompt_len:]
    return tok.decode(gen_ids, skip_special_tokens=True).strip()

def main():
    parser = argparse.ArgumentParser(
        description="Validate RAG on clean vs poison indices"
    )
    parser.add_argument("--dataset", default="hotpotqa", help="BEIR dataset name (also used in paths)")
    parser.add_argument("--adv_json", default=None, help="Path to adv_targeted_results/<dataset>.json (auto-fill if not provided)")
    parser.add_argument("--clean_root", default="./Storage/contriever", help="Root dir that contains clean index/meta")
    parser.add_argument("--poison_root", default="./Storage/Poison/contriever", help="Root dir that contains poisoned index/meta")
    parser.add_argument("--model_emb", default="/root/autodl-tmp/MODEL/contriever", help="SentenceTransformer embedding model path")
    parser.add_argument("--model_llm", default="/root/autodl-tmp/MODEL/Llama-2-7b-chat-hf", help="Causal-LM path for answer generation")
    parser.add_argument("--top_k", type=int, default=5, help="# contexts retrieved")
    parser.add_argument("--max_tokens", type=int, default=32, help="max_new_tokens for generation")
    parser.add_argument("--output", default=None, help="Output JSON path (auto-fill if not provided)")
    args = parser.parse_args()

    DATASET     = args.dataset
    ADV_JSON    = args.adv_json or f"adv_targeted_results/{DATASET}.json"
    CLEAN_DIR   = os.path.join(args.clean_root, DATASET)
    POISON_DIR  = os.path.join(args.poison_root, DATASET)
    OUT_JSON    = args.output or f"validation_{DATASET}.json"

    assert os.path.exists(ADV_JSON),          f"Missing {ADV_JSON}"
    assert os.path.isdir(CLEAN_DIR),          f"Missing clean dir {CLEAN_DIR}"
    assert os.path.isdir(POISON_DIR),         f"Missing poison dir {POISON_DIR}"

    # 1) 模型加载
    print(" Loading models …")
    embedder = SentenceTransformer(args.model_emb, device="cuda")
    tok      = AutoTokenizer.from_pretrained(args.model_llm, trust_remote_code=True)
    llm      = AutoModelForCausalLM.from_pretrained(
        args.model_llm, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True
    ).eval()

    # 2) 索引 + 元数据
    print(" Loading FAISS indices …")
    idx_clean  = faiss.read_index(os.path.join(CLEAN_DIR, "faiss_index.bin"))
    meta_clean = build_id_map(os.path.join(CLEAN_DIR, "docid2meta.jsonl"))
    idx_poison  = faiss.read_index(os.path.join(POISON_DIR, "faiss_index.bin"))
    meta_poison = build_id_map(os.path.join(POISON_DIR, "docid2meta.jsonl"))

    # 3) 读取 adversarial queries
    adv_data: Dict[str, Dict] = json.load(open(ADV_JSON, encoding="utf-8"))

    results = []
    for qid, item in tqdm(adv_data.items(), desc="Processing queries"):
        query = item["question"]

        # clean
        c_ids, c_ctx = faiss_search(idx_clean, meta_clean, embedder,
                                    query, args.top_k)
        c_ans = answer_llm(tok, llm, query, c_ctx, args.max_tokens)

        # poison
        p_ids, p_ctx = faiss_search(idx_poison, meta_poison, embedder,
                                    query, args.top_k)
        p_ans = answer_llm(tok, llm, query, p_ctx, args.max_tokens)

        results.append({
            "id":              item["id"],
            "question":        query,
            "correct answer":  item["correct answer"],
            "incorrect answer":item["incorrect answer"],
            "clean RAG output":c_ans,
            "poison RAG output":p_ans,
            "poison content":   p_ctx    
        })

    # 4) 保存
    os.makedirs(os.path.dirname(OUT_JSON) or ".", exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as fw:
        json.dump(results, fw, ensure_ascii=False, indent=2)
    print(f"\n Saved validation output → {OUT_JSON}")


if __name__ == "__main__":
    main()
