#!/bin/bash
python run_language_model.py --ckpt_dir "H:\Models\Llama-3.2-1B" --temperature 0.6 --top_p 0.9 --max_seq_len 4096 --max_gen_len 256 --path "Health-rag"
