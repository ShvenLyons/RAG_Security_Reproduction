#  Follow My Instruction and Spill the Beans

**论文信息**  
Qi, Z., Zhang, H., Xing, E., Kakade, S., & Lakkaraju, H. (2024).  
*Follow my instruction and spill the beans: Scalable data extraction from retrieval-augmented generation systems.*  
arXiv preprint [arXiv:2402.17840](https://arxiv.org/abs/2402.17840)

---

##  实验配置

| 项目         | 内容                                                                 |
|--------------|----------------------------------------------------------------------|
| 数据集        | Wikipedia-News / Harry Potter                                        |
| 切分方式       | 256 token / 128 token overlap                                        |
| 索引方式       | Top-1（k = 1）                                                       |
| 数据库        | Lucene-BM25                                                          |
| 攻击对象       | RICLM 的输入文本 / GPT 的上下文                                     |
| Query 生成方式 | - WikiQA 数据集中挑选的 250 条开放域问答问题  <br> - GPT 生成 100 条覆盖各章节问题，再迭代抽取 |

---

###  评估指标说明

| 指标类别       | 代码名称                                      | 说明                                                                 |
|----------------|-----------------------------------------------|----------------------------------------------------------------------|
| Token 集合指标 | token_set_precision / token_set_recall / token_set_f1 | 将预测与参考句去重为词集合，交集为 TP，参考独有为 FP，预测独有为 FN，计算 P/R/F1 |
|                | token_set_f1_sem                              | 上述 F1 的标准误（Standard Error of Mean）                          |
| n-gram 重叠计数| n_ngrams_match_1 / 2 / 3                       | 逐句统计 1-gram / 2-gram / 3-gram 最小重叠次数                      |
|                | num_true_words / num_pred_words               | 参考与预测文本的平均长度                                            |
| 生成质量指标   | bleu_score (+ _sem)                           | 使用 SACRE-BLEU，逐句计算后取平均及标准误                           |
|                | rougeL_score (+ _sem)                         | 只计算 per-sentence 的 ROUGE-L 分数后求均值和标准误                 |
|                | bert_score (+ _sem)                           | BERTScore-F1，评估语义相似度                                        |
|                | exact_match (+ _sem)                          | 完全匹配比例（逐句完全一致的占比）及其标准误                        |

###  复现结果

| 指标类别   | Qwen3-14B | Qwen3-14B | Qwen3-8B | Qwen3-8B | Qwen3-4B | Qwen3-4B | Qwen2.5-7B | Qwen2.5-7B |
|------------|------------------|------------------|------------------|------------------|
|            | Output-1 | Output-2 | Output-1 | Output-2 | Output-1 | Output-2 | Output-1 | Output-2 |
| ROUGE-L    | 0.4752   | 0.4869   | 0.4518   | 0.4584   | 0.4725   | 0.4620   | 0.7489   | 0.8221   |
| BLEU       | 0.3204   | 0.3282   | 0.3014   | 0.3078   | 0.3109   | 0.3066   | 0.5896   | 0.7002   |
| F1         | 0.4072 ± 0.0070 | 0.4210 ± 0.0081 | 0.3876 ± 0.0056 | 0.3955 ± 0.0064 | 0.4258 ± 0.0071 | 0.4068 ± 0.0067 | 0.7100 ± 0.0087 | 0.8099 ± 0.0098 |
| BERTScore  | 0.9046 ± 0.0007 | 0.9161 ± 0.0007 | 0.9056 ± 0.0007 | 0.9164 ± 0.0007 | 0.9037 ± 0.0007 | 0.9165 ± 0.0007 | 0.9273 ± 0.0011 | 0.9544 ± 0.0013 |
