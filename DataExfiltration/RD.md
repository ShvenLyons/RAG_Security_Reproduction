#  Follow My Instruction and Spill the Beans

## ./SpillTheBeans

**论文信息**  
Qi, Z., Zhang, H., Xing, E., Kakade, S., & Lakkaraju, H. (2024).  
*Follow my instruction and spill the beans: Scalable data extraction from retrieval-augmented generation systems.*  
arXiv preprint [arXiv:2402.17840](https://arxiv.org/abs/2402.17840)

---

##  实验配置

| 项目         | 数据集1          |           数据集2             |
|--------------|--------------------------|----------------------|
| 数据集         | Wikipedia-News          | Harry Potter        |
| 切分方式       | 256 token / 128 overlap |       256/128       |
| 索引方式       | Top-1（k = 1）           |     Top-1           |
| 数据库         | Lucene-BM25             |          BM25       |
| 攻击对象       | RICLM 的输入文本 | GPT 的用户提示                |
| Query 生成方式 | WikiQA 数据集中挑选的 250 条开放域问答问题  | GPT 生成 100 条覆盖各章节问题，再迭代抽取 |

---

###  评估指标说明

| 指标类别       | 代码名称                                      | 说明                                                                 |
|----------------|-----------------------------------------------|----------------------------------------------------------------------|
| Token 指标     | token_precision / token_recall / token_f1      | 将预测与参考句去重为词集合，交集为 TP，参考独有为 FP，预测独有为 FN，计算 P/R/F1 |
|                | token_set_f1_sem                              | 上述 F1 的标准误（Standard Error of Mean）                           |
| n-gram 重叠     | n_ngrams_match_1 / 2 / 3                       | 逐句统计 1-gram / 2-gram / 3-gram 最小重叠次数                      |
|                | num_true_words / num_pred_words               | 参考与预测文本的平均长度                                              |
| 生成质量指标    | bleu_score (+ _sem)                           | 使用 SACRE-BLEU，逐句计算后取平均及标准误                              |
|                | rougeL_score (+ _sem)                         | 只计算 per-sentence 的 ROUGE-L 分数后求均值和标准误                   |
|                | bert_score (+ _sem)                           | BERTScore-F1，评估语义相似度                                         |
|                | exact_match (+ _sem)                          | 完全匹配比例（逐句完全一致的占比）及其标准误                            |

###  复现结果
QWEN3
| 指标类别   | Qwen3-14B-1      | Qwen3-14B-2      | Qwen3-8B-1       | Qwen3-8B-2       | Qwen3-4B-1       | Qwen3-4B-2       | 
|------------|------------------|------------------|------------------|------------------|------------------|------------------|
| ROUGE-L    | 0.4752           | 0.4869           | 0.4518           | 0.4584           | 0.4725           | 0.4620           |
| BLEU       | 0.3204           | 0.3282           | 0.3014           | 0.3078           | 0.3109           | 0.3066           | 
| F1         | 0.4072 ± 0.0070  | 0.4210 ± 0.0081  | 0.3876 ± 0.0056  | 0.3955 ± 0.0064  | 0.4258 ± 0.0071  | 0.4068 ± 0.0067  |
| BERTScore  | 0.9046 ± 0.0007  | 0.9161 ± 0.0007  | 0.9056 ± 0.0007  | 0.9164 ± 0.0007  | 0.9037 ± 0.0007  | 0.9165 ± 0.0007  |

QWEN2.5
| 指标类别   | Qwen2.5-7B-1     | Qwen2.5-7B-2    | Qwen2.5-3B-1      | Qwen2.5-3B-2     | Qwen2.5-1.5B-1   | Qwen2.5-1.5B-2   | 
|------------|------------------|------------------|------------------|------------------|------------------|------------------|
| ROUGE-L    | 0.7489           | 0.8221           | 0.7477	          |  0.7950          | 0.7372           | 0.7815           |
| BLEU       | 0.5896           | 0.7002           | 0.5861	          |  0.6508          | 0.5680        	  | 0.6314           | 
| F1         | 0.7100 ± 0.0087  | 0.8099 ± 0.0098  | 0.7136 ± 0.0092  |  0.7774 ± 0.0116 | 0.6990 ± 0.0096  |	0.7579 ± 0.0104  |
| BERTScore  | 0.9273 ± 0.0011  | 0.9544 ± 0.0013  | 0.9286 ± 0.0010	|  0.9505 ± 0.0015 | 0.9267 ± 0.0010  | 0.9480 ± 0.0013  |

REST MODEL
| 指标类别  | Qwen1.5-1.8B-1  | Qwen1.5-1.8B-2  | LLaMA2-7B-1     | LLaMA2-7B-2     |
| --------- | --------------- | --------------- | --------------- | --------------- |
| ROUGE-L   | 0.6123          | 0.7928          | 0.7964          | 0.8242          |
| BLEU      | 0.4176          | 0.6155          | 0.6589          | 0.6880          |
| F1        | 0.5561 ± 0.0097 | 0.7842 ± 0.0132 | 0.7694 ± 0.0078 | 0.8137 ± 0.0088 |
| BERTScore | 0.9094 ± 0.0011 | 0.9521 ± 0.0017 | 0.9342 ± 0.0010 | 0.9534 ± 0.0013 |


注：模型编号如 `Qwen3-14B-1` 与 `Qwen3-14B-2` 中的数字后缀代表 Prompt 插入位置：`1 = beginning`，`2 = end`。

<img src="image/Spill_beginning_end.jpg" width="50%">

---

###  原始项目链接与说明

- 原论文项目代码开源地址：[github.com/zhentingqi/rag-privacy](https://github.com/zhentingqi/rag-privacy)

本目录中的实验流程基于上述原始代码复现，但为了更好地统一管理与分析：
- **重新组织为线性化 Pipeline**（预处理 → 检索 → 生成 → 抽取）；
- **未包含原项目中的完整评估代码**，仅指出关键指标计算与必要函数的表示。

如需完整对照实验，请参考原项目仓库。

---

#  The Good and The Bad

## ./GoodAndBad

**论文信息**  
Zeng, Shenglai, Jiankun Zhang, Pengfei He, Yue Xing, Yiding Liu, Han Xu, Jie Ren et al. (2024). 
*The good and the bad: Exploring privacy issues in retrieval-augmented generation (rag).*
arXiv preprint [arXiv:2402.16893](https://arxiv.org/abs/2402.16893)
