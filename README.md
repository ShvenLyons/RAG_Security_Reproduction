# RAG_Security_Reproduction

复现当前 RAG 安全攻击方法，包括数据窃取、检索投毒、提示注入、成员推断

---

##  文件说明

| 文件夹名称            | 攻击类别       | 功能描述                                                                 |
|-----------------------|----------------|--------------------------------------------------------------------------|
| `DataExfiltration/`   | 数据提取   | 复现基于指令/Query 的知识库内容泄露攻击。详见 [`RD.md`](./DataExfiltration/RD.md)。 |
| `PoisonAttack/`       | 检索投毒   | 实现检索环节的后门/污染攻击，包括 Chunk Injection、Embedding Manipulation 等策略。 |
| `MembershipInference/`| 成员推理   | 用于实现针对嵌入空间或生成响应的成员信息泄露攻击实验。               |
| `PromptInjection/`    | 提示注入   | 复现提示污染（例如 Jailbreak Prompt）对 RAG 检索与生成的劫持能力。   |

---

##  Attack ⇐⇒ 论文

- 每类攻击文件夹包含：
  - 攻击与对应论文简介（`RD.md` ）；
  - 每个攻击方式复现均为单独子文件夹；

---

##  当前TODO

- [ ] 整理 The Good and The Bad 结果；
- [√] 复现 PoisonedRAG ；
- [ ]
