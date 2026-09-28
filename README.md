# ReflexRAG · 自适应反思式检索问答系统

> 面向劳动法领域的 Agentic RAG 问答系统:Agent 自主决策「何时检索、怎么检索」,
> 不够就自我纠正地补检索, 每个结论可溯源到法条原文, 并配套自动化评测流水线。

## 为什么不是又一个「LangChain + Chroma 读 PDF」

学生项目里最常见的 RAG demo 是:加载文档 → 切块 → 存向量库 → 模板拼接 → 出答案。
它回答不了三个面试官必问的问题:

1. **每个问题都值得检索吗?都只检索一次吗?** —— ReflexRAG 用查询分析器做四路路由
   (闲聊直答 / 单跳 / 多跳分解 / 超范围拒答),多跳问题自动拆子查询并**迭代补检**;
2. **你怎么知道检索质量好坏?** —— 配套**评测流水线**(金标集 + recall@k / MRR +
   LLM-as-judge 忠实度 + 多配置对比),朴素 RAG 基线 vs 完整系统的每项收益有数字;
3. **答案可信吗?** —— 生成强制**引用编号**,引用越界由代码零成本拦截,陈述忠实性由
   LLM 审查器二次校验,问题会触发重生成。

## 架构

```
                    ┌────────────────────────────────────────────┐
   用户问题 ──────► │ ① 查询分析器(LLM)                          │
                    │   route: chitchat / simple / multi_hop /   │
                    │          out_of_scope                      │
                    │   + 查询规范化 + 多跳查询分解(2~4子查询)    │
                    └───────────────┬────────────────────────────┘
              chitchat│out_of_scope │ simple│multi_hop
                 直答/拒答            ▼
                    ┌────────────────────────────────────────────┐
                    │ ② 迭代检索执行器                            │
                    │   多子查询 → 混合检索 ─┐                    │
                    │   充分性判定(LLM)◄───┘ 不足→补检(≤N轮)   │
                    └───────────────┬────────────────────────────┘
                                    ▼
                    ┌────────────────────────────────────────────┐
                    │ ③ 混合检索:向量(Qdrant) + BM25(jieba)    │
                    │    → RRF 融合 → 可选 Rerank(API)          │
                    └───────────────┬────────────────────────────┘
                                    ▼
                    ┌────────────────────────────────────────────┐
                    │ ④ 流式生成(逐字 SSE) + 强制引用 [n]       │
                    │ ⑤ 审查器:引用越界(代码) + 忠实性(LLM)   │
                    │    越界 → 自动修正重生成一次                │
                    └────────────────────────────────────────────┘
   全程: TraceEvent 逐阶段上报(路由/轮次/耗时/用量), API 以 SSE 实时推送前端
```

几个值得展开的设计:

- **结构感知分块**:法条是天然语义单元,分块器按「第X条」切分、连续条文打包到
  ~500 字,章节/条号/发布日期进入元数据 —— 而非无差别递归切割;
- **两级融合 RRF**:每个子查询内部向量路 + BM25 路 RRF 融合,跨子查询再做一次 RRF,
  多个子查询共同命中的权威条文自然浮到顶部;
- **充分性驱动的补检**:不是固定次数的多轮检索,缺口(missing aspects)直接驱动
  下一轮查询生成,轮数按路由差异化(simple ≤1 轮补检,multi_hop ≤2);
- **免检索路径**:寒暄与超范围问题不消耗检索与长上下文,trace 里能看到每次决策依据。

## 语料

劳动法域 13 部公开法律法规(劳动法 / 劳动合同法 / 劳动争议调解仲裁法 / 社会保险法 /
工伤保险条例 / 最高法劳动争议司法解释(一)(二) 等,含 2025 年 7 月新规),
来自开源项目 [LawRefBook/Laws](https://github.com/LawRefBook/Laws),见 `scripts/fetch_corpus.py`。

## 快速开始

```bash
# 1. 环境(Python ≥3.11)
uv venv .venv && uv pip install -e .

# 2. 配置: 复制 .env.example 为 .env, 填入任意 OpenAI 兼容平台的 key
#    (LLM 与 Embedding 可用不同供应商, Embedding 必须支持 /embeddings)

# 3. 抓取语料并建索引
python scripts/fetch_corpus.py
python scripts/ingest.py --rebuild

# 4. 命令行问答
python scripts/ask.py "试用期最长多久" --trace

# 5. Web 演示(带 Agent 思考过程可视化)
uvicorn reflex_rag.api.main:app --reload
# 浏览器打开 http://127.0.0.1:8000
```

## 评测(开发中)

`data/golden/` 金标集: 单跳事实 / 多跳综合 / 闲聊 / 超范围拒答 四类问题,
标注证据条文与期望路由。评测流水线输出:

- 检索: recall@{5,10}、MRR@10、nDCG@10(基于金标 chunk);
- 路由准确率、拒答正确率;
- 生成: LLM-as-judge 忠实度与相关性、引用精确率;
- 效率: p50/p95 延迟、token 用量与成本;
- 多配置对比(朴素向量 RAG / +BM25 / +rerank / 完整系统)报告与图表。

## 项目结构

```
src/reflex_rag/
├── config.py              # 全局配置(.env)
├── schemas.py             # Chunk / RetrievedChunk
├── llm/client.py          # 异步 LLM 客户端(重试/JSON/流式/用量)
├── embeddings/            # OpenAI 兼容嵌入客户端
├── ingestion/
│   ├── loaders.py         # frontmatter 语料加载
│   ├── chunker.py         # 结构感知分块(条文原子)
│   └── pipeline.py        # 摄取编排
├── retrieval/
│   ├── vector_store.py    # Qdrant 封装
│   ├── bm25.py            # jieba + BM25Okapi
│   ├── fusion.py          # RRF
│   ├── reranker.py        # 可插拔 rerank
│   └── hybrid.py          # 混合检索编排
├── agent/
│   ├── prompts.py         # 提示词集中管理
│   ├── analyzer.py        # 路由 + 查询分解
│   ├── executor.py        # 迭代检索 + 充分性判定
│   ├── generator.py       # 流式生成 + 引用提取
│   ├── critic.py          # 两级答案审查
│   └── pipeline.py        # 全链路编排 + Trace
└── api/main.py            # FastAPI + SSE
```

## License

MIT
