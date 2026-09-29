# 🤖 AI Agent Framework

![tests](https://github.com/iaanimo/ai-agent/actions/workflows/tests.yml/badge.svg)

基于 **LLM + LangChain** 搭建的模块化 AI 智能体框架，集成了规划、记忆、工具调用、RAG、思维链推理和多智能体协作等核心能力。

---

## 📐 架构总览

```
┌─────────────────────────────────────────────────────────────┐
│                        AI Agent Framework                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ Planner  │◄──►│  Agent   │◄──►│ Memory   │              │
│  │ 任务规划  │    │  核心引擎 │    │ 三层记忆  │              │
│  └──────────┘    └────┬─────┘    └──────────┘              │
│                       │                                     │
│         ┌─────────────┼─────────────┐                      │
│         ▼             ▼             ▼                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                 │
│  │  CoT     │  │ Executor │  │   RAG    │                 │
│  │ 思维链    │  │ 工具执行  │  │ 检索增强  │                 │
│  └──────────┘  └──────────┘  └──────────┘                 │
│                       │                                     │
│         ┌─────────────┼─────────────┐                      │
│         ▼             ▼             ▼                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                 │
│  │ Search   │  │ Calculator│ │ CodeExec │                 │
│  │ 网络搜索  │  │ 数学计算  │  │ 代码执行  │                 │
│  └──────────┘  └──────────┘  └──────────┘                 │
│                                                             │
│  ┌─────────────────────────────────────────────┐           │
│  │           Multi-Agent Orchestrator           │           │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐    │           │
│  │  │Researcher│ │  Coder   │ │ Reviewer │    │           │
│  │  │ 研究员   │ │ 程序员   │ │ 审查员    │    │           │
│  │  └──────────┘ └──────────┘ └──────────┘    │           │
│  └─────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
cd ai-agent
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件，填入你的 API Key
```

支持的 LLM 提供商：
| 提供商 | 环境变量 | 说明 |
|--------|---------|------|
| DeepSeek | `DEEPSEEK_API_KEY` | 默认，性价比高 |
| OpenAI | `OPENAI_API_KEY` | GPT-4o 等 |
| Local (Ollama) | `LOCAL_BASE_URL` | 本地部署 |

### 3. 运行

```bash
# 交互式聊天（ReAct 模式）
python main.py

# 指定模式
python main.py --mode react      # ReAct 模式
python main.py --mode plan       # Plan-and-Execute 模式
python main.py --mode direct     # 直接回答模式

# 运行演示
python main.py --demo rag        # RAG 演示
python main.py --demo cot        # CoT 推理演示
python main.py --demo multi      # 多智能体协作演示
python main.py --demo plan       # 规划执行演示
```

---

## 🧩 核心模块详解

### 1. Agent — 核心引擎

三种运行模式：

| 模式 | 说明 | 适用场景 |
|------|------|---------|
| **ReAct** | 推理→行动→观察 循环 | 需要工具调用的复杂任务 |
| **Plan-and-Execute** | 先规划再执行，支持动态重规划 | 多步骤复杂任务 |
| **Direct** | 直接回答，无工具调用 | 简单问答 |

```python
from core.agent import Agent, AgentMode
from tools import get_all_tools

agent = Agent(
    mode=AgentMode.REACT,
    tools=get_all_tools(),
    enable_cot=True,
    enable_memory=True,
)

result = await agent.run("帮我分析这个数据...")
```

### 2. Memory — 三层记忆系统

```
┌─────────────────────────────────────┐
│         Memory (统一接口)            │
├──────────┬──────────┬───────────────┤
│ 短期记忆  │ 工作记忆  │   长期记忆    │
│(滑动窗口) │(草稿本)   │(向量持久化)   │
│          │          │              │
│ 最近20条  │ 当前任务   │ 历史经验     │
│ 对话历史  │ 上下文     │ 知识积累     │
└──────────┴──────────┴───────────────┘
```

- **短期记忆 (ConversationBuffer)**：滑动窗口，保留最近 N 轮对话
- **工作记忆 (WorkingMemory)**：当前任务的草稿本，存储中间结果
- **长期记忆 (LongTermMemory)**：持久化存储，支持中文关键词检索、自动去重

**自动记忆**：每轮对话后，Agent 会用一次轻量 LLM 调用提取值得记住的用户事实（偏好、项目、约束）并持久化，下次自动带进上下文。可用 `AUTO_MEMORY=false` 关闭。

交互命令：

```
/remember <fact>    # 手动记住一条事实
/memories           # 查看记住了什么（含 id 和访问次数）
/forget <id|关键词>  # 按 id 或关键词删除记忆
```

### 3. Planner — 任务规划器

LLM 驱动的任务分解引擎，将复杂目标拆解为可执行步骤：

```python
from core.planner import Planner

planner = Planner(llm, available_tools=["search", "calculator", "code_executor"])
plan = await planner.create_plan("分析 2024 年 AI 行业趋势并生成报告")

print(plan.to_display())
# 📋 Plan for: 分析 2024 年 AI 行业趋势并生成报告 [0/5]
#   ⏳ Step 0: 搜索 2024 年 AI 行业新闻
#   ⏳ Step 1: 提取关键趋势和数据
#   ⏳ Step 2: 分析各趋势的影响
#   ⏳ Step 3: 生成可视化图表
#   ⏳ Step 4: 撰写综合报告
```

支持 **动态重规划**：步骤失败时自动调整计划。

### 4. Chain of Thought (CoT) — 思维链推理

四种 CoT 策略：

| 策略 | 说明 | 特点 |
|------|------|------|
| **Zero-shot CoT** | "Let's think step by step" | 最简单，通用 |
| **Few-shot CoT** | 带示例的推理 | 更可控 |
| **Self-consistency** | 单次调用内展开多条路径、由模型自评收敛（**不采样多次**） | 更可靠 |
| **Tree-of-Thought** | 树形探索 | 最深入 |

```python
from core.reasoning import ChainOfThought, CoTStrategy

cot = ChainOfThought(strategy=CoTStrategy.TREE_OF_THOUGHT)
chain = await cot.reason("如果一个水池有两个进水管和一个出水管...")
print(chain.to_display())
```

### 5. Tool System — 工具系统

内置工具：

| 工具 | 功能 | 示例 |
|------|------|------|
| **search** | DuckDuckGo 网络搜索 | 搜索最新信息 |
| **calculator** | 数学表达式计算 | `sqrt(144) + 25% of 800` |
| **code_executor** | Python 代码执行 | 运行任意 Python 代码 |
| **file_ops** | 文件读写操作 | 读取/写入/列出文件 |

工具调用走**自研 JSON 协议**（ReAct 循环内解析，**三层解析降级 + 格式纠正重试**），
不依赖模型的 Function Calling 接口 —— 这样换任何一家模型都能跑。

```python
from tools import get_all_tools, create_search_tool

agent = Agent(tools=get_all_tools())

# 或注册自定义工具
from langchain_core.tools import StructuredTool

my_tool = StructuredTool(
    name="my_tool",
    description="My custom tool",
    func=lambda x: f"Result: {x}",
)
agent.register_tools([my_tool])
```

### 6. RAG — 检索增强生成

RAG 流程：文档加载 → 分块 → 向量化 → 检索 → 增强生成。

**向量化走三级降级**，缺依赖时逐级下落，保证零依赖也能跑：

1. `HuggingFaceEmbeddings`（本地语义模型，需 `sentence-transformers`）
2. `OpenAIEmbeddings`（仅当 provider 支持 embedding 接口时）
3. **`SimpleEmbeddings`** —— 纯 stdlib + numpy 的**哈希分桶词频向量**，零依赖

> ⚠️ **当前环境走第 3 条**（`sentence-transformers` 未安装），因此检索是
> **词法级**的、不是语义级 —— 同义词、改写问法召回不到。要上语义检索，
> 装上 `sentence-transformers` 即自动切到第 1 条，业务代码不用动。

```python
from rag import RAGRetriever

rag = RAGRetriever()

# 加载文档
rag.ingest("./knowledge_base/")           # 加载目录
rag.ingest("./document.pdf")              # 加载 PDF
rag.ingest("https://example.com/article") # 加载网页

# 查询
answer = await rag.query("什么是 RAG？")
```

支持的向量存储：
- **FAISS**：轻量级，适合中小规模（**当前环境已安装并在用**）
- **ChromaDB**：持久化，适合生产环境（**接口已接，当前环境未安装**）

### 7. Multi-Agent — 多智能体协作

三种协作模式：

```
Sequential（串行）:
  Researcher ──► Coder ──► Reviewer ──► Final Answer

Parallel（并行）:
  Researcher ──┐
  Coder    ───┼──► Synthesizer ──► Final Answer
  Reviewer ──┘

Debate（辩论）:
  Agent A ◄──► Agent B ◄──► Agent C
       (cross-review & refine)
```

```python
from multi_agent import Orchestrator, ResearcherAgent, CoderAgent, ReviewerAgent
from multi_agent.orchestrator import CollaborationMode

orchestrator = Orchestrator()
orchestrator.add_agent(ResearcherAgent())
orchestrator.add_agent(CoderAgent())
orchestrator.add_agent(ReviewerAgent())

result = await orchestrator.run(
    "设计一个电商推荐系统",
    mode=CollaborationMode.SEQUENTIAL
)
```

**交互命令**：在对话里直接触发多智能体协作（无需写代码）：

```
/multi 帮我设计一个电商推荐系统                      # 默认串行
/multi parallel 调研并实现一个 REST API              # 并行
/multi debate 该用 SQL 还是 NoSQL                    # 辩论模式
```

---

## 📁 项目结构

```
ai-agent/
├── main.py                 # 主入口
├── requirements.txt        # 依赖
├── .env.example            # 环境变量模板
├── README.md               # 本文件
│
├── config/                 # 配置管理
│   ├── __init__.py
│   └── settings.py
│
├── core/                   # 核心引擎
│   ├── agent.py            # Agent 主类
│   ├── memory.py           # 三层记忆系统
│   ├── planner.py          # 任务规划器
│   ├── reasoning.py        # CoT 推理引擎
│   ├── executor.py         # 工具执行器
│   └── llm_factory.py      # LLM 工厂
│
├── tools/                  # 内置工具
│   ├── search.py           # 网络搜索
│   ├── calculator.py       # 数学计算
│   ├── code_executor.py    # 代码执行
│   └── file_ops.py         # 文件操作
│
├── rag/                    # RAG 系统
│   ├── document_loader.py  # 文档加载器
│   ├── vectorstore.py      # 向量存储
│   └── retriever.py        # 检索器
│
├── multi_agent/            # 多智能体
│   ├── orchestrator.py     # 协调器
│   └── agents/
│       ├── researcher.py   # 研究员
│       ├── coder.py        # 程序员
│       └── reviewer.py     # 审查员
│
├── examples/               # 示例
│   ├── single_agent_demo.py
│   ├── rag_demo.py
│   ├── cot_demo.py
│   ├── multi_agent_demo.py
│   └── plan_and_execute_demo.py
│
└── data/                   # 数据目录
    ├── knowledge_base/     # 知识库文档
    ├── vector_store/       # 向量索引
    └── memory/             # 长期记忆
```

---

## 🔧 扩展指南

### 添加自定义工具

```python
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class MyToolInput(BaseModel):
    query: str = Field(description="Input query")

def my_function(query: str) -> str:
    """My custom tool description."""
    return f"Processed: {query}"

custom_tool = StructuredTool(
    name="my_custom_tool",
    description="Description of what this tool does",
    func=my_function,
    args_schema=MyToolInput,
)

agent.register_tools([custom_tool])
```

### 添加自定义 Agent

```python
from multi_agent.orchestrator import SpecializedAgent

class MyAgent(SpecializedAgent):
    def __init__(self, llm=None):
        super().__init__(
            name="MyAgent",
            role="Custom Role",
            llm=llm or create_llm(),
            system_prompt="You are a specialized agent for...",
        )

orchestrator.add_agent(MyAgent())
```

---

## 🧪 测试

```bash
python -m pytest        # 或 .venv/Scripts/python.exe -m pytest
```

测试覆盖工具安全（计算器 AST 防逃逸、文件路径封闭、代码执行超时）、记忆（去重/中文检索/自动提取）、Agent 行为（落盘写报告、重规划保留结果、ReAct 重试）、多智能体工具循环。全部使用假 LLM 和临时目录，**不联网、不调用真实 API、不碰真实 data/**。

---

## 🙏 致谢

- [LangChain](https://github.com/langchain-ai/langchain) — 核心框架
- [FAISS](https://github.com/facebookresearch/faiss) — 向量检索
- [ChromaDB](https://github.com/chroma-core/chroma) — 向量数据库
- [DuckDuckGo Search](https://github.com/deedy5/duckduckgo_search) — 网络搜索
- [Rich](https://github.com/Textualize/rich) — 终端美化

---
## Web 部署

项目自带一个常驻 Web 服务器 + 聊天界面（FastAPI + Uvicorn），依赖已独立安装在项目自己的 .venv 里。

### 一键启动

双击 start.bat，或在命令行运行：

```bat
cd /d ai-agent
.venv\Scripts\python.exe server.py
```

然后浏览器打开 http://127.0.0.1:8000 即可聊天。

### 开机自启

注册为计划任务，登录 Windows 后自动启动：

```bat
.venv\Scripts\python.exe autostart.py install
```

其他命令：autostart.py uninstall 移除，autostart.py status 查看状态。

### 局域网访问

```bat
.venv\Scripts\python.exe server.py --host 0.0.0.0
```

> 注意：当前服务没有鉴权，请不要直接暴露到公网。
