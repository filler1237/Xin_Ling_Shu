# 心灵树：本地 RAG 问答 & 心理支持统一智能体

一个可本地部署的中文问答/心理支持演示项目：用本地向量库做检索增强生成（RAG），并在回答前进行安全评估（危机识别与越狱/不当内容拦截）。项目包含通用 RAG、心理支持多知识库，以及一个将四个知识库融合的“统一智能体”（A/B 路径 + 仲裁融合）。

> 免责声明：本项目用于学习与课程实践，不构成医疗建议或诊断。若你或他人存在自伤/自杀风险或紧急危险，请立即联系当地紧急服务或专业机构。

## 功能概览

- 本地 RAG：将文档切分、向量化、写入 Chroma，并按 Top-k 检索后交给 LLM 生成回答
- 心理支持链路：symptom/method/strategy 三库检索 + 策略编排生成支持性回复
- 统一智能体（四库）：rag\_docs + symptom\_db + method\_db + strategy\_db 联合检索；A/B 两条生成路径并由仲裁器融合输出
- 安全评估：生成前进行危机识别与安全拦截；配套 pytest 回归样例
- Gradio UI：提供交互界面、证据/参考、运行状态与会话存储

## 技术栈

- LLM：Ollama（默认 `qwen2.5:7b`）
- 向量库：ChromaDB（HNSW 索引）
- 框架：LangChain（RAG/向量库封装）
- 向量模型：BGE（默认 `BAAI/bge-small-zh-v1.5`，通过 sentence-transformers）
- UI：Gradio

## 目录结构（核心）

- `main.py`：命令行入口（build/ui/psy-build/psy-chat/unified-build/unified-chat）
- `rag/`：通用 RAG pipeline（加载、切分、向量库、检索、生成）
- `psych_support/`：心理支持链路（分析、安全、知识库、多库检索、回复生成）
- `unified_agent/`：统一智能体（四库检索 + A/B + 仲裁融合 + 会话）
- `ui/`：Gradio 界面
- `tests/`：安全样例回归测试
- `data/`：运行时数据（向量库、会话、上传文件等；通常不建议提交到 Git）

## 快速开始（Windows / PowerShell）

### 1) 安装依赖

建议使用虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r .\requirements.txt
```

### 2) 安装并启动 Ollama

1. 安装 Ollama：<https://ollama.com/>
2. 拉取模型（默认配置使用 `qwen2.5:7b`）：

```powershell
ollama pull qwen2.5:7b
```

确保 Ollama 服务在运行，并可通过默认地址访问：`http://localhost:11434`

### 3) 构建向量库（推荐：统一智能体四库）

首次运行建议先构建四个库：

```powershell
python .\main.py unified-build
```

默认会使用：

- 通用文档目录：`./新建文件夹`
- 通用向量库目录：`./data/db`
- 心理支持向量库目录：`./data/psych_db`

### 4) 启动 UI

```powershell
python .\main.py ui
```

启动后按终端输出打开本地地址即可。

## 使用方式

### 通用 RAG

- 构建/更新向量库：

```powershell
python .\main.py build
```

- 启动 UI（统一界面，默认走统一智能体）：

```powershell
python .\main.py ui
```

### 心理支持智能体

- 构建心理支持三库：

```powershell
python .\main.py psy-build
```

- 命令行对话演示：

```powershell
python .\main.py psy-chat
```

### 统一智能体（四库）

- 构建四库：

```powershell
python .\main.py unified-build
```

- 命令行对话演示：

```powershell
python .\main.py unified-chat
```

## 配置说明

项目默认配置在：

- 通用 RAG：`rag/config.py`
- 心理支持：`psych_support/config.py`

常用启动参数（对部分子命令生效）：

- `--docs_dir`：修改通用文档目录
- `--db_dir`：修改向量库持久化目录（通用库/心理库会随子命令使用不同默认值）
- `--ollama_model`：修改 Ollama 模型名
- `--chunk_size` / `--chunk_overlap`：切分参数
- `--top_k`：检索条数

示例（把通用文档放到 `data/docs`，并指定向量库目录）：

```powershell
python .\main.py unified-build --docs_dir data/docs --db_dir data/db
```

## 测试

运行安全样例回归测试：

```powershell
pytest
```

## 常见问题（Troubleshooting）

### 1) Chroma 报错：`Error loading hnsw index`

该问题通常是向量库目录中的 HNSW 索引损坏/不完整，或多进程/多实例同时占用导致。

处理步骤：

1. 关闭所有 Gradio UI / Python 进程（确保没有两个实例在用同一个库目录）
2. 删除向量库目录（整文件夹）：
   - `./data/db`
   - `./data/psych_db`
3. 重新构建：

```powershell
python .\main.py unified-build
```

### 2) Windows 下删不掉 `data/db` 或 `data/psych_db`

通常是文件句柄被占用（UI 未关闭或后台仍在跑）。先完全退出相关进程后再删；必要时重启系统后删除再重建。

## 致谢

- LangChain / ChromaDB / Gradio / Ollama
- BAAI BGE 系列向量模型
