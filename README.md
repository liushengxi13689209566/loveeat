# loveeat · AI 智能选餐

本地运行的智能选餐 Web 应用：用自然语言描述用餐需求（如"想吃点辣的，不要猪肉，来两道菜"），本地 Ollama 大模型从内置菜品库中挑出合适的菜。全程本地推理，无外部 API，数据不出本机。

## 运行前提

1. **安装 Go（1.20+）**：从 https://go.dev/dl/ 下载，或 macOS 上执行 `brew install go`
2. **安装 Ollama**：从 https://ollama.com/download 下载安装包（macOS 安装后默认开机自动在后台运行）
3. **拉取模型**（约 2GB，首次需等待）：

   ```bash
   ollama pull qwen2.5:3b
   ```

## 启动

```bash
# 若 Ollama 未在后台运行，先启动它（macOS 安装包版一般已自动运行，可跳过）
ollama serve

# 在项目目录启动后端（须在本目录运行，GET / 会读取这里的 index.html）
go run main.go
```

浏览器访问 **http://localhost:8080**，输入需求，点击"AI 帮我选"。

## 配置（可选，环境变量）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama 服务地址 |
| `OLLAMA_MODEL` | `qwen2.5:3b` | 使用的本地模型 |
| `LISTEN_ADDR` | `:8080` | 后端监听地址 |

示例：`OLLAMA_MODEL=qwen2.5:7b go run main.go`

## 接口

- `GET /`：前端页面
- `POST /api/recommend`（一次性返回）：

  ```json
  // 请求
  {"query": "想吃辣的，不要猪肉，来两道"}
  // 响应
  {"dishes": [{"name": "水煮鱼", "tastes": ["辣"], "ingredients": ["鱼", "豆芽", "豆瓣酱", "干辣椒"], "category": "荤菜"}]}
  ```

- `POST /api/recommend/stream`（SSE 流式，前端默认使用）：逐 token 实时推送模型生成过程，事件约定
  - `token`：`{"text": "..."}` 模型增量输出片段
  - `status`：`{"message": "..."}` 过程状态（如解析失败自动重试）
  - `result`：`{"dishes": [...]}` 最终推荐菜品（与普通接口一致）
  - `error`：`{"error": "..."}` 失败信息

## 常见问题

- **提示"无法连接本地 Ollama"**：确认 `ollama serve` 在运行、模型已 `ollama pull qwen2.5:3b`。
- **推荐很慢**：首次调用需要把模型加载进内存；3B 模型一般 5~30 秒出结果。
- **"AI 暂时没想好"**：小模型偶发输出格式不稳定，后端已自动重试一次；换个更明确的说法再试。
- **想看 AI 的输入输出**：后端日志会打印完整的模型输入（Prompt）、模型原始输出（含耗时）、模型返回的菜名及被剔除的幻觉菜名，直接看 `go run main.go` 的终端输出即可。

## 文件说明

- `main.go`：后端（菜品库、忌口解析、searchDishes 工具定义与实现、Agent 工具调用循环、Prompt 构造、Ollama /api/chat 调用、数量解析、JSON 提取与校验、SSE 路由）
- `index.html`：前端单页（原生 JS/CSS，内嵌于后端同目录，由 `GET /` 直接返回）

## 实现说明

- **Function Calling（工具查库）**：菜品库不再整体塞进提示词。后端通过 Ollama `/api/chat` 的 `tools` 定义 `searchDishes(taste, exclude_ingredient, category)` 工具，模型自主决定何时调用、传什么参数，后端在本地对菜品切片执行过滤并回传结果（单次最多 50 条，防止撑爆上下文）。菜品库可以无限扩大，而提示词长度保持不变（要求模型支持工具调用，qwen2.5 系列支持）。
- **忌口三重保障**：① 用户输入解析出的忌口关键词在工具实现层强制生效（与模型传入的 exclude_ingredient 合并）；② 工具只返回过滤后的菜，被禁食材根本不在模型视野里；③ 最终菜名必须出自工具返回过的集合（防编造）。
- **流式输出**：后端把模型逐段生成的内容经 SSE 实时转发给前端（`fetch` + `ReadableStream` 读取），页面上的“AI 思考过程”框会实时打印工具查询过程（如“AI 正在查询菜品库：taste=辣…”）与生成内容，生成完毕后渲染菜品卡片；解析失败时自动重试一次。
- **调试日志**：每次请求打印系统提示词、模型工具调用参数、工具返回条数、模型最终输出（含耗时）、被剔除菜名等，直接看 `go run main.go` 的终端输出。
