<div align="center">

# OneClick Distill 🧊

**开源 · 一键 · 模型蒸馏** — 把大模型的能力蒸馏进小模型，零门槛、零配置、Agent 友好。

`oneclick-distill` 是完整的数据导入 → 数据合成 → 微调训练 → GGUF 量化导出的桌面级蒸馏流水线。GUI（内置 Web UI）、CLI、MCP 三种入口共用同一套 job 契约，普通用户一键点选，AI Agent 也能直接驱动。

</div>

## ✨ 特性

- **三步傻瓜式流程**：导入素材 & 选教师模型 → 选规格 & 硬件自检 → 启动蒸馏 & A/B 对比
- **CPU 保底 / GPU 加速双后端**：无 NVIDIA 显卡也能用（transformers 自研 LoRA），有显卡自动切换 Unsloth QLoRA
- **显存防爆（保命模式）**：硬件探针 → 动态超参数下探 → 试运行探路 → 运行时 OOM 自动降级
- **开箱即用**：`pip install oneclick-distill` 即可运行，无额外编译依赖
- **Agent 友好**：标准 CLI + MCP stdio server，Codex / OpenCode / OpenManus 可直接下发蒸馏指令
- **GGUF 一键分发**：调用 llama.cpp 官方工具导出量化模型，可导入 Ollama

## 🚀 快速开始

```bash
git clone https://github.com/yxtan5022-maker/one-click-distill.git
cd one-click-distill
pip install -e .

# 1. 硬件自检（显示显存/内存/防爆策略）
oneclick-distill hardware

# 2. 一键演示：内置数据，CPU 上跑通冒烟流程（~2MB 小模型）
oneclick-distill demo --max-steps 5

# 3. 启动内置 Web UI（三步卡片流 + 实时进度 + A/B Playground）
oneclick-distill serve
# 浏览器打开 http://127.0.0.1:8080
```

### 完整蒸馏流程（CLI）

```bash
oneclick-distill pipeline \
  --data ./docs ./notes.md \
  --teacher deepseek --teacher-api-key sk-xxx \
  --size balanced \
  --out-dir runs/my-first-distill
```

带教师模型时会先调用 DeepSeek/OpenAI/本地 Ollama 合成问答训练数据，再微调学生模型，最后导出 GGUF。

## 🤖 Agent 集成（MCP）

```bash
oneclick-distill mcp
```

MCP 暴露工具：`hardware` / `start_pipeline` / `pipeline_status` / `local_server_start` / `local_server_stop` / `local_server_status` / `evaluate`。配置示例（Codex / OpenCode / Claude Desktop）：

```json
{
  "mcpServers": {
    "oneclick-distill": {
      "command": "oneclick-distill",
      "args": ["mcp"]
    }
  }
}
```

Agent 下发的任务会出现在 Web UI 的 **"Agent 托管中"** 横幅中并实时同步日志。

## 🗺️ 路线图

- [x] MVP：CLI + FastAPI/WS + 内置 Web UI + MCP + CPU/GPU 双后端
- [x] Electron 桌面外壳（内置 Web UI + 显存占用图）
- [x] Windows 一键安装包（嵌入式 Python 运行时 + 预编译 llama.cpp 工具）
- [x] llama-server 本地 API 节点（OpenAI 兼容 /v1）
- [x] 自动 A/B 评测指标（响应时延、一致性）
- [x] macOS / Linux 安装包（CI 矩阵构建 dmg / AppImage 已产出）
  - 待办：macOS/Linux 的 llama.cpp 量化工具链二进制与平台端到端冒烟测试

### 桌面应用与打包

```bash
# 构建便携 Python 运行时（Windows）
powershell -ExecutionPolicy Bypass -File scripts\build_runtime.ps1
# macOS / Linux
bash scripts/build_runtime.sh

cd desktop
npm ci
npm run dist        # Windows NSIS 安装包（macOS: npx electron-builder --mac dmg；Linux: --linux AppImage）
npm run dist:zip    # Windows 免安装 zip（解压即用，推荐：比 NSIS 安装快约 10 倍）
```

安装包内置 Python 3.12 + torch(CPU) + 全部依赖 + llama.cpp 工具链，安装后无需联网即可离线使用；静默安装 / 卸载分别对应 `Setup.exe /S /D=<dir>` 与 `Uninstall.exe /S`。

**Windows 分发两种形态**：

| 形态 | 命令 | 用户操作 | 实测耗时 |
|---|---|---|---|
| NSIS 安装包（`Setup.exe`） | `npm run dist` | 双击安装 | ~14 分钟 |
| 免安装 zip | `npm run dist:zip` | 解压后双击 `OneClick Distill.exe` | 解压 ~1 分钟 + 启动 ~6 秒 |

zip 体积更大（约 330MB vs 245MB），但无需安装步骤、可任意位置运行、删目录即卸载，适合想快速上手或不想改注册表的用户。

## 📦 数据格式

- **输入**：`.txt` / `.md` / `.json` / `.jsonl` / `.pdf`（PDF 需 `pip install pypdf`）
- **问答对格式**（不使用教师模型时）：每行 `{"prompt": "...", "response": "..."}`

## ⚙️ 配置

复制 `.env.example` 为 `.env`：

| 变量 | 说明 |
|---|---|
| `TEACHER_BASE_URL` / `TEACHER_MODEL` / `TEACHER_API_KEY` | 教师模型 API（数据合成用） |
| `HF_TOKEN` | HuggingFace 令牌（下载 gated 模型用） |
| `HOST` / `PORT` | 服务监听地址 |

## 🧪 测试

```bash
pip install -e .[dev]
pytest
```

## 📄 License

MIT License — 详见 [LICENSE](LICENSE)。

## 🙏 致谢

- [Unsloth](https://github.com/unslothai/unsloth) — 极致微调速度
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — GGUF 量化与推理
- [HuggingFace Transformers](https://github.com/huggingface/transformers) — 模型生态
