# deepDemoV2 — MultiModel Framework

开源多模型组合框架：将对话、意图识别、文生图、扩图等能力通过可插拔 Provider 编排为统一工作流。当前附带 Tkinter 桌面 Demo 作为参考应用。

## 特性

- **多模型组合**：Router 按意图分发，Pipeline 支持多步串联（如文生图 → 扩图）
- **可插拔 Provider**：新增厂商/模型无需改动编排核心
- **配置驱动**：通过 `config/models.yaml` 声明模型与参数（规划中）
- **参考应用**：`deepImageDemo.py` 演示 DeepSeek + 阿里云万相组合

## 快速开始

### 首次安装

```powershell
cd C:\Users\Administrator\Documents\GitHub\deepDemoV2
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 启动项目

已创建虚拟环境后，在项目根目录执行（**推荐，无需先激活 venv**）：

```powershell
.\.venv\Scripts\python.exe deepImageDemo.py
```

或先激活虚拟环境再启动：

```powershell
.\.venv\Scripts\Activate.ps1
python deepImageDemo.py
```

启动后点击 **「设置」** 填入 DeepSeek 与 DashScope API Key。密钥会经 Fernet 加密后写入 `.apikey`（密文），解密密钥保存在本机 `.mmf_keystore`。**这两个文件均不会、也不应提交到 Git。**

也可通过环境变量 `DEEPSEEK_API_KEY`、`DASHSCOPE_API_KEY` 注入（适合 CI/服务器）。

## 文档

| 文档 | 说明 |
|------|------|
| [PROJECT_SPEC.md](PROJECT_SPEC.md) | 框架架构、Capability/Provider 规范、迁移路线图 |
| `docs/` | 用户与开发者文档（建设中） |

## 架构概览

```
Applications (GUI/CLI/API)
        ↓
Framework Core (Orchestrator / Router / Pipeline / Registry)
        ↓
Providers (deepseek, dashscope, community…)
```

## 参与贡献

欢迎实现新的 Provider 或示例应用。开发前请阅读 [PROJECT_SPEC.md](PROJECT_SPEC.md) 中的 Provider 开发规范与开源协作约定。

## 许可证

[Unlicense](LICENSE) — 公共领域，可自由使用与修改。
