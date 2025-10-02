# 百万智能体社交应用 - Million Agents Social App

基于CAMEL框架的百万级智能体社交应用

## 项目结构

```
million-agents/
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── social_agent.py
│   ├── social/
│   │   ├── __init__.py
│   │   ├── network.py
│   │   ├── interactions.py
│   │   └── communities.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
├── tests/
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_agents.py
│   ├── test_social.py
│   └── conftest.py
├── .env
├── .gitignore
├── requirements.txt
├── pyproject.toml
└── README.md
```

## 功能特性

- 基于CAMEL框架的多智能体系统
- 社交网络交互和社区形成
- 百万级智能体扩展能力
- 开放的API接口
- 完整的测试覆盖

## 环境配置

1. 复制 `.env.example` 到 `.env`
2. 填入OpenAI API密钥
3. 安装依赖：`pip install -r requirements.txt`
4. 运行测试：`pytest`

## 开发进度

- [x] 基础项目结构
- [ ] 环境配置
- [ ] 基础智能体类
- [ ] 社交网络功能
- [ ] 百万级扩展
- [ ] 测试覆盖