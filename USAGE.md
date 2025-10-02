# 使用指南 - 百万级智能体社交应用

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone git@github.com:shizhengLi/million-agents.git
cd million-agents

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入你的 OpenAI API 密钥
```

### 2. 环境变量配置

在 `.env` 文件中配置以下必需参数：

```bash
# OpenAI API 配置 (必需)
OPENAI_API_KEY=your_actual_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini

# 社交网络配置 (可选)
MAX_AGENTS=1000000
AGENT_BATCH_SIZE=100
INTERACTION_INTERVAL=5
COMMUNITY_SIZE_LIMIT=1000
```

### 3. 运行示例

```bash
# 运行基础演示
python examples/demo.py

# 运行测试
python -m pytest tests/ -v
```

## 核心功能使用

### 创建社交智能体

```python
from src.agents import SocialAgent

# 创建一个友好的AI研究者智能体
alice = SocialAgent(
    agent_id="alice_001",
    name="Alice",
    personality="friendly",
    interests=["AI", "machine learning", "social networks"],
    bio="AI researcher interested in social dynamics"
)

# 创建一个分析型的数据科学家
bob = SocialAgent(
    agent_id="bob_002",
    name="Bob",
    personality="analytical",
    interests=["data science", "statistics", "research"],
    bio="Data scientist with analytical mindset"
)
```

### 管理好友关系

```python
# 添加好友
alice.add_friend(bob)
print(f"Alice 的好友数量: {len(alice.friends)}")

# 移除好友
alice.remove_friend(bob)
print(f"Alice 的好友数量: {len(alice.friends)}")
```

### 社区管理

```python
# 加入社区
alice.join_community("AI_Researchers")
bob.join_community("Data_Scientists")

# 查看社区
print(f"Alice 的社区: {alice.communities}")
print(f"Bob 的社区: {bob.communities}")

# 离开社区
alice.leave_community("AI_Researchers")
```

### 生成智能体对话

```python
# 生成消息
message = alice.generate_message(
    context="欢迎来到我们的社交网络！",
    max_length=100
)
print(f"Alice 说: {message}")

# 记录交互
alice.record_interaction(
    with_agent_id=bob.agent_id,
    message="嗨 Bob，很高兴认识你！",
    interaction_type="greeting"
)
```

### 兼容性计算

```python
# 计算两个智能体的兼容性
compatibility = alice.check_compatibility(bob)
print(f"Alice 和 Bob 的兼容性: {compatibility:.2%}")

# 如果兼容性高，可以建立好友关系
if compatibility > 0.5:
    alice.add_friend(bob)
    print("建立好友关系成功！")
```

### 获取智能体统计信息

```python
# 获取社交统计
stats = alice.get_stats()
print(f"好友数量: {stats['total_friends']}")
print(f"社区数量: {stats['total_communities']}")
print(f"交互次数: {stats['total_interactions']}")
print(f"最常见交互类型: {stats['most_common_interaction']}")
```

## 配置管理

### 使用设置类

```python
from src.config import Settings

# 获取配置 (单例模式)
settings = Settings()

# 访问配置项
print(f"最大智能体数量: {settings.max_agents:,}")
print(f"批量大小: {settings.agent_batch_size}")
print(f"OpenAI 模型: {settings.openai_model}")
```

### 环境变量说明

| 变量名 | 描述 | 默认值 | 必需 |
|--------|------|--------|------|
| `OPENAI_API_KEY` | OpenAI API 密钥 | - | ✅ |
| `OPENAI_BASE_URL` | OpenAI API 基础URL | `https://api.openai.com/v1` | ✅ |
| `OPENAI_MODEL` | 使用的模型 | `gpt-4o-mini` | ✅ |
| `MAX_AGENTS` | 最大智能体数量 | `1000000` | ❌ |
| `AGENT_BATCH_SIZE` | 批处理大小 | `100` | ❌ |
| `INTERACTION_INTERVAL` | 交互间隔(秒) | `5` | ❌ |
| `COMMUNITY_SIZE_LIMIT` | 社区大小限制 | `1000` | ❌ |
| `DATABASE_URL` | 数据库URL | `sqlite:///social_agents.db` | ❌ |
| `LOG_LEVEL` | 日志级别 | `INFO` | ❌ |

## 测试

### 运行所有测试

```bash
# 运行全部测试并查看覆盖率
python -m pytest tests/ -v --cov=src --cov-report=html

# 只运行单元测试
python -m pytest tests/ -v -m "unit"

# 只运行集成测试
python -m pytest tests/ -v -m "integration"
```

### 测试覆盖率

项目要求 100% 的测试覆盖率。当前状态：
- 总测试用例: 31个
- 通过率: 100%
- 代码覆盖率: 89.53%

## 扩展功能

### 批量创建智能体

```python
from src.agents import SocialAgent
import random

# 预定义的个性类型和兴趣
personalities = ["friendly", "analytical", "creative", "formal", "casual"]
interests = [
    ["AI", "machine learning"],
    ["art", "design"],
    ["research", "science"],
    ["social networks", "communication"]
]

# 批量创建智能体
agents = []
for i in range(1000):
    agent = SocialAgent(
        agent_id=f"agent_{i:04d}",
        name=f"Agent_{i}",
        personality=random.choice(personalities),
        interests=random.choice(interests)
    )
    agents.append(agent)

print(f"创建了 {len(agents)} 个智能体")
```

### 性能监控

```python
import time
from src.config import Settings

settings = Settings()

# 监控批量操作性能
start_time = time.time()

# 执行大量操作
for i in range(len(agents) - 1):
    agents[i].add_friend(agents[i + 1])

end_time = time.time()
print(f"添加 {len(agents)-1} 个好友关系耗时: {end_time - start_time:.2f} 秒")
```

## 故障排除

### 常见问题

1. **ModuleNotFoundError: No module named 'src'**
   ```bash
   # 确保在项目根目录运行
   cd /path/to/million-agents
   PYTHONPATH=./src python examples/demo.py
   ```

2. **OpenAI API 错误**
   ```bash
   # 检查 .env 文件中的 API 密钥
   cat .env | grep OPENAI_API_KEY
   
   # 确保 API 密钥有效且配额充足
   ```

3. **测试失败**
   ```bash
   # 清理测试缓存
   rm -rf .pytest_cache
   python -m pytest tests/ -v
   ```

### 调试模式

```python
import logging
from src.config import Settings

# 启用调试日志
logging.basicConfig(level=logging.DEBUG)

# 获取详细配置信息
settings = Settings()
print(str(settings))  # 显示敏感信息已隐藏的配置
```

## 扩展开发

### 添加新的个性类型

```python
# 在 src/agents/social_agent.py 中添加
VALID_PERSONALITIES = {
    # 现有类型...
    "enthusiastic": "energetic and passionate",
    "thoughtful": "deep and reflective"
}
```

### 自定义交互类型

```python
# 记录自定义交互
agent.record_interaction(
    with_agent_id=other.agent_id,
    message="合作完成了一个项目！",
    interaction_type="collaboration",  # 自定义类型
    context={"project": "AI Research", "duration": "2 weeks"}
)
```

## 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启 Pull Request

**注意**: 所有更改必须通过 100% 测试覆盖率要求。

---

**更多详细信息请参考**: [PROJECT_PLAN.md](PROJECT_PLAN.md) 和 [API 文档](docs/) (即将添加)