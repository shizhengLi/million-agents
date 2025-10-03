# 🚀 百万智能体项目 - 快速开始指南

## 🎯 这个项目是什么？

这是一个**百万级智能体社交网络模拟平台**，可以创建、管理和分析大规模智能体社区。

### 核心能力
- 🤖 **创建百万级智能体** - 批量创建具有不同个性的智能体
- 🕸️ **社交网络构建** - 建立复杂的社交关系网络
- 📈 **消息传播模拟** - 模拟信息在社交网络中的传播
- 🎯 **影响力分析** - 找出最具影响力的关键节点
- 🌐 **实时可视化** - 通过Web界面实时观察系统状态

## 🎮 如何使用（3种方式）

### 方式1: Web界面（最简单！）

```bash
# 1. 启动Web服务器
python -m uvicorn src.web_interface.api.app:create_app --factory --host 0.0.0.0 --port 8000 --reload

# 2. 打开浏览器访问
open http://localhost:8000
```

**Web界面功能：**
- 🎮 **传播控制面板** - 设置传播参数，点击"开始传播模拟"
- 🕸️ **网络拓扑图** - 可视化智能体关系网络
- 📊 **实时指标** - 查看活跃智能体、连接数等
- 📈 **传播分析** - 实时图表展示传播过程
- 🤖 **智能体管理** - 查看智能体状态和详情

### 方式2: 运行Demo脚本

```bash
# 快速开始演示（推荐新手）
python examples/quick_start_demo.py

# 产品传播模拟
python examples/product_viral_demo.py

# 批量智能体管理
python examples/batch_demo.py

# 异步处理演示
python examples/async_demo.py
```

### 方式3: Python代码集成

```python
# 在你的项目中使用
from src.agents import SocialAgent
from src.agents.batch_manager import BatchAgentManager
from src.message_propagation import ViralPropagationModel

# 创建智能体
agent = SocialAgent(
    agent_id="user_001",
    name="Alice",
    personality="friendly",
    interests=["AI", "社交网络"]
)

# 批量创建
manager = BatchAgentManager(max_agents=10000)
agents = manager.create_batch_agents(count=100, name_prefix="User")

# 传播模拟
model = ViralPropagationModel(network)
result = model.propagate_full_simulation()
```

## 🎯 实际应用场景

### 1. 市场营销分析
```bash
# 模拟新产品在社交网络中的传播
python examples/product_viral_demo.py
```
**应用场景：** 测试不同营销策略的效果，找到最优传播路径

### 2. 政策影响评估
模拟政策在不同人群中的接受度，预测实施效果

### 3. 社交研究
研究信息传播、群体行为、社会影响力等

### 4. 教育培训
作为复杂系统、网络科学、AI伦理的教学工具

## 📊 系统要求

- **Python:** 3.9+
- **内存:** 推荐8GB+（用于大规模模拟）
- **API:** OpenAI API Key（可选，用于智能对话）
- **浏览器:** 现代浏览器（Chrome/Firefox/Safari）

## 🔧 安装和配置

### 1. 克隆项目
```bash
git clone <项目地址>
cd million-agents
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 配置环境变量
```bash
# 复制配置文件
cp .env.example .env

# 编辑 .env 文件，添加API密钥（可选）
# OPENAI_API_KEY=your_api_key_here
```

### 4. 验证安装
```bash
# 运行快速演示
python examples/quick_start_demo.py
```

## 🎨 系统架构概览

```
百万智能体系统
├── 智能体层 (Agents)
│   ├── 社交智能体 (SocialAgent)
│   ├── 批量管理器 (BatchAgentManager)
│   └── 异步管理器 (AsyncAgentManager)
├── 网络层 (Social Network)
│   ├── 网络分析算法
│   ├── 社区发现
│   └── 影响力计算
├── 传播层 (Message Propagation)
│   ├── 病毒式传播模型
│   ├── 信息扩散模型
│   └── 影响力最大化
├── 可视化层 (Web Interface)
│   ├── 实时仪表板
│   ├── 网络拓扑可视化
│   └── 传播过程图表
└── API层 (RESTful API)
    ├── 系统状态API
    ├── 传播模拟API
    └── 数据导出API
```

## 📈 性能指标

- **智能体创建速度:** 100+ 智能体/秒
- **最大支持规模:** 1,000,000+ 智能体
- **API响应时间:** < 100ms
- **内存效率:** ~1KB/智能体
- **并发处理:** 支持异步批量操作

## 🛠️ 扩展开发

### 添加新的智能体类型
```python
from src.agents.base import BaseAgent

class CustomAgent(BaseAgent):
    def __init__(self, custom_attributes, **kwargs):
        super().__init__(**kwargs)
        self.custom_attributes = custom_attributes
```

### 添加新的传播模型
```python
from src.message_propagation.base import BasePropagationModel

class CustomPropagationModel(BasePropagationModel):
    def propagate_full_simulation(self):
        # 自定义传播逻辑
        pass
```

## 📚 学习资源

1. **文档目录:** `docs/`
   - `web-interface-visualization-guide.md` - Web界面使用指南
   - `usage-guide-and-demos.md` - 详细使用指南和Demo

2. **示例代码:** `examples/`
   - `quick_start_demo.py` - 快速开始
   - `product_viral_demo.py` - 产品传播模拟

3. **API文档:** http://localhost:8000/docs

## 🤝 社区和支持

- 📧 **问题反馈:** 提交GitHub Issue
- 📖 **文档更新:** 贡献改进文档
- 🎯 **功能请求:** 提出新功能建议
- 🧪 **Bug报告:** 提供详细的错误信息

## 🚀 下一步

1. **立即体验:** 运行 `python examples/quick_start_demo.py`
2. **Web界面:** 启动服务器访问 http://localhost:8000
3. **深入阅读:** 查看 `docs/usage-guide-and-demos.md`
4. **开始实验:** 修改Demo参数，创建自己的模拟场景

---

**记住：** 这是一个功能强大的研究平台，既适合简单的演示，也能支持百万级的复杂模拟！

🎉 **开始你的百万智能体之旅吧！**