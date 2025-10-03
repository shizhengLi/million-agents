# 多智能体消息传播可视化Web接口

## 📋 概述

这是一个基于FastAPI和现代Web技术栈构建的多智能体消息传播系统可视化Web界面。该系统提供了完整的消息传播模拟、影响力最大化分析、网络拓扑可视化等功能，支持实时交互和数据分析。

### 🎯 核心功能

- **消息传播模拟** - 支持病毒式传播和信息扩散模型
- **影响力最大化分析** - 多种算法找出最优种子智能体
- **实时网络可视化** - 动态展示网络拓扑和传播过程
- **智能体管理** - 监控和管理多智能体系统状态
- **数据分析仪表板** - 实时指标监控和历史数据管理

## 🚀 快速开始

### 启动Web服务器

```bash
# 进入项目目录
cd million-agents

# 启动Web服务器
python -m uvicorn src.web_interface.api.app:create_app --factory --host 0.0.0.0 --port 8000 --reload
```

### 访问界面

- **主页仪表板：** http://localhost:8000
- **API文档：** http://localhost:8000/docs
- **传播仪表板：** http://localhost:8000/dashboard

## 🌐 Web界面功能详解

### 1. 主仪表板界面

访问 http://localhost:8000 即可看到完整的功能仪表板，包含以下模块：

#### 📊 关键指标卡片
- **活跃智能体** - 当前系统中活跃的智能体数量
- **网络连接数** - 智能体之间的连接关系总数
- **传播会话** - 历史传播模拟会话数量
- **平均影响力** - 所有智能体的平均声誉分数

#### 🎛️ 传播控制面板
- **传播消息** - 自定义要传播的消息内容
- **种子智能体数量** - 选择初始传播的智能体数量（1-10）
- **传播模型** - 选择传播模型：
  - 病毒式传播模型 (Viral Propagation)
  - 信息扩散模型 (Information Diffusion)
- **感染概率** - 调节传播概率（0-1）
- **最大传播步数** - 设置传播过程的最大迭代次数

#### 🕸️ 网络拓扑可视化
- **实时网络图** - 使用vis.js展示智能体网络拓扑
- **节点信息** - 鼠标悬停显示智能体详细信息
- **连接关系** - 显示智能体之间的连接强度
- **交互操作** - 支持拖拽、缩放、点击等交互

#### 📈 传播过程分析
- **实时图表** - 使用Chart.js展示传播过程曲线
- **传播摘要** - 显示传播结果的统计信息
- **步进分析** - 每一步传播的详细数据

#### 📝 实时传播日志
- **事件记录** - 实时显示传播过程中的关键事件
- **时间戳** - 每条日志的精确时间记录
- **日志级别** - 信息、警告、错误等不同级别

#### 📚 历史会话管理
- **会话列表** - 显示所有历史传播模拟会话
- **详细信息** - 每个会话的参数、结果、时间等
- **操作功能** - 查看、删除、导出等操作

### 2. 影响力最大化功能

#### 🔍 算法选择
- **贪心算法** - 基于边际收益最大化的经典算法
- **度数启发式** - 选择度数最高的节点作为种子
- **CELF算法** - Cost-Effective Lazy Forward算法优化

#### 📊 分析结果
- **最优种子集合** - 算法推荐的最优种子智能体
- **预期影响力** - 种子集合的预期影响范围
- **计算时间** - 算法执行时间统计
- **网络统计** - 分析时的网络状态信息

## 🔧 API接口文档

### 系统状态API

#### 获取系统统计信息
```http
GET /api/stats/system
```

**响应示例：**
```json
{
  "agents": {
    "total_agents": 3,
    "active_agents": 2,
    "inactive_agents": 1,
    "agent_types": {
      "social": 1,
      "content": 1,
      "hybrid": 1
    }
  },
  "social_network": {
    "total_connections": 3,
    "network_density": 0.5,
    "average_strength": 0.6
  },
  "reputation_system": {
    "total_agents": 3,
    "active_agents": 2,
    "average_reputation": 75.6,
    "reputation_distribution": {
      "high": 1,
      "medium": 2,
      "low": 0
    }
  }
}
```

#### 获取智能体列表
```http
GET /api/agents
```

**响应示例：**
```json
[
  {
    "id": "agent_1",
    "name": "Social Agent Alpha",
    "type": "social",
    "status": "active",
    "reputation_score": 85.5,
    "description": "Active social interaction agent",
    "created_at": "2025-09-03T10:25:42.962274",
    "updated_at": "2025-10-03T08:25:42.962433",
    "last_active": "2025-10-03T10:10:42.962434"
  }
]
```

### 网络分析API

#### 获取网络指标
```http
GET /api/network/metrics
```

**响应示例：**
```json
{
  "node_count": 3,
  "edge_count": 2,
  "average_degree": 1.33,
  "clustering_coefficient": 0.3,
  "average_path_length": 3.5,
  "network_density": 0.67,
  "connected_components": 1,
  "largest_component_size": 3
}
```

### 传播模拟API

#### 启动传播模拟
```http
POST /api/propagation/start
```

**请求体：**
```json
{
  "message": "测试传播消息",
  "seed_agents": ["agent_1", "agent_2"],
  "model_type": "viral",
  "parameters": {
    "infection_probability": 0.2
  },
  "max_steps": 50
}
```

**响应示例：**
```json
{
  "session_id": "prop_session_12345",
  "status": "completed",
  "message": "测试传播消息",
  "model_type": "viral",
  "seed_agents": ["agent_1", "agent_2"],
  "influenced_agents": ["agent_1", "agent_2", "agent_3"],
  "propagation_steps": 5,
  "statistics": {
    "total_influenced": 3,
    "seed_count": 2,
    "propagation_ratio": 1.0,
    "propagation_steps": 5,
    "model_parameters": {
      "infection_probability": 0.2
    }
  },
  "created_at": "2025-10-03T10:30:00.000000"
}
```

#### 获取传播会话详情
```http
GET /api/propagation/session/{session_id}
```

#### 获取活跃传播会话
```http
GET /api/propagation/sessions
```

### 影响力最大化API

#### 计算影响力最大化
```http
POST /api/influence-maximization
```

**请求体：**
```json
{
  "seed_count": 2,
  "algorithm": "greedy",
  "model_parameters": {
    "infection_probability": 0.1
  }
}
```

**响应示例：**
```json
{
  "optimal_seeds": ["agent_1", "agent_3"],
  "expected_influence": 4,
  "algorithm_used": "greedy",
  "computation_time": 0.05,
  "network_stats": {
    "node_count": 5,
    "edge_count": 6,
    "average_degree": 2.4
  }
}
```

## 🎮 使用指南

### 基础操作流程

1. **启动系统**
   ```bash
   python -m uvicorn src.web_interface.api.app:create_app --factory --host 0.0.0.0 --port 8000 --reload
   ```

2. **访问界面**
   - 打开浏览器访问 http://localhost:8000

3. **查看系统状态**
   - 观察顶部的关键指标卡片
   - 查看网络拓扑可视化
   - 检查智能体列表和状态

4. **进行传播模拟**
   - 在控制面板中输入传播消息
   - 选择种子智能体数量
   - 选择传播模型和参数
   - 点击"开始传播模拟"

5. **分析传播结果**
   - 查看传播过程图表
   - 阅读实时传播日志
   - 分析传播统计数据

6. **影响力最大化**
   - 点击"计算最优种子"
   - 选择算法类型
   - 查看推荐结果

### 高级功能

#### 自定义网络拓扑
- 系统支持动态加载智能体数据
- 网络连接关系可以实时更新
- 支持多种网络结构分析

#### 批量传播实验
- 可以进行多次传播模拟
- 比较不同参数的传播效果
- 导出实验数据进行分析

#### 实时监控
- WebSocket连接提供实时更新
- 系统状态变化即时反映
- 支持长时间运行监控

## 🧪 测试验证

### 功能测试
系统包含完整的测试套件，确保功能可靠性：

```bash
# 运行Web API测试
python -m pytest tests/web_interface/test_message_propagation_api.py -v

# 测试覆盖率
python -m pytest tests/web_interface/ --cov=src.web_interface --cov-report=html
```

### 测试结果
- ✅ **25/25** 测试通过 (100%通过率)
- ✅ 传播模拟API测试
- ✅ 影响力最大化API测试
- ✅ 网络指标API测试
- ✅ 数据验证测试
- ✅ 集成测试

## 🛠️ 技术架构

### 前端技术栈
- **HTML5/CSS3** - 现代网页结构和样式
- **Tailwind CSS** - 实用优先的CSS框架
- **JavaScript ES6+** - 现代JavaScript特性
- **Chart.js** - 数据可视化图表库
- **vis.js** - 网络拓扑可视化库
- **Font Awesome** - 图标库

### 后端技术栈
- **FastAPI** - 现代高性能Web框架
- **Pydantic** - 数据验证和序列化
- **Uvicorn** - ASGI服务器
- **WebSockets** - 实时双向通信

### 消息传播系统
- **病毒式传播模型** - SIR模型变体
- **信息扩散模型** - Independent Cascade模型
- **影响力最大化算法** - 贪心、度数启发式、CELF
- **社交网络分析** - 网络指标计算

## 📊 性能特性

- **实时响应** - API响应时间 < 100ms
- **并发支持** - 支持多个用户同时使用
- **数据可视化** - 流畅的图表和网络渲染
- **内存优化** - 高效的数据结构管理
- **可扩展性** - 支持大规模网络模拟

## 🔒 安全特性

- **输入验证** - 严格的数据验证机制
- **错误处理** - 完善的异常处理
- **API限制** - 防止恶意请求
- **数据隔离** - 模拟数据与生产数据分离

## 🚨 故障排除

### 常见问题

#### 1. 页面显示数据为0
**问题：** 指标卡片显示0值
**解决：** 检查API响应数据格式，确保字段映射正确

#### 2. 网络图不显示
**问题：** 网络拓扑可视化区域空白
**解决：** 检查vis.js库是否正确加载，确认智能体数据存在

#### 3. 传播模拟失败
**问题：** 点击传播按钮没有反应
**解决：** 检查种子智能体ID是否正确，确认传播模型参数有效

#### 4. WebSocket连接问题
**问题：** 实时更新不工作
**解决：** 检查浏览器是否支持WebSocket，确认服务器WebSocket服务正常

### 调试方法

1. **浏览器开发者工具**
   - 查看Network标签页检查API请求
   - 查看Console标签页检查JavaScript错误

2. **服务器日志**
   ```bash
   # 查看服务器输出日志
   # API请求和响应会在终端显示
   ```

3. **API测试**
   ```bash
   # 直接测试API端点
   curl http://localhost:8000/api/stats/system
   curl http://localhost:8000/api/agents
   ```

## 📈 未来改进计划

### 短期目标
- [ ] 添加更多传播模型支持
- [ ] 优化大数据量网络渲染性能
- [ ] 增加数据导出格式支持
- [ ] 完善用户界面交互体验

### 长期目标
- [ ] 支持多用户协作分析
- [ ] 集成机器学习预测模型
- [ ] 添加3D网络可视化
- [ ] 支持分布式计算集群

## 📞 支持与反馈

如果您在使用过程中遇到问题或有改进建议，请：

1. 检查本文档的故障排除部分
2. 查看API文档了解更多技术细节
3. 运行测试套件验证系统状态
4. 提供详细的错误信息和复现步骤

---

**文档版本：** 1.0.0
**最后更新：** 2025-10-03
**系统版本：** million-agents v1.0.0