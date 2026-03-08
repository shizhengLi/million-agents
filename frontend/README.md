# 百万级智能体社交网络管理平台

## 项目简介

这是一个百万级智能体社交网络模拟平台，能够创建、管理和分析大规模智能体社区。项目具备从单个智能体交互到百万级群体模拟的完整能力。

## 技术栈

- **前端**: React 18 + TypeScript + Vite
- **UI框架**: Ant Design 5
- **图表**: Recharts
- **网络可视化**: vis-network
- **后端**: FastAPI (Python)
- **数据库**: SQLite

## 功能模块

### 1. 仪表板 (Dashboard)
- 智能体总数、活跃数统计
- 社交连接数、平均度数展示
- 智能体类型分布饼图
- 智能体状态分布柱状图

### 2. 传播模拟 (Propagation)
- 支持病毒式传播模型和信息扩散模型
- 可配置感染概率、恢复概率、最大传播步数
- 实时查看模拟结果和历史记录
- 种子智能体选择

### 3. 网络可视化 (Network)
- 交互式社交网络拓扑图
- 节点拖拽、缩放、聚焦
- 智能体搜索和定位
- 连接强度展示

### 4. 智能体管理 (Agents)
- 查看所有智能体列表
- 添加、编辑、删除智能体
- 智能体类型和状态标签
- 声誉分数排序

### 5. 影响力分析 (Influence)
- 影响力最大化计算
- 三种算法支持：
  - 贪心算法 (Greedy)
  - 度启发式 (Degree)
  - CELF算法

## 快速开始

### 前端启动

```bash
cd frontend
npm install
npm run dev
```

访问 http://localhost:3000

### 后端启动

```bash
# 确保已初始化数据库
python init_db.py

# 启动后端服务
python -m uvicorn src.web_interface.api.app:create_app --factory --host 0.0.0.0 --port 8000 --reload
```

## API端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/agents/` | GET | 获取所有智能体 |
| `/api/agents/` | POST | 创建智能体 |
| `/api/agents/stats` | GET | 获取智能体统计 |
| `/api/network/data` | GET | 获取网络数据 |
| `/api/propagation/start` | POST | 启动传播模拟 |
| `/api/influence/calculate` | POST | 计算影响力最大化 |

## 项目结构

```
million-agents/
├── frontend/                 # React前端
│   ├── src/
│   │   ├── components/       # 组件
│   │   ├── pages/           # 页面
│   │   ├── services/        # API服务
│   │   ├── types/           # 类型定义
│   │   └── App.tsx         # 主应用
│   └── package.json
├── src/
│   ├── agents/              # 智能体模块
│   ├── database/            # 数据库模型
│   ├── web_interface/       # Web API
│   ├── message_propagation/ # 传播模型
│   └── ...
├── init_db.py               # 数据库初始化脚本
└── README.md
```

## 使用说明

1. 启动后端API服务
2. 初始化数据库: `python init_db.py`
3. 启动前端: `cd frontend && npm run dev`
4. 访问 http://localhost:3000
5. 选择种子智能体并启动传播模拟

## 扩展

- 支持更多传播模型
- 添加实时WebSocket推送
- 实现更复杂的影响力算法
- 添加用户认证和权限管理
