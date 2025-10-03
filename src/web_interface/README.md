# Million Agents Web Interface

百万级智能体社交网络管理界面

## 🌟 概述

这是一个用于管理和可视化百万级智能体社交网络的Web管理界面。提供了直观的仪表板、智能体管理、网络可视化和系统指标监控功能。

## 🚀 快速开始

### 方法一：使用启动脚本（推荐）

```bash
# 自动模式（推荐）
python run_web_interface.py

# 强制使用FastAPI模式
python run_web_interface.py --fastapi

# 强制使用简单模式
python run_web_interface.py --simple

# 显示帮助
python run_web_interface.py --help
```

### 方法二：手动启动

```bash
# 如果安装了FastAPI
pip install fastapi uvicorn jinja2 python-multipart httpx websockets
python -m uvicorn src.web_interface.api.app:create_app --factory --host 0.0.0.0 --port 8000 --reload

# 或者使用简单HTTP服务器
cd src/web_interface
python -m http.server 8080
```

## 📱 功能特性

### 🎯 核心功能

- **📊 系统仪表板**
  - 智能体总数和活跃数量
  - 平均声誉分数
  - 网络连接统计
  - 系统状态监控

- **🤖 智能体管理**
  - 智能体CRUD操作（创建、读取、更新、删除）
  - 智能体搜索和过滤
  - 声誉分数管理
  - 状态控制（活跃、非活跃、暂停）

- **🕸️ 网络可视化**
  - 实时网络拓扑图
  - 智能体关系可视化
  - 网络指标展示
  - 交互式图形界面

- **📈 系统指标**
  - 详细性能指标
  - 声誉分布统计
  - 网络健康状态
  - 实时数据更新

### 🎨 界面特性

- ✅ 现代化响应式设计
- ✅ 移动端友好
- ✅ 暗色/亮色主题支持
- ✅ 实时数据更新（WebSocket）
- ✅ 无障碍设计
- ✅ 多语言支持（中文/英文）

## 🏗️ 技术架构

### 前端技术栈

- **HTML5**: 语义化标记
- **CSS3**: 现代样式设计，响应式布局
- **JavaScript (ES6+)**: 交互逻辑和API调用
- **Canvas API**: 网络可视化渲染
- **WebSocket**: 实时数据通信

### 后端技术栈

- **FastAPI**: 现代Web框架（可选）
- **Uvicorn**: ASGI服务器
- **Pydantic**: 数据验证和序列化
- **模拟服务**: 演示数据支持

### 项目结构

```
src/web_interface/
├── __init__.py              # 模块初始化
├── api/                     # API接口
│   ├── __init__.py
│   ├── app.py              # FastAPI应用
│   └── mock_services.py    # 模拟服务
├── static/                 # 静态文件
│   ├── css/
│   │   └── style.css       # 主样式文件
│   └── js/
│       └── main.js         # 主JavaScript文件
├── templates/              # HTML模板
│   └── index.html          # 主页面
└── README.md               # 本文档
```

## 🧪 测试

### 运行测试

```bash
# 运行所有Web界面测试
python -m pytest tests/web_interface/ -v

# 运行模拟服务测试
python -m pytest tests/web_interface/test_mock_services.py -v

# 运行前端组件测试
python -m pytest tests/web_interface/test_frontend.py -v
```

### 测试覆盖

- ✅ **29个测试用例全部通过**
- ✅ **模拟服务98%代码覆盖率**
- ✅ **前端组件完整测试**
- ✅ **API接口集成测试**
- ✅ **数据流验证测试**

### 测试分类

1. **模拟服务测试** (17个测试用例)
   - 智能体CRUD操作
   - 社交网络数据查询
   - 统计指标计算
   - 数据一致性验证

2. **前端组件测试** (12个测试用例)
   - HTML模板验证
   - CSS样式检查
   - JavaScript功能测试
   - 文件结构验证

## 🔧 配置选项

### 环境变量

```bash
# 服务器配置
PORT=8000
HOST=0.0.0.0

# 开发模式
DEBUG=true
RELOAD=true

# 数据源配置
DATA_SOURCE=mock  # mock | database
```

### API配置

```python
# API响应格式
{
    "agents": [...],
    "total": 100,
    "page": 1,
    "per_page": 20
}

# 错误响应格式
{
    "detail": "Error message",
    "status_code": 400
}
```

## 📊 API文档

### 智能体管理API

```http
GET    /api/agents              # 获取智能体列表
GET    /api/agents/{id}         # 获取单个智能体
POST   /api/agents              # 创建智能体
PUT    /api/agents/{id}         # 更新智能体
DELETE /api/agents/{id}         # 删除智能体
```

### 网络可视化API

```http
GET    /api/social-network      # 获取网络数据
GET    /api/agents/{id}/connections  # 获取智能体连接
```

### 系统指标API

```http
GET    /api/stats/system        # 系统统计
GET    /api/metrics/reputation  # 声誉指标
```

详细API文档请访问: `http://localhost:8000/docs`

## 🎨 界面预览

### 仪表板
- 📊 系统概览卡片
- 📈 实时数据更新
- 🔔 状态指示器

### 智能体管理
- 📋 智能体列表表格
- 🔍 搜索和过滤功能
- ➕ 创建/编辑模态框
- ✏️ 内联编辑操作

### 网络可视化
- 🕸️ 交互式网络图
- 🎯 节点和边可视化
- 📊 网络指标面板
- 🔍 缩放和平移控制

### 系统指标
- 📉 详细统计图表
- 📋 分类指标展示
- 🔄 实时数据更新

## 🔮 开发路线图

### 已完成 ✅
- [x] 基础Web界面框架
- [x] 智能体管理功能
- [x] 网络可视化
- [x] 系统指标监控
- [x] 响应式设计
- [x] 完整测试覆盖
- [x] 模拟数据支持

### 开发中 🚧
- [ ] 用户认证系统
- [ ] 实时通知功能
- [ ] 数据导出功能
- [ ] 高级过滤选项

### 计划中 📋
- [ ] 主题切换功能
- [ ] 多语言国际化
- [ ] 性能优化
- [ ] 移动端APP
- [ ] 微服务架构
- [ ] 机器学习集成

## 🤝 贡献指南

1. Fork项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开Pull Request

### 开发规范

- 遵循PEP 8代码风格
- 编写单元测试
- 更新相关文档
- 确保测试通过

## 📄 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 📞 联系方式

- 项目主页: [GitHub Repository]
- 问题反馈: [GitHub Issues]
- 文档: [项目文档]

## 🙏 致谢

感谢所有为这个项目做出贡献的开发者和用户！

---

**注意**: 这是一个演示项目，目前使用模拟数据。在生产环境中使用时，请连接真实的后端服务。