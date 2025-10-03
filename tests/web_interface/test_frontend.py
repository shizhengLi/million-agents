"""
Frontend Tests
前端界面组件测试
"""

import pytest
import os
from unittest.mock import Mock, patch


class TestFrontendComponents:
    """前端组件测试"""

    def test_html_template_exists(self):
        """测试HTML模板文件是否存在"""
        template_path = "src/web_interface/templates/index.html"
        assert os.path.exists(template_path), "HTML模板文件不存在"

    def test_css_file_exists(self):
        """测试CSS文件是否存在"""
        css_path = "src/web_interface/static/css/style.css"
        assert os.path.exists(css_path), "CSS文件不存在"

    def test_js_file_exists(self):
        """测试JavaScript文件是否存在"""
        js_path = "src/web_interface/static/js/main.js"
        assert os.path.exists(js_path), "JavaScript文件不存在"

    def test_html_template_content(self):
        """测试HTML模板内容"""
        template_path = "src/web_interface/templates/index.html"
        with open(template_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 验证关键HTML元素
        assert '<!DOCTYPE html>' in content, "缺少DOCTYPE声明"
        assert '<title>Million Agents Web Interface</title>' in content, "缺少页面标题"
        assert '<meta charset="UTF-8">' in content, "缺少字符集声明"
        assert '<meta name="viewport"' in content, "缺少响应式视口设置"
        assert '<link rel="stylesheet" href="/static/css/style.css">' in content, "缺少CSS引用"
        assert '<script src="/static/js/main.js">' in content, "缺少JavaScript引用"

        # 验证导航结构
        assert '<nav>' in content, "缺少导航元素"
        assert '<a href="#dashboard"' in content, "缺少仪表板链接"
        assert '<a href="#agents"' in content, "缺少智能体管理链接"
        assert '<a href="#network"' in content, "缺少网络可视化链接"
        assert '<a href="#metrics"' in content, "缺少系统指标链接"

        # 验证主要部分（允许有style属性）
        assert 'id="dashboard"' in content, "缺少仪表板部分"
        assert 'id="agents"' in content, "缺少智能体管理部分"
        assert 'id="network"' in content, "缺少网络可视化部分"
        assert 'id="metrics"' in content, "缺少系统指标部分"

        # 验证智能体表格
        assert '<table class="agent-table">' in content, "缺少智能体表格"
        assert '<tbody id="agent-table-body">' in content, "缺少表格体"

        # 验证模态框
        assert '<div id="agent-modal" class="modal">' in content, "缺少智能体模态框"
        assert '<form id="agent-form">' in content, "缺少智能体表单"

        # 验证网络可视化
        assert '<canvas id="network-canvas">' in content, "缺少网络画布"
        assert 'id="network-metrics"' in content, "缺少网络指标容器"

    def test_css_content(self):
        """测试CSS内容"""
        css_path = "src/web_interface/static/css/style.css"
        with open(css_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 验证关键CSS类
        assert '.container' in content, "缺少容器样式"
        assert '.dashboard' in content, "缺少仪表板样式"
        assert '.stat-card' in content, "缺少统计卡片样式"
        assert '.agent-table' in content, "缺少智能体表格样式"
        assert '.modal' in content, "缺少模态框样式"
        assert '.network-container' in content, "缺少网络容器样式"

        # 验证响应式设计
        assert '@media' in content, "缺少响应式媒体查询"
        assert 'max-width' in content, "缺少断点设置"

        # 验证状态样式
        assert '.status-active' in content, "缺少活跃状态样式"
        assert '.status-inactive' in content, "缺少非活跃状态样式"
        assert '.reputation-high' in content, "缺少高声誉样式"

    def test_javascript_content(self):
        """测试JavaScript内容"""
        js_path = "src/web_interface/static/js/main.js"
        with open(js_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 验证关键函数
        assert 'function loadDashboard' in content, "缺少加载仪表板函数"
        assert 'function loadAgents' in content, "缺少加载智能体函数"
        assert 'function renderAgentTable' in content, "缺少渲染智能体表格函数"
        assert 'function loadNetworkVisualization' in content, "缺少加载网络可视化函数"
        assert 'async function fetchAPI' in content, "缺少API请求函数"

        # 验证事件处理
        assert 'addEventListener' in content, "缺少事件监听器"
        assert 'handleNavigation' in content, "缺少导航处理函数"
        assert 'handleAgentFormSubmit' in content, "缺少表单提交处理函数"

        # 验证WebSocket支持
        assert 'WebSocket' in content, "缺少WebSocket支持"
        assert 'initializeWebSocket' in content, "缺少WebSocket初始化函数"

        # 验证错误处理
        assert 'showError' in content, "缺少错误显示函数"
        assert 'showSuccess' in content, "缺少成功显示函数"
        assert 'showLoading' in content, "缺少加载状态函数"

        # 验证DOM操作
        assert 'getElementById' in content, "缺少DOM元素获取"
        assert 'createElement' in content, "缺少DOM元素创建"

    def test_mock_services_integration(self):
        """测试模拟服务集成"""
        from src.web_interface.api.mock_services import MockAgentService, MockSocialNetworkService

        # 验证服务实例化
        agent_service = MockAgentService()
        network_service = MockSocialNetworkService()

        assert agent_service is not None, "智能体服务实例化失败"
        assert network_service is not None, "网络服务实例化失败"

        # 验证初始数据
        assert len(agent_service.agents) >= 3, "智能体服务初始数据不足"
        assert len(network_service.connections) >= 3, "网络服务初始数据不足"

    @pytest.mark.asyncio
    async def test_frontend_data_flow(self):
        """测试前端数据流"""
        from src.web_interface.api.mock_services import MockAgentService, MockSocialNetworkService

        agent_service = MockAgentService()
        network_service = MockSocialNetworkService()

        # 测试智能体数据获取
        agents = await agent_service.get_all_agents()
        assert len(agents) > 0, "智能体数据为空"
        assert all('id' in agent for agent in agents), "智能体数据缺少ID字段"
        assert all('name' in agent for agent in agents), "智能体数据缺少名称字段"

        # 测试网络数据获取
        network_data = await network_service.get_network_data()
        assert 'nodes' in network_data, "网络数据缺少节点"
        assert 'edges' in network_data, "网络数据缺少边"
        assert 'metrics' in network_data, "网络数据缺少指标"

    def test_file_structure(self):
        """测试文件结构"""
        required_files = [
            "src/web_interface/__init__.py",
            "src/web_interface/api/__init__.py",
            "src/web_interface/api/mock_services.py",
            "src/web_interface/templates/index.html",
            "src/web_interface/static/css/style.css",
            "src/web_interface/static/js/main.js"
        ]

        for file_path in required_files:
            assert os.path.exists(file_path), f"必需文件不存在: {file_path}"

    def test_static_file_accessibility(self):
        """测试静态文件可访问性"""
        css_path = "src/web_interface/static/css/style.css"
        js_path = "src/web_interface/static/js/main.js"

        # 验证文件大小合理
        assert os.path.getsize(css_path) > 1000, "CSS文件太小，可能内容不完整"
        assert os.path.getsize(js_path) > 2000, "JavaScript文件太小，可能内容不完整"

    def test_html_semantics(self):
        """测试HTML语义化"""
        template_path = "src/web_interface/templates/index.html"
        with open(template_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 验证语义化标签 - 使用更宽松的匹配
        assert '<header' in content, "缺少header标签"
        assert '<main' in content, "缺少main标签"
        assert '<section' in content, "缺少section标签"
        assert '<nav' in content, "缺少nav标签"
        assert '<h1' in content, "缺少h1标签"
        assert '<h2' in content, "缺少h2标签"
        assert '<h3' in content, "缺少h3标签"
        assert '<p' in content, "缺少p标签"
        assert '<ul' in content, "缺少ul标签"
        assert '<li' in content, "缺少li标签"

        # 验证无障碍属性
        assert 'meta name="description"' in content, "缺少页面描述"
        assert 'target="_blank"' in content, "缺少外部链接标识"

    def test_responsive_design_elements(self):
        """测试响应式设计元素"""
        css_path = "src/web_interface/static/css/style.css"
        with open(css_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 验证响应式设计特性
        responsive_features = [
            '@media', 'max-width', 'flex', 'grid',
            'display: none', 'text-align', 'padding'
        ]

        for feature in responsive_features:
            assert feature in content, f"缺少响应式设计特性: {feature}"