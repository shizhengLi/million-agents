/**
 * Million Agents Web Interface JavaScript
 * 百万级智能体社交网络管理界面脚本
 */

// 全局变量
let currentAgents = [];
let currentNetworkData = null;
let networkVisualization = null;

// DOM元素
const elements = {};

// 初始化
document.addEventListener('DOMContentLoaded', function() {
    initializeElements();
    initializeEventListeners();
    loadDashboard();
});

/**
 * 初始化DOM元素引用
 */
function initializeElements() {
    elements.agentTableBody = document.getElementById('agent-table-body');
    elements.totalAgentsSpan = document.getElementById('total-agents');
    elements.activeAgentsSpan = document.getElementById('active-agents');
    elements.avgReputationSpan = document.getElementById('avg-reputation');
    elements.loadingDiv = document.getElementById('loading');
    elements.errorDiv = document.getElementById('error-message');
    elements.successDiv = document.getElementById('success-message');
    elements.agentModal = document.getElementById('agent-modal');
    elements.networkCanvas = document.getElementById('network-canvas');
    elements.networkMetrics = document.getElementById('network-metrics');
}

/**
 * 初始化事件监听器
 */
function initializeEventListeners() {
    // 导航事件
    document.querySelectorAll('nav a').forEach(link => {
        link.addEventListener('click', handleNavigation);
    });

    // 智能体管理事件
    document.getElementById('create-agent-btn')?.addEventListener('click', showCreateAgentModal);
    document.getElementById('agent-form')?.addEventListener('submit', handleAgentFormSubmit);
    document.querySelector('.close')?.addEventListener('click', hideModal);

    // 搜索和过滤事件
    document.getElementById('search-input')?.addEventListener('input', handleSearch);
    document.getElementById('filter-select')?.addEventListener('change', handleFilter);

    // 网络可视化事件
    document.getElementById('refresh-network')?.addEventListener('click', loadNetworkVisualization);
    document.getElementById('reset-zoom')?.addEventListener('click', resetNetworkZoom);

    // 点击模态框外部关闭
    window.addEventListener('click', function(event) {
        if (event.target === elements.agentModal) {
            hideModal();
        }
    });
}

/**
 * 处理导航事件
 */
function handleNavigation(event) {
    event.preventDefault();
    const target = event.target.getAttribute('href');

    // 更新导航状态
    document.querySelectorAll('nav a').forEach(link => {
        link.classList.remove('active');
    });
    event.target.classList.add('active');

    // 显示对应内容
    showSection(target);
}

/**
 * 显示指定部分
 */
function showSection(section) {
    // 隐藏所有部分
    document.querySelectorAll('main > section').forEach(sec => {
        sec.style.display = 'none';
    });

    // 显示目标部分
    const targetSection = document.querySelector(section) || document.getElementById('dashboard');
    if (targetSection) {
        targetSection.style.display = 'block';
    }

    // 加载对应数据
    switch(section) {
        case '#agents':
            loadAgents();
            break;
        case '#network':
            loadNetworkVisualization();
            break;
        case '#metrics':
            loadMetrics();
            break;
        default:
            loadDashboard();
    }
}

/**
 * 加载仪表板数据
 */
async function loadDashboard() {
    try {
        showLoading();

        const [agentsStats, networkStats, reputationMetrics] = await Promise.all([
            fetchAPI('/api/stats/system'),
            fetchAPI('/api/metrics/reputation'),
            fetchAPI('/api/social-network')
        ]);

        updateDashboardStats(agentsStats, reputationMetrics, networkStats);
        hideLoading();

    } catch (error) {
        showError('加载仪表板数据失败: ' + error.message);
        hideLoading();
    }
}

/**
 * 更新仪表板统计
 */
function updateDashboardStats(agentStats, reputationMetrics, networkStats) {
    if (elements.totalAgentsSpan) {
        elements.totalAgentsSpan.textContent = agentStats?.agents?.total_agents || 0;
    }
    if (elements.activeAgentsSpan) {
        elements.activeAgentsSpan.textContent = agentStats?.agents?.active_agents || 0;
    }
    if (elements.avgReputationSpan) {
        elements.avgReputationSpan.textContent = reputationMetrics?.average_reputation?.toFixed(1) || '0.0';
    }
}

/**
 * 加载智能体列表
 */
async function loadAgents() {
    try {
        showLoading();

        const agents = await fetchAPI('/api/agents');
        currentAgents = agents;

        renderAgentTable(agents);
        hideLoading();

    } catch (error) {
        showError('加载智能体列表失败: ' + error.message);
        hideLoading();
    }
}

/**
 * 渲染智能体表格
 */
function renderAgentTable(agents) {
    if (!elements.agentTableBody) return;

    elements.agentTableBody.innerHTML = '';

    if (agents.length === 0) {
        elements.agentTableBody.innerHTML = `
            <tr>
                <td colspan="6" class="text-center">暂无智能体数据</td>
            </tr>
        `;
        return;
    }

    agents.forEach(agent => {
        const row = createAgentRow(agent);
        elements.agentTableBody.appendChild(row);
    });
}

/**
 * 创建智能体表格行
 */
function createAgentRow(agent) {
    const row = document.createElement('tr');

    const statusClass = `status-${agent.status}`;
    const reputationClass = getReputationClass(agent.reputation_score);

    row.innerHTML = `
        <td>${agent.id}</td>
        <td>
            <strong>${agent.name}</strong>
            ${agent.description ? `<br><small class="text-muted">${agent.description}</small>` : ''}
        </td>
        <td><span class="status-badge ${statusClass}">${agent.status}</span></td>
        <td><span class="reputation-score ${reputationClass}">${agent.reputation_score.toFixed(1)}</span></td>
        <td>${agent.type}</td>
        <td>
            <button class="btn btn-sm btn-secondary" onclick="editAgent('${agent.id}')">编辑</button>
            <button class="btn btn-sm btn-danger" onclick="deleteAgent('${agent.id}')">删除</button>
        </td>
    `;

    return row;
}

/**
 * 获取声誉分数样式类
 */
function getReputationClass(score) {
    if (score >= 80) return 'reputation-high';
    if (score >= 60) return 'reputation-medium';
    return 'reputation-low';
}

/**
 * 显示创建智能体模态框
 */
function showCreateAgentModal() {
    const modal = document.getElementById('agent-modal');
    const form = document.getElementById('agent-form');
    const title = document.getElementById('modal-title');

    if (title) title.textContent = '创建新智能体';
    if (form) form.reset();
    if (form) form.setAttribute('data-agent-id', '');

    if (modal) modal.style.display = 'block';
}

/**
 * 编辑智能体
 */
async function editAgent(agentId) {
    try {
        const agent = await fetchAPI(`/api/agents/${agentId}`);

        const modal = document.getElementById('agent-modal');
        const form = document.getElementById('agent-form');
        const title = document.getElementById('modal-title');

        if (title) title.textContent = '编辑智能体';
        if (form) {
            form.setAttribute('data-agent-id', agentId);
            form.getElementById('agent-name').value = agent.name;
            form.getElementById('agent-type').value = agent.type;
            form.getElementById('agent-description').value = agent.description || '';
            form.getElementById('agent-status').value = agent.status;
        }

        if (modal) modal.style.display = 'block';

    } catch (error) {
        showError('获取智能体信息失败: ' + error.message);
    }
}

/**
 * 删除智能体
 */
async function deleteAgent(agentId) {
    if (!confirm('确定要删除这个智能体吗？此操作不可撤销。')) {
        return;
    }

    try {
        await fetchAPI(`/api/agents/${agentId}`, { method: 'DELETE' });
        showSuccess('智能体删除成功');
        await loadAgents(); // 重新加载列表

    } catch (error) {
        showError('删除智能体失败: ' + error.message);
    }
}

/**
 * 处理智能体表单提交
 */
async function handleAgentFormSubmit(event) {
    event.preventDefault();

    const form = event.target;
    const agentId = form.getAttribute('data-agent-id');
    const isEdit = agentId && agentId !== '';

    const formData = {
        name: form.getElementById('agent-name').value,
        type: form.getElementById('agent-type').value,
        description: form.getElementById('agent-description').value,
        status: form.getElementById('agent-status').value
    };

    try {
        if (isEdit) {
            await fetchAPI(`/api/agents/${agentId}`, {
                method: 'PUT',
                body: JSON.stringify(formData)
            });
            showSuccess('智能体更新成功');
        } else {
            const newAgent = await fetchAPI('/api/agents', {
                method: 'POST',
                body: JSON.stringify(formData)
            });
            showSuccess('智能体创建成功');
        }

        hideModal();
        await loadAgents(); // 重新加载列表

    } catch (error) {
        showError(`${isEdit ? '更新' : '创建'}智能体失败: ` + error.message);
    }
}

/**
 * 隐藏模态框
 */
function hideModal() {
    const modal = document.getElementById('agent-modal');
    if (modal) modal.style.display = 'none';
}

/**
 * 处理搜索
 */
function handleSearch(event) {
    const searchTerm = event.target.value.toLowerCase();
    const filteredAgents = currentAgents.filter(agent =>
        agent.name.toLowerCase().includes(searchTerm) ||
        agent.id.toLowerCase().includes(searchTerm) ||
        (agent.description && agent.description.toLowerCase().includes(searchTerm))
    );
    renderAgentTable(filteredAgents);
}

/**
 * 处理过滤
 */
function handleFilter(event) {
    const filterValue = event.target.value;
    let filteredAgents = currentAgents;

    switch(filterValue) {
        case 'active':
            filteredAgents = currentAgents.filter(agent => agent.status === 'active');
            break;
        case 'inactive':
            filteredAgents = currentAgents.filter(agent => agent.status === 'inactive');
            break;
        case 'high-reputation':
            filteredAgents = currentAgents.filter(agent => agent.reputation_score >= 80);
            break;
        case 'social':
            filteredAgents = currentAgents.filter(agent => agent.type === 'social');
            break;
        case 'content':
            filteredAgents = currentAgents.filter(agent => agent.type === 'content');
            break;
        case 'hybrid':
            filteredAgents = currentAgents.filter(agent => agent.type === 'hybrid');
            break;
    }

    renderAgentTable(filteredAgents);
}

/**
 * 加载网络可视化
 */
async function loadNetworkVisualization() {
    try {
        showLoading();

        const networkData = await fetchAPI('/api/social-network?limit=50');
        currentNetworkData = networkData;

        renderNetworkVisualization(networkData);
        updateNetworkMetrics(networkData.metrics);
        hideLoading();

    } catch (error) {
        showError('加载网络可视化失败: ' + error.message);
        hideLoading();
    }
}

/**
 * 渲染网络可视化
 */
function renderNetworkVisualization(networkData) {
    if (!elements.networkCanvas) return;

    const canvas = elements.networkCanvas;
    const ctx = canvas.getContext('2d');

    // 设置画布大小
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;

    // 清空画布
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 简单的网络可视化实现
    const nodes = networkData.nodes || [];
    const edges = networkData.edges || [];

    // 计算节点位置
    const positions = calculateNodePositions(nodes, canvas.width, canvas.height);

    // 绘制边
    ctx.strokeStyle = '#ddd';
    ctx.lineWidth = 1;
    edges.forEach(edge => {
        const sourcePos = positions[edge.source];
        const targetPos = positions[edge.target];

        if (sourcePos && targetPos) {
            ctx.beginPath();
            ctx.moveTo(sourcePos.x, sourcePos.y);
            ctx.lineTo(targetPos.x, targetPos.y);
            ctx.stroke();
        }
    });

    // 绘制节点
    nodes.forEach(node => {
        const pos = positions[node.id];
        if (!pos) return;

        // 节点颜色根据类型
        const colors = {
            social: '#667eea',
            content: '#28a745',
            hybrid: '#ffc107'
        };
        const color = colors[node.type] || '#6c757d';

        // 绘制节点圆圈
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, 15, 0, 2 * Math.PI);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.stroke();

        // 绘制节点标签
        ctx.fillStyle = '#333';
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(node.name, pos.x, pos.y + 30);
    });
}

/**
 * 计算节点位置
 */
function calculateNodePositions(nodes, width, height) {
    const positions = {};
    const centerX = width / 2;
    const centerY = height / 2;
    const radius = Math.min(width, height) * 0.3;

    nodes.forEach((node, index) => {
        const angle = (2 * Math.PI * index) / nodes.length;
        positions[node.id] = {
            x: centerX + radius * Math.cos(angle),
            y: centerY + radius * Math.sin(angle)
        };
    });

    return positions;
}

/**
 * 更新网络指标
 */
function updateNetworkMetrics(metrics) {
    if (!elements.networkMetrics) return;

    elements.networkMetrics.innerHTML = `
        <div class="network-metric">
            <h4>总节点数</h4>
            <div class="value">${metrics.total_nodes || 0}</div>
        </div>
        <div class="network-metric">
            <h4>总连接数</h4>
            <div class="value">${metrics.total_edges || 0}</div>
        </div>
        <div class="network-metric">
            <h4>网络密度</h4>
            <div class="value">${(metrics.density || 0).toFixed(3)}</div>
        </div>
        <div class="network-metric">
            <h4>聚类系数</h4>
            <div class="value">${(metrics.clustering_coefficient || 0).toFixed(3)}</div>
        </div>
    `;
}

/**
 * 重置网络缩放
 */
function resetNetworkZoom() {
    if (currentNetworkData) {
        renderNetworkVisualization(currentNetworkData);
    }
}

/**
 * 加载指标数据
 */
async function loadMetrics() {
    try {
        showLoading();

        const [systemStats, reputationMetrics] = await Promise.all([
            fetchAPI('/api/stats/system'),
            fetchAPI('/api/metrics/reputation')
        ]);

        renderDetailedMetrics(systemStats, reputationMetrics);
        hideLoading();

    } catch (error) {
        showError('加载指标数据失败: ' + error.message);
        hideLoading();
    }
}

/**
 * 渲染详细指标
 */
function renderDetailedMetrics(systemStats, reputationMetrics) {
    const metricsContainer = document.getElementById('detailed-metrics');
    if (!metricsContainer) return;

    metricsContainer.innerHTML = `
        <div class="metrics-grid">
            <div class="metric-section">
                <h3>智能体统计</h3>
                <div class="metric-item">
                    <span class="metric-label">总智能体数:</span>
                    <span class="metric-value">${systemStats?.agents?.total_agents || 0}</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">活跃智能体:</span>
                    <span class="metric-value">${systemStats?.agents?.active_agents || 0}</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">非活跃智能体:</span>
                    <span class="metric-value">${systemStats?.agents?.inactive_agents || 0}</span>
                </div>
            </div>

            <div class="metric-section">
                <h3>声誉指标</h3>
                <div class="metric-item">
                    <span class="metric-label">平均声誉分数:</span>
                    <span class="metric-value">${reputationMetrics?.average_reputation?.toFixed(1) || '0.0'}</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">高声誉智能体:</span>
                    <span class="metric-value">${reputationMetrics?.reputation_distribution?.high || 0}</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">中等声誉智能体:</span>
                    <span class="metric-value">${reputationMetrics?.reputation_distribution?.medium || 0}</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">低声誉智能体:</span>
                    <span class="metric-value">${reputationMetrics?.reputation_distribution?.low || 0}</span>
                </div>
            </div>

            <div class="metric-section">
                <h3>社交网络统计</h3>
                <div class="metric-item">
                    <span class="metric-label">总连接数:</span>
                    <span class="metric-value">${systemStats?.social_network?.total_connections || 0}</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">网络密度:</span>
                    <span class="metric-value">${(systemStats?.social_network?.network_density || 0).toFixed(3)}</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">平均连接强度:</span>
                    <span class="metric-value">${(systemStats?.social_network?.average_strength || 0).toFixed(3)}</span>
                </div>
            </div>
        </div>
    `;
}

/**
 * API请求封装
 */
async function fetchAPI(url, options = {}) {
    const defaultOptions = {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
        },
    };

    const finalOptions = { ...defaultOptions, ...options };

    if (finalOptions.body && typeof finalOptions.body === 'string') {
        finalOptions.body = finalOptions.body;
    }

    const response = await fetch(url, finalOptions);

    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
}

/**
 * 显示加载状态
 */
function showLoading() {
    if (elements.loadingDiv) {
        elements.loadingDiv.style.display = 'block';
    }
}

/**
 * 隐藏加载状态
 */
function hideLoading() {
    if (elements.loadingDiv) {
        elements.loadingDiv.style.display = 'none';
    }
}

/**
 * 显示错误消息
 */
function showError(message) {
    if (elements.errorDiv) {
        elements.errorDiv.textContent = message;
        elements.errorDiv.style.display = 'block';
        setTimeout(() => {
            elements.errorDiv.style.display = 'none';
        }, 5000);
    }
}

/**
 * 显示成功消息
 */
function showSuccess(message) {
    if (elements.successDiv) {
        elements.successDiv.textContent = message;
        elements.successDiv.style.display = 'block';
        setTimeout(() => {
            elements.successDiv.style.display = 'none';
        }, 3000);
    }
}

/**
 * WebSocket连接（用于实时更新）
 */
function initializeWebSocket() {
    if (typeof WebSocket === 'undefined') {
        console.warn('WebSocket not supported');
        return;
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    const ws = new WebSocket(wsUrl);

    ws.onopen = function(event) {
        console.log('WebSocket connected');
        ws.send('subscribe:agents');
        ws.send('subscribe:network');
    };

    ws.onmessage = function(event) {
        const data = JSON.parse(event.data);
        handleWebSocketMessage(data);
    };

    ws.onerror = function(error) {
        console.error('WebSocket error:', error);
    };

    ws.onclose = function(event) {
        console.log('WebSocket disconnected');
        // 尝试重连
        setTimeout(initializeWebSocket, 5000);
    };
}

/**
 * 处理WebSocket消息
 */
function handleWebSocketMessage(data) {
    switch(data.type) {
        case 'agent_update':
            if (document.getElementById('agents').style.display !== 'none') {
                loadAgents();
            }
            break;
        case 'network_update':
            if (document.getElementById('network').style.display !== 'none') {
                loadNetworkVisualization();
            }
            break;
        case 'metrics_update':
            if (document.getElementById('dashboard').style.display !== 'none') {
                loadDashboard();
            }
            break;
    }
}

// 导出全局函数供HTML调用
window.editAgent = editAgent;
window.deleteAgent = deleteAgent;
window.resetNetworkZoom = resetNetworkZoom;