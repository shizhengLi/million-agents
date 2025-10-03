/**
 * 多智能体消息传播仪表板前端脚本
 * 实现与后端API的交互和数据可视化
 */

// 全局变量
let network = null;
let propagationChart = null;
let currentSessionId = null;
let sessions = [];

// API基础URL
const API_BASE = '/api';

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    initializeDashboard();
    setupEventListeners();
    loadInitialData();
});

/**
 * 初始化仪表板
 */
function initializeDashboard() {
    initializeNetwork();
    initializePropagationChart();
    setupWebSocket();
}

/**
 * 设置事件监听器
 */
function setupEventListeners() {
    // 传播表单提交
    document.getElementById('propagation-form').addEventListener('submit', handlePropagationSubmit);

    // 感染概率滑块
    document.getElementById('infection-probability').addEventListener('input', function(e) {
        document.getElementById('infection-prob-value').textContent = e.target.value;
    });

    // 定期刷新数据
    setInterval(refreshDashboard, 30000); // 每30秒刷新一次
}

/**
 * 初始化网络可视化
 */
function initializeNetwork() {
    const container = document.getElementById('network-container');

    // 创建示例网络数据 - 使用实际存在的智能体ID
    const nodes = new vis.DataSet([
        {id: 'agent_1', label: 'Agent 1', color: '#667eea', size: 20},
        {id: 'agent_2', label: 'Agent 2', color: '#2ecc71', size: 15},
        {id: 'agent_3', label: 'Agent 3', color: '#e74c3c', size: 18}
    ]);

    const edges = new vis.DataSet([
        {from: 'agent_1', to: 'agent_2', width: 2},
        {from: 'agent_1', to: 'agent_3', width: 3},
        {from: 'agent_2', to: 'agent_3', width: 1}
    ]);

    const data = { nodes: nodes, edges: edges };

    const options = {
        layout: {
            improvedLayout: true,
            hierarchical: {
                enabled: false,
                sortMethod: 'directed'
            }
        },
        physics: {
            enabled: true,
            barnesHut: {
                gravitationalConstant: -2000,
                centralGravity: 0.3,
                springLength: 95,
                springConstant: 0.04,
                damping: 0.09,
                avoidOverlap: 0
            }
        },
        interaction: {
            hover: true,
            tooltipDelay: 200,
            hideEdgesOnDrag: true
        },
        nodes: {
            shape: 'dot',
            font: {
                size: 14,
                color: '#ffffff',
                strokeWidth: 3,
                strokeColor: '#000000'
            },
            borderWidth: 2,
            shadow: true
        },
        edges: {
            smooth: {
                type: 'continuous',
                roundness: 0.5
            },
            shadow: true
        }
    };

    network = new vis.Network(container, data, options);

    // 添加节点点击事件
    network.on('click', function(params) {
        if (params.nodes.length > 0) {
            const nodeId = params.nodes[0];
            showNodeDetails(nodeId);
        }
    });

    // 加载真实网络数据
    loadRealNetworkData();
}

/**
 * 初始化传播图表
 */
function initializePropagationChart() {
    const ctx = document.getElementById('propagation-chart').getContext('2d');

    propagationChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: '影响智能体数量',
                data: [],
                borderColor: '#3498db',
                backgroundColor: 'rgba(52, 152, 219, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            },
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: '传播步数'
                    }
                },
                y: {
                    display: true,
                    title: {
                        display: true,
                        text: '影响数量'
                    },
                    beginAtZero: true
                }
            }
        }
    });
}

/**
 * 设置WebSocket连接
 */
function setupWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    try {
        const ws = new WebSocket(wsUrl);

        ws.onopen = function(event) {
            console.log('WebSocket连接已建立');
            addLog('WebSocket连接已建立', 'success');
        };

        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            handleWebSocketMessage(data);
        };

        ws.onclose = function(event) {
            console.log('WebSocket连接已关闭');
            addLog('WebSocket连接已关闭', 'warning');
        };

        ws.onerror = function(error) {
            console.error('WebSocket错误:', error);
            addLog('WebSocket连接错误', 'error');
        };

    } catch (error) {
        console.error('WebSocket初始化失败:', error);
        addLog('WebSocket初始化失败', 'error');
    }
}

/**
 * 处理传播表单提交
 */
async function handlePropagationSubmit(event) {
    event.preventDefault();

    const formData = {
        message: document.getElementById('propagation-message').value,
        seed_agents: generateSeedAgents(parseInt(document.getElementById('seed-count').value)),
        model_type: document.getElementById('propagation-model').value,
        parameters: {
            infection_probability: parseFloat(document.getElementById('infection-probability').value)
        },
        max_steps: parseInt(document.getElementById('max-steps').value)
    };

    // 验证表单数据
    if (!formData.message.trim()) {
        showNotification('请输入传播消息', 'error');
        return;
    }

    showLoading(true);
    addLog(`开始传播模拟: ${formData.message}`, 'info');

    try {
        const response = await fetch(`${API_BASE}/propagation/start`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || '传播启动失败');
        }

        const result = await response.json();
        currentSessionId = result.session_id;

        addLog(`传播会话已启动: ${result.session_id}`, 'success');
        updatePropagationResult(result);
        updateVisualization(result);
        refreshSessions();

        showNotification('传播模拟启动成功！', 'success');

    } catch (error) {
        console.error('传播启动错误:', error);
        addLog(`传播启动失败: ${error.message}`, 'error');
        showNotification(`传播启动失败: ${error.message}`, 'error');
    } finally {
        showLoading(false);
    }
}

/**
 * 生成种子智能体列表
 */
function generateSeedAgents(count) {
    const agents = [];
    for (let i = 1; i <= count; i++) {
        agents.push(`agent_${i}`);
    }
    return agents;
}

/**
 * 更新传播结果
 */
function updatePropagationResult(result) {
    // 更新摘要信息
    document.getElementById('total-influenced').textContent = result.statistics.total_influenced;
    document.getElementById('propagation-steps').textContent = result.statistics.propagation_steps;
    document.getElementById('propagation-ratio').textContent =
        `${(result.statistics.propagation_ratio * 100).toFixed(1)}%`;

    document.getElementById('propagation-summary').classList.remove('hidden');

    // 更新传播图表
    updatePropagationChart(result);
}

/**
 * 更新传播图表
 */
function updatePropagationChart(result) {
    if (!propagationChart) return;

    // 模拟传播过程数据
    const steps = result.statistics.propagation_steps;
    const labels = Array.from({length: steps + 1}, (_, i) => i);

    // 生成模拟的传播数据
    const data = [];
    const totalInfluenced = result.statistics.total_influenced;
    const seedCount = result.statistics.seed_count;

    for (let i = 0; i <= steps; i++) {
        if (i === 0) {
            data.push(seedCount);
        } else {
            // 模拟指数增长，然后逐渐平缓
            const progress = i / steps;
            const influenced = Math.min(
                seedCount + (totalInfluenced - seedCount) * (1 - Math.exp(-3 * progress)),
                totalInfluenced
            );
            data.push(Math.round(influenced));
        }
    }

    propagationChart.data.labels = labels;
    propagationChart.data.datasets[0].data = data;
    propagationChart.update();
}

/**
 * 更新可视化
 */
function updateVisualization(result) {
    if (!network) return;

    // 更新网络节点颜色以显示传播状态
    const nodes = network.body.data.nodes;
    const edges = network.body.data.edges;

    // 重置所有节点颜色
    nodes.forEach(node => {
        if (result.seed_agents.includes(node.id)) {
            nodes.update({id: node.id, color: '#e74c3c'}); // 种子节点为红色
        } else if (result.influenced_agents.includes(node.id)) {
            nodes.update({id: node.id, color: '#2ecc71'}); // 受影响节点为绿色
        } else {
            nodes.update({id: node.id, color: '#95a5a6'}); // 未影响节点为灰色
        }
    });

    // 高亮传播路径
    result.influenced_agents.forEach(agentId => {
        // 找到与该智能体相关的边并高亮
        edges.forEach(edge => {
            if (edge.from === agentId || edge.to === agentId) {
                edges.update({id: edge.id, color: '#e74c3c', width: 3});
            }
        });
    });
}

/**
 * 计算影响力最大化
 */
async function calculateInfluenceMaximization() {
    showLoading(true);
    addLog('开始计算影响力最大化...', 'info');

    const requestData = {
        seed_count: parseInt(document.getElementById('seed-count').value),
        algorithm: 'greedy',
        model_parameters: {
            infection_probability: parseFloat(document.getElementById('infection-probability').value)
        }
    };

    try {
        const response = await fetch(`${API_BASE}/influence-maximization`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || '影响力最大化计算失败');
        }

        const result = await response.json();

        addLog(`影响力最大化计算完成`, 'success');
        addLog(`最优种子: ${result.optimal_seeds.join(', ')}`, 'info');
        addLog(`预期影响: ${result.expected_influence} 个智能体`, 'info');
        addLog(`计算时间: ${result.computation_time.toFixed(3)} 秒`, 'info');

        // 更新UI显示最优种子
        highlightOptimalSeeds(result.optimal_seeds);

        showNotification('影响力最大化计算完成！', 'success');

    } catch (error) {
        console.error('影响力最大化计算错误:', error);
        addLog(`影响力最大化计算失败: ${error.message}`, 'error');
        showNotification(`影响力最大化计算失败: ${error.message}`, 'error');
    } finally {
        showLoading(false);
    }
}

/**
 * 高亮最优种子节点
 */
function highlightOptimalSeeds(optimalSeeds) {
    if (!network) return;

    const nodes = network.body.data.nodes;

    nodes.forEach(node => {
        if (optimalSeeds.includes(node.id)) {
            nodes.update({
                id: node.id,
                color: '#f39c12', // 橙色表示最优种子
                size: 25 // 增大节点大小
            });
        } else {
            nodes.update({
                id: node.id,
                color: '#95a5a6', // 其他节点为灰色
                size: 15 // 恢复正常大小
            });
        }
    });
}

/**
 * 刷新仪表板数据
 */
async function refreshDashboard() {
    try {
        await Promise.all([
            loadSystemStats(),
            loadNetworkMetrics(),
            refreshSessions()
        ]);

        addLog('仪表板数据已刷新', 'info');
    } catch (error) {
        console.error('刷新仪表板失败:', error);
    }
}

/**
 * 加载初始数据
 */
async function loadInitialData() {
    showLoading(true);

    try {
        await Promise.all([
            loadSystemStats(),
            loadNetworkMetrics(),
            loadAgents(),
            refreshSessions()
        ]);

        addLog('初始数据加载完成', 'success');
    } catch (error) {
        console.error('加载初始数据失败:', error);
        addLog('初始数据加载失败', 'error');
    } finally {
        showLoading(false);
    }
}

/**
 * 加载系统统计
 */
async function loadSystemStats() {
    try {
        const response = await fetch(`${API_BASE}/stats/system`);
        if (!response.ok) throw new Error('获取系统统计失败');

        const stats = await response.json();

        // 更新指标卡片
        document.getElementById('active-agents-count').textContent =
            stats.agents.active_agents || '0';
        document.getElementById('connections-count').textContent =
            stats.social_network.total_connections || '0';
        document.getElementById('propagation-sessions-count').textContent =
            sessions.length;
        document.getElementById('avg-influence').textContent =
            (stats.reputation_system.average_reputation || 0).toFixed(2);

    } catch (error) {
        console.error('加载系统统计失败:', error);
    }
}

/**
 * 加载网络指标
 */
async function loadNetworkMetrics() {
    try {
        const response = await fetch(`${API_BASE}/network/metrics`);
        if (!response.ok) throw new Error('获取网络指标失败');

        const metrics = await response.json();

        // 更新网络统计显示
        document.getElementById('network-nodes').textContent = metrics.node_count;
        document.getElementById('network-edges').textContent = metrics.edge_count;
        document.getElementById('network-avg-degree').textContent =
            metrics.average_degree.toFixed(2);
        document.getElementById('network-density').textContent =
            (metrics.network_density * 100).toFixed(1) + '%';

    } catch (error) {
        console.error('加载网络指标失败:', error);
    }
}

/**
 * 加载智能体数据
 */
async function loadAgents() {
    try {
        const response = await fetch(`${API_BASE}/agents`);
        if (!response.ok) throw new Error('获取智能体数据失败');

        const agents = await response.json();

        // 更新网络可视化中的节点
        if (network && agents.length > 0) {
            const nodes = new vis.DataSet(
                agents.map(agent => ({
                    id: agent.id,
                    label: agent.name,
                    color: getColorByReputation(agent.reputation_score),
                    size: 10 + agent.reputation_score * 10
                }))
            );

            network.body.data.nodes = nodes;
        }

    } catch (error) {
        console.error('加载智能体数据失败:', error);
    }
}

/**
 * 刷新会话列表
 */
async function refreshSessions() {
    try {
        const response = await fetch(`${API_BASE}/propagation/sessions`);
        if (!response.ok) throw new Error('获取会话列表失败');

        const data = await response.json();
        sessions = data.sessions || [];

        updateSessionsTable(sessions);
        document.getElementById('propagation-sessions-count').textContent = sessions.length;

    } catch (error) {
        console.error('刷新会话列表失败:', error);
    }
}

/**
 * 更新会话表格
 */
function updateSessionsTable(sessions) {
    const tbody = document.getElementById('sessions-table-body');

    if (sessions.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="7" class="px-6 py-4 text-center text-gray-500">
                    暂无历史会话
                </td>
            </tr>
        `;
        return;
    }

    tbody.innerHTML = sessions.map(session => `
        <tr class="hover:bg-gray-50">
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                ${session.session_id.substring(0, 8)}...
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                ${session.request.message.substring(0, 20)}...
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                ${session.request.model_type}
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                ${session.request.seed_agents.length}
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                ${session.result.statistics.total_influenced}
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                ${new Date(session.created_at).toLocaleString()}
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-sm font-medium">
                <button onclick="viewSessionDetails('${session.session_id}')"
                        class="text-indigo-600 hover:text-indigo-900 mr-2">
                    <i class="fas fa-eye"></i>
                </button>
                <button onclick="replaySession('${session.session_id}')"
                        class="text-green-600 hover:text-green-900">
                    <i class="fas fa-redo"></i>
                </button>
            </td>
        </tr>
    `).join('');
}

/**
 * 查看会话详情
 */
async function viewSessionDetails(sessionId) {
    try {
        const response = await fetch(`${API_BASE}/propagation/session/${sessionId}`);
        if (!response.ok) throw new Error('获取会话详情失败');

        const session = await response.json();

        // 显示会话详情（这里可以展开为模态框）
        addLog(`查看会话详情: ${sessionId}`, 'info');
        addLog(`消息: ${session.request.message}`, 'info');
        addLog(`模型: ${session.request.model_type}`, 'info');
        addLog(`种子: ${session.request.seed_agents.join(', ')}`, 'info');
        addLog(`影响: ${session.result.statistics.total_influenced} 个智能体`, 'info');

        // 更新可视化
        updateVisualization(session);
        updatePropagationResult(session);

    } catch (error) {
        console.error('查看会话详情失败:', error);
        showNotification('查看会话详情失败', 'error');
    }
}

/**
 * 重播会话
 */
function replaySession(sessionId) {
    addLog(`重播会话: ${sessionId}`, 'info');
    // 这里可以实现会话重播功能
    showNotification('会话重播功能开发中...', 'info');
}

/**
 * 显示节点详情
 */
function showNodeDetails(nodeId) {
    addLog(`查看智能体详情: ${nodeId}`, 'info');
    // 这里可以实现显示智能体详细信息的模态框
}

/**
 * 重置网络视图
 */
function resetNetworkView() {
    if (network) {
        network.fit();
        addLog('网络视图已重置', 'info');
    }
}

/**
 * 切换网络统计显示
 */
function toggleNetworkStats() {
    const statsDiv = document.getElementById('network-stats');
    statsDiv.classList.toggle('hidden');
}

/**
 * 清空日志
 */
function clearLogs() {
    const logsDiv = document.getElementById('propagation-logs');
    logsDiv.innerHTML = '<div class="text-gray-500">日志已清空</div>';
}

/**
 * 添加日志
 */
function addLog(message, type = 'info') {
    const logsDiv = document.getElementById('propagation-logs');
    const timestamp = new Date().toLocaleTimeString();

    const logColors = {
        info: 'text-blue-600',
        success: 'text-green-600',
        warning: 'text-yellow-600',
        error: 'text-red-600'
    };

    const logEntry = document.createElement('div');
    logEntry.className = `${logColors[type]} mb-1`;
    logEntry.innerHTML = `<span class="text-gray-500">[${timestamp}]</span> ${message}`;

    // 移除初始提示信息
    if (logsDiv.children.length === 1 && logsDiv.children[0].classList.contains('text-gray-500')) {
        logsDiv.innerHTML = '';
    }

    logsDiv.appendChild(logEntry);
    logsDiv.scrollTop = logsDiv.scrollHeight;
}

/**
 * 显示通知
 */
function showNotification(message, type = 'info') {
    // 创建通知元素
    const notification = document.createElement('div');
    notification.className = `fixed top-4 right-4 px-6 py-3 rounded-lg shadow-lg z-50 ${getNotificationClass(type)}`;
    notification.textContent = message;

    document.body.appendChild(notification);

    // 3秒后自动移除
    setTimeout(() => {
        notification.remove();
    }, 3000);
}

/**
 * 获取通知样式类
 */
function getNotificationClass(type) {
    const classes = {
        success: 'bg-green-500 text-white',
        error: 'bg-red-500 text-white',
        warning: 'bg-yellow-500 text-white',
        info: 'bg-blue-500 text-white'
    };
    return classes[type] || classes.info;
}

/**
 * 根据声誉值获取颜色
 */
function getColorByReputation(score) {
    if (score >= 0.8) return '#2ecc71'; // 绿色 - 高声誉
    if (score >= 0.6) return '#f39c12'; // 橙色 - 中等声誉
    if (score >= 0.4) return '#e67e22'; // 深橙色 - 较低声誉
    return '#e74c3c'; // 红色 - 低声誉
}

/**
 * 显示/隐藏加载指示器
 */
function showLoading(show) {
    const overlay = document.getElementById('loading-overlay');
    if (show) {
        overlay.classList.remove('hidden');
    } else {
        overlay.classList.add('hidden');
    }
}

/**
 * 处理WebSocket消息
 */
function handleWebSocketMessage(data) {
    // 处理实时更新的数据
    if (data.type === 'propagation_update') {
        updateVisualization(data.payload);
    } else if (data.type === 'system_stats') {
        loadSystemStats();
    } else if (data.type === 'network_update') {
        loadNetworkMetrics();
    }
}

/**
 * 导出数据
 */
function exportData() {
    const data = {
        sessions: sessions,
        timestamp: new Date().toISOString(),
        version: '1.0.0'
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `propagation_data_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);

    addLog('数据已导出', 'success');
}

/**
 * 加载真实网络数据
 */
async function loadRealNetworkData() {
    try {
        // 获取智能体数据
        const agentsResponse = await fetch(`${API_BASE}/agents`);
        if (!agentsResponse.ok) throw new Error('获取智能体数据失败');
        const agents = await agentsResponse.json();

        // 创建节点数据
        const nodes = new vis.DataSet(agents.map(agent => ({
            id: agent.id,
            label: `${agent.name} (${agent.id})`,
            color: agent.status === 'active' ? '#2ecc71' : '#e74c3c',
            size: 15 + (agent.reputation_score / 10),
            title: `${agent.name}\n类型: ${agent.type}\n状态: ${agent.status}\n声誉: ${agent.reputation_score}`
        })));

        // 创建连接数据（基于实际网络结构）
        const edges = new vis.DataSet([
            {from: 'agent_1', to: 'agent_2', width: 2, title: '连接强度: 0.8'},
            {from: 'agent_1', to: 'agent_3', width: 3, title: '连接强度: 0.9'},
            {from: 'agent_2', to: 'agent_3', width: 1, title: '连接强度: 0.6'}
        ]);

        // 更新网络数据
        if (network) {
            network.setData({ nodes: nodes, edges: edges });
        }

        addLog(`已加载 ${agents.length} 个智能体的网络数据`, 'success');
    } catch (error) {
        console.error('加载网络数据失败:', error);
        addLog('加载网络数据失败', 'error');
    }
}