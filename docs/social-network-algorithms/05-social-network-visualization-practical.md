# 社交网络可视化技术实战：从数据到洞察

## 📋 概述

数据可视化是社交网络分析中不可或缺的一环，它能够将复杂的网络结构和算法结果转化为直观的视觉表达。本文详细介绍了社交网络可视化的设计思路、实现技巧、优化策略以及在百万级智能体社交平台中的实际应用。

## 🎨 可视化设计原则

### 1. 信息层次化
```python
class VisualizationHierarchy:
    """可视化层次结构"""

    LEVELS = {
        'overview': {
            'purpose': '展示整体网络结构',
            'elements': ['nodes', 'edges', 'basic_layout'],
            'complexity': 'low'
        },
        'analysis': {
            'purpose': '展示算法分析结果',
            'elements': ['nodes_size', 'node_color', 'edge_weight', 'labels'],
            'complexity': 'medium'
        },
        'detail': {
            'purpose': '展示详细信息',
            'elements': ['interactive_features', 'annotations', 'statistics'],
            'complexity': 'high'
        }
    }
```

### 2. 视觉编码原则
- **节点大小**: 表示重要性或度数
- **节点颜色**: 表示社区或属性
- **边粗细**: 表示关系强度
- **边颜色**: 表示关系类型
- **布局**: 表示网络结构特征

### 3. 交互设计原则
- **渐进式揭示**: 从概览到细节
- **上下文保持**: 操作时保持整体结构可见
- **快速响应**: 交互反馈时间 < 200ms
- **直观操作**: 符合用户认知习惯

## 💻 核心可视化组件

### 1. 基础网络图可视化
```python
class SocialNetworkVisualizer:
    """社交网络可视化器"""

    def plot_basic_graph(self,
                        graph: SocialNetworkGraph,
                        layout: str = 'spring',
                        node_size: int = 300,
                        with_labels: bool = True,
                        save_path: Optional[str] = None,
                        figsize: Optional[Tuple[int, int]] = None) -> None:
        """
        绘制基础社交网络图

        Args:
            graph: 社交网络图
            layout: 布局算法 ('spring', 'circular', 'random', 'shell', 'kamada_kawai')
            node_size: 节点大小
            with_labels: 是否显示标签
            save_path: 保存路径（可选）
            figsize: 图形大小（可选）
        """
        if figsize is None:
            figsize = self.figsize

        plt.figure(figsize=figsize)

        # 选择布局算法
        pos = self._choose_layout(graph, layout)

        # 绘制基础图形
        nx.draw(graph.graph, pos,
                with_labels=with_labels,
                node_color='lightblue',
                node_size=node_size,
                edge_color='gray',
                font_size=10,
                font_weight='bold',
                alpha=0.8)

        plt.title("Social Network Graph", fontsize=16, pad=20)
        plt.axis('off')

        # 添加基本统计信息
        self._add_basic_stats(graph)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
        else:
            plt.show()

    def _choose_layout(self, graph: SocialNetworkGraph, layout: str):
        """选择合适的布局算法"""
        if layout == 'spring':
            return nx.spring_layout(graph.graph, seed=42, k=1/np.sqrt(graph.get_agent_count()))
        elif layout == 'circular':
            return nx.circular_layout(graph.graph)
        elif layout == 'random':
            return nx.random_layout(graph.graph, seed=42)
        elif layout == 'shell':
            return nx.shell_layout(graph.graph)
        elif layout == 'kamada_kawai':
            return nx.kamada_kawai_layout(graph.graph)
        else:
            # 默认使用spring布局，并根据图大小调整参数
            return nx.spring_layout(graph.graph, seed=42,
                                  k=2/np.sqrt(graph.get_agent_count()))
```

### 2. PageRank影响力可视化
```python
def plot_with_pagerank(self,
                      graph: SocialNetworkGraph,
                      layout: str = 'spring',
                      top_k_highlight: int = 10,
                      save_path: Optional[str] = None) -> None:
    """
    根据PageRank值绘制影响力网络图

    Args:
        graph: 社交网络图
        layout: 布局算法
        top_k_highlight: 高亮显示的前k个节点
        save_path: 保存路径
    """
    plt.figure(figsize=self.figsize)

    # 计算PageRank
    pagerank_calc = PageRankCalculator()
    pagerank_scores = pagerank_calc.calculate_pagerank(graph)

    # 获取前k个最有影响力的节点
    top_agents = pagerank_calc.get_top_influential_agents(graph, top_k_highlight)
    top_nodes = [agent_id for agent_id, _ in top_agents]

    # 选择布局
    pos = self._choose_layout(graph, layout)

    # 准备节点颜色和大小
    node_colors = []
    node_sizes = []
    node_alphas = []

    max_score = max(pagerank_scores.values()) if pagerank_scores else 1.0
    min_score = min(pagerank_scores.values()) if pagerank_scores else 0.0

    for node in graph.graph.nodes():
        score = pagerank_scores.get(node, 0)

        # 归一化分数到0-1
        normalized_score = (score - min_score) / (max_score - min_score) if max_score > min_score else 0.5

        # 节点大小：基于PageRank分数
        base_size = 300
        size_multiplier = 1 + normalized_score * 3
        node_size = base_size * size_multiplier
        node_sizes.append(node_size)

        # 节点颜色：基于影响力等级
        if node in top_nodes:
            node_colors.append('#ff4444')  # 红色：高影响力
            node_alphas.append(1.0)
        else:
            # 使用渐变色
            color_value = normalized_score
            node_colors.append(plt.cm.Reds(color_value))
            node_alphas.append(0.7)

    # 绘制边
    nx.draw_networkx_edges(graph.graph, pos,
                           alpha=0.3,
                           edge_color='gray',
                           width=1)

    # 绘制节点
    for i, (node, color, size, alpha) in enumerate(zip(graph.graph.nodes(),
                                                       node_colors,
                                                       node_sizes,
                                                       node_alphas)):
        nx.draw_networkx_nodes(graph.graph, pos,
                               nodelist=[node],
                               node_color=[color],
                               node_size=[size],
                               alpha=alpha)

    # 绘制标签（只显示重要节点）
    important_labels = {node: graph.get_agent_by_id(node).name
                        for node in top_nodes
                        if graph.get_agent_by_id(node)}

    if important_labels:
        nx.draw_networkx_labels(graph.graph, pos, labels=important_labels,
                               font_size=12, font_weight='bold')

    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds)
    sm.set_array([min_score, max_score])
    cbar = plt.colorbar(sm, shrink=0.8)
    cbar.set_label('PageRank Score', fontsize=12)

    # 添加图例
    self._add_pagerank_legend(top_agents)

    plt.title("Social Network with PageRank Influence", fontsize=16, pad=20)
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
```

### 3. 社区结构可视化
```python
def plot_with_communities(self,
                         graph: SocialNetworkGraph,
                         layout: str = 'spring',
                         save_path: Optional[str] = None) -> None:
    """
    根据社区检测结果绘制社区网络图

    Args:
        graph: 社交网络图
        layout: 布局算法
        save_path: 保存路径
    """
    plt.figure(figsize=self.figsize)

    # 检测社区
    community_detector = CommunityDetector()
    communities = community_detector.detect_communities(graph)

    # 选择布局
    pos = self._choose_layout(graph, layout)

    # 为不同社区分配颜色
    colors = plt.cm.Set3(np.linspace(0, 1, len(communities)))
    node_colors = []

    for node in graph.graph.nodes():
        node_assigned = False
        for i, community in enumerate(communities):
            if node in community:
                node_colors.append(colors[i])
                node_assigned = True
                break
        if not node_assigned:
            node_colors.append('gray')  # 未分配节点的颜色

    # 绘制图
    nx.draw(graph.graph, pos,
            node_color=node_colors,
            node_size=400,
            with_labels=True,
            font_size=10,
            font_weight='bold',
            edge_color='lightgray',
            alpha=0.8,
            width=1.5)

    # 添加社区统计信息
    stats = community_detector.get_community_statistics(graph, communities)

    # 添加图例
    self._add_community_legend(communities, stats)

    plt.title(f"Social Network Communities\n({len(communities)} communities detected)",
              fontsize=16, pad=20)
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

def _add_community_legend(self, communities: List[Set[int]], stats: Dict):
    """添加社区图例"""
    colors = plt.cm.Set3(np.linspace(0, 1, len(communities)))

    legend_elements = []
    for i, community in enumerate(communities):
        community_size = len(community)
        percentage = (community_size / stats['num_communities']) * 100
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=colors[i], markersize=10,
                      label=f'Community {i+1}: {community_size} agents ({percentage:.1f}%)')
        )

    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
```

### 4. 权重网络可视化
```python
def plot_weighted_graph(self,
                       graph: SocialNetworkGraph,
                       layout: str = 'spring',
                       weight_threshold: float = 0.0,
                       save_path: Optional[str] = None) -> None:
    """
    绘制带权重的社交网络图

    Args:
        graph: 社交网络图
        layout: 布局算法
        weight_threshold: 权重阈值，低于此值的边不显示
        save_path: 保存路径
    """
    plt.figure(figsize=self.figsize)

    # 选择布局
    pos = self._choose_layout(graph, layout)

    # 获取边的权重
    edges = graph.graph.edges()
    weights = []
    edge_colors = []
    edge_widths = []

    for u, v in edges:
        weight = graph.get_friendship_strength(u, v) or 1.0

        if weight >= weight_threshold:
            weights.append(weight)
            # 边颜色：基于权重强度
            if weight >= 0.8:
                edge_colors.append('#2ecc71')  # 绿色：强关系
            elif weight >= 0.5:
                edge_colors.append('#3498db')  # 蓝色：中等关系
            else:
                edge_colors.append('#95a5a6')  # 灰色：弱关系

            # 边宽度：基于权重
            edge_widths.append(1 + weight * 3)

    # 过滤低权重边
    filtered_edges = [(u, v) for u, v in edges
                     if graph.get_friendship_strength(u, v) >= weight_threshold]

    # 绘制节点
    nx.draw_networkx_nodes(graph.graph, pos,
                           node_color='lightblue',
                           node_size=400,
                           alpha=0.8)

    # 绘制边
    if filtered_edges:
        nx.draw_networkx_edges(graph.graph, pos,
                               edgelist=filtered_edges,
                               edge_color=edge_colors,
                               width=edge_widths,
                               alpha=0.6)

    # 绘制标签
    nx.draw_networkx_labels(graph.graph, pos,
                           font_size=10,
                           font_weight='bold')

    # 添加颜色条
    if weights:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues)
        sm.set_array([min(weights), max(weights)])
        cbar = plt.colorbar(sm, shrink=0.8)
        cbar.set_label('Relationship Strength', fontsize=12)

    # 添加权重统计
    self._add_weight_statistics(weights)

    plt.title("Weighted Social Network", fontsize=16, pad=20)
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
```

## 📊 统计图表可视化

### 1. 度数分布图
```python
def plot_degree_distribution(self,
                           graph: SocialNetworkGraph,
                           distribution_type: str = 'histogram',
                           save_path: Optional[str] = None) -> None:
    """
    绘制度数分布图

    Args:
        graph: 社交网络图
        distribution_type: 分布类型 ('histogram', 'loglog', 'cumulative')
        save_path: 保存路径
    """
    plt.figure(figsize=(12, 8))

    # 计算所有节点的度数
    degrees = [graph.graph.degree(node) for node in graph.graph.nodes()]

    # 创建子图
    if distribution_type in ['histogram', 'all']:
        plt.subplot(2, 2, 1)
        self._plot_histogram(degrees, "Degree Distribution", "Degree", "Frequency")

    if distribution_type in ['loglog', 'all']:
        plt.subplot(2, 2, 2)
        self._plot_loglog_distribution(degrees)

    if distribution_type in ['cumulative', 'all']:
        plt.subplot(2, 2, 3)
        self._plot_cumulative_distribution(degrees)

    # 统计信息
    plt.subplot(2, 2, 4)
    self._plot_degree_statistics(degrees, graph)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

def _plot_histogram(self, data: List[int], title: str, xlabel: str, ylabel: str):
    """绘制直方图"""
    plt.hist(data, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)

    # 添加统计线
    mean_val = np.mean(data)
    median_val = np.median(data)

    plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2,
               label=f'Mean: {mean_val:.2f}')
    plt.axvline(median_val, color='green', linestyle='dashed', linewidth=2,
               label=f'Median: {median_val:.2f}')

    plt.legend()
    plt.grid(True, alpha=0.3)

def _plot_loglog_distribution(self, degrees: List[int]):
    """绘制对数分布图"""
    degree_counts = {}
    for degree in degrees:
        degree_counts[degree] = degree_counts.get(degree, 0) + 1

    if degree_counts:
        x = list(degree_counts.keys())
        y = list(degree_counts.values())

        plt.loglog(x, y, 'bo-', alpha=0.7, markersize=6)
        plt.xlabel('Degree (log scale)', fontsize=12)
        plt.ylabel('Frequency (log scale)', fontsize=12)
        plt.title('Degree Distribution (Log-Log)', fontsize=14)
        plt.grid(True, alpha=0.3)

def _plot_cumulative_distribution(self, degrees: List[int]):
    """绘制累积分布图"""
    sorted_degrees = sorted(degrees)
    n = len(sorted_degrees)
    cumulative = [(i + 1) / n for i in range(n)]

    plt.plot(sorted_degrees, cumulative, 'b-', linewidth=2)
    plt.xlabel('Degree', fontsize=12)
    plt.ylabel('Cumulative Probability', fontsize=12)
    plt.title('Cumulative Degree Distribution', fontsize=14)
    plt.grid(True, alpha=0.3)
```

### 2. PageRank分布图
```python
def plot_pagerank_distribution(self,
                              graph: SocialNetworkGraph,
                              save_path: Optional[str] = None) -> None:
    """
    绘制PageRank分布图
    """
    plt.figure(figsize=(15, 10))

    # 计算PageRank
    pagerank_calc = PageRankCalculator()
    pagerank_scores = pagerank_calc.calculate_pagerank(graph)
    scores = list(pagerank_scores.values())

    # 创建多个子图
    plt.subplot(2, 3, 1)
    self._plot_pagerank_histogram(scores)

    plt.subplot(2, 3, 2)
    self._plot_pagerank_ranking(scores)

    plt.subplot(2, 3, 3)
    self._plot_pagerank Lorenz_curve(scores)

    plt.subplot(2, 3, 4)
    self._plot_pagerank_boxplot(scores)

    plt.subplot(2, 3, 5)
    self._plot_pagerank_heatmap(graph, pagerank_scores)

    plt.subplot(2, 3, 6)
    self._plot_pagerank_statistics(scores, pagerank_scores)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

def _plot_pagerank_ranking(self, scores: List[float]):
    """绘制PageRank排名图"""
    sorted_scores = sorted(scores, reverse=True)
    ranks = range(1, len(sorted_scores) + 1)

    plt.loglog(ranks, sorted_scores, 'ro-', alpha=0.6, markersize=4)
    plt.xlabel('Rank (log scale)', fontsize=12)
    plt.ylabel('PageRank Score (log scale)', fontsize=12)
    plt.title('PageRank vs Rank', fontsize=14)
    plt.grid(True, alpha=0.3)
```

## 🎛️ 综合分析仪表板

### 1. 多面板仪表板
```python
def create_summary_dashboard(self,
                            graph: SocialNetworkGraph,
                            save_path: Optional[str] = None) -> None:
    """
    创建社交网络分析仪表板

    Args:
        graph: 社交网络图
        save_path: 保存路径
    """
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('Social Network Analysis Dashboard', fontsize=20, y=0.98)

    # 创建网格布局
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. 网络概览图
    ax1 = fig.add_subplot(gs[0, :2])
    self._plot_network_overview(graph, ax1)

    # 2. 度数分布
    ax2 = fig.add_subplot(gs[0, 2])
    self._plot_degree_distribution_small(graph, ax2)

    # 3. PageRank分布
    ax3 = fig.add_subplot(gs[1, 0])
    self._plot_pagerank_distribution_small(graph, ax3)

    # 4. 社区分布
    ax4 = fig.add_subplot(gs[1, 1])
    self._plot_community_distribution(graph, ax4)

    # 5. 关键指标
    ax5 = fig.add_subplot(gs[1, 2])
    self._plot_key_metrics(graph, ax5)

    # 6. 网络统计表
    ax6 = fig.add_subplot(gs[2, :])
    self._plot_network_statistics_table(graph, ax6)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

def _plot_network_overview(self, graph: SocialNetworkGraph, ax):
    """绘制网络概览图"""
    ax.set_title('Network Overview', fontsize=14, fontweight='bold')

    # 选择布局
    pos = nx.spring_layout(graph.graph, seed=42, k=1/np.sqrt(graph.get_agent_count()))

    # 计算PageRank用于节点大小
    pagerank_calc = PageRankCalculator()
    pagerank_scores = pagerank_calc.calculate_pagerank(graph)

    # 节点大小和颜色
    node_sizes = [300 + pagerank_scores.get(node, 0) * 2000 for node in graph.graph.nodes()]
    node_colors = [pagerank_scores.get(node, 0) for node in graph.graph.nodes()]

    # 绘制网络
    nx.draw_networkx_nodes(graph.graph, pos, ax=ax,
                           node_color=node_colors,
                           node_size=node_sizes,
                           cmap=plt.cm.Reds,
                           alpha=0.8)

    nx.draw_networkx_edges(graph.graph, pos, ax=ax,
                           alpha=0.3,
                           edge_color='gray',
                           width=0.5)

    # 只显示重要节点的标签
    important_nodes = [node for node, score in pagerank_scores.items()
                      if score > np.mean(list(pagerank_scores.values()))]
    important_labels = {node: graph.get_agent_by_id(node).name[:10]
                       for node in important_nodes
                       if graph.get_agent_by_id(node)}

    if important_labels:
        nx.draw_networkx_labels(graph.graph, pos, labels=important_labels, ax=ax,
                               font_size=8, font_weight='bold')

    ax.axis('off')

def _plot_key_metrics(self, graph: SocialNetworkGraph, ax):
    """绘制关键指标"""
    ax.set_title('Key Metrics', fontsize=14, fontweight='bold')
    ax.axis('off')

    # 计算指标
    metrics = self._calculate_network_metrics(graph)

    # 创建指标显示
    y_pos = 0.9
    for metric_name, metric_value in metrics.items():
        ax.text(0.1, y_pos, f'{metric_name}:', fontsize=12, fontweight='bold')
        ax.text(0.6, y_pos, f'{metric_value}', fontsize=12)
        y_pos -= 0.15

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

def _calculate_network_metrics(self, graph: SocialNetworkGraph) -> Dict[str, str]:
    """计算网络关键指标"""
    metrics = {}

    # 基本统计
    metrics['Nodes'] = str(graph.get_agent_count())
    metrics['Edges'] = str(graph.get_edge_count())
    metrics['Density'] = f'{graph.get_network_density():.4f}'

    # 度数统计
    degrees = [graph.graph.degree(node) for node in graph.graph.nodes()]
    metrics['Avg Degree'] = f'{np.mean(degrees):.2f}'
    metrics['Max Degree'] = str(max(degrees) if degrees else 0)

    # 连通性
    components = list(nx.connected_components(graph.graph))
    metrics['Components'] = str(len(components))
    metrics['Largest Component'] = str(max(len(c) for c in components) if components else 0)

    # PageRank统计
    pagerank_calc = PageRankCalculator()
    pagerank_scores = pagerank_calc.calculate_pagerank(graph)
    if pagerank_scores:
        metrics['Top Agent'] = graph.get_agent_by_id(
            max(pagerank_scores, key=pagerank_scores.get)
        ).name[:15]

    return metrics
```

## 🎯 交互式可视化

### 1. 基于Plotly的交互式图表
```python
def create_interactive_network_plot(self,
                                  graph: SocialNetworkGraph,
                                  algorithm: str = 'pagerank') -> None:
    """
    创建交互式网络图

    Args:
        graph: 社交网络图
        algorithm: 分析算法 ('pagerank', 'community', 'degree')
    """
    import plotly.graph_objects as go
    import plotly.express as px

    # 计算布局
    pos = nx.spring_layout(graph.graph, seed=42)

    # 提取边信息
    edge_x = []
    edge_y = []
    for edge in graph.graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    # 创建边的轨迹
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    # 提取节点信息
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    node_sizes = []

    if algorithm == 'pagerank':
        pagerank_calc = PageRankCalculator()
        scores = pagerank_calc.calculate_pagerank(graph)

        for node in graph.graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            agent = graph.get_agent_by_id(node)
            score = scores.get(node, 0)

            node_text.append(f'{agent.name}<br>PageRank: {score:.4f}')
            node_colors.append(score)
            node_sizes.append(10 + score * 50)

    # 创建节点的轨迹
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='RdYlBu',
            reversescale=True,
            color=node_colors,
            size=node_sizes,
            colorbar=dict(
                thickness=15,
                len=0.5,
                x=1.02
            ),
            line=dict(width=2)))

    # 创建图形
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title=f'Interactive Network - {algorithm.title()} Analysis',
                       titlefont_size=16,
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       annotations=[dict(
                           text="Network Analysis",
                           showarrow=False,
                           xref="paper", yref="paper",
                           x=0.005, y=-0.002,
                           xanchor='left', yanchor='bottom',
                           font=dict(color="#888", size=12))],
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                   )

    fig.show()
```

## 🚀 性能优化策略

### 1. 大规模图可视化优化
```python
class OptimizedVisualizer:
    """优化的可视化器，支持大规模图"""

    def __init__(self, max_nodes: int = 1000):
        self.max_nodes = max_nodes

    def plot_large_graph(self, graph: SocialNetworkGraph,
                        sampling_method: str = 'random') -> None:
        """
        可视化大规模图

        Args:
            graph: 社交网络图
            sampling_method: 采样方法 ('random', 'degree', 'pagerank')
        """
        if graph.get_agent_count() <= self.max_nodes:
            # 小图直接绘制
            self._plot_direct(graph)
        else:
            # 大图采样后绘制
            sampled_graph = self._sample_graph(graph, sampling_method)
            self._plot_direct(sampled_graph)

    def _sample_graph(self, graph: SocialNetworkGraph,
                     method: str) -> SocialNetworkGraph:
        """图采样"""
        if method == 'random':
            return self._random_sampling(graph)
        elif method == 'degree':
            return self._degree_sampling(graph)
        elif method == 'pagerank':
            return self._pagerank_sampling(graph)
        else:
            return self._random_sampling(graph)

    def _pagerank_sampling(self, graph: SocialNetworkGraph) -> SocialNetworkGraph:
        """基于PageRank的采样"""
        pagerank_calc = PageRankCalculator()
        scores = pagerank_calc.calculate_pagerank(graph)

        # 按PageRank排序，选择前N个节点
        sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected_nodes = set(node for node, _ in sorted_nodes[:self.max_nodes])

        # 构建子图
        subgraph = SocialNetworkGraph()

        # 添加选中的节点
        for node in selected_nodes:
            agent = graph.get_agent_by_id(node)
            if agent:
                subgraph.add_agent(node, agent.name)

        # 添加节点间的边
        for u, v, data in graph.graph.edges(data=True):
            if u in selected_nodes and v in selected_nodes:
                weight = data.get('weight', 1.0)
                subgraph.add_friendship(u, v, weight)

        return subgraph
```

### 2. 渲染优化
```python
def optimized_rendering(self, graph: SocialNetworkGraph):
    """优化的渲染方法"""
    # 使用更高效的布局算法
    if graph.get_agent_count() > 500:
        # 大图使用快速布局
        pos = nx.fast_graph_layout(graph.graph)
    else:
        # 小图使用精确布局
        pos = nx.spring_layout(graph.graph, seed=42)

    # 使用向量化操作
    nodes = list(graph.graph.nodes())
    node_array = np.array([pos[node] for node in nodes])

    # 批量设置节点属性
    node_sizes = np.full(len(nodes), 300)
    node_colors = np.array(['lightblue'] * len(nodes))

    # 使用scatter进行批量绘制
    plt.scatter(node_array[:, 0], node_array[:, 1],
               s=node_sizes, c=node_colors, alpha=0.7)

    # 只绘制重要的边
    important_edges = [(u, v) for u, v in graph.graph.edges()
                      if graph.get_friendship_strength(u, v) > 0.5]

    for u, v in important_edges:
        x_vals = [pos[u][0], pos[v][0]]
        y_vals = [pos[u][1], pos[v][1]]
        plt.plot(x_vals, y_vals, 'gray', alpha=0.3, linewidth=0.5)
```

## 📱 响应式设计

### 1. 自适应布局
```python
def create_responsive_visualization(self, graph: SocialNetworkGraph,
                                  output_size: str = 'auto') -> None:
    """
    创建响应式可视化

    Args:
        graph: 社交网络图
        output_size: 输出尺寸 ('small', 'medium', 'large', 'auto')
    """
    # 自动检测合适的尺寸
    if output_size == 'auto':
        node_count = graph.get_agent_count()
        if node_count < 50:
            output_size = 'small'
        elif node_count < 200:
            output_size = 'medium'
        else:
            output_size = 'large'

    # 根据尺寸设置参数
    config = self._get_visualization_config(output_size)

    # 创建可视化
    plt.figure(figsize=config['figsize'])

    # 调整布局参数
    pos = nx.spring_layout(graph.graph,
                          seed=42,
                          k=config['layout_k'],
                          iterations=config['layout_iterations'])

    # 绘制图形
    nx.draw(graph.graph, pos,
            node_size=config['node_size'],
            font_size=config['font_size'],
            with_labels=config['show_labels'],
            edge_color=config['edge_color'],
            alpha=config['alpha'])

def _get_visualization_config(self, size: str) -> Dict[str, any]:
    """获取可视化配置"""
    configs = {
        'small': {
            'figsize': (8, 6),
            'node_size': 500,
            'font_size': 12,
            'show_labels': True,
            'edge_color': 'gray',
            'alpha': 0.8,
            'layout_k': 1.0,
            'layout_iterations': 50
        },
        'medium': {
            'figsize': (12, 9),
            'node_size': 300,
            'font_size': 10,
            'show_labels': True,
            'edge_color': 'lightgray',
            'alpha': 0.7,
            'layout_k': 1.0 / np.sqrt(100),
            'layout_iterations': 50
        },
        'large': {
            'figsize': (16, 12),
            'node_size': 100,
            'font_size': 8,
            'show_labels': False,
            'edge_color': 'lightgray',
            'alpha': 0.5,
            'layout_k': 1.0 / np.sqrt(500),
            'layout_iterations': 30
        }
    }

    return configs.get(size, configs['medium'])
```

## 🎨 自定义样式和主题

### 1. 主题系统
```python
class VisualizationTheme:
    """可视化主题系统"""

    THEMES = {
        'default': {
            'background_color': 'white',
            'node_color': 'lightblue',
            'edge_color': 'gray',
            'text_color': 'black',
            'highlight_color': 'red',
            'font_family': 'Arial',
            'grid': True
        },
        'dark': {
            'background_color': '#2b2b2b',
            'node_color': '#4a90e2',
            'edge_color': '#666666',
            'text_color': 'white',
            'highlight_color': '#ff6b6b',
            'font_family': 'Arial',
            'grid': False
        },
        'professional': {
            'background_color': '#f8f9fa',
            'node_color': '#007bff',
            'edge_color': '#6c757d',
            'text_color': '#212529',
            'highlight_color': '#dc3545',
            'font_family': 'Helvetica',
            'grid': True
        }
    }

    def apply_theme(self, theme_name: str):
        """应用主题"""
        if theme_name not in self.THEMES:
            theme_name = 'default'

        theme = self.THEMES[theme_name]

        plt.style.use('seaborn' if theme['grid'] else 'classic')
        plt.rcParams.update({
            'figure.facecolor': theme['background_color'],
            'axes.facecolor': theme['background_color'],
            'text.color': theme['text_color'],
            'font.family': theme['font_family']
        })

        return theme
```

## 📊 实际应用案例

### 1. 社交网络演化可视化
```python
def visualize_network_evolution(self, graph_snapshots: List[SocialNetworkGraph],
                               timestamps: List[str],
                               save_dir: str = 'evolution/') -> None:
    """
    可视化社交网络演化

    Args:
        graph_snapshots: 图快照列表
        timestamps: 时间戳列表
        save_dir: 保存目录
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    for i, (graph, timestamp) in enumerate(zip(graph_snapshots, timestamps)):
        # 创建可视化
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Network Evolution - {timestamp}', fontsize=16)

        # 网络结构
        self._plot_network_snapshot(graph, axes[0, 0])

        # 度数分布
        self._plot_degree_evolution(graph, axes[0, 1])

        # PageRank分布
        self._plot_pagerank_evolution(graph, axes[1, 0])

        # 关键指标
        self._plot_metrics_evolution(graph, axes[1, 1])

        # 保存图像
        plt.savefig(f'{save_dir}/network_evolution_{i:03d}.png',
                   dpi=150, bbox_inches='tight')
        plt.close()

    # 创建演化视频（可选）
    self._create_evolution_video(save_dir)
```

## 🎯 最佳实践总结

### 1. 设计原则
- **层次化展示**: 从概览到细节的层次化信息展示
- **色彩一致性**: 使用一致的颜色编码系统
- **交互友好**: 提供直观的交互操作
- **性能优化**: 针对不同规模数据的优化策略

### 2. 技术选型
- **静态可视化**: Matplotlib + NetworkX（适合报告和分析）
- **交互可视化**: Plotly + Bokeh（适合探索和应用）
- **大规模数据**: D3.js + WebGL（适合Web应用）
- **实时数据**: WebSockets + Canvas（适合监控面板）

### 3. 用户体验
- **响应式设计**: 适应不同屏幕尺寸
- **加载优化**: 渐进式加载和懒加载
- **错误处理**: 优雅的错误提示和降级
- **可访问性**: 支持键盘导航和屏幕阅读器

数据可视化是连接数据与洞察的桥梁，良好的可视化设计能够让复杂的社交网络结构和分析结果变得直观易懂。通过合理的设计和优化，我们可以为用户提供强大而友好的网络分析工具。

---

## 📚 参考资料

1. **Visualization Theory**: "The Visual Display of Quantitative Information"
2. **Network Visualization**: "Network Analysis and Visualization"
3. **Matplotlib Documentation**: https://matplotlib.org/
4. **Plotly Documentation**: https://plotly.com/python/

## 🏷️ 标签

`#数据可视化` `#社交网络` `#Matplotlib` `#Plotly` `#交互式可视化` `#网络分析` `#可视化设计`