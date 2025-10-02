"""
社交网络可视化模块
"""

from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from .graph import SocialNetworkGraph
from .algorithms import PageRankCalculator, CommunityDetector


class SocialNetworkVisualizer:
    """社交网络可视化器"""

    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        初始化可视化器

        Args:
            figsize: 图形大小
        """
        self.figsize = figsize

    def plot_basic_graph(self,
                        graph: SocialNetworkGraph,
                        layout: str = 'spring',
                        node_size: int = 300,
                        with_labels: bool = True,
                        save_path: Optional[str] = None) -> None:
        """
        绘制基础社交网络图

        Args:
            graph: 社交网络图
            layout: 布局算法 ('spring', 'circular', 'random', 'shell')
            node_size: 节点大小
            with_labels: 是否显示标签
            save_path: 保存路径（可选）
        """
        plt.figure(figsize=self.figsize)

        # 选择布局
        if layout == 'spring':
            pos = nx.spring_layout(graph.graph, seed=42)
        elif layout == 'circular':
            pos = nx.circular_layout(graph.graph)
        elif layout == 'random':
            pos = nx.random_layout(graph.graph, seed=42)
        elif layout == 'shell':
            pos = nx.shell_layout(graph.graph)
        else:
            pos = nx.spring_layout(graph.graph, seed=42)

        # 绘制图
        nx.draw(graph.graph, pos,
                with_labels=with_labels,
                node_color='lightblue',
                node_size=node_size,
                edge_color='gray',
                font_size=10,
                font_weight='bold')

        plt.title("Social Network Graph", fontsize=16)
        plt.axis('off')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def plot_with_pagerank(self,
                          graph: SocialNetworkGraph,
                          layout: str = 'spring',
                          save_path: Optional[str] = None) -> None:
        """
        根据PageRank值绘制社交网络图

        Args:
            graph: 社交网络图
            layout: 布局算法
            save_path: 保存路径（可选）
        """
        plt.figure(figsize=self.figsize)

        # 计算PageRank
        pagerank_calc = PageRankCalculator()
        pagerank_scores = pagerank_calc.calculate_pagerank(graph)

        # 选择布局
        if layout == 'spring':
            pos = nx.spring_layout(graph.graph, seed=42)
        elif layout == 'circular':
            pos = nx.circular_layout(graph.graph)
        else:
            pos = nx.spring_layout(graph.graph, seed=42)

        # 获取所有节点的PageRank值，用于颜色映射
        scores = [pagerank_scores.get(node, 0) for node in graph.graph.nodes()]

        # 绘制图
        nodes = nx.draw_networkx_nodes(
            graph.graph, pos,
            node_color=scores,
            node_size=300 + np.array(scores) * 2000,  # 节点大小反映影响力
            cmap=plt.cm.Reds,
            alpha=0.8
        )

        nx.draw_networkx_edges(graph.graph, pos, alpha=0.5)
        nx.draw_networkx_labels(graph.graph, pos, font_size=10, font_weight='bold')

        # 添加颜色条
        plt.colorbar(nodes, label='PageRank Score')

        plt.title("Social Network with PageRank Influence", fontsize=16)
        plt.axis('off')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def plot_with_communities(self,
                             graph: SocialNetworkGraph,
                             layout: str = 'spring',
                             save_path: Optional[str] = None) -> None:
        """
        根据社区检测结果绘制社交网络图

        Args:
            graph: 社交网络图
            layout: 布局算法
            save_path: 保存路径（可选）
        """
        plt.figure(figsize=self.figsize)

        # 检测社区
        community_detector = CommunityDetector()
        communities = community_detector.detect_communities(graph)

        # 选择布局
        if layout == 'spring':
            pos = nx.spring_layout(graph.graph, seed=42)
        elif layout == 'circular':
            pos = nx.circular_layout(graph.graph)
        else:
            pos = nx.spring_layout(graph.graph, seed=42)

        # 为不同社区分配颜色
        colors = plt.cm.Set3(np.linspace(0, 1, len(communities)))
        node_colors = []

        for node in graph.graph.nodes():
            for i, community in enumerate(communities):
                if node in community:
                    node_colors.append(colors[i])
                    break

        # 绘制图
        nx.draw(graph.graph, pos,
                node_color=node_colors,
                node_size=300,
                with_labels=True,
                font_size=10,
                font_weight='bold',
                edge_color='gray',
                alpha=0.8)

        plt.title(f"Social Network with {len(communities)} Communities", fontsize=16)
        plt.axis('off')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def plot_weighted_graph(self,
                           graph: SocialNetworkGraph,
                           layout: str = 'spring',
                           save_path: Optional[str] = None) -> None:
        """
        绘制带权重的社交网络图

        Args:
            graph: 社交网络图
            layout: 布局算法
            save_path: 保存路径（可选）
        """
        plt.figure(figsize=self.figsize)

        # 选择布局
        if layout == 'spring':
            pos = nx.spring_layout(graph.graph, weight='weight', seed=42)
        elif layout == 'circular':
            pos = nx.circular_layout(graph.graph)
        else:
            pos = nx.spring_layout(graph.graph, weight='weight', seed=42)

        # 获取边的权重
        edges = graph.graph.edges()
        weights = [graph.graph[u][v].get('weight', 1.0) for u, v in edges]

        # 绘制图
        nx.draw(graph.graph, pos,
                with_labels=True,
                node_color='lightblue',
                node_size=300,
                edge_color=weights,
                edge_cmap=plt.cm.Blues,
                width=2 + np.array(weights) * 3,  # 边的宽度反映权重
                font_size=10,
                font_weight='bold')

        # 添加颜色条
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues)
        sm.set_array(weights)
        plt.colorbar(sm, label='Relationship Strength')

        plt.title("Weighted Social Network", fontsize=16)
        plt.axis('off')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def plot_degree_distribution(self,
                               graph: SocialNetworkGraph,
                               save_path: Optional[str] = None) -> None:
        """
        绘制度数分布图

        Args:
            graph: 社交网络图
            save_path: 保存路径（可选）
        """
        plt.figure(figsize=self.figsize)

        # 计算所有节点的度数
        degrees = [graph.graph.degree(node) for node in graph.graph.nodes()]

        # 绘制直方图
        plt.hist(degrees, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Degree (Number of Friends)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Degree Distribution', fontsize=16)

        # 添加统计信息
        plt.axvline(np.mean(degrees), color='red', linestyle='dashed',
                   linewidth=2, label=f'Mean: {np.mean(degrees):.2f}')
        plt.axvline(np.median(degrees), color='green', linestyle='dashed',
                   linewidth=2, label=f'Median: {np.median(degrees):.2f}')
        plt.legend()

        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def plot_pagerank_distribution(self,
                                  graph: SocialNetworkGraph,
                                  save_path: Optional[str] = None) -> None:
        """
        绘制PageRank分布图

        Args:
            graph: 社交网络图
            save_path: 保存路径（可选）
        """
        plt.figure(figsize=self.figsize)

        # 计算PageRank
        pagerank_calc = PageRankCalculator()
        pagerank_scores = pagerank_calc.calculate_pagerank(graph)

        # 提取分数
        scores = list(pagerank_scores.values())

        # 绘制直方图
        plt.hist(scores, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        plt.xlabel('PageRank Score', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('PageRank Distribution', fontsize=16)

        # 添加统计信息
        plt.axvline(np.mean(scores), color='red', linestyle='dashed',
                   linewidth=2, label=f'Mean: {np.mean(scores):.4f}')
        plt.axvline(np.median(scores), color='green', linestyle='dashed',
                   linewidth=2, label=f'Median: {np.median(scores):.4f}')
        plt.legend()

        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def create_summary_dashboard(self,
                                graph: SocialNetworkGraph,
                                save_path: Optional[str] = None) -> None:
        """
        创建社交网络分析仪表板

        Args:
            graph: 社交网络图
            save_path: 保存路径（可选）
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. 基础网络图
        ax1 = axes[0, 0]
        pos = nx.spring_layout(graph.graph, seed=42)
        nx.draw(graph.graph, pos, ax=ax1,
                node_color='lightblue', node_size=200,
                with_labels=False, edge_color='gray')
        ax1.set_title("Network Overview")

        # 2. 度数分布
        ax2 = axes[0, 1]
        degrees = [graph.graph.degree(node) for node in graph.graph.nodes()]
        ax2.hist(degrees, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_xlabel('Degree')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Degree Distribution')
        ax2.grid(True, alpha=0.3)

        # 3. PageRank分布
        ax3 = axes[1, 0]
        pagerank_calc = PageRankCalculator()
        pagerank_scores = pagerank_calc.calculate_pagerank(graph)
        scores = list(pagerank_scores.values())
        ax3.hist(scores, bins=15, alpha=0.7, color='lightcoral', edgecolor='black')
        ax3.set_xlabel('PageRank Score')
        ax3.set_ylabel('Frequency')
        ax3.set_title('PageRank Distribution')
        ax3.grid(True, alpha=0.3)

        # 4. 网络统计信息
        ax4 = axes[1, 1]
        ax4.axis('off')

        # 计算统计信息
        stats_text = f"""
        Network Statistics:
        ─────────────────
        Nodes: {graph.get_agent_count()}
        Edges: {graph.get_edge_count()}
        Density: {graph.get_network_density():.4f}
        Avg Degree: {np.mean(degrees):.2f}
        Max Degree: {max(degrees) if degrees else 0}

        Top Influential Agents:
        ─────────────────"""

        # 添加前5个有影响力的Agent
        top_agents = pagerank_calc.get_top_influential_agents(graph, top_k=5)
        for i, (agent_id, score) in enumerate(top_agents, 1):
            agent_name = graph.get_agent_by_id(agent_id).name if graph.get_agent_by_id(agent_id) else f"Agent{agent_id}"
            stats_text += f"\n{i}. {agent_name}: {score:.4f}"

        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
                fontsize=12, verticalalignment='top', fontfamily='monospace')

        plt.tight_layout()
        plt.suptitle("Social Network Analysis Dashboard", fontsize=18, y=1.02)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def export_graph_data(self,
                         graph: SocialNetworkGraph,
                         format: str = 'gexf',
                         output_path: str = 'network_graph') -> None:
        """
        导出图数据

        Args:
            graph: 社交网络图
            format: 导出格式 ('gexf', 'graphml', 'gml', 'json')
            output_path: 输出路径（不包含扩展名）
        """
        if format == 'gexf':
            nx.write_gexf(graph.graph, f"{output_path}.gexf")
        elif format == 'graphml':
            nx.write_graphml(graph.graph, f"{output_path}.graphml")
        elif format == 'gml':
            nx.write_gml(graph.graph, f"{output_path}.gml")
        elif format == 'json':
            data = nx.node_link_data(graph.graph)
            import json
            with open(f"{output_path}.json", 'w') as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"Graph exported to {output_path}.{format}")