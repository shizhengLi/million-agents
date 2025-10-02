"""
社交网络可视化测试
"""

import pytest
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from src.social_network.graph import SocialNetworkGraph
from src.social_network.visualization import SocialNetworkVisualizer


class TestSocialNetworkVisualizer:
    """测试社交网络可视化"""

    def setup_method(self):
        """测试前设置"""
        plt.close('all')  # 关闭所有图形

        # 创建测试图
        self.graph = SocialNetworkGraph()

        # 添加节点
        for i in range(1, 8):
            self.graph.add_agent(i, f"agent{i}")

        # 添加边形成社交网络
        # 中心节点1连接到其他节点
        for i in range(2, 8):
            self.graph.add_friendship(1, i, strength=0.9)

        # 添加一些额外连接形成社区
        self.graph.add_friendship(2, 3, strength=0.8)
        self.graph.add_friendship(4, 5, strength=0.7)
        self.graph.add_friendship(6, 7, strength=0.6)

    def test_visualizer_initialization(self):
        """测试可视化器初始化"""
        visualizer = SocialNetworkVisualizer()
        assert visualizer.figsize == (12, 8)

        visualizer_custom = SocialNetworkVisualizer(figsize=(10, 6))
        assert visualizer_custom.figsize == (10, 6)

    def test_plot_basic_graph(self):
        """测试基础图形绘制"""
        visualizer = SocialNetworkVisualizer()

        # 测试不同布局
        layouts = ['spring', 'circular', 'random', 'shell']
        for layout in layouts:
            visualizer.plot_basic_graph(self.graph, layout=layout)
            plt.close()

        # 测试不同参数
        visualizer.plot_basic_graph(
            self.graph,
            node_size=500,
            with_labels=False
        )
        plt.close()

    def test_plot_with_pagerank(self):
        """测试带PageRank的图形绘制"""
        visualizer = SocialNetworkVisualizer()

        visualizer.plot_with_pagerank(self.graph)
        plt.close()

        visualizer.plot_with_pagerank(self.graph, layout='circular')
        plt.close()

    def test_plot_with_communities(self):
        """测试带社区的图形绘制"""
        visualizer = SocialNetworkVisualizer()

        visualizer.plot_with_communities(self.graph)
        plt.close()

        visualizer.plot_with_communities(self.graph, layout='circular')
        plt.close()

    def test_plot_weighted_graph(self):
        """测试带权重的图形绘制"""
        visualizer = SocialNetworkVisualizer()

        visualizer.plot_weighted_graph(self.graph)
        plt.close()

        visualizer.plot_weighted_graph(self.graph, layout='circular')
        plt.close()

    def test_plot_degree_distribution(self):
        """测试度数分布图"""
        visualizer = SocialNetworkVisualizer()

        visualizer.plot_degree_distribution(self.graph)
        plt.close()

    def test_plot_pagerank_distribution(self):
        """测试PageRank分布图"""
        visualizer = SocialNetworkVisualizer()

        visualizer.plot_pagerank_distribution(self.graph)
        plt.close()

    def test_create_summary_dashboard(self):
        """测试创建分析仪表板"""
        visualizer = SocialNetworkVisualizer()

        visualizer.create_summary_dashboard(self.graph)
        plt.close()

    def test_export_graph_data(self):
        """测试导出图数据"""
        import tempfile
        import os

        visualizer = SocialNetworkVisualizer()

        with tempfile.TemporaryDirectory() as temp_dir:
            # 测试不同格式
            formats = ['gexf', 'graphml', 'gml', 'json']

            for format in formats:
                output_path = os.path.join(temp_dir, f"test_network.{format}")
                base_path = os.path.join(temp_dir, "test_network")

                visualizer.export_graph_data(self.graph, format, base_path)

                # 验证文件是否创建
                assert os.path.exists(f"{base_path}.{format}")

    def test_empty_graph_visualization(self):
        """测试空图的可视化"""
        empty_graph = SocialNetworkGraph()
        visualizer = SocialNetworkVisualizer()

        # 基础图
        visualizer.plot_basic_graph(empty_graph)
        plt.close()

        # 度数分布
        visualizer.plot_degree_distribution(empty_graph)
        plt.close()

        # PageRank分布
        visualizer.plot_pagerank_distribution(empty_graph)
        plt.close()

        # 仪表板
        visualizer.create_summary_dashboard(empty_graph)
        plt.close()

    def test_single_node_visualization(self):
        """测试单节点图的可视化"""
        single_graph = SocialNetworkGraph()
        single_graph.add_agent(1, "single_agent")

        visualizer = SocialNetworkVisualizer()

        # 基础图
        visualizer.plot_basic_graph(single_graph)
        plt.close()

        # PageRank图
        visualizer.plot_with_pagerank(single_graph)
        plt.close()

        # 社区图
        visualizer.plot_with_communities(single_graph)
        plt.close()

        # 度数分布
        visualizer.plot_degree_distribution(single_graph)
        plt.close()

        # PageRank分布
        visualizer.plot_pagerank_distribution(single_graph)
        plt.close()

        # 仪表板
        visualizer.create_summary_dashboard(single_graph)
        plt.close()

    def test_invalid_export_format(self):
        """测试无效的导出格式"""
        visualizer = SocialNetworkVisualizer()

        with pytest.raises(ValueError, match="Unsupported format"):
            visualizer.export_graph_data(self.graph, format='invalid')

    def test_complete_graph_visualization(self):
        """测试完全图的可视化"""
        complete_graph = SocialNetworkGraph()

        # 创建完全图
        for i in range(1, 6):
            complete_graph.add_agent(i, f"agent{i}")

        for i in range(1, 6):
            for j in range(i + 1, 6):
                complete_graph.add_friendship(i, j, strength=0.8)

        visualizer = SocialNetworkVisualizer()

        # 基础图
        visualizer.plot_basic_graph(complete_graph)
        plt.close()

        # PageRank图
        visualizer.plot_with_pagerank(complete_graph)
        plt.close()

        # 社区图
        visualizer.plot_with_communities(complete_graph)
        plt.close()

        # 权重图
        visualizer.plot_weighted_graph(complete_graph)
        plt.close()

        # 度数分布
        visualizer.plot_degree_distribution(complete_graph)
        plt.close()

        # PageRank分布
        visualizer.plot_pagerank_distribution(complete_graph)
        plt.close()

        # 仪表板
        visualizer.create_summary_dashboard(complete_graph)
        plt.close()