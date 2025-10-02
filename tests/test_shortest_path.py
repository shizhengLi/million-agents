"""
最短路径算法测试
"""

import pytest
from src.social_network.graph import SocialNetworkGraph
from src.social_network.algorithms import ShortestPathCalculator


class TestShortestPathCalculator:
    """测试最短路径算法"""

    def test_simple_shortest_path(self):
        """测试简单的最短路径计算"""
        # 创建一条线：1-2-3-4
        graph = SocialNetworkGraph()

        # 添加节点
        for i in range(1, 5):
            graph.add_agent(i, f"agent{i}")

        # 添加边（形成一条线）
        graph.add_friendship(1, 2)
        graph.add_friendship(2, 3)
        graph.add_friendship(3, 4)

        # 计算最短路径
        calculator = ShortestPathCalculator()
        path = calculator.calculate_shortest_path(graph, 1, 4)

        # 验证结果
        assert path == [1, 2, 3, 4]

    def test_direct_connection(self):
        """测试直接连接的最短路径"""
        graph = SocialNetworkGraph()

        # 添加节点
        graph.add_agent(1, "agent1")
        graph.add_agent(2, "agent2")

        # 直接连接
        graph.add_friendship(1, 2)

        # 计算最短路径
        calculator = ShortestPathCalculator()
        path = calculator.calculate_shortest_path(graph, 1, 2)

        # 验证结果
        assert path == [1, 2]

    def test_no_path_exists(self):
        """测试不存在路径的情况"""
        graph = SocialNetworkGraph()

        # 添加两个不连通的节点
        graph.add_agent(1, "agent1")
        graph.add_agent(2, "agent2")

        # 计算最短路径
        calculator = ShortestPathCalculator()
        path = calculator.calculate_shortest_path(graph, 1, 2)

        # 验证结果
        assert path is None

    def test_multiple_paths(self):
        """测试多条路径的情况，选择最短的一条"""
        graph = SocialNetworkGraph()

        # 添加节点
        for i in range(1, 6):
            graph.add_agent(i, f"agent{i}")

        # 创建多条路径：
        # 路径1：1-2-5 (长度2)
        graph.add_friendship(1, 2)
        graph.add_friendship(2, 5)

        # 路径2：1-3-4-5 (长度3)
        graph.add_friendship(1, 3)
        graph.add_friendship(3, 4)
        graph.add_friendship(4, 5)

        # 计算最短路径
        calculator = ShortestPathCalculator()
        path = calculator.calculate_shortest_path(graph, 1, 5)

        # 应该选择较短的路径
        assert path == [1, 2, 5]

    def test_same_start_and_end(self):
        """测试起点和终点相同的情况"""
        graph = SocialNetworkGraph()

        # 添加节点
        graph.add_agent(1, "agent1")

        # 计算最短路径
        calculator = ShortestPathCalculator()
        path = calculator.calculate_shortest_path(graph, 1, 1)

        # 验证结果
        assert path == [1]

    def test_weighted_shortest_path(self):
        """测试带权重的最短路径"""
        graph = SocialNetworkGraph()

        # 添加节点
        for i in range(1, 5):
            graph.add_agent(i, f"agent{i}")

        # 创建带权重的边
        # 路径1：1-2-4，权重都是0.9（总权重高）
        graph.add_friendship(1, 2, strength=0.9)
        graph.add_friendship(2, 4, strength=0.9)

        # 路径2：1-3-4，权重都是0.1（总权重低，但强度小意味着关系弱）
        graph.add_friendship(1, 3, strength=0.1)
        graph.add_friendship(3, 4, strength=0.1)

        # 计算最短路径（在社交网络中，强度越高通常意味着越短的距离）
        calculator = ShortestPathCalculator()
        path = calculator.calculate_shortest_path(graph, 1, 4, use_weights=True)

        # 验证结果（应该选择强度高的路径）
        assert path == [1, 2, 4]

    def test_get_all_shortest_paths(self):
        """测试获取所有节点对之间的最短路径"""
        graph = SocialNetworkGraph()

        # 添加节点
        for i in range(1, 5):
            graph.add_agent(i, f"agent{i}")

        # 创建一个简单的图
        graph.add_friendship(1, 2)
        graph.add_friendship(2, 3)
        graph.add_friendship(3, 4)

        # 计算所有最短路径
        calculator = ShortestPathCalculator()
        all_paths = calculator.get_all_shortest_paths(graph)

        # 验证结果
        assert len(all_paths) > 0

        # 检查特定路径
        assert (1, 4) in all_paths
        assert all_paths[(1, 4)] == [1, 2, 3, 4]

    def test_get_path_length(self):
        """测试获取路径长度"""
        graph = SocialNetworkGraph()

        # 添加节点
        for i in range(1, 5):
            graph.add_agent(i, f"agent{i}")

        # 创建路径
        graph.add_friendship(1, 2)
        graph.add_friendship(2, 3)
        graph.add_friendship(3, 4)

        # 计算最短路径
        calculator = ShortestPathCalculator()
        path = calculator.calculate_shortest_path(graph, 1, 4)
        length = calculator.get_path_length(path)

        # 验证结果
        assert length == 3  # 1-2-3-4 有3条边

    def test_get_path_weight(self):
        """测试获取路径权重"""
        graph = SocialNetworkGraph()

        # 添加节点
        for i in range(1, 4):
            graph.add_agent(i, f"agent{i}")

        # 创建带权重的路径
        graph.add_friendship(1, 2, strength=0.8)
        graph.add_friendship(2, 3, strength=0.6)

        # 计算最短路径
        calculator = ShortestPathCalculator()
        path = calculator.calculate_shortest_path(graph, 1, 3)
        weight = calculator.get_path_weight(graph, path)

        # 验证结果
        assert weight == 1.4  # 0.8 + 0.6

    def test_calculate_average_path_length(self):
        """测试计算平均路径长度"""
        graph = SocialNetworkGraph()

        # 创建一个完全图
        for i in range(1, 5):
            graph.add_agent(i, f"agent{i}")

        for i in range(1, 5):
            for j in range(i + 1, 5):
                graph.add_friendship(i, j)

        # 计算平均路径长度
        calculator = ShortestPathCalculator()
        avg_length = calculator.calculate_average_path_length(graph)

        # 在完全图中，平均路径长度应该接近1
        assert 1.0 <= avg_length <= 1.5

    def test_get_diameter(self):
        """测试获取图的直径"""
        graph = SocialNetworkGraph()

        # 创建一条线：1-2-3-4-5
        for i in range(1, 6):
            graph.add_agent(i, f"agent{i}")

        for i in range(1, 5):
            graph.add_friendship(i, i + 1)

        # 计算图的直径
        calculator = ShortestPathCalculator()
        diameter = calculator.get_diameter(graph)

        # 在线性图中，直径应该是4（1到5的距离）
        assert diameter == 4

    def test_empty_graph_shortest_path(self):
        """测试空图的最短路径计算"""
        graph = SocialNetworkGraph()

        calculator = ShortestPathCalculator()

        # 尝试计算不存在的节点之间的路径
        path = calculator.calculate_shortest_path(graph, 1, 2)

        # 验证结果
        assert path is None

    def test_single_node_shortest_path(self):
        """测试单节点图的最短路径计算"""
        graph = SocialNetworkGraph()
        graph.add_agent(1, "single")

        calculator = ShortestPathCalculator()

        # 计算到自身的路径
        path = calculator.calculate_shortest_path(graph, 1, 1)

        # 验证结果
        assert path == [1]

    def test_invalid_nodes(self):
        """测试无效节点的处理"""
        graph = SocialNetworkGraph()

        calculator = ShortestPathCalculator()

        # 尝试计算不存在节点的路径
        path1 = calculator.calculate_shortest_path(graph, 999, 1)
        path2 = calculator.calculate_shortest_path(graph, 1, 999)

        # 验证结果
        assert path1 is None
        assert path2 is None