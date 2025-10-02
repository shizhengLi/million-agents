"""
PageRank算法测试
"""

import pytest
from src.social_network.graph import SocialNetworkGraph
from src.social_network.algorithms import PageRankCalculator


class TestPageRankCalculator:
    """测试PageRank算法"""

    def test_simple_pagerank_calculation(self):
        """测试简单的PageRank计算"""
        # 创建一个简单的图：1 <-> 2 <-> 3
        graph = SocialNetworkGraph()

        # 添加节点
        graph.add_agent(1, "agent1")
        graph.add_agent(2, "agent2")
        graph.add_agent(3, "agent3")

        # 添加边（形成一条线）
        graph.add_friendship(1, 2)
        graph.add_friendship(2, 3)

        # 计算PageRank
        calculator = PageRankCalculator()
        rankings = calculator.calculate_pagerank(graph)

        # 验证结果
        assert len(rankings) == 3
        assert all(agent_id in rankings for agent_id in [1, 2, 3])

        # PageRank值应该在0到1之间
        for score in rankings.values():
            assert 0 <= score <= 1

        # 所有PageRank值的和应该接近1
        total_score = sum(rankings.values())
        assert abs(total_score - 1.0) < 0.01

    def test_star_graph_pagerank(self):
        """测试星形图的PageRank计算"""
        graph = SocialNetworkGraph()

        # 创建星形图：中心节点1连接到其他所有节点
        graph.add_agent(1, "center")
        for i in range(2, 6):
            graph.add_agent(i, f"agent{i}")
            graph.add_friendship(1, i)

        # 计算PageRank
        calculator = PageRankCalculator()
        rankings = calculator.calculate_pagerank(graph)

        # 中心节点应该有最高的PageRank值
        assert rankings[1] > rankings[2]
        assert rankings[1] > rankings[3]
        assert rankings[1] > rankings[4]
        assert rankings[1] > rankings[5]

        # 叶子节点的PageRank值应该相等
        leaf_scores = [rankings[i] for i in range(2, 6)]
        assert all(abs(score - leaf_scores[0]) < 0.001 for score in leaf_scores)

    def test_complete_graph_pagerank(self):
        """测试完全图的PageRank计算"""
        graph = SocialNetworkGraph()

        # 创建完全图：所有节点互相连接
        for i in range(1, 5):
            graph.add_agent(i, f"agent{i}")

        for i in range(1, 5):
            for j in range(i + 1, 5):
                graph.add_friendship(i, j)

        # 计算PageRank
        calculator = PageRankCalculator()
        rankings = calculator.calculate_pagerank(graph)

        # 在完全图中，所有节点的PageRank应该相等
        scores = list(rankings.values())
        for score in scores:
            assert abs(score - scores[0]) < 0.001

    def test_pagerank_with_dangling_node(self):
        """测试包含悬挂节点的PageRank计算"""
        graph = SocialNetworkGraph()

        # 创建图：1-2-3，4是孤立的
        graph.add_agent(1, "agent1")
        graph.add_agent(2, "agent2")
        graph.add_agent(3, "agent3")
        graph.add_agent(4, "isolated")

        graph.add_friendship(1, 2)
        graph.add_friendship(2, 3)
        # 节点4没有连接

        # 计算PageRank
        calculator = PageRankCalculator()
        rankings = calculator.calculate_pagerank(graph)

        # 验证所有节点都有PageRank值
        assert len(rankings) == 4
        assert all(agent_id in rankings for agent_id in [1, 2, 3, 4])

        # 悬挂节点应该有最低的PageRank
        assert rankings[4] < rankings[1]
        assert rankings[4] < rankings[2]
        assert rankings[4] < rankings[3]

    def test_pagerank_with_weighted_edges(self):
        """测试带权重边的PageRank计算"""
        graph = SocialNetworkGraph()

        # 添加节点
        graph.add_agent(1, "agent1")
        graph.add_agent(2, "agent2")
        graph.add_agent(3, "agent3")

        # 添加带权重的边
        graph.add_friendship(1, 2, strength=0.9)  # 强连接
        graph.add_friendship(2, 3, strength=0.1)  # 弱连接

        # 计算PageRank
        calculator = PageRankCalculator()
        rankings = calculator.calculate_pagerank(graph)

        # 验证结果
        assert len(rankings) == 3

        # 强连接应该影响PageRank分布
        # 节点2位于强连接和弱连接之间，应该有合理的PageRank值

    def test_pagerank_empty_graph(self):
        """测试空图的PageRank计算"""
        graph = SocialNetworkGraph()

        calculator = PageRankCalculator()
        rankings = calculator.calculate_pagerank(graph)

        # 空图应该返回空字典
        assert rankings == {}

    def test_pagerank_single_node(self):
        """测试单节点图的PageRank计算"""
        graph = SocialNetworkGraph()
        graph.add_agent(1, "single_agent")

        calculator = PageRankCalculator()
        rankings = calculator.calculate_pagerank(graph)

        # 单节点图的PageRank应该是1
        assert len(rankings) == 1
        assert rankings[1] == 1.0

    def test_pagerank_custom_parameters(self):
        """测试自定义参数的PageRank计算"""
        graph = SocialNetworkGraph()

        # 创建简单图
        for i in range(1, 4):
            graph.add_agent(i, f"agent{i}")

        graph.add_friendship(1, 2)
        graph.add_friendship(2, 3)

        # 使用自定义参数
        calculator = PageRankCalculator()
        rankings = calculator.calculate_pagerank(
            graph,
            damping_factor=0.8,  # 默认是0.85
            max_iterations=50,    # 默认是100
            tolerance=1e-5        # 默认是1e-6
        )

        # 验证结果
        assert len(rankings) == 3
        total_score = sum(rankings.values())
        assert abs(total_score - 1.0) < 0.01

    def test_get_top_influential_agents(self):
        """测试获取影响力最大的Agent"""
        graph = SocialNetworkGraph()

        # 创建一个有明显影响力的图
        graph.add_agent(1, "hub_agent")
        for i in range(2, 8):
            graph.add_agent(i, f"agent{i}")
            graph.add_friendship(1, i)

        # 添加一些额外连接
        graph.add_friendship(2, 3)
        graph.add_friendship(4, 5)

        calculator = PageRankCalculator()
        top_agents = calculator.get_top_influential_agents(graph, top_k=3)

        # 验证结果
        assert len(top_agents) <= 3
        assert top_agents[0][0] == 1  # hub_agent应该排在第一位
        assert len(top_agents[0]) == 2  # (agent_id, pagerank_score)

        # 按PageRank值降序排列
        for i in range(len(top_agents) - 1):
            assert top_agents[i][1] >= top_agents[i + 1][1]