"""
社区发现算法测试
"""

import pytest
from src.social_network.graph import SocialNetworkGraph
from src.social_network.algorithms import CommunityDetector


class TestCommunityDetector:
    """测试社区发现算法"""

    def test_simple_two_communities(self):
        """测试简单的两个社区"""
        # 创建两个明显分离的社区
        graph = SocialNetworkGraph()

        # 社区1: 节点1,2,3完全连接
        graph.add_agent(1, "agent1")
        graph.add_agent(2, "agent2")
        graph.add_agent(3, "agent3")

        graph.add_friendship(1, 2)
        graph.add_friendship(1, 3)
        graph.add_friendship(2, 3)

        # 社区2: 节点4,5,6完全连接
        graph.add_agent(4, "agent4")
        graph.add_agent(5, "agent5")
        graph.add_agent(6, "agent6")

        graph.add_friendship(4, 5)
        graph.add_friendship(4, 6)
        graph.add_friendship(5, 6)

        # 社区1和社区2之间只有一个弱连接
        graph.add_friendship(3, 4, strength=0.1)

        # 检测社区
        detector = CommunityDetector()
        communities = detector.detect_communities(graph)

        # 验证结果
        assert len(communities) >= 2  # 应该至少检测到2个社区

        # 验证所有节点都被分配到某个社区
        all_nodes = set()
        for community in communities:
            all_nodes.update(community)

        assert all_nodes == {1, 2, 3, 4, 5, 6}

    def test_single_large_community(self):
        """测试单个大社区"""
        graph = SocialNetworkGraph()

        # 创建一个完全连接的图（应该是一个社区）
        for i in range(1, 7):
            graph.add_agent(i, f"agent{i}")

        # 完全连接
        for i in range(1, 7):
            for j in range(i + 1, 7):
                graph.add_friendship(i, j)

        # 检测社区
        detector = CommunityDetector()
        communities = detector.detect_communities(graph)

        # 在完全连接的图中，应该检测到1个主要社区
        assert len(communities) >= 1

        # 所有节点应该都在一个社区中（或社区数量很少）
        if len(communities) == 1:
            assert len(communities[0]) == 6

    def test_disconnected_communities(self):
        """测试完全断开的社区"""
        graph = SocialNetworkGraph()

        # 社区1: 三个完全连接的节点
        for i in range(1, 4):
            graph.add_agent(i, f"agent{i}")

        for i in range(1, 4):
            for j in range(i + 1, 4):
                graph.add_friendship(i, j)

        # 社区2: 两个完全连接的节点
        graph.add_agent(4, "agent4")
        graph.add_agent(5, "agent5")
        graph.add_friendship(4, 5)

        # 孤立节点
        graph.add_agent(6, "isolated")

        # 检测社区
        detector = CommunityDetector()
        communities = detector.detect_communities(graph)

        # 应该检测到多个社区
        assert len(communities) >= 3

        # 验证节点分配
        all_nodes = set()
        for community in communities:
            all_nodes.update(community)

        assert all_nodes == {1, 2, 3, 4, 5, 6}

    def test_star_community_detection(self):
        """测试星形社区的检测"""
        graph = SocialNetworkGraph()

        # 创建星形图：中心节点连接所有其他节点
        graph.add_agent(1, "center")
        for i in range(2, 8):
            graph.add_agent(i, f"leaf{i}")
            graph.add_friendship(1, i)

        # 检测社区
        detector = CommunityDetector()
        communities = detector.detect_communities(graph)

        # 星形图通常被识别为单个社区
        assert len(communities) >= 1

        # 验证所有节点都被分配
        all_nodes = set()
        for community in communities:
            all_nodes.update(community)

        assert all_nodes == set(range(1, 8))

    def test_line_graph_communities(self):
        """测试线性图的社区检测"""
        graph = SocialNetworkGraph()

        # 创建线性图: 1-2-3-4-5-6
        for i in range(1, 7):
            graph.add_agent(i, f"agent{i}")

        for i in range(1, 6):
            graph.add_friendship(i, i + 1)

        # 检测社区
        detector = CommunityDetector()
        communities = detector.detect_communities(graph)

        # 线性图可能被分割成多个社区
        assert len(communities) >= 1

        # 验证节点分配
        all_nodes = set()
        for community in communities:
            all_nodes.update(community)

        assert all_nodes == set(range(1, 7))

    def test_empty_graph_community_detection(self):
        """测试空图的社区检测"""
        graph = SocialNetworkGraph()

        detector = CommunityDetector()
        communities = detector.detect_communities(graph)

        # 空图应该返回空列表
        assert communities == []

    def test_single_node_community_detection(self):
        """测试单节点图的社区检测"""
        graph = SocialNetworkGraph()
        graph.add_agent(1, "single")

        detector = CommunityDetector()
        communities = detector.detect_communities(graph)

        # 单节点图应该返回一个包含该节点的社区
        assert len(communities) == 1
        assert communities[0] == {1}

    def test_weighted_community_detection(self):
        """测试带权重边的社区检测"""
        graph = SocialNetworkGraph()

        # 添加节点
        for i in range(1, 9):
            graph.add_agent(i, f"agent{i}")

        # 社区1: 强连接 (权重0.9)
        for i in range(1, 5):
            for j in range(i + 1, 5):
                graph.add_friendship(i, j, strength=0.9)

        # 社区2: 强连接 (权重0.9)
        for i in range(5, 9):
            for j in range(i + 1, 9):
                graph.add_friendship(i, j, strength=0.9)

        # 社区间弱连接 (权重0.1)
        graph.add_friendship(4, 5, strength=0.1)

        # 检测社区
        detector = CommunityDetector()
        communities = detector.detect_communities(graph)

        # 应该检测到两个主要社区
        assert len(communities) >= 2

    def test_get_community_statistics(self):
        """测试社区统计信息"""
        graph = SocialNetworkGraph()

        # 创建两个不同大小的社区
        # 社区1: 4个节点
        for i in range(1, 5):
            graph.add_agent(i, f"agent{i}")

        for i in range(1, 5):
            for j in range(i + 1, 5):
                graph.add_friendship(i, j)

        # 社区2: 2个节点
        graph.add_agent(5, "agent5")
        graph.add_agent(6, "agent6")
        graph.add_friendship(5, 6)

        # 社区间弱连接
        graph.add_friendship(2, 5, strength=0.1)

        detector = CommunityDetector()
        communities = detector.detect_communities(graph)
        stats = detector.get_community_statistics(graph, communities)

        # 验证统计信息
        assert 'num_communities' in stats
        assert 'community_sizes' in stats
        assert 'largest_community_size' in stats
        assert 'smallest_community_size' in stats
        assert 'average_community_size' in stats

        assert stats['num_communities'] >= 2
        assert stats['largest_community_size'] >= 4
        assert stats['smallest_community_size'] >= 2

    def test_get_agent_community_assignment(self):
        """测试获取Agent的社区分配"""
        graph = SocialNetworkGraph()

        # 创建两个明显分离的社区
        # 社区1
        for i in range(1, 4):
            graph.add_agent(i, f"agent{i}")

        for i in range(1, 4):
            for j in range(i + 1, 4):
                graph.add_friendship(i, j)

        # 社区2
        for i in range(4, 7):
            graph.add_agent(i, f"agent{i}")

        for i in range(4, 7):
            for j in range(i + 1, 7):
                graph.add_friendship(i, j)

        detector = CommunityDetector()
        communities = detector.detect_communities(graph)
        assignment = detector.get_agent_community_assignment(communities)

        # 验证分配结果
        assert len(assignment) == 6

        # 社区1的节点应该分配到同一个社区ID
        community_1_id = assignment[1]
        assert assignment[2] == community_1_id
        assert assignment[3] == community_1_id

        # 社区2的节点应该分配到另一个社区ID
        community_2_id = assignment[4]
        assert assignment[5] == community_2_id
        assert assignment[6] == community_2_id

        # 两个社区的ID应该不同
        assert community_1_id != community_2_id

    def test_louvain_method_parameters(self):
        """测试Louvain方法的参数"""
        graph = SocialNetworkGraph()

        # 创建简单的测试图
        for i in range(1, 7):
            graph.add_agent(i, f"agent{i}")

        # 创建两个三角形连接
        for i in range(1, 4):
            for j in range(i + 1, 4):
                graph.add_friendship(i, j)

        for i in range(4, 7):
            for j in range(i + 1, 7):
                graph.add_friendship(i, j)

        graph.add_friendship(3, 4)

        detector = CommunityDetector()

        # 测试不同参数
        communities1 = detector.detect_communities(graph, resolution=1.0)
        communities2 = detector.detect_communities(graph, resolution=0.5)

        # 不同参数可能产生不同的社区划分
        assert len(communities1) >= 1
        assert len(communities2) >= 1

        # 验证所有节点都被分配
        for communities in [communities1, communities2]:
            all_nodes = set()
            for community in communities:
                all_nodes.update(community)
            assert all_nodes == set(range(1, 7))