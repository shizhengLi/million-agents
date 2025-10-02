"""
社交网络图算法测试
"""

import pytest
import networkx as nx
from typing import Dict, List, Tuple

class TestSocialNetworkGraph:
    """测试社交网络图基础功能"""

    def test_graph_creation_with_agents(self):
        """测试创建包含Agent的图"""
        from src.social_network.graph import SocialNetworkGraph

        # 创建社交网络图
        graph = SocialNetworkGraph()

        # 添加Agent节点
        graph.add_agent(1, "agent1")
        graph.add_agent(2, "agent2")
        graph.add_agent(3, "agent3")

        # 验证节点数量
        assert graph.get_agent_count() == 3

        # 验证Agent存在
        assert graph.has_agent(1)
        assert graph.has_agent(2)
        assert graph.has_agent(3)
        assert not graph.has_agent(4)

    def test_add_friendship_edges(self):
        """测试添加好友关系边"""
        from src.social_network.graph import SocialNetworkGraph

        graph = SocialNetworkGraph()

        # 添加Agent
        graph.add_agent(1, "agent1")
        graph.add_agent(2, "agent2")
        graph.add_agent(3, "agent3")

        # 添加好友关系
        graph.add_friendship(1, 2, strength=0.8)
        graph.add_friendship(2, 3, strength=0.6)

        # 验证边数量
        assert graph.get_edge_count() == 2

        # 验证好友关系
        assert graph.are_friends(1, 2)
        assert graph.are_friends(2, 3)
        assert not graph.are_friends(1, 3)

    def test_get_agent_friends(self):
        """测试获取Agent的好友列表"""
        from src.social_network.graph import SocialNetworkGraph

        graph = SocialNetworkGraph()

        # 创建社交网络
        graph.add_agent(1, "agent1")
        graph.add_agent(2, "agent2")
        graph.add_agent(3, "agent3")
        graph.add_agent(4, "agent4")

        graph.add_friendship(1, 2)
        graph.add_friendship(1, 3)
        graph.add_friendship(2, 4)

        # 获取agent1的好友
        friends_1 = graph.get_agent_friends(1)
        assert len(friends_1) == 2
        assert 2 in friends_1
        assert 3 in friends_1

        # 获取agent2的好友
        friends_2 = graph.get_agent_friends(2)
        assert len(friends_2) == 2
        assert 1 in friends_2
        assert 4 in friends_2

    def test_remove_friendship(self):
        """测试删除好友关系"""
        from src.social_network.graph import SocialNetworkGraph

        graph = SocialNetworkGraph()

        graph.add_agent(1, "agent1")
        graph.add_agent(2, "agent2")

        graph.add_friendship(1, 2)
        assert graph.are_friends(1, 2)

        graph.remove_friendship(1, 2)
        assert not graph.are_friends(1, 2)

    def test_get_agent_degree(self):
        """测试获取Agent的度数（好友数量）"""
        from src.social_network.graph import SocialNetworkGraph

        graph = SocialNetworkGraph()

        graph.add_agent(1, "agent1")
        graph.add_agent(2, "agent2")
        graph.add_agent(3, "agent3")

        graph.add_friendship(1, 2)
        graph.add_friendship(1, 3)

        assert graph.get_agent_degree(1) == 2
        assert graph.get_agent_degree(2) == 1
        assert graph.get_agent_degree(3) == 1