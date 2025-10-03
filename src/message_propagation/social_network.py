"""
简化的社交网络类

用于消息传播模型的社交网络基础结构
"""

from typing import List, Dict, Set, Optional
from abc import ABC, abstractmethod


class SocialNetwork(ABC):
    """社交网络抽象基类"""

    @abstractmethod
    def get_neighbors(self, agent_id: str) -> List[str]:
        """
        获取智能体的邻居

        Args:
            agent_id: 智能体ID

        Returns:
            List[str]: 邻居智能体ID列表
        """
        pass

    @abstractmethod
    def get_agent_count(self) -> int:
        """
        获取网络中智能体总数

        Returns:
            int: 智能体总数
        """
        pass

    @abstractmethod
    def get_all_agents(self) -> List[str]:
        """
        获取所有智能体ID

        Returns:
            List[str]: 智能体ID列表
        """
        pass

    @abstractmethod
    def get_agent(self, agent_id: str) -> Optional[Dict]:
        """
        获取智能体信息

        Args:
            agent_id: 智能体ID

        Returns:
            Optional[Dict]: 智能体信息，如果不存在则返回None
        """
        pass


class MockSocialNetwork(SocialNetwork):
    """模拟社交网络实现"""

    def __init__(self, agents: List[str] = None, adjacency_list: Dict[str, List[str]] = None):
        """
        初始化模拟社交网络

        Args:
            agents: 智能体ID列表
            adjacency_list: 邻接表，表示网络连接关系
        """
        self.agents = agents or []
        self.adjacency_list = adjacency_list or {}
        self.agent_info: Dict[str, Dict] = {}

        # 初始化智能体信息
        for agent in self.agents:
            self.agent_info[agent] = {
                'id': agent,
                'type': 'social_agent',
                'status': 'active'
            }

    def get_neighbors(self, agent_id: str) -> List[str]:
        """获取智能体的邻居"""
        return self.adjacency_list.get(agent_id, [])

    def get_agent_count(self) -> int:
        """获取网络中智能体总数"""
        return len(self.agents)

    def get_all_agents(self) -> List[str]:
        """获取所有智能体ID"""
        return self.agents.copy()

    def get_agent(self, agent_id: str) -> Optional[Dict]:
        """获取智能体信息"""
        return self.agent_info.get(agent_id)

    def add_agent(self, agent_id: str, neighbors: List[str] = None) -> None:
        """
        添加智能体到网络

        Args:
            agent_id: 智能体ID
            neighbors: 邻居列表
        """
        if agent_id not in self.agents:
            self.agents.append(agent_id)
            self.agent_info[agent_id] = {
                'id': agent_id,
                'type': 'social_agent',
                'status': 'active'
            }

        if neighbors:
            self.adjacency_list[agent_id] = neighbors

    def add_connection(self, agent1: str, agent2: str) -> None:
        """
        添加连接关系

        Args:
            agent1: 智能体1
            agent2: 智能体2
        """
        if agent1 not in self.adjacency_list:
            self.adjacency_list[agent1] = []
        if agent2 not in self.adjacency_list:
            self.adjacency_list[agent2] = []

        if agent2 not in self.adjacency_list[agent1]:
            self.adjacency_list[agent1].append(agent2)
        if agent1 not in self.adjacency_list[agent2]:
            self.adjacency_list[agent2].append(agent1)

    def remove_agent(self, agent_id: str) -> None:
        """
        从网络中移除智能体

        Args:
            agent_id: 智能体ID
        """
        if agent_id in self.agents:
            self.agents.remove(agent_id)
        if agent_id in self.agent_info:
            del self.agent_info[agent_id]
        if agent_id in self.adjacency_list:
            del self.adjacency_list[agent_id]

        # 从其他智能体的邻居列表中移除
        for neighbors in self.adjacency_list.values():
            if agent_id in neighbors:
                neighbors.remove(agent_id)

    def get_degree(self, agent_id: str) -> int:
        """
        获取智能体的度数

        Args:
            agent_id: 智能体ID

        Returns:
            int: 度数
        """
        return len(self.get_neighbors(agent_id))

    def get_network_density(self) -> float:
        """
        计算网络密度

        Returns:
            float: 网络密度 (0-1)
        """
        n = len(self.agents)
        if n <= 1:
            return 0.0

        total_edges = sum(len(neighbors) for neighbors in self.adjacency_list.values())
        max_edges = n * (n - 1) / 2
        return total_edges / max_edges if max_edges > 0 else 0.0