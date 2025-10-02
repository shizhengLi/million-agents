"""
社交网络图数据结构
"""

from typing import Dict, List, Set, Optional, Tuple
import networkx as nx
from dataclasses import dataclass


@dataclass
class Agent:
    """Agent节点信息"""
    id: int
    name: str


class SocialNetworkGraph:
    """社交网络图类"""

    def __init__(self):
        """初始化社交网络图"""
        self.graph = nx.Graph()
        self.agents: Dict[int, Agent] = {}

    def add_agent(self, agent_id: int, name: str) -> None:
        """添加Agent节点到图中"""
        if agent_id in self.agents:
            raise ValueError(f"Agent {agent_id} already exists")

        agent = Agent(id=agent_id, name=name)
        self.agents[agent_id] = agent
        self.graph.add_node(agent_id, name=name)

    def has_agent(self, agent_id: int) -> bool:
        """检查Agent是否存在"""
        return agent_id in self.agents

    def get_agent_count(self) -> int:
        """获取Agent数量"""
        return len(self.agents)

    def add_friendship(self, agent1_id: int, agent2_id: int, strength: float = 1.0) -> None:
        """添加好友关系（边）"""
        if not self.has_agent(agent1_id):
            raise ValueError(f"Agent {agent1_id} does not exist")
        if not self.has_agent(agent2_id):
            raise ValueError(f"Agent {agent2_id} does not exist")
        if agent1_id == agent2_id:
            raise ValueError("Agent cannot be friends with themselves")

        # 添加无向边，权重为关系强度
        self.graph.add_edge(agent1_id, agent2_id, weight=strength)

    def are_friends(self, agent1_id: int, agent2_id: int) -> bool:
        """检查两个Agent是否是好友"""
        return self.graph.has_edge(agent1_id, agent2_id)

    def get_edge_count(self) -> int:
        """获取边的数量（好友关系数量）"""
        return self.graph.number_of_edges()

    def get_agent_friends(self, agent_id: int) -> List[int]:
        """获取Agent的所有好友ID列表"""
        if not self.has_agent(agent_id):
            raise ValueError(f"Agent {agent_id} does not exist")

        return list(self.graph.neighbors(agent_id))

    def remove_friendship(self, agent1_id: int, agent2_id: int) -> None:
        """删除好友关系"""
        if self.graph.has_edge(agent1_id, agent2_id):
            self.graph.remove_edge(agent1_id, agent2_id)

    def get_agent_degree(self, agent_id: int) -> int:
        """获取Agent的度数（好友数量）"""
        if not self.has_agent(agent_id):
            raise ValueError(f"Agent {agent_id} does not exist")

        return self.graph.degree(agent_id)

    def get_friendship_strength(self, agent1_id: int, agent2_id: int) -> Optional[float]:
        """获取好友关系强度"""
        if not self.are_friends(agent1_id, agent2_id):
            return None

        return self.graph[agent1_id][agent2_id].get('weight', 1.0)

    def update_friendship_strength(self, agent1_id: int, agent2_id: int, strength: float) -> None:
        """更新好友关系强度"""
        if not self.are_friends(agent1_id, agent2_id):
            raise ValueError("Agents are not friends")

        if not (0.0 <= strength <= 1.0):
            raise ValueError("Strength must be between 0.0 and 1.0")

        self.graph[agent1_id][agent2_id]['weight'] = strength

    def get_all_agents(self) -> List[Agent]:
        """获取所有Agent列表"""
        return list(self.agents.values())

    def get_agent_by_id(self, agent_id: int) -> Optional[Agent]:
        """根据ID获取Agent"""
        return self.agents.get(agent_id)

    def get_network_density(self) -> float:
        """计算网络密度"""
        if self.get_agent_count() < 2:
            return 0.0

        return nx.density(self.graph)

    def get_connected_components(self) -> List[Set[int]]:
        """获取连通分量"""
        return [set(component) for component in nx.connected_components(self.graph)]

    def get_largest_component_size(self) -> int:
        """获取最大连通分量的大小"""
        if not self.agents:
            return 0

        return max(len(component) for component in nx.connected_components(self.graph))

    def remove_agent(self, agent_id: int) -> None:
        """删除Agent及其所有关系"""
        if not self.has_agent(agent_id):
            raise ValueError(f"Agent {agent_id} does not exist")

        self.graph.remove_node(agent_id)
        del self.agents[agent_id]

    def clear(self) -> None:
        """清空图"""
        self.graph.clear()
        self.agents.clear()

    def __len__(self) -> int:
        """返回Agent数量"""
        return self.get_agent_count()

    def __contains__(self, agent_id: int) -> bool:
        """检查Agent是否存在"""
        return self.has_agent(agent_id)