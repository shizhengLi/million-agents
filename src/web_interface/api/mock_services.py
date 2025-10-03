"""
Mock Services for Testing
用于测试的模拟服务
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import random


class MockAgentService:
    """模拟智能体服务"""

    def __init__(self):
        self.agents = self._initialize_mock_agents()
        self.next_id = len(self.agents) + 1

    def _initialize_mock_agents(self) -> List[Dict[str, Any]]:
        """初始化模拟智能体数据"""
        return [
            {
                "id": "agent_1",
                "name": "Social Agent Alpha",
                "type": "social",
                "status": "active",
                "reputation_score": 85.5,
                "description": "Active social interaction agent",
                "created_at": datetime.utcnow() - timedelta(days=30),
                "updated_at": datetime.utcnow() - timedelta(hours=2),
                "last_active": datetime.utcnow() - timedelta(minutes=15)
            },
            {
                "id": "agent_2",
                "name": "Content Creator Beta",
                "type": "content",
                "status": "active",
                "reputation_score": 72.3,
                "description": "Content generation and curation agent",
                "created_at": datetime.utcnow() - timedelta(days=25),
                "updated_at": datetime.utcnow() - timedelta(hours=5),
                "last_active": datetime.utcnow() - timedelta(minutes=30)
            },
            {
                "id": "agent_3",
                "name": "Hybrid Agent Gamma",
                "type": "hybrid",
                "status": "inactive",
                "reputation_score": 68.9,
                "description": "Multi-purpose hybrid agent",
                "created_at": datetime.utcnow() - timedelta(days=20),
                "updated_at": datetime.utcnow() - timedelta(days=1),
                "last_active": datetime.utcnow() - timedelta(hours=3)
            }
        ]

    async def get_all_agents(self) -> List[Dict[str, Any]]:
        """获取所有智能体"""
        return self.agents.copy()

    async def get_agent_by_id(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取智能体"""
        for agent in self.agents:
            if agent["id"] == agent_id:
                return agent.copy()
        return None

    async def create_agent(self, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建智能体"""
        new_agent = {
            "id": f"agent_{self.next_id}",
            "name": agent_data["name"],
            "type": agent_data["type"],
            "status": "active",
            "reputation_score": 50.0,
            "description": agent_data.get("description", ""),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "last_active": datetime.utcnow()
        }
        self.agents.append(new_agent)
        self.next_id += 1
        return new_agent.copy()

    async def update_agent(self, agent_id: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """更新智能体"""
        for i, agent in enumerate(self.agents):
            if agent["id"] == agent_id:
                if "name" in update_data:
                    self.agents[i]["name"] = update_data["name"]
                if "description" in update_data:
                    self.agents[i]["description"] = update_data["description"]
                if "status" in update_data:
                    self.agents[i]["status"] = update_data["status"]
                self.agents[i]["updated_at"] = datetime.utcnow()
                return self.agents[i].copy()
        return None

    async def delete_agent(self, agent_id: str) -> bool:
        """删除智能体"""
        for i, agent in enumerate(self.agents):
            if agent["id"] == agent_id:
                del self.agents[i]
                return True
        return False

    async def get_agent_stats(self) -> Dict[str, Any]:
        """获取智能体统计"""
        total = len(self.agents)
        active = len([a for a in self.agents if a["status"] == "active"])

        type_distribution = {}
        for agent in self.agents:
            agent_type = agent["type"]
            type_distribution[agent_type] = type_distribution.get(agent_type, 0) + 1

        return {
            "total_agents": total,
            "active_agents": active,
            "inactive_agents": total - active,
            "agent_types": type_distribution
        }

    async def get_reputation_metrics(self) -> Dict[str, Any]:
        """获取声誉指标"""
        if not self.agents:
            return {
                "total_agents": 0,
                "active_agents": 0,
                "average_reputation": 0.0,
                "reputation_distribution": {"high": 0, "medium": 0, "low": 0},
                "recent_changes": []
            }

        total = len(self.agents)
        active = len([a for a in self.agents if a["status"] == "active"])
        avg_reputation = sum(a["reputation_score"] for a in self.agents) / total

        # 声誉分布
        high = len([a for a in self.agents if a["reputation_score"] >= 80])
        medium = len([a for a in self.agents if 60 <= a["reputation_score"] < 80])
        low = len([a for a in self.agents if a["reputation_score"] < 60])

        # 模拟最近变化
        recent_changes = []
        if self.agents:
            agent = random.choice(self.agents)
            recent_changes.append({
                "agent_id": agent["id"],
                "old_score": round(agent["reputation_score"] - random.uniform(1, 5), 1),
                "new_score": agent["reputation_score"],
                "timestamp": datetime.utcnow() - timedelta(minutes=random.randint(5, 60))
            })

        return {
            "total_agents": total,
            "active_agents": active,
            "average_reputation": round(avg_reputation, 1),
            "reputation_distribution": {"high": high, "medium": medium, "low": low},
            "recent_changes": recent_changes
        }


class MockSocialNetworkService:
    """模拟社交网络服务"""

    def __init__(self):
        self.connections = self._initialize_mock_connections()

    def _initialize_mock_connections(self) -> List[Dict[str, Any]]:
        """初始化模拟连接数据"""
        return [
            {
                "source": "agent_1",
                "target": "agent_2",
                "weight": 0.8,
                "relationship_type": "friend",
                "last_interaction": datetime.utcnow() - timedelta(hours=1)
            },
            {
                "source": "agent_1",
                "target": "agent_3",
                "weight": 0.6,
                "relationship_type": "colleague",
                "last_interaction": datetime.utcnow() - timedelta(hours=3)
            },
            {
                "source": "agent_2",
                "target": "agent_3",
                "weight": 0.4,
                "relationship_type": "acquaintance",
                "last_interaction": datetime.utcnow() - timedelta(days=1)
            }
        ]

    async def get_network_data(self, limit: int = 100) -> Dict[str, Any]:
        """获取网络数据"""
        nodes = [
            {
                "id": "agent_1",
                "name": "Social Agent Alpha",
                "group": "social",
                "reputation_score": 85.5,
                "status": "active"
            },
            {
                "id": "agent_2",
                "name": "Content Creator Beta",
                "group": "content",
                "reputation_score": 72.3,
                "status": "active"
            },
            {
                "id": "agent_3",
                "name": "Hybrid Agent Gamma",
                "group": "hybrid",
                "reputation_score": 68.9,
                "status": "inactive"
            }
        ]

        edges = [
            {
                "source": "agent_1",
                "target": "agent_2",
                "weight": 0.8,
                "relationship_type": "friend",
                "last_interaction": datetime.utcnow() - timedelta(hours=1)
            },
            {
                "source": "agent_1",
                "target": "agent_3",
                "weight": 0.6,
                "relationship_type": "colleague",
                "last_interaction": datetime.utcnow() - timedelta(hours=3)
            }
        ]

        metrics = {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "density": round(len(edges) / (len(nodes) * (len(nodes) - 1) / 2), 3),
            "clustering_coefficient": round(random.uniform(0.2, 0.8), 3)
        }

        return {
            "nodes": nodes[:limit],
            "edges": edges[:limit * 2],
            "metrics": metrics
        }

    async def get_agent_connections(self, agent_id: str) -> List[Dict[str, Any]]:
        """获取智能体连接"""
        connections = []
        for conn in self.connections:
            if conn["source"] == agent_id:
                target_name = f"Agent {conn['target'].split('_')[1]}"
                connections.append({
                    "target_id": conn["target"],
                    "target_name": target_name,
                    "relationship_type": conn["relationship_type"],
                    "strength": conn["weight"],
                    "last_interaction": conn["last_interaction"]
                })
        return connections

    async def get_network_stats(self) -> Dict[str, Any]:
        """获取网络统计"""
        return {
            "total_connections": len(self.connections),
            "network_density": round(len(self.connections) / 6, 3),  # 假设最多6个可能连接
            "average_strength": round(sum(c["weight"] for c in self.connections) / len(self.connections), 3) if self.connections else 0
        }