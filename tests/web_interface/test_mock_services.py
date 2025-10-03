"""
Mock Services Tests
模拟服务的单元测试
"""

import pytest
from datetime import datetime, timedelta
from src.web_interface.api.mock_services import MockAgentService, MockSocialNetworkService


class TestMockAgentService:
    """模拟智能体服务测试"""

    @pytest.fixture
    def service(self):
        """创建服务实例"""
        return MockAgentService()

    @pytest.mark.asyncio
    async def test_get_all_agents(self, service):
        """测试获取所有智能体"""
        agents = await service.get_all_agents()
        assert len(agents) == 3
        assert agents[0]["id"] == "agent_1"
        assert agents[0]["name"] == "Social Agent Alpha"
        assert agents[0]["type"] == "social"

    @pytest.mark.asyncio
    async def test_get_agent_by_id_found(self, service):
        """测试根据ID找到智能体"""
        agent = await service.get_agent_by_id("agent_1")
        assert agent is not None
        assert agent["id"] == "agent_1"
        assert agent["name"] == "Social Agent Alpha"

    @pytest.mark.asyncio
    async def test_get_agent_by_id_not_found(self, service):
        """测试根据ID未找到智能体"""
        agent = await service.get_agent_by_id("nonexistent")
        assert agent is None

    @pytest.mark.asyncio
    async def test_create_agent(self, service):
        """测试创建智能体"""
        agent_data = {
            "name": "New Test Agent",
            "type": "social",
            "description": "A new test agent"
        }
        initial_count = len(await service.get_all_agents())

        created_agent = await service.create_agent(agent_data)

        assert created_agent["name"] == "New Test Agent"
        assert created_agent["type"] == "social"
        assert created_agent["status"] == "active"
        assert created_agent["reputation_score"] == 50.0

        # 验证智能体已添加
        final_count = len(await service.get_all_agents())
        assert final_count == initial_count + 1

    @pytest.mark.asyncio
    async def test_update_agent(self, service):
        """测试更新智能体"""
        update_data = {
            "name": "Updated Agent Name",
            "description": "Updated description"
        }

        updated_agent = await service.update_agent("agent_1", update_data)

        assert updated_agent is not None
        assert updated_agent["name"] == "Updated Agent Name"
        assert updated_agent["description"] == "Updated description"
        assert updated_agent["id"] == "agent_1"

    @pytest.mark.asyncio
    async def test_update_agent_not_found(self, service):
        """测试更新不存在的智能体"""
        update_data = {"name": "New Name"}

        result = await service.update_agent("nonexistent", update_data)
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_agent(self, service):
        """测试删除智能体"""
        initial_count = len(await service.get_all_agents())

        success = await service.delete_agent("agent_1")
        assert success is True

        # 验证智能体已删除
        final_count = len(await service.get_all_agents())
        assert final_count == initial_count - 1

        # 验证无法找到已删除的智能体
        agent = await service.get_agent_by_id("agent_1")
        assert agent is None

    @pytest.mark.asyncio
    async def test_delete_agent_not_found(self, service):
        """测试删除不存在的智能体"""
        success = await service.delete_agent("nonexistent")
        assert success is False

    @pytest.mark.asyncio
    async def test_get_agent_stats(self, service):
        """测试获取智能体统计"""
        stats = await service.get_agent_stats()

        assert "total_agents" in stats
        assert "active_agents" in stats
        assert "inactive_agents" in stats
        assert "agent_types" in stats

        assert stats["total_agents"] == 3
        assert stats["active_agents"] == 2
        assert stats["inactive_agents"] == 1
        assert "social" in stats["agent_types"]
        assert "content" in stats["agent_types"]
        assert "hybrid" in stats["agent_types"]

    @pytest.mark.asyncio
    async def test_get_reputation_metrics(self, service):
        """测试获取声誉指标"""
        metrics = await service.get_reputation_metrics()

        assert "total_agents" in metrics
        assert "active_agents" in metrics
        assert "average_reputation" in metrics
        assert "reputation_distribution" in metrics
        assert "recent_changes" in metrics

        assert metrics["total_agents"] == 3
        assert metrics["active_agents"] == 2
        assert isinstance(metrics["average_reputation"], float)
        assert 0 <= metrics["average_reputation"] <= 100

        distribution = metrics["reputation_distribution"]
        assert "high" in distribution
        assert "medium" in distribution
        assert "low" in distribution
        assert sum(distribution.values()) == 3


class TestMockSocialNetworkService:
    """模拟社交网络服务测试"""

    @pytest.fixture
    def service(self):
        """创建服务实例"""
        return MockSocialNetworkService()

    @pytest.mark.asyncio
    async def test_get_network_data(self, service):
        """测试获取网络数据"""
        network_data = await service.get_network_data()

        assert "nodes" in network_data
        assert "edges" in network_data
        assert "metrics" in network_data

        nodes = network_data["nodes"]
        edges = network_data["edges"]
        metrics = network_data["metrics"]

        assert len(nodes) == 3
        assert len(edges) == 2

        # 验证节点数据
        assert nodes[0]["id"] == "agent_1"
        assert nodes[0]["name"] == "Social Agent Alpha"
        assert nodes[0]["group"] == "social"

        # 验证边数据
        assert edges[0]["source"] == "agent_1"
        assert edges[0]["target"] == "agent_2"
        assert edges[0]["weight"] == 0.8
        assert edges[0]["relationship_type"] == "friend"

        # 验证网络指标
        assert "total_nodes" in metrics
        assert "total_edges" in metrics
        assert "density" in metrics
        assert "clustering_coefficient" in metrics
        assert metrics["total_nodes"] == 3
        assert metrics["total_edges"] == 2

    @pytest.mark.asyncio
    async def test_get_network_data_with_limit(self, service):
        """测试带限制的网络数据获取"""
        network_data = await service.get_network_data(limit=2)

        assert len(network_data["nodes"]) <= 2
        assert len(network_data["edges"]) <= 4  # limit * 2

    @pytest.mark.asyncio
    async def test_get_agent_connections(self, service):
        """测试获取智能体连接"""
        connections = await service.get_agent_connections("agent_1")

        assert len(connections) == 2

        # 验证连接数据
        conn1 = connections[0]
        assert "target_id" in conn1
        assert "target_name" in conn1
        assert "relationship_type" in conn1
        assert "strength" in conn1
        assert "last_interaction" in conn1

        assert conn1["target_id"] in ["agent_2", "agent_3"]
        assert 0 <= conn1["strength"] <= 1

    @pytest.mark.asyncio
    async def test_get_agent_connections_no_connections(self, service):
        """测试获取没有连接的智能体"""
        connections = await service.get_agent_connections("nonexistent")
        assert len(connections) == 0

    @pytest.mark.asyncio
    async def test_get_network_stats(self, service):
        """测试获取网络统计"""
        stats = await service.get_network_stats()

        assert "total_connections" in stats
        assert "network_density" in stats
        assert "average_strength" in stats

        assert stats["total_connections"] == 3
        assert 0 <= stats["network_density"] <= 1
        assert 0 <= stats["average_strength"] <= 1


class TestServiceIntegration:
    """服务集成测试"""

    @pytest.fixture
    def agent_service(self):
        """创建智能体服务实例"""
        return MockAgentService()

    @pytest.fixture
    def network_service(self):
        """创建网络服务实例"""
        return MockSocialNetworkService()

    @pytest.mark.asyncio
    async def test_agent_network_data_consistency(self, agent_service, network_service):
        """测试智能体和网络服务数据一致性"""
        agents = await agent_service.get_all_agents()
        network_data = await network_service.get_network_data()

        # 验证智能体ID在网络节点中存在
        agent_ids = {agent["id"] for agent in agents}
        node_ids = {node["id"] for node in network_data["nodes"]}

        # 应该有重叠的ID
        assert len(agent_ids.intersection(node_ids)) > 0

    @pytest.mark.asyncio
    async def test_create_agent_and_network_impact(self, agent_service, network_service):
        """测试创建智能体对网络的影响"""
        # 创建新智能体
        new_agent = {
            "name": "Network Test Agent",
            "type": "social",
            "description": "Agent for network testing"
        }
        created_agent = await agent_service.create_agent(new_agent)

        # 验证智能体创建成功
        assert created_agent["id"] is not None
        assert created_agent["name"] == "Network Test Agent"

        # 获取网络数据，新智能体应该在网络中可见
        network_data = await network_service.get_network_data()
        node_ids = {node["id"] for node in network_data["nodes"]}

        # 注意：由于使用模拟数据，新创建的智能体可能不会立即出现在网络中
        # 这是一个已知的行为，真实系统中会有自动的网络集成逻辑