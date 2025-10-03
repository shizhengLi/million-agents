"""
病毒式传播模型测试

测试病毒式传播算法的各种场景和边界条件
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
from src.message_propagation.viral_propagation import ViralPropagationModel
from src.message_propagation.social_network import SocialNetwork


class TestViralPropagationModel:
    """病毒式传播模型测试类"""

    @pytest.fixture
    def mock_social_network(self):
        """创建模拟社交网络"""
        network = Mock(spec=SocialNetwork)
        network.get_neighbors.return_value = []
        network.get_agent.return_value = None
        return network

    @pytest.fixture
    def propagation_model(self, mock_social_network):
        """创建传播模型实例"""
        return ViralPropagationModel(mock_social_network)

    def test_initialization(self, propagation_model, mock_social_network):
        """测试传播模型初始化"""
        assert propagation_model.network == mock_social_network
        assert propagation_model.infection_probability == 0.1
        assert propagation_model.recovery_probability == 0.05
        assert propagation_model.max_iterations == 100
        assert len(propagation_model.infected_agents) == 0
        assert len(propagation_model.recovered_agents) == 0
        assert len(propagation_model.susceptible_agents) == 0

    def test_initialization_with_custom_parameters(self, mock_social_network):
        """测试自定义参数初始化"""
        model = ViralPropagationModel(
            network=mock_social_network,
            infection_probability=0.2,
            recovery_probability=0.1,
            max_iterations=200
        )
        assert model.infection_probability == 0.2
        assert model.recovery_probability == 0.1
        assert model.max_iterations == 200

    def test_set_initial_infected_empty_list(self, propagation_model):
        """测试设置空感染列表"""
        with pytest.raises(ValueError, match="至少需要指定一个初始感染智能体"):
            propagation_model.set_initial_infected([])

    def test_set_initial_infected_single_agent(self, propagation_model):
        """测试设置单个初始感染智能体"""
        propagation_model.set_initial_infected(["agent_001"])
        assert "agent_001" in propagation_model.infected_agents
        assert propagation_model.infection_time["agent_001"] is not None

    def test_set_initial_infected_multiple_agents(self, propagation_model):
        """测试设置多个初始感染智能体"""
        initial_agents = ["agent_001", "agent_002", "agent_003"]
        propagation_model.set_initial_infected(initial_agents)
        assert len(propagation_model.infected_agents) == 3
        for agent in initial_agents:
            assert agent in propagation_model.infected_agents
            assert propagation_model.infection_time[agent] is not None

    def test_set_initial_infected_duplicate_agents(self, propagation_model):
        """测试设置重复的初始感染智能体"""
        initial_agents = ["agent_001", "agent_001", "agent_002"]
        propagation_model.set_initial_infected(initial_agents)
        assert len(propagation_model.infected_agents) == 2
        assert "agent_001" in propagation_model.infected_agents
        assert "agent_002" in propagation_model.infected_agents

    def test_propagate_step_no_initial_infected(self, propagation_model):
        """测试未设置初始感染时的传播步骤"""
        with pytest.raises(ValueError, match="需要先设置初始感染智能体"):
            propagation_model.propagate_step()

    def test_propagate_step_no_neighbors(self, propagation_model, mock_social_network):
        """测试感染智能体无邻居的情况"""
        propagation_model.set_initial_infected(["agent_001"])
        mock_social_network.get_neighbors.return_value = []

        infected_count = propagation_model.propagate_step()
        assert infected_count == 0
        assert len(propagation_model.infected_agents) == 1

    def test_propagate_step_with_neighbors_infection(self, propagation_model, mock_social_network):
        """测试有邻居感染的情况"""
        propagation_model.set_initial_infected(["agent_001"])
        mock_social_network.get_neighbors.return_value = ["agent_002", "agent_003"]

        # 模拟感染概率为1确保感染
        propagation_model.infection_probability = 1.0
        with patch('random.random', return_value=0.5):
            infected_count = propagation_model.propagate_step()

        assert infected_count == 2
        assert "agent_002" in propagation_model.infected_agents
        assert "agent_003" in propagation_model.infected_agents

    def test_propagate_step_with_neighbors_no_infection(self, propagation_model, mock_social_network):
        """测试有邻居但不感染的情况"""
        propagation_model.set_initial_infected(["agent_001"])
        mock_social_network.get_neighbors.return_value = ["agent_002", "agent_003"]

        # 模拟感染概率为0确保不感染
        propagation_model.infection_probability = 0.0
        with patch('random.random', return_value=0.5):
            infected_count = propagation_model.propagate_step()

        assert infected_count == 0
        assert len(propagation_model.infected_agents) == 1

    def test_propagate_step_already_infected_neighbors(self, propagation_model, mock_social_network):
        """测试邻居已经被感染的情况"""
        propagation_model.set_initial_infected(["agent_001", "agent_002"])
        mock_social_network.get_neighbors.return_value = ["agent_002", "agent_003"]

        infected_count = propagation_model.propagate_step()
        assert infected_count <= 1  # 只有agent_003可能被感染
        assert "agent_002" in propagation_model.infected_agents

    def test_propagate_step_recovery(self, propagation_model):
        """测试智能体恢复"""
        propagation_model.set_initial_infected(["agent_001"])

        # 模拟恢复概率为1确保恢复
        propagation_model.recovery_probability = 1.0
        with patch('random.random', return_value=0.5):
            propagation_model.propagate_step()

        assert "agent_001" not in propagation_model.infected_agents
        assert "agent_001" in propagation_model.recovered_agents

    def test_propagate_full_simulation_empty_network(self, propagation_model):
        """测试空网络的完整传播模拟"""
        with pytest.raises(ValueError, match="需要先设置初始感染智能体"):
            propagation_model.propagate_full_simulation()

    def test_propagate_full_simulation_simple_network(self, propagation_model, mock_social_network):
        """测试简单网络的完整传播模拟"""
        propagation_model.set_initial_infected(["agent_001"])

        # 模拟简单的邻居关系
        def get_neighbors_side_effect(agent_id):
            if agent_id == "agent_001":
                return ["agent_002"]
            elif agent_id == "agent_002":
                return ["agent_003"]
            else:
                return []

        mock_social_network.get_neighbors.side_effect = get_neighbors_side_effect

        # 设置高感染概率确保传播
        propagation_model.infection_probability = 1.0
        propagation_model.recovery_probability = 0.0
        propagation_model.max_iterations = 5

        history = propagation_model.propagate_full_simulation()

        assert len(history) > 0
        assert all('infected' in record for record in history)
        assert all('recovered' in record for record in history)
        assert all('new_infections' in record for record in history)

    def test_get_infection_statistics_empty(self, propagation_model):
        """测试空网络的感染统计"""
        stats = propagation_model.get_infection_statistics()
        assert stats['total_infected'] == 0
        assert stats['total_recovered'] == 0
        assert stats['peak_infection'] == 0
        assert stats['infection_rate'] == 0.0

    def test_get_infection_statistics_with_data(self, propagation_model):
        """测试有数据的感染统计"""
        propagation_model.infected_agents = {"agent_001", "agent_002"}
        propagation_model.recovered_agents = {"agent_003", "agent_004"}
        propagation_model.peak_infection = 5
        propagation_model.total_infected_count = 4  # total affected

        stats = propagation_model.get_infection_statistics()
        assert stats['total_infected'] == 2
        assert stats['total_recovered'] == 2
        assert stats['peak_infection'] == 5
        assert stats['total_affected'] == 4
        # infection_rate = infected / total_affected = 2/4 = 0.5
        assert abs(stats['infection_rate'] - 0.5) < 0.001

    def test_reset_simulation(self, propagation_model):
        """测试重置模拟"""
        propagation_model.set_initial_infected(["agent_001", "agent_002"])
        propagation_model.propagate_step()  # 添加一些传播历史

        propagation_model.reset_simulation()

        assert len(propagation_model.infected_agents) == 0
        assert len(propagation_model.recovered_agents) == 0
        assert len(propagation_model.susceptible_agents) == 0
        assert len(propagation_model.infection_time) == 0
        assert len(propagation_model.propagation_history) == 0

    def test_get_propagation_tree_empty(self, propagation_model):
        """测试空传播树"""
        tree = propagation_model.get_propagation_tree()
        assert tree == {}

    def test_get_propagation_tree_simple(self, propagation_model):
        """测试简单传播树"""
        propagation_model.set_initial_infected(["agent_001"])
        propagation_model.infection_source["agent_002"] = "agent_001"
        propagation_model.infection_source["agent_003"] = "agent_001"

        tree = propagation_model.get_propagation_tree()
        assert "agent_001" in tree
        assert "agent_002" in tree["agent_001"]
        assert "agent_003" in tree["agent_001"]

    def test_calculate_reproduction_number(self, propagation_model):
        """测试基本再生数计算"""
        propagation_model.infection_source = {
            "agent_002": "agent_001",
            "agent_003": "agent_001",
            "agent_004": "agent_002",
            "agent_005": "agent_002"
        }

        r0 = propagation_model.calculate_reproduction_number()
        assert r0 == 2.0  # agent_001感染2个，agent_002感染2个，平均2个

    def test_calculate_reproduction_number_no_infections(self, propagation_model):
        """测试无感染情况的基本再生数"""
        r0 = propagation_model.calculate_reproduction_number()
        assert r0 == 0.0

    def test_calculate_reproduction_number_only_sources_no_targets(self, propagation_model):
        """测试只有感染源但没有感染目标的情况"""
        # 只有初始感染，没有传播
        propagation_model.infection_source = {}
        r0 = propagation_model.calculate_reproduction_number()
        assert r0 == 0.0

    @pytest.mark.asyncio
    async def test_async_propagate_simulation(self, propagation_model, mock_social_network):
        """测试异步传播模拟"""
        propagation_model.set_initial_infected(["agent_001"])
        mock_social_network.get_neighbors.return_value = []

        history = await propagation_model.async_propagate_simulation()
        assert isinstance(history, list)
        assert len(history) > 0

    def test_model_validation_invalid_probability(self, mock_social_network):
        """测试无效概率参数"""
        with pytest.raises(ValueError, match="感染概率必须在0和1之间"):
            ViralPropagationModel(mock_social_network, infection_probability=1.5)

        with pytest.raises(ValueError, match="恢复概率必须在0和1之间"):
            ViralPropagationModel(mock_social_network, recovery_probability=-0.1)

    def test_model_validation_invalid_iterations(self, mock_social_network):
        """测试无效迭代次数"""
        with pytest.raises(ValueError, match="最大迭代次数必须大于0"):
            ViralPropagationModel(mock_social_network, max_iterations=0)

    def test_complex_propagation_scenario(self, propagation_model, mock_social_network):
        """测试复杂传播场景"""
        # 设置星形网络拓扑
        def get_neighbors_side_effect(agent_id):
            if agent_id == "agent_001":  # 中心节点
                return ["agent_002", "agent_003", "agent_004", "agent_005"]
            else:  # 叶子节点
                return ["agent_001"]

        mock_social_network.get_neighbors.side_effect = get_neighbors_side_effect

        propagation_model.set_initial_infected(["agent_001"])
        propagation_model.infection_probability = 0.5
        propagation_model.recovery_probability = 0.1

        with patch('random.random', return_value=0.3):  # 低于感染概率
            infected_count = propagation_model.propagate_step()

        # 中心节点感染4个邻居
        assert infected_count == 4
        assert len(propagation_model.infected_agents) == 5

    def test_get_infection_time_series(self, propagation_model):
        """测试获取感染时间序列"""
        propagation_model.set_initial_infected(["agent_001", "agent_002"])

        # 手动设置一些感染时间
        from datetime import datetime, timedelta
        base_time = datetime.now()
        propagation_model.infection_time["agent_001"] = base_time
        propagation_model.infection_time["agent_002"] = base_time + timedelta(minutes=5)

        time_series = propagation_model.get_infection_time_series()

        assert len(time_series) == 2
        assert time_series[0][0] == "agent_001"  # 应该按时间排序
        assert time_series[1][0] == "agent_002"

    def test_get_infection_time_series_empty(self, propagation_model):
        """测试空感染时间序列"""
        time_series = propagation_model.get_infection_time_series()
        assert time_series == []

    def test_get_superspreaders(self, propagation_model):
        """测试获取超级传播者"""
        # 设置感染源，agent_001感染了6个，agent_002感染了3个
        propagation_model.infection_source = {
            "agent_002": "agent_001",
            "agent_003": "agent_001",
            "agent_004": "agent_001",
            "agent_005": "agent_001",
            "agent_006": "agent_001",
            "agent_007": "agent_001",
            "agent_008": "agent_002",
            "agent_009": "agent_002",
            "agent_010": "agent_002"
        }

        superspreaders = propagation_model.get_superspreaders(threshold=5)
        assert len(superspreaders) == 1
        assert "agent_001" in superspreaders

    def test_get_superspreaders_different_threshold(self, propagation_model):
        """测试不同阈值的超级传播者"""
        propagation_model.infection_source = {
            "agent_002": "agent_001",
            "agent_003": "agent_001",
            "agent_004": "agent_001",  # agent_001感染3个
            "agent_005": "agent_002"   # agent_002感染1个
        }

        # 阈值为1，两个都是超级传播者
        superspreaders = propagation_model.get_superspreaders(threshold=1)
        assert len(superspreaders) == 2
        assert "agent_001" in superspreaders
        assert "agent_002" in superspreaders

        # 阈值为3，只有agent_001是超级传播者
        superspreaders = propagation_model.get_superspreaders(threshold=3)
        assert len(superspreaders) == 1
        assert "agent_001" in superspreaders

    def test_get_superspreaders_empty(self, propagation_model):
        """测试空超级传播者"""
        superspreaders = propagation_model.get_superspreaders()
        assert superspreaders == []

    def test_get_network_metrics_empty_history(self, propagation_model):
        """测试空传播历史的网络指标"""
        metrics = propagation_model.get_network_metrics()

        assert metrics['outbreak_size'] == 0
        assert metrics['outbreak_duration'] == 0
        assert metrics['peak_time'] is None
        assert metrics['growth_rate'] == 0.0
        assert metrics['reproduction_number'] == 0.0

    def test_get_network_metrics_with_history(self, propagation_model, mock_social_network):
        """测试有传播历史的网络指标"""
        # 设置传播历史
        from datetime import datetime, timedelta
        base_time = datetime.now()

        propagation_model.set_initial_infected(["agent_001"])
        propagation_model.total_infected_count = 10
        propagation_model.propagation_history = [
            {'timestamp': base_time, 'infected': 1, 'new_infections': 1},
            {'timestamp': base_time + timedelta(minutes=1), 'infected': 5, 'new_infections': 4},
            {'timestamp': base_time + timedelta(minutes=2), 'infected': 8, 'new_infections': 3}
        ]

        # 设置感染源以计算再生数
        propagation_model.infection_source = {
            "agent_002": "agent_001",
            "agent_003": "agent_001",
            "agent_004": "agent_001",
            "agent_005": "agent_002"
        }

        metrics = propagation_model.get_network_metrics()

        assert metrics['outbreak_size'] == 10
        assert metrics['outbreak_duration'] == 3
        assert metrics['peak_time'] is not None
        assert metrics['growth_rate'] > 0.0
        assert metrics['reproduction_number'] == 2.0  # agent_001感染3个，agent_002感染1个，平均2个

    def test_get_network_metrics_single_step(self, propagation_model):
        """测试单步传播的网络指标"""
        from datetime import datetime
        base_time = datetime.now()

        propagation_model.total_infected_count = 5
        propagation_model.propagation_history = [
            {'timestamp': base_time, 'infected': 5, 'new_infections': 5}
        ]

        metrics = propagation_model.get_network_metrics()

        assert metrics['outbreak_size'] == 5
        assert metrics['outbreak_duration'] == 1
        assert metrics['growth_rate'] == 0.0  # 只有一步，增长率为0