"""
信息扩散预测模型测试

测试信息扩散预测算法的各种场景
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
from src.message_propagation.information_diffusion import InformationDiffusionModel
from src.message_propagation.social_network import SocialNetwork


class TestInformationDiffusionModel:
    """信息扩散预测模型测试类"""

    @pytest.fixture
    def mock_social_network(self):
        """创建模拟社交网络"""
        network = Mock(spec=SocialNetwork)
        network.get_neighbors.return_value = []
        network.get_agent_count.return_value = 100
        network.get_all_agents.return_value = []
        return network

    @pytest.fixture
    def diffusion_model(self, mock_social_network):
        """创建扩散模型实例"""
        return InformationDiffusionModel(mock_social_network)

    def test_initialization(self, diffusion_model, mock_social_network):
        """测试扩散模型初始化"""
        assert diffusion_model.network == mock_social_network
        assert diffusion_model.adoption_probability == 0.1
        assert diffusion_model.abandon_probability == 0.05
        assert diffusion_model.max_time_steps == 100
        assert len(diffusion_model.adopted_agents) == 0
        assert len(diffusion_model.abandoned_agents) == 0
        assert len(diffusion_model.unaware_agents) == 0

    def test_initialization_with_custom_parameters(self, mock_social_network):
        """测试自定义参数初始化"""
        model = InformationDiffusionModel(
            network=mock_social_network,
            adoption_probability=0.2,
            abandon_probability=0.1,
            max_time_steps=200
        )
        assert model.adoption_probability == 0.2
        assert model.abandon_probability == 0.1
        assert model.max_time_steps == 200

    def test_set_initial_adopters_empty_list(self, diffusion_model):
        """测试设置空初始采用者列表"""
        with pytest.raises(ValueError, match="至少需要指定一个初始采用者"):
            diffusion_model.set_initial_adopters([])

    def test_set_initial_adopters_single_agent(self, diffusion_model):
        """测试设置单个初始采用者"""
        diffusion_model.set_initial_adopters(["agent_001"])
        assert "agent_001" in diffusion_model.adopted_agents
        assert diffusion_model.adoption_time["agent_001"] is not None

    def test_set_initial_adopters_multiple_agents(self, diffusion_model):
        """测试设置多个初始采用者"""
        initial_adopters = ["agent_001", "agent_002", "agent_003"]
        diffusion_model.set_initial_adopters(initial_adopters)
        assert len(diffusion_model.adopted_agents) == 3
        for agent in initial_adopters:
            assert agent in diffusion_model.adopted_agents
            assert diffusion_model.adoption_time[agent] is not None

    def test_diffuse_step_no_initial_adopters(self, diffusion_model):
        """测试未设置初始采用者时的扩散步骤"""
        with pytest.raises(ValueError, match="需要先设置初始采用者"):
            diffusion_model.diffuse_step()

    def test_diffuse_step_no_neighbors(self, diffusion_model, mock_social_network):
        """测试采用者无邻居的情况"""
        diffusion_model.set_initial_adopters(["agent_001"])
        mock_social_network.get_neighbors.return_value = []

        adopted_count = diffusion_model.diffuse_step()
        assert adopted_count == 0
        assert len(diffusion_model.adopted_agents) == 1

    def test_diffuse_step_with_neighbors_adoption(self, diffusion_model, mock_social_network):
        """测试有邻居采用的情况"""
        diffusion_model.set_initial_adopters(["agent_001"])
        mock_social_network.get_neighbors.return_value = ["agent_002", "agent_003"]

        # 模拟采用概率为1确保采用
        diffusion_model.adoption_probability = 1.0
        with patch('random.random', return_value=0.5):
            adopted_count = diffusion_model.diffuse_step()

        assert adopted_count == 2
        assert "agent_002" in diffusion_model.adopted_agents
        assert "agent_003" in diffusion_model.adopted_agents

    def test_diffuse_step_with_neighbors_no_adoption(self, diffusion_model, mock_social_network):
        """测试有邻居但不采用的情况"""
        diffusion_model.set_initial_adopters(["agent_001"])
        mock_social_network.get_neighbors.return_value = ["agent_002", "agent_003"]

        # 模拟采用概率为0确保不采用
        diffusion_model.adoption_probability = 0.0
        with patch('random.random', return_value=0.5):
            adopted_count = diffusion_model.diffuse_step()

        assert adopted_count == 0
        assert len(diffusion_model.adopted_agents) == 1

    def test_diffuse_step_abandonment(self, diffusion_model):
        """测试智能体放弃"""
        diffusion_model.set_initial_adopters(["agent_001"])

        # 模拟放弃概率为1确保放弃
        diffusion_model.abandon_probability = 1.0
        with patch('random.random', return_value=0.5):
            diffusion_model.diffuse_step()

        assert "agent_001" not in diffusion_model.adopted_agents
        assert "agent_001" in diffusion_model.abandoned_agents

    def test_predict_diffusion_complete(self, diffusion_model, mock_social_network):
        """测试完整的扩散预测"""
        diffusion_model.set_initial_adopters(["agent_001"])

        # 模拟简单的邻居关系
        def get_neighbors_side_effect(agent_id):
            if agent_id == "agent_001":
                return ["agent_002"]
            elif agent_id == "agent_002":
                return ["agent_003"]
            else:
                return []

        mock_social_network.get_neighbors.side_effect = get_neighbors_side_effect

        # 设置高采用概率确保传播
        diffusion_model.adoption_probability = 1.0
        diffusion_model.abandon_probability = 0.0
        diffusion_model.max_time_steps = 5

        prediction = diffusion_model.predict_diffusion()

        assert len(prediction) > 0
        assert all('adopted' in record for record in prediction)
        assert all('abandoned' in record for record in prediction)
        assert all('new_adoptions' in record for record in prediction)

    def test_calculate_adoption_rate_empty(self, diffusion_model):
        """测试空网络的采用率计算"""
        rate = diffusion_model.calculate_adoption_rate()
        assert rate == 0.0

    def test_calculate_adoption_rate_with_data(self, diffusion_model, mock_social_network):
        """测试有数据的采用率计算"""
        mock_social_network.get_agent_count.return_value = 100
        diffusion_model.adopted_agents = {"agent_001", "agent_002", "agent_003"}

        rate = diffusion_model.calculate_adoption_rate()
        assert rate == 0.03  # 3/100

    def test_get_diffusion_speed_empty(self, diffusion_model):
        """测试空扩散速度计算"""
        speed = diffusion_model.get_diffusion_speed()
        assert speed == 0.0

    def test_get_diffusion_speed_with_history(self, diffusion_model):
        """测试有历史的扩散速度计算"""
        diffusion_model.diffusion_history = [
            {'new_adoptions': 5},
            {'new_adoptions': 10},
            {'new_adoptions': 8}
        ]

        speed = diffusion_model.get_diffusion_speed()
        assert speed == 23 / 3  # (5+10+8)/3

    def test_predict_time_to_saturation_empty(self, diffusion_model):
        """测试空网络的饱和时间预测"""
        time_to_saturation = diffusion_model.predict_time_to_saturation()
        assert time_to_saturation == 0

    def test_predict_time_to_saturation_with_history(self, diffusion_model, mock_social_network):
        """测试有历史的饱和时间预测"""
        mock_social_network.get_agent_count.return_value = 100

        # 模拟扩散历史，在第5步达到饱和
        diffusion_model.diffusion_history = [
            {'adopted': 5, 'cumulative_adoptions': 5},
            {'adopted': 15, 'cumulative_adoptions': 15},
            {'adopted': 30, 'cumulative_adoptions': 30},
            {'adopted': 60, 'cumulative_adoptions': 60},
            {'adopted': 90, 'cumulative_adoptions': 90},
            {'adopted': 95, 'cumulative_adoptions': 95},
            {'adopted': 96, 'cumulative_adoptions': 96}
        ]

        time_to_saturation = diffusion_model.predict_time_to_saturation()
        assert time_to_saturation == 6  # 在第6步达到95%

    def test_get_influence_probability_empty(self, diffusion_model):
        """测试空影响力概率计算"""
        prob = diffusion_model.get_influence_probability("agent_001")
        assert prob == 0.0

    def test_get_influence_probability_with_data(self, diffusion_model):
        """测试有数据的影响力概率计算"""
        diffusion_model.influence_success = {"agent_001": 8}
        diffusion_model.influence_attempts = {"agent_001": 10}

        prob = diffusion_model.get_influence_probability("agent_001")
        assert prob == 0.8

    def test_reset_diffusion(self, diffusion_model):
        """测试重置扩散模拟"""
        diffusion_model.set_initial_adopters(["agent_001", "agent_002"])
        diffusion_model.diffuse_step()  # 添加一些扩散历史

        diffusion_model.reset_diffusion()

        assert len(diffusion_model.adopted_agents) == 0
        assert len(diffusion_model.abandoned_agents) == 0
        assert len(diffusion_model.unaware_agents) == 0
        assert len(diffusion_model.adoption_time) == 0
        assert len(diffusion_model.diffusion_history) == 0

    def test_get_critical_mass_empty(self, diffusion_model, mock_social_network):
        """测试空网络的关键规模计算"""
        critical_mass = diffusion_model.get_critical_mass()
        assert critical_mass == 0

    def test_get_critical_mass_with_data(self, diffusion_model, mock_social_network):
        """测试有数据的关键规模计算"""
        mock_social_network.get_agent_count.return_value = 100
        diffusion_model.adopted_agents = {"agent_001", "agent_002", "agent_003"}

        critical_mass = diffusion_model.get_critical_mass()
        assert critical_mass == 3  # 当前采用者数量

    def test_complex_diffusion_scenario(self, diffusion_model, mock_social_network):
        """测试复杂扩散场景"""
        # 设置链式网络拓扑
        def get_neighbors_side_effect(agent_id):
            neighbors_map = {
                "agent_001": ["agent_002"],
                "agent_002": ["agent_001", "agent_003"],
                "agent_003": ["agent_002", "agent_004"],
                "agent_004": ["agent_003"]
            }
            return neighbors_map.get(agent_id, [])

        mock_social_network.get_neighbors.side_effect = get_neighbors_side_effect
        mock_social_network.get_agent_count.return_value = 4

        diffusion_model.set_initial_adopters(["agent_001"])
        diffusion_model.adoption_probability = 0.7
        diffusion_model.abandon_probability = 0.1

        with patch('random.random', return_value=0.5):  # 低于采用概率
            adopted_count = diffusion_model.diffuse_step()

        # agent_001感染agent_002
        assert adopted_count == 1
        assert len(diffusion_model.adopted_agents) == 2

    @pytest.mark.asyncio
    async def test_async_predict_diffusion(self, diffusion_model, mock_social_network):
        """测试异步扩散预测"""
        diffusion_model.set_initial_adopters(["agent_001"])
        mock_social_network.get_neighbors.return_value = []

        prediction = await diffusion_model.async_predict_diffusion()
        assert isinstance(prediction, list)
        assert len(prediction) > 0

    def test_model_validation_invalid_probability(self, mock_social_network):
        """测试无效概率参数"""
        with pytest.raises(ValueError, match="采用概率必须在0和1之间"):
            InformationDiffusionModel(mock_social_network, adoption_probability=1.5)

        with pytest.raises(ValueError, match="放弃概率必须在0和1之间"):
            InformationDiffusionModel(mock_social_network, abandon_probability=-0.1)

    def test_model_validation_invalid_time_steps(self, mock_social_network):
        """测试无效时间步数"""
        with pytest.raises(ValueError, match="最大时间步数必须大于0"):
            InformationDiffusionModel(mock_social_network, max_time_steps=0)

    def test_get_diffusion_statistics_empty(self, diffusion_model):
        """测试空扩散统计"""
        stats = diffusion_model.get_diffusion_statistics()
        assert stats['total_adopted'] == 0
        assert stats['total_abandoned'] == 0
        assert stats['adoption_rate'] == 0.0
        assert stats['peak_adoption'] == 0

    def test_get_diffusion_statistics_with_data(self, diffusion_model):
        """测试有数据的扩散统计"""
        diffusion_model.adopted_agents = {"agent_001", "agent_002"}
        diffusion_model.abandoned_agents = {"agent_003"}
        diffusion_model.peak_adoption = 5

        stats = diffusion_model.get_diffusion_statistics()
        assert stats['total_adopted'] == 2
        assert stats['total_abandoned'] == 1
        assert stats['peak_adoption'] == 5

    def test_predict_diffusion_without_initial_adopters(self, diffusion_model):
        """测试未设置初始采用者时预测扩散"""
        with pytest.raises(ValueError, match="需要先设置初始采用者"):
            diffusion_model.predict_diffusion()

    def test_get_adoption_time_series(self, diffusion_model):
        """测试获取采用时间序列"""
        diffusion_model.set_initial_adopters(["agent_001", "agent_002"])

        # 手动设置一些采用时间
        from datetime import datetime, timedelta
        base_time = datetime.now()
        diffusion_model.adoption_time["agent_001"] = base_time
        diffusion_model.adoption_time["agent_002"] = base_time + timedelta(minutes=5)

        time_series = diffusion_model.get_adoption_time_series()

        assert len(time_series) == 2
        assert time_series[0][0] == "agent_001"  # 应该按时间排序
        assert time_series[1][0] == "agent_002"

    def test_get_adoption_time_series_empty(self, diffusion_model):
        """测试空采用时间序列"""
        time_series = diffusion_model.get_adoption_time_series()
        assert time_series == []

    def test_get_influential_agents(self, diffusion_model):
        """测试获取有影响力的智能体"""
        # 设置影响成功和尝试次数
        diffusion_model.influence_success = {
            "agent_001": 8,
            "agent_002": 3,
            "agent_003": 1
        }
        diffusion_model.influence_attempts = {
            "agent_001": 10,
            "agent_002": 10,
            "agent_003": 10
        }

        # 阈值0.5，只有agent_001有影响力
        influential = diffusion_model.get_influential_agents(threshold=0.5)
        assert len(influential) == 1
        assert "agent_001" in influential

        # 阈值0.1，三个都有影响力
        influential = diffusion_model.get_influential_agents(threshold=0.1)
        assert len(influential) == 3
        assert "agent_001" in influential
        assert "agent_002" in influential
        assert "agent_003" in influential

    def test_get_influential_agents_empty(self, diffusion_model):
        """测试空影响力智能体"""
        influential = diffusion_model.get_influential_agents()
        assert influential == []

    def test_calculate_diffusion_threshold(self, diffusion_model):
        """测试计算扩散阈值"""
        diffusion_model.diffusion_history = [
            {'adopted': 5},
            {'adopted': 10},
            {'adopted': 8},
            {'adopted': 15}
        ]

        threshold = diffusion_model.calculate_diffusion_threshold()
        assert threshold == 15 / 100  # 默认100个智能体

    def test_calculate_diffusion_threshold_with_different_count(self, diffusion_model, mock_social_network):
        """测试不同智能体数量的扩散阈值"""
        mock_social_network.get_agent_count.return_value = 50
        diffusion_model.diffusion_history = [
            {'adopted': 10},
            {'adopted': 15},
            {'adopted': 25}
        ]

        threshold = diffusion_model.calculate_diffusion_threshold()
        assert threshold == 25 / 50

    def test_calculate_diffusion_threshold_empty_history(self, diffusion_model):
        """测试空历史的扩散阈值"""
        threshold = diffusion_model.calculate_diffusion_threshold()
        assert threshold == 0.0

    def test_get_diffusion_network_metrics_empty(self, diffusion_model):
        """测试空扩散网络指标"""
        metrics = diffusion_model.get_diffusion_network_metrics()

        assert metrics['cascade_size'] == 0
        assert metrics['cascade_depth'] == 0
        assert metrics['branching_factor'] == 0.0
        assert metrics['influence_entropy'] == 0.0
        assert metrics['diffusion_threshold'] == 0.0
        assert metrics['critical_mass'] == 0

    def test_get_diffusion_network_metrics_with_data(self, diffusion_model, mock_social_network):
        """测试有数据的扩散网络指标"""
        mock_social_network.get_agent_count.return_value = 100

        # 设置扩散数据
        diffusion_model.adopted_agents = {"agent_001", "agent_002"}
        diffusion_model.total_adoption_count = 10
        diffusion_model.diffusion_history = [
            {'adopted': 2, 'new_adoptions': 2},
            {'adopted': 5, 'new_adoptions': 3},
            {'adopted': 8, 'new_adoptions': 3}
        ]

        diffusion_model.influence_success = {
            "agent_001": 5,
            "agent_002": 3
        }
        diffusion_model.influence_attempts = {
            "agent_001": 8,
            "agent_002": 5
        }

        metrics = diffusion_model.get_diffusion_network_metrics()

        assert metrics['cascade_size'] == 10
        assert metrics['cascade_depth'] == 3
        assert metrics['branching_factor'] == 6.5  # 13/2
        assert metrics['influence_entropy'] > 0.0
        assert metrics['diffusion_threshold'] == 8 / 100
        assert metrics['critical_mass'] == 2