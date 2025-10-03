"""
消息传播模型集成测试

测试各个模块之间的集成和协同工作
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
from src.message_propagation.viral_propagation import ViralPropagationModel
from src.message_propagation.information_diffusion import InformationDiffusionModel
from src.message_propagation.influence_maximization import InfluenceMaximization
from src.message_propagation.propagation_tracker import PropagationTracker
from src.message_propagation.social_network import SocialNetwork


class TestMessagePropagationIntegration:
    """消息传播模型集成测试类"""

    @pytest.fixture
    def mock_social_network(self):
        """创建模拟社交网络"""
        network = Mock(spec=SocialNetwork)
        network.get_neighbors.return_value = []
        network.get_agent_count.return_value = 10
        network.get_all_agents.return_value = []
        network.get_agent.return_value = None
        return network

    @pytest.fixture
    def integrated_system(self, mock_social_network):
        """创建集成系统实例"""
        return {
            'viral': ViralPropagationModel(mock_social_network),
            'diffusion': InformationDiffusionModel(mock_social_network),
            'influence': InfluenceMaximization(mock_social_network),
            'tracker': PropagationTracker(mock_social_network)
        }

    def test_viral_propagation_with_tracker(self, integrated_system, mock_social_network):
        """测试病毒式传播与追踪器集成"""
        viral_model = integrated_system['viral']
        tracker = integrated_system['tracker']

        # 设置网络拓扑
        def get_neighbors_side_effect(agent_id):
            neighbors_map = {
                "agent_001": ["agent_002", "agent_003"],
                "agent_002": ["agent_004"],
                "agent_003": ["agent_004", "agent_005"]
            }
            return neighbors_map.get(agent_id, [])

        mock_social_network.get_neighbors.side_effect = get_neighbors_side_effect

        # 初始化病毒传播
        viral_model.set_initial_infected(["agent_001"])
        viral_model.infection_probability = 1.0  # 确保传播
        viral_model.recovery_probability = 0.0   # 禁止恢复

        # 执行传播步骤
        with patch('random.random', return_value=0.5):
            infected_count = viral_model.propagate_step()

        # 追踪传播路径
        for infected_agent in viral_model.infected_agents:
            if infected_agent != "agent_001":  # 排除初始感染
                tracker.track_infection("agent_001", infected_agent)

        # 验证集成结果
        assert infected_count > 0
        assert len(tracker.propagation_edges) > 0
        assert len(tracker.infection_sources) > 0

    def test_diffusion_with_influence_maximization(self, integrated_system, mock_social_network):
        """测试信息扩散与影响力最大化集成"""
        diffusion_model = integrated_system['diffusion']
        influence_model = integrated_system['influence']

        # 设置网络拓扑
        agents = ["agent_001", "agent_002", "agent_003", "agent_004"]
        mock_social_network.get_all_agents.return_value = agents

        def get_neighbors_side_effect(agent_id):
            neighbors_map = {
                "agent_001": ["agent_002", "agent_003"],
                "agent_002": ["agent_003", "agent_004"],
                "agent_003": ["agent_004"],
                "agent_004": []
            }
            return neighbors_map.get(agent_id, [])

        mock_social_network.get_neighbors.side_effect = get_neighbors_side_effect

        # 找到最优种子
        seeds = influence_model.degree_heuristic(2)
        assert len(seeds) == 2

        # 使用最优种子初始化扩散
        diffusion_model.set_initial_adopters(seeds)
        diffusion_model.adoption_probability = 1.0  # 确保扩散
        diffusion_model.abandon_probability = 0.0   # 禁止放弃

        # 执行扩散步骤
        with patch('random.random', return_value=0.5):
            adopted_count = diffusion_model.diffuse_step()

        # 验证集成结果
        assert adopted_count >= 0
        assert len(diffusion_model.adopted_agents) >= len(seeds)

    def test_full_propagation_pipeline(self, integrated_system, mock_social_network):
        """测试完整的传播流水线"""
        viral_model = integrated_system['viral']
        diffusion_model = integrated_system['diffusion']
        influence_model = integrated_system['influence']
        tracker = integrated_system['tracker']

        # 设置复杂网络拓扑
        agents = [f"agent_{i:03d}" for i in range(1, 11)]
        mock_social_network.get_all_agents.return_value = agents

        def get_neighbors_side_effect(agent_id):
            # 创建一个复杂的网络结构
            index = int(agent_id.split('_')[1]) - 1
            neighbors = []
            if index < len(agents) - 1:
                neighbors.append(agents[index + 1])  # 连接到下一个节点
            if index > 0:
                neighbors.append(agents[index - 1])  # 连接到上一个节点
            if index % 2 == 0 and index + 2 < len(agents):
                neighbors.append(agents[index + 2])  # 偶数节点额外连接
            return neighbors

        mock_social_network.get_neighbors.side_effect = get_neighbors_side_effect

        # 步骤1: 影响力最大化找到最优种子
        seeds = influence_model.degree_heuristic(2)
        assert len(seeds) == 2

        # 步骤2: 病毒式传播模拟
        viral_model.set_initial_infected(seeds)
        viral_model.infection_probability = 0.8
        viral_model.recovery_probability = 0.1

        # 执行几步病毒传播
        for _ in range(3):
            with patch('random.random', return_value=0.3):
                viral_model.propagate_step()

        # 步骤3: 追踪传播路径
        for infected_agent in viral_model.infected_agents:
            if infected_agent not in seeds:
                # 找到可能的感染源
                for source in seeds:
                    if source in mock_social_network.get_neighbors(infected_agent):
                        tracker.track_infection(source, infected_agent)
                        break

        # 步骤4: 信息扩散模拟
        diffusion_model.set_initial_adopters(viral_model.infected_agents)
        diffusion_model.adoption_probability = 0.6
        diffusion_model.abandon_probability = 0.05

        with patch('random.random', return_value=0.4):
            diffusion_model.diffuse_step()

        # 验证流水线结果
        assert len(seeds) == 2
        assert len(viral_model.infected_agents) >= 2
        assert len(diffusion_model.adopted_agents) >= len(viral_model.infected_agents)
        assert len(tracker.propagation_edges) >= 0

        # 获取综合统计
        viral_stats = viral_model.get_infection_statistics()
        diffusion_stats = diffusion_model.get_diffusion_statistics()
        tracker_stats = tracker.get_propagation_statistics()

        assert viral_stats['total_infected'] >= 2
        assert diffusion_stats['total_adopted'] >= 0
        assert tracker_stats['total_infections'] >= 0

    def test_propagation_models_consistency(self, integrated_system, mock_social_network):
        """测试传播模型的一致性"""
        viral_model = integrated_system['viral']
        diffusion_model = integrated_system['diffusion']

        # 设置相同的网络拓扑
        agents = ["agent_001", "agent_002", "agent_003", "agent_004"]
        mock_social_network.get_all_agents.return_value = agents

        def get_neighbors_side_effect(agent_id):
            neighbors_map = {
                "agent_001": ["agent_002", "agent_003"],
                "agent_002": ["agent_003", "agent_004"],
                "agent_003": ["agent_004"],
                "agent_004": []
            }
            return neighbors_map.get(agent_id, [])

        mock_social_network.get_neighbors.side_effect = get_neighbors_side_effect

        # 设置相同的初始条件
        initial_agents = ["agent_001"]
        viral_model.set_initial_infected(initial_agents)
        diffusion_model.set_initial_adopters(initial_agents)

        # 设置相同的传播参数
        viral_model.infection_probability = 0.5
        diffusion_model.adoption_probability = 0.5
        viral_model.recovery_probability = 0.0
        diffusion_model.abandon_probability = 0.0

        # 执行传播
        with patch('random.random', return_value=0.3):
            viral_infected = viral_model.propagate_step()
            diffusion_adopted = diffusion_model.diffuse_step()

        # 验证一致性（虽然算法不同，但应该在合理范围内）
        assert viral_infected >= 0
        assert diffusion_adopted >= 0
        # 两个模型的传播范围应该相近
        assert abs(viral_infected - diffusion_adopted) <= 2

    def test_tracker_with_multiple_models(self, integrated_system, mock_social_network):
        """测试追踪器与多个模型的集成"""
        viral_model = integrated_system['viral']
        diffusion_model = integrated_system['diffusion']
        tracker = integrated_system['tracker']

        # 设置网络拓扑
        def get_neighbors_side_effect(agent_id):
            return ["agent_002", "agent_003"] if agent_id == "agent_001" else []

        mock_social_network.get_neighbors.side_effect = get_neighbors_side_effect

        # 病毒式传播
        viral_model.set_initial_infected(["agent_001"])
        viral_model.infection_probability = 1.0

        with patch('random.random', return_value=0.5):
            viral_model.propagate_step()

        # 信息扩散
        diffusion_model.set_initial_adopters(["agent_001"])
        diffusion_model.adoption_probability = 1.0

        with patch('random.random', return_value=0.5):
            diffusion_model.diffuse_step()

        # 统一追踪所有传播
        all_propagated = (viral_model.infected_agents |
                         diffusion_model.adopted_agents -
                         {"agent_001"})

        # 确保实际要追踪的智能体集合
        actual_tracked = set()
        for agent in all_propagated:
            if agent != "agent_001":  # 确保不追踪相同智能体
                tracker.track_infection("agent_001", agent)
                actual_tracked.add(agent)

        # 验证追踪结果
        assert len(tracker.propagation_edges) == len(actual_tracked)
        assert len(tracker.infection_sources) == len(actual_tracked)

        # 获取追踪统计
        stats = tracker.get_propagation_statistics()
        assert stats['total_infections'] == len(actual_tracked)
        assert stats['unique_sources'] == 1  # 只有agent_001

    def test_performance_with_large_network(self, integrated_system, mock_social_network):
        """测试大型网络的性能"""
        viral_model = integrated_system['viral']
        influence_model = integrated_system['influence']
        tracker = integrated_system['tracker']

        # 创建大型网络
        agents = [f"agent_{i:04d}" for i in range(1, 101)]  # 100个智能体
        mock_social_network.get_all_agents.return_value = agents

        def get_neighbors_side_effect(agent_id):
            # 每个智能体连接到接下来的5个智能体
            index = int(agent_id.split('_')[1]) - 1
            neighbors = []
            for i in range(1, 6):
                if index + i < len(agents):
                    neighbors.append(agents[index + i])
            return neighbors

        mock_social_network.get_neighbors.side_effect = get_neighbors_side_effect

        # 测试影响力最大化的性能
        import time
        start_time = time.time()
        seeds = influence_model.degree_heuristic(5)
        influence_time = time.time() - start_time

        # 测试病毒传播的性能
        viral_model.set_initial_infected(seeds)
        viral_model.infection_probability = 0.1

        start_time = time.time()
        viral_model.propagate_step()
        viral_time = time.time() - start_time

        # 测试追踪器的性能
        start_time = time.time()
        for infected in viral_model.infected_agents:
            if infected not in seeds:
                tracker.track_infection(seeds[0], infected)
        tracker_time = time.time() - start_time

        # 验证性能在合理范围内
        assert influence_time < 5.0  # 影响力计算应该在5秒内
        assert viral_time < 1.0       # 病毒传播应该在1秒内
        assert tracker_time < 0.1     # 追踪应该在0.1秒内

        # 验证功能正确性
        assert len(seeds) == 5
        assert len(viral_model.infected_agents) >= 5

    def test_error_handling_integration(self, integrated_system, mock_social_network):
        """测试错误处理的集成"""
        viral_model = integrated_system['viral']
        diffusion_model = integrated_system['diffusion']
        tracker = integrated_system['tracker']

        # 测试空网络的错误处理
        mock_social_network.get_all_agents.return_value = []
        mock_social_network.get_neighbors.return_value = []

        # 病毒传播错误处理
        with pytest.raises(ValueError, match="需要先设置初始感染智能体"):
            viral_model.propagate_step()

        # 扩散模型错误处理
        with pytest.raises(ValueError, match="需要先设置初始采用者"):
            diffusion_model.diffuse_step()

        # 追踪器错误处理
        with pytest.raises(ValueError, match="感染源不能为空"):
            tracker.track_infection("", "target_001")

        # 修复网络后应该正常工作
        mock_social_network.get_all_agents.return_value = ["agent_001"]

        viral_model.set_initial_infected(["agent_001"])
        viral_model.propagate_step()  # 应该不抛出异常

        diffusion_model.set_initial_adopters(["agent_001"])
        diffusion_model.diffuse_step()  # 应该不抛出异常

        with pytest.raises(ValueError, match="感染源和感染目标不能相同"):
            tracker.track_infection("agent_001", "agent_001")

    def test_concurrent_propagation_simulation(self, integrated_system, mock_social_network):
        """测试并发传播模拟"""
        import asyncio
        viral_model = integrated_system['viral']
        diffusion_model = integrated_system['diffusion']

        # 设置网络拓扑
        agents = ["agent_001", "agent_002", "agent_003"]
        mock_social_network.get_all_agents.return_value = agents
        mock_social_network.get_neighbors.return_value = agents

        # 初始化模型
        viral_model.set_initial_infected(["agent_001"])
        diffusion_model.set_initial_adopters(["agent_002"])

        viral_model.infection_probability = 0.5
        diffusion_model.adoption_probability = 0.5

        # 并发执行传播
        async def run_concurrent_simulation():
            with patch('random.random', return_value=0.3):
                viral_task = asyncio.create_task(viral_model.async_propagate_simulation())
                diffusion_task = asyncio.create_task(diffusion_model.async_predict_diffusion())

                viral_result, diffusion_result = await asyncio.gather(
                    viral_task, diffusion_task
                )

                return viral_result, diffusion_result

        # 运行并发模拟
        viral_result, diffusion_result = asyncio.run(run_concurrent_simulation())

        # 验证结果
        assert isinstance(viral_result, list)
        assert isinstance(diffusion_result, list)
        assert len(viral_result) > 0
        assert len(diffusion_result) > 0