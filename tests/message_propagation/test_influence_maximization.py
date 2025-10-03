"""
影响力最大化算法测试

测试影响力最大化算法的各种场景
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from src.message_propagation.influence_maximization import InfluenceMaximization
from src.message_propagation.social_network import SocialNetwork


class TestInfluenceMaximization:
    """影响力最大化算法测试类"""

    @pytest.fixture
    def mock_social_network(self):
        """创建模拟社交网络"""
        network = Mock(spec=SocialNetwork)
        network.get_neighbors.return_value = []
        network.get_all_agents.return_value = []
        network.get_agent_count.return_value = 10
        return network

    @pytest.fixture
    def influence_model(self, mock_social_network):
        """创建影响力模型实例"""
        return InfluenceMaximization(mock_social_network)

    def test_initialization(self, influence_model, mock_social_network):
        """测试影响力模型初始化"""
        assert influence_model.network == mock_social_network
        assert influence_model.propagation_probability == 0.1
        assert influence_model.simulation_rounds == 100
        assert len(influence_model.selected_seeds) == 0

    def test_initialization_with_custom_parameters(self, mock_social_network):
        """测试自定义参数初始化"""
        model = InfluenceMaximization(
            network=mock_social_network,
            propagation_probability=0.2,
            simulation_rounds=200
        )
        assert model.propagation_probability == 0.2
        assert model.simulation_rounds == 200

    def test_greedy_algorithm_empty_seed_set(self, influence_model):
        """测试空种子集的贪心算法"""
        with pytest.raises(ValueError, match="种子数量必须大于0"):
            influence_model.greedy_algorithm(0)

    def test_greedy_algorithm_negative_seed_count(self, influence_model):
        """测试负数种子数量的贪心算法"""
        with pytest.raises(ValueError, match="种子数量必须大于0"):
            influence_model.greedy_algorithm(-5)

    def test_greedy_algorithm_insufficient_agents(self, influence_model, mock_social_network):
        """测试智能体数量不足的情况"""
        mock_social_network.get_all_agents.return_value = ["agent_001", "agent_002"]

        with pytest.raises(ValueError, match="种子数量不能超过总智能体数量"):
            influence_model.greedy_algorithm(5)

    def test_greedy_algorithm_single_seed(self, influence_model, mock_social_network):
        """测试单个种子的贪心算法"""
        mock_social_network.get_all_agents.return_value = ["agent_001", "agent_002", "agent_003"]

        # 模拟影响力评估
        with patch.object(influence_model, 'estimate_influence', return_value=5):
            seeds = influence_model.greedy_algorithm(1)

        assert len(seeds) == 1
        assert seeds[0] in ["agent_001", "agent_002", "agent_003"]

    def test_greedy_algorithm_multiple_seeds(self, influence_model, mock_social_network):
        """测试多个种子的贪心算法"""
        agents = ["agent_001", "agent_002", "agent_003", "agent_004"]
        mock_social_network.get_all_agents.return_value = agents

        # 模拟不同智能体的影响力
        def estimate_influence_side_effect(seed_set):
            influence_map = {
                frozenset(["agent_001"]): 10,
                frozenset(["agent_002"]): 8,
                frozenset(["agent_003"]): 6,
                frozenset(["agent_004"]): 4,
                frozenset(["agent_001", "agent_002"]): 15,
                frozenset(["agent_001", "agent_003"]): 14,
            }
            return influence_map.get(frozenset(seed_set), 0)

        with patch.object(influence_model, 'estimate_influence',
                         side_effect=estimate_influence_side_effect):
            seeds = influence_model.greedy_algorithm(2)

        assert len(seeds) == 2
        assert "agent_001" in seeds  # 最高影响力的智能体

    def test_estimate_influence_empty_seed_set(self, influence_model):
        """测试空种子集的影响力评估"""
        influence = influence_model.estimate_influence([])
        assert influence == 0

    def test_estimate_influence_single_seed(self, influence_model, mock_social_network):
        """测试单个种子的影响力评估"""
        mock_social_network.get_neighbors.return_value = ["agent_002", "agent_003"]

        # 模拟传播过程
        with patch('random.random', return_value=0.05):  # 低于传播概率
            influence = influence_model.estimate_influence(["agent_001"])

        assert influence >= 1  # 至少包含种子本身

    def test_estimate_influence_multiple_seeds(self, influence_model, mock_social_network):
        """测试多个种子的影响力评估"""
        mock_social_network.get_neighbors.return_value = ["agent_003", "agent_004"]

        # 模拟传播过程
        with patch('random.random', return_value=0.05):  # 低于传播概率
            influence = influence_model.estimate_influence(["agent_001", "agent_002"])

        assert influence >= 2  # 至少包含两个种子

    def test_degree_heuristic_empty_network(self, influence_model, mock_social_network):
        """测试空网络的度启发式算法"""
        mock_social_network.get_all_agents.return_value = []

        seeds = influence_model.degree_heuristic(3)
        assert len(seeds) == 0

    def test_degree_heuristic_single_agent(self, influence_model, mock_social_network):
        """测试单个智能体的度启发式算法"""
        mock_social_network.get_all_agents.return_value = ["agent_001"]
        mock_social_network.get_neighbors.return_value = []

        seeds = influence_model.degree_heuristic(1)
        assert len(seeds) == 1
        assert seeds[0] == "agent_001"

    def test_degree_heuristic_multiple_agents(self, influence_model, mock_social_network):
        """测试多个智能体的度启发式算法"""
        agents = ["agent_001", "agent_002", "agent_003"]
        mock_social_network.get_all_agents.return_value = agents

        # 模拟不同的度数
        def get_neighbors_side_effect(agent_id):
            degree_map = {
                "agent_001": ["n1", "n2", "n3"],  # 度为3
                "agent_002": ["n4", "n5"],        # 度为2
                "agent_003": ["n6"]               # 度为1
            }
            return degree_map.get(agent_id, [])

        mock_social_network.get_neighbors.side_effect = get_neighbors_side_effect

        seeds = influence_model.degree_heuristic(2)
        assert len(seeds) == 2
        assert "agent_001" in seeds  # 度数最高的智能体

    def test_celf_algorithm_empty_network(self, influence_model, mock_social_network):
        """测试空网络的CELF算法"""
        mock_social_network.get_all_agents.return_value = []

        seeds = influence_model.celf_algorithm(3)
        assert len(seeds) == 0

    def test_celf_algorithm_single_seed(self, influence_model, mock_social_network):
        """测试单个种子的CELF算法"""
        mock_social_network.get_all_agents.return_value = ["agent_001", "agent_002"]

        # 模拟影响力评估
        with patch.object(influence_model, 'estimate_influence', return_value=5):
            seeds = influence_model.celf_algorithm(1)

        assert len(seeds) == 1
        assert seeds[0] in ["agent_001", "agent_002"]

    def test_celf_algorithm_multiple_seeds(self, influence_model, mock_social_network):
        """测试多个种子的CELF算法"""
        agents = ["agent_001", "agent_002", "agent_003"]
        mock_social_network.get_all_agents.return_value = agents

        # 模拟影响力评估
        def estimate_influence_side_effect(seed_set):
            if len(seed_set) == 1:
                if "agent_001" in seed_set:
                    return 10
                elif "agent_002" in seed_set:
                    return 8
                else:
                    return 6
            elif len(seed_set) == 2:
                return 15
            return 0

        with patch.object(influence_model, 'estimate_influence',
                         side_effect=estimate_influence_side_effect):
            seeds = influence_model.celf_algorithm(2)

        assert len(seeds) == 2
        assert "agent_001" in seeds  # 最高影响力的智能体

    def test_simulate_propagation_empty_seeds(self, influence_model):
        """测试空种子集的传播模拟"""
        influenced = influence_model.simulate_propagation([])
        assert len(influenced) == 0

    def test_simulate_propagation_single_seed(self, influence_model, mock_social_network):
        """测试单个种子的传播模拟"""
        mock_social_network.get_neighbors.return_value = ["agent_002", "agent_003"]

        # 模拟传播过程
        with patch('random.random', return_value=0.05):  # 低于传播概率
            influenced = influence_model.simulate_propagation(["agent_001"])

        assert len(influenced) >= 1
        assert "agent_001" in influenced

    def test_simulate_propagation_no_propagation(self, influence_model, mock_social_network):
        """测试无传播的情况"""
        mock_social_network.get_neighbors.return_value = ["agent_002", "agent_003"]

        # 模拟传播概率为0
        influence_model.propagation_probability = 0.0
        with patch('random.random', return_value=0.5):
            influenced = influence_model.simulate_propagation(["agent_001"])

        assert len(influenced) == 1  # 只有种子本身
        assert "agent_001" in influenced

    def test_simulate_propagation_full_propagation(self, influence_model, mock_social_network):
        """测试完全传播的情况"""
        mock_social_network.get_neighbors.return_value = ["agent_002", "agent_003"]

        # 模拟传播概率为1
        influence_model.propagation_probability = 1.0
        with patch('random.random', return_value=0.5):
            influenced = influence_model.simulate_propagation(["agent_001"])

        assert len(influenced) >= 1
        assert "agent_001" in influenced

    def test_calculate_marginal_gain_empty_current_set(self, influence_model):
        """测试空当前集合的边际增益计算"""
        gain = influence_model.calculate_marginal_gain("agent_001", [])
        # 这应该等于agent_001单独的影响力
        assert isinstance(gain, (int, float))

    def test_calculate_marginal_gain_with_current_set(self, influence_model):
        """测试有当前集合的边际增益计算"""
        # 模拟影响力评估
        with patch.object(influence_model, 'estimate_influence') as mock_estimate:
            mock_estimate.side_effect = lambda seeds: len(seeds) * 5  # 简单的影响力模型

            gain = influence_model.calculate_marginal_gain("agent_001", ["agent_002"])
            assert gain == 5  # 10 - 5

    def test_get_influence_statistics_empty(self, influence_model):
        """测试空影响力统计"""
        stats = influence_model.get_influence_statistics()
        assert stats['total_seeds'] == 0
        assert stats['estimated_influence'] == 0
        assert stats['average_marginal_gain'] == 0.0

    def test_get_influence_statistics_with_seeds(self, influence_model):
        """测试有种子的影响力统计"""
        influence_model.selected_seeds = ["agent_001", "agent_002"]
        influence_model.estimated_influence = 50
        influence_model.marginal_gains = [15, 10]

        stats = influence_model.get_influence_statistics()
        assert stats['total_seeds'] == 2
        assert stats['estimated_influence'] == 50
        assert stats['average_marginal_gain'] == 12.5

    def test_reset_influence_model(self, influence_model):
        """测试重置影响力模型"""
        influence_model.selected_seeds = ["agent_001", "agent_002"]
        influence_model.estimated_influence = 50

        influence_model.reset_influence_model()

        assert len(influence_model.selected_seeds) == 0
        assert influence_model.estimated_influence == 0

    def test_model_validation_invalid_probability(self, mock_social_network):
        """测试无效传播概率"""
        with pytest.raises(ValueError, match="传播概率必须在0和1之间"):
            InfluenceMaximization(mock_social_network, propagation_probability=1.5)

        with pytest.raises(ValueError, match="传播概率必须在0和1之间"):
            InfluenceMaximization(mock_social_network, propagation_probability=-0.1)

    def test_model_validation_invalid_simulation_rounds(self, mock_social_network):
        """测试无效模拟轮数"""
        with pytest.raises(ValueError, match="模拟轮数必须大于0"):
            InfluenceMaximization(mock_social_network, simulation_rounds=0)

    def test_complex_influence_scenario(self, influence_model, mock_social_network):
        """测试复杂影响力场景"""
        # 设置星形网络拓扑
        agents = ["center", "leaf1", "leaf2", "leaf3", "leaf4"]
        mock_social_network.get_all_agents.return_value = agents

        def get_neighbors_side_effect(agent_id):
            if agent_id == "center":
                return ["leaf1", "leaf2", "leaf3", "leaf4"]
            else:
                return ["center"]

        mock_social_network.get_neighbors.side_effect = get_neighbors_side_effect

        # 中心节点应该有最高的影响力
        with patch('random.random', return_value=0.05):  # 低于传播概率
            seeds = influence_model.degree_heuristic(1)

        assert len(seeds) == 1
        assert seeds[0] == "center"  # 度数最高的中心节点

    @pytest.mark.asyncio
    async def test_async_greedy_algorithm(self, influence_model, mock_social_network):
        """测试异步贪心算法"""
        mock_social_network.get_all_agents.return_value = ["agent_001", "agent_002"]

        with patch.object(influence_model, 'estimate_influence', return_value=5):
            seeds = await influence_model.async_greedy_algorithm(1)

        assert len(seeds) == 1
        assert seeds[0] in ["agent_001", "agent_002"]

    def test_get_degree_centrality(self, influence_model, mock_social_network):
        """测试计算度中心性"""
        mock_social_network.get_all_agents.return_value = ["agent_001", "agent_002", "agent_003"]

        def get_neighbors_side_effect(agent_id):
            neighbors_map = {
                "agent_001": ["agent_002", "agent_003"],  # 度为2
                "agent_002": ["agent_001"],                # 度为1
                "agent_003": []                            # 度为0
            }
            return neighbors_map.get(agent_id, [])

        mock_social_network.get_neighbors.side_effect = get_neighbors_side_effect

        centrality = influence_model.get_degree_centrality()

        assert centrality["agent_001"] == 2 / 2  # 2/(3-1) = 1.0
        assert centrality["agent_002"] == 1 / 2  # 1/(3-1) = 0.5
        assert centrality["agent_003"] == 0 / 2   # 0/(3-1) = 0.0

    def test_get_degree_centrality_single_agent(self, influence_model, mock_social_network):
        """测试单个智能体的度中心性"""
        mock_social_network.get_all_agents.return_value = ["agent_001"]
        mock_social_network.get_neighbors.return_value = []

        centrality = influence_model.get_degree_centrality()
        assert centrality["agent_001"] == 0.0

    def test_get_betweenness_centrality(self, influence_model, mock_social_network):
        """测试计算介数中心性"""
        mock_social_network.get_all_agents.return_value = ["A", "B", "C", "D"]

        def get_neighbors_side_effect(agent_id):
            neighbors_map = {
                "A": ["B"],
                "B": ["A", "C"],
                "C": ["B", "D"],
                "D": ["C"]
            }
            return neighbors_map.get(agent_id, [])

        mock_social_network.get_neighbors.side_effect = get_neighbors_side_effect

        betweenness = influence_model.get_betweenness_centrality()

        # 在线性链 A-B-C-D 中，B和C的介数中心性应该最高
        assert betweenness["B"] > 0
        assert betweenness["C"] > 0
        assert betweenness["A"] >= 0
        assert betweenness["D"] >= 0

    def test_get_betweenness_centrality_empty(self, influence_model, mock_social_network):
        """测试空网络的介数中心性"""
        mock_social_network.get_all_agents.return_value = []
        mock_social_network.get_neighbors.return_value = []

        betweenness = influence_model.get_betweenness_centrality()
        assert betweenness == {}

    def test_find_shortest_path(self, influence_model, mock_social_network):
        """测试查找最短路径"""
        mock_social_network.get_all_agents.return_value = ["A", "B", "C", "D"]

        def get_neighbors_side_effect(agent_id):
            neighbors_map = {
                "A": ["B"],
                "B": ["A", "C"],
                "C": ["B", "D"],
                "D": ["C"]
            }
            return neighbors_map.get(agent_id, [])

        mock_social_network.get_neighbors.side_effect = get_neighbors_side_effect

        # 测试存在的路径
        path = influence_model.find_shortest_path("A", "D")
        assert path == ["A", "B", "C", "D"]

        # 测试相邻节点
        path = influence_model.find_shortest_path("A", "B")
        assert path == ["A", "B"]

        # 测试相同节点
        path = influence_model.find_shortest_path("A", "A")
        assert path == ["A"]

    def test_find_shortest_path_no_path(self, influence_model, mock_social_network):
        """测试不存在的路径"""
        mock_social_network.get_all_agents.return_value = ["A", "B", "C"]
        mock_social_network.get_neighbors.return_value = []

        path = influence_model.find_shortest_path("A", "C")
        assert path == []

    def test_compare_algorithms(self, influence_model, mock_social_network):
        """测试比较不同算法的性能"""
        mock_social_network.get_all_agents.return_value = ["agent_001", "agent_002", "agent_003"]
        mock_social_network.get_neighbors.return_value = []

        # 模拟不同的影响力评估
        def estimate_influence_side_effect(seeds):
            influence_map = {
                frozenset(["agent_001"]): 10,
                frozenset(["agent_002"]): 8,
                frozenset(["agent_003"]): 6,
                frozenset(["agent_001", "agent_002"]): 15,
                frozenset(["agent_001", "agent_003"]): 14,
                frozenset(["agent_002", "agent_003"]): 12
            }
            return influence_map.get(frozenset(seeds), 0)

        with patch.object(influence_model, 'estimate_influence',
                         side_effect=estimate_influence_side_effect):
            results = influence_model.compare_algorithms(2)

        assert 'greedy' in results
        assert 'degree' in results
        assert 'celf' in results
        assert 'comparison' in results

        # 验证结果结构
        for algorithm in ['greedy', 'degree', 'celf']:
            assert 'seeds' in results[algorithm]
            assert 'influence' in results[algorithm]
            assert 'time' in results[algorithm]

    def test_validate_seed_set_valid(self, influence_model, mock_social_network):
        """测试有效种子集合验证"""
        mock_social_network.get_all_agents.return_value = ["agent_001", "agent_002", "agent_003"]

        is_valid = influence_model.validate_seed_set(["agent_001", "agent_002"])
        assert is_valid == True

    def test_validate_seed_set_invalid_agent(self, influence_model, mock_social_network):
        """测试包含无效智能体的种子集合验证"""
        mock_social_network.get_all_agents.return_value = ["agent_001", "agent_002"]

        is_valid = influence_model.validate_seed_set(["agent_001", "agent_999"])
        assert is_valid == False

    def test_validate_seed_set_duplicate(self, influence_model, mock_social_network):
        """测试重复智能体的种子集合验证"""
        mock_social_network.get_all_agents.return_value = ["agent_001", "agent_002"]

        is_valid = influence_model.validate_seed_set(["agent_001", "agent_001"])
        assert is_valid == False  # 因为长度和集合大小不同