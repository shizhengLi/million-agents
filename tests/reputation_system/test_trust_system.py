"""
信任系统的测试用例

使用TDD方法实现智能体信任计算和信任传播机制
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch
import sys
import os
from datetime import datetime, timedelta

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from reputation_system.trust_system import TrustSystem, TrustNetwork, TrustNode


class TestTrustSystem:
    """信任系统测试类"""

    def setup_method(self):
        """每个测试方法执行前的设置"""
        self.trust_system = TrustSystem()

    def test_initialization(self):
        """测试信任系统初始化"""
        # Given & When
        trust_system = TrustSystem()

        # Then
        assert trust_system is not None
        assert hasattr(trust_system, 'trust_network')
        assert hasattr(trust_system, 'trust_decay_factor')
        assert hasattr(trust_system, 'propagation_depth')

    def test_create_trust_node(self):
        """测试创建信任节点"""
        # Given
        agent_id = "agent1"

        # When
        node = self.trust_system.create_trust_node(agent_id)

        # Then
        assert isinstance(node, TrustNode)
        assert node.agent_id == agent_id
        assert 0 <= node.base_trust_score <= 100
        assert node.created_at is not None

    def test_calculate_direct_trust(self):
        """测试直接信任计算"""
        # Given
        agent_a = "agent1"
        agent_b = "agent2"

        # 创建信任节点
        self.trust_system.create_trust_node(agent_a)
        self.trust_system.create_trust_node(agent_b)

        # 添加交互历史
        interactions = [
            {'agent_a': agent_a, 'agent_b': agent_b, 'outcome': 'positive', 'weight': 1.0},
            {'agent_a': agent_a, 'agent_b': agent_b, 'outcome': 'positive', 'weight': 1.0},
            {'agent_a': agent_a, 'agent_b': agent_b, 'outcome': 'negative', 'weight': 0.5},
        ]

        for interaction in interactions:
            self.trust_system.add_interaction(interaction)

        # When
        trust_score = self.trust_system.calculate_direct_trust(agent_a, agent_b)

        # Then
        assert isinstance(trust_score, float)
        assert 0 <= trust_score <= 100

    def test_calculate_direct_trust_no_history(self):
        """测试无交互历史的直接信任"""
        # Given
        agent_a = "agent1"
        agent_b = "agent2"

        self.trust_system.create_trust_node(agent_a)
        self.trust_system.create_trust_node(agent_b)

        # When
        trust_score = self.trust_system.calculate_direct_trust(agent_a, agent_b)

        # Then
        assert trust_score == 50.0  # 中性信任分数

    def test_trust_propagation_short_path(self):
        """测试短路径信任传播"""
        # Given
        agents = ['agent1', 'agent2', 'agent3']
        interactions = [
            {'agent_a': 'agent1', 'agent_b': 'agent2', 'outcome': 'positive', 'weight': 1.0},
            {'agent_a': 'agent2', 'agent_b': 'agent3', 'outcome': 'positive', 'weight': 1.0},
        ]

        # 创建节点和交互
        for agent_id in agents:
            self.trust_system.create_trust_node(agent_id)

        for interaction in interactions:
            self.trust_system.add_interaction(interaction)

        # When
        propagated_trust = self.trust_system.calculate_propagated_trust('agent1', 'agent3')

        # Then
        assert isinstance(propagated_trust, float)
        assert 0 <= propagated_trust <= 100
        assert propagated_trust > 0  # 应该有传播的信任

    def test_trust_propagation_long_path(self):
        """测试长路径信任传播"""
        # Given
        agents = [f'agent{i}' for i in range(6)]  # agent1 到 agent6 的链式信任
        interactions = [
            {'agent_a': f'agent{i}', 'agent_b': f'agent{i+1}', 'outcome': 'positive', 'weight': 1.0}
            for i in range(5)
        ]

        # 创建节点和交互
        for agent_id in agents:
            self.trust_system.create_trust_node(agent_id)

        for interaction in interactions:
            self.trust_system.add_interaction(interaction)

        # When
        propagated_trust = self.trust_system.calculate_propagated_trust('agent1', 'agent6')

        # Then
        assert isinstance(propagated_trust, float)
        assert 0 <= propagated_trust <= 100
        # 长路径传播的信任应该较低
        direct_trust = self.trust_system.calculate_direct_trust('agent1', 'agent2')
        assert propagated_trust < direct_trust

    def test_trust_propagation_no_path(self):
        """测试无路径的信任传播"""
        # Given
        agent_a = "agent1"
        agent_b = "agent2"

        self.trust_system.create_trust_node(agent_a)
        self.trust_system.create_trust_node(agent_b)

        # When
        propagated_trust = self.trust_system.calculate_propagated_trust(agent_a, agent_b)

        # Then
        assert propagated_trust == 50.0  # 无路径时返回中性分数

    def test_update_trust_with_positive_interaction(self):
        """测试正交互更新信任"""
        # Given
        agent_a = "agent1"
        agent_b = "agent2"

        self.trust_system.create_trust_node(agent_a)
        self.trust_system.create_trust_node(agent_b)

        initial_trust = self.trust_system.calculate_direct_trust(agent_a, agent_b)

        # When
        interaction = {
            'agent_a': agent_a,
            'agent_b': agent_b,
            'outcome': 'positive',
            'weight': 1.0,
            'timestamp': datetime.now()
        }
        self.trust_system.add_interaction(interaction)

        updated_trust = self.trust_system.calculate_direct_trust(agent_a, agent_b)

        # Then
        assert updated_trust > initial_trust

    def test_update_trust_with_negative_interaction(self):
        """测试负交互更新信任"""
        # Given
        agent_a = "agent1"
        agent_b = "agent2"

        # 先建立一些正交互
        for _ in range(5):
            interaction = {
                'agent_a': agent_a,
                'agent_b': agent_b,
                'outcome': 'positive',
                'weight': 1.0,
                'timestamp': datetime.now()
            }
            self.trust_system.add_interaction(interaction)

        self.trust_system.create_trust_node(agent_a)
        self.trust_system.create_trust_node(agent_b)

        initial_trust = self.trust_system.calculate_direct_trust(agent_a, agent_b)

        # When
        negative_interaction = {
            'agent_a': agent_a,
            'agent_b': agent_b,
            'outcome': 'negative',
            'weight': 2.0,  # 高权重负交互
            'timestamp': datetime.now()
        }
        self.trust_system.add_interaction(negative_interaction)

        updated_trust = self.trust_system.calculate_direct_trust(agent_a, agent_b)

        # Then
        assert updated_trust < initial_trust

    def test_trust_decay_over_time(self):
        """测试信任随时间衰减"""
        # Given
        agent_a = "agent1"
        agent_b = "agent2"

        # 创建一个过去的交互
        past_time = datetime.now() - timedelta(days=60)
        old_interaction = {
            'agent_a': agent_a,
            'agent_b': agent_b,
            'outcome': 'positive',
            'weight': 1.0,
            'timestamp': past_time
        }

        self.trust_system.create_trust_node(agent_a)
        self.trust_system.create_trust_node(agent_b)
        self.trust_system.add_interaction(old_interaction)

        initial_trust = self.trust_system.calculate_direct_trust(agent_a, agent_b)

        # When - 应用时间衰减
        self.trust_system.apply_trust_decay(days_elapsed=30)

        # Then
        decayed_trust = self.trust_system.calculate_direct_trust(agent_a, agent_b)
        assert decayed_trust < initial_trust

    def test_batch_trust_calculation(self):
        """测试批量信任计算"""
        # Given
        source_agent = "agent1"
        target_agents = ['agent2', 'agent3', 'agent4', 'agent5']

        # 创建节点和交互
        self.trust_system.create_trust_node(source_agent)
        for agent_id in target_agents:
            self.trust_system.create_trust_node(agent_id)
            interaction = {
                'agent_a': source_agent,
                'agent_b': agent_id,
                'outcome': 'positive',
                'weight': 1.0,
                'timestamp': datetime.now()
            }
            self.trust_system.add_interaction(interaction)

        # When
        trust_scores = self.trust_system.batch_calculate_trust(source_agent, target_agents)

        # Then
        assert isinstance(trust_scores, dict)
        assert len(trust_scores) == len(target_agents)
        assert all(0 <= score <= 100 for score in trust_scores.values())
        assert all(agent_id in trust_scores for agent_id in target_agents)

    def test_trust_network_analysis(self):
        """测试信任网络分析"""
        # Given
        # 创建一个小型信任网络
        network_size = 10
        agents = [f'agent{i}' for i in range(network_size)]

        # 创建节点
        for agent_id in agents:
            self.trust_system.create_trust_node(agent_id)

        # 创建随机交互
        np.random.seed(42)
        for i in range(network_size):
            for j in range(i + 1, min(i + 4, network_size)):  # 每个节点与后续3个节点连接
                outcome = 'positive' if np.random.random() > 0.2 else 'negative'
                interaction = {
                    'agent_a': f'agent{i}',
                    'agent_b': f'agent{j}',
                    'outcome': outcome,
                    'weight': np.random.uniform(0.5, 1.0),
                    'timestamp': datetime.now()
                }
                self.trust_system.add_interaction(interaction)

        # When
        network_stats = self.trust_system.analyze_trust_network()

        # Then
        assert 'total_nodes' in network_stats
        assert 'total_edges' in network_stats
        assert 'average_trust_score' in network_stats
        assert 'network_density' in network_stats
        assert 'clustering_coefficient' in network_stats
        assert network_stats['total_nodes'] == network_size

    def test_find_trust_path(self):
        """测试信任路径查找"""
        # Given
        # 创建信任链: A -> B -> C -> D
        agents = ['agentA', 'agentB', 'agentC', 'agentD']
        interactions = [
            {'agent_a': 'agentA', 'agent_b': 'agentB', 'outcome': 'positive', 'weight': 1.0},
            {'agent_a': 'agentB', 'agent_b': 'agentC', 'outcome': 'positive', 'weight': 1.0},
            {'agent_a': 'agentC', 'agent_b': 'agentD', 'outcome': 'positive', 'weight': 1.0},
        ]

        for agent_id in agents:
            self.trust_system.create_trust_node(agent_id)

        for interaction in interactions:
            self.trust_system.add_interaction(interaction)

        # When
        trust_path = self.trust_system.find_trust_path('agentA', 'agentD')

        # Then
        assert isinstance(trust_path, list)
        assert len(trust_path) == 4  # A -> B -> C -> D
        assert trust_path[0] == 'agentA'
        assert trust_path[-1] == 'agentD'

    def test_find_multiple_trust_paths(self):
        """测试多条信任路径查找"""
        # Given
        # 创建多个信任路径的网络
        agents = ['A', 'B', 'C', 'D', 'E', 'F']
        interactions = [
            # 路径1: A -> B -> D
            {'agent_a': 'A', 'agent_b': 'B', 'outcome': 'positive', 'weight': 1.0},
            {'agent_a': 'B', 'agent_b': 'D', 'outcome': 'positive', 'weight': 1.0},
            # 路径2: A -> C -> E -> D
            {'agent_a': 'A', 'agent_b': 'C', 'outcome': 'positive', 'weight': 1.0},
            {'agent_a': 'C', 'agent_b': 'E', 'outcome': 'positive', 'weight': 1.0},
            {'agent_a': 'E', 'agent_b': 'D', 'outcome': 'positive', 'weight': 1.0},
            # 路径3: A -> F -> D (直接)
            {'agent_a': 'A', 'agent_b': 'F', 'outcome': 'positive', 'weight': 1.0},
            {'agent_a': 'F', 'agent_b': 'D', 'outcome': 'positive', 'weight': 1.0},
        ]

        for agent_id in agents:
            self.trust_system.create_trust_node(agent_id)

        for interaction in interactions:
            self.trust_system.add_interaction(interaction)

        # When
        trust_paths = self.trust_system.find_all_trust_paths('A', 'D')

        # Then
        assert isinstance(trust_paths, list)
        assert len(trust_paths) >= 3  # 至少找到3条路径
        for path in trust_paths:
            assert path[0] == 'A'
            assert path[-1] == 'D'

    def test_trust_node_serialization(self):
        """测试信任节点序列化"""
        # Given
        agent_id = "agent1"
        node = self.trust_system.create_trust_node(agent_id)

        # When
        node_dict = node.to_dict()

        # Then
        assert isinstance(node_dict, dict)
        assert 'agent_id' in node_dict
        assert 'base_trust_score' in node_dict
        assert 'created_at' in node_dict
        assert 'interaction_count' in node_dict

    def test_trust_node_deserialization(self):
        """测试信任节点反序列化"""
        # Given
        node_dict = {
            'agent_id': 'agent1',
            'base_trust_score': 75.5,
            'created_at': datetime.now().isoformat(),
            'interaction_count': 10
        }

        # When
        node = TrustNode.from_dict(node_dict)

        # Then
        assert node.agent_id == 'agent1'
        assert node.base_trust_score == 75.5
        assert node.interaction_count == 10

    def test_calculate_trust_confidence(self):
        """测试信任度计算"""
        # Given
        agent_a = "agent1"
        agent_b = "agent2"

        self.trust_system.create_trust_node(agent_a)
        self.trust_system.create_trust_node(agent_b)

        # 少量交互
        for _ in range(3):
            interaction = {
                'agent_a': agent_a,
                'agent_b': agent_b,
                'outcome': 'positive',
                'weight': 1.0,
                'timestamp': datetime.now()
            }
            self.trust_system.add_interaction(interaction)

        # When
        trust_score = self.trust_system.calculate_direct_trust(agent_a, agent_b)
        confidence = self.trust_system.calculate_trust_confidence(agent_a, agent_b)

        # Then
        assert isinstance(trust_score, float)
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 100
        # 少量交互时置信度应该较低
        assert confidence < 80

    def test_trust_confidence_with_many_interactions(self):
        """测试大量交互时的信任度"""
        # Given
        agent_a = "agent1"
        agent_b = "agent2"

        self.trust_system.create_trust_node(agent_a)
        self.trust_system.create_trust_node(agent_b)

        # 大量交互
        for _ in range(100):
            interaction = {
                'agent_a': agent_a,
                'agent_b': agent_b,
                'outcome': 'positive',
                'weight': 1.0,
                'timestamp': datetime.now()
            }
            self.trust_system.add_interaction(interaction)

        # When
        confidence = self.trust_system.calculate_trust_confidence(agent_a, agent_b)

        # Then
        assert confidence > 90  # 大量交互时置信度应该很高

    def test_detect_trust_anomaly(self):
        """测试信任异常检测"""
        # Given
        agent_a = "agent1"
        agent_b = "agent2"

        # 建立稳定的信任关系
        self.trust_system.create_trust_node(agent_a)
        self.trust_system.create_trust_node(agent_b)

        for _ in range(20):
            interaction = {
                'agent_a': agent_a,
                'agent_b': agent_b,
                'outcome': 'positive',
                'weight': 1.0,
                'timestamp': datetime.now()
            }
            self.trust_system.add_interaction(interaction)

        # When - 突然出现异常交互模式
        for _ in range(5):
            interaction = {
                'agent_a': agent_a,
                'agent_b': agent_b,
                'outcome': 'negative',
                'weight': 2.0,  # 异常高权重
                'timestamp': datetime.now()
            }
            self.trust_system.add_interaction(interaction)

        # Then
        anomaly_score = self.trust_system.detect_trust_anomaly(agent_a, agent_b)
        assert isinstance(anomaly_score, float)
        assert 0 <= anomaly_score <= 100
        assert anomaly_score > 60  # 应该检测到异常

    def test_contextual_trust_calculation(self):
        """测试上下文感知的信任计算"""
        # Given
        agent_a = "agent1"
        agent_b = "agent2"

        self.trust_system.create_trust_node(agent_a)
        self.trust_system.create_trust_node(agent_b)

        # 在不同上下文中的交互
        contexts = ['technical', 'creative', 'business']
        for context in contexts:
            interaction = {
                'agent_a': agent_a,
                'agent_b': agent_b,
                'outcome': 'positive',
                'weight': 1.0,
                'context': context,
                'timestamp': datetime.now()
            }
            self.trust_system.add_interaction(interaction)

        # When
        overall_trust = self.trust_system.calculate_direct_trust(agent_a, agent_b)
        contextual_trusts = self.trust_system.calculate_contextual_trust(agent_a, agent_b)

        # Then
        assert isinstance(contextual_trusts, dict)
        assert len(contextual_trusts) == len(contexts)
        assert all(context in contextual_trusts for context in contexts)
        assert all(0 <= score <= 100 for score in contextual_trusts.values())

    def test_trust_recommendation_integration(self):
        """测试信任与推荐系统的集成"""
        # Given
        source_agent = "agent1"
        candidate_agents = ['agent2', 'agent3', 'agent4', 'agent5']

        # 创建信任网络
        self.trust_system.create_trust_node(source_agent)
        for agent_id in candidate_agents:
            self.trust_system.create_trust_node(agent_id)

        # 建立不同的信任关系
        trust_scores = [90.0, 60.0, 30.0, 75.0]
        for agent_id, trust in zip(candidate_agents, trust_scores):
            # 根据目标信任分数创建相应数量的正交互
            num_interactions = int(trust / 10)
            for _ in range(num_interactions):
                outcome = 'positive' if trust > 50 else 'negative'
                interaction = {
                    'agent_a': source_agent,
                    'agent_b': agent_id,
                    'outcome': outcome,
                    'weight': 1.0,
                    'timestamp': datetime.now()
                }
                self.trust_system.add_interaction(interaction)

        # When
        trust_based_recommendations = self.trust_system.get_trust_based_recommendations(
            source_agent, candidate_agents
        )

        # Then
        assert isinstance(trust_based_recommendations, list)
        assert len(trust_based_recommendations) == len(candidate_agents)
        # 应该按信任分数降序排列
        for i in range(len(trust_based_recommendations) - 1):
            current_score = self.trust_system.calculate_direct_trust(
                source_agent, trust_based_recommendations[i]['agent_id']
            )
            next_score = self.trust_system.calculate_direct_trust(
                source_agent, trust_based_recommendations[i + 1]['agent_id']
            )
            assert current_score >= next_score

    def test_trust_system_performance(self):
        """测试信任系统性能"""
        # Given
        num_agents = 100
        num_interactions = 50

        # 创建大型信任网络
        start_time = time.time()

        for i in range(num_agents):
            agent_id = f"agent_{i}"
            self.trust_system.create_trust_node(agent_id)

            for j in range(num_interactions):
                target_id = f"agent_{(i + j + 1) % num_agents}"
                interaction = {
                    'agent_a': agent_id,
                    'agent_b': target_id,
                    'outcome': 'positive' if j % 4 != 0 else 'negative',
                    'weight': np.random.uniform(0.5, 1.0),
                    'timestamp': datetime.now()
                }
                self.trust_system.add_interaction(interaction)

        setup_time = time.time() - start_time

        # When - 批量计算信任
        batch_start_time = time.time()
        for i in range(min(10, num_agents)):
            source_id = f"agent_{i}"
            target_ids = [f"agent_{j}" for j in range(min(10, num_agents)) if j != i]
            self.trust_system.batch_calculate_trust(source_id, target_ids)

        batch_time = time.time() - batch_start_time

        # Then
        assert setup_time < 30.0  # 设置应该在30秒内完成
        assert batch_time < 10.0   # 批量计算应该在10秒内完成

    def test_trust_system_persistence(self):
        """测试信任系统持久化"""
        # Given
        agent_ids = ['agent1', 'agent2', 'agent3']

        # 创建信任网络
        for agent_id in agent_ids:
            self.trust_system.create_trust_node(agent_id)

        for i, agent_a in enumerate(agent_ids):
            for j, agent_b in enumerate(agent_ids):
                if i != j:
                    interaction = {
                        'agent_a': agent_a,
                        'agent_b': agent_b,
                        'outcome': 'positive' if (i + j) % 2 == 0 else 'negative',
                        'weight': 1.0,
                        'timestamp': datetime.now()
                    }
                    self.trust_system.add_interaction(interaction)

        # When - 保存和加载信任系统
        trust_data = self.trust_system.serialize_trust_system()

        new_trust_system = TrustSystem()
        new_trust_system.deserialize_trust_system(trust_data)

        # Then
        for agent_a in agent_ids:
            for agent_b in agent_ids:
                if agent_a != agent_b:
                    original_trust = self.trust_system.calculate_direct_trust(agent_a, agent_b)
                    loaded_trust = new_trust_system.calculate_direct_trust(agent_a, agent_b)
                    assert abs(original_trust - loaded_trust) < 0.01

    def test_concurrent_trust_updates(self):
        """测试并发信任更新"""
        # Given
        agent_a = "agent1"
        agent_b = "agent2"

        self.trust_system.create_trust_node(agent_a)
        self.trust_system.create_trust_node(agent_b)

        def add_interaction():
            interaction = {
                'agent_a': agent_a,
                'agent_b': agent_b,
                'outcome': 'positive',
                'weight': 1.0,
                'timestamp': datetime.now()
            }
            self.trust_system.add_interaction(interaction)

        # When - 并发添加交互
        import threading
        threads = []
        for _ in range(20):
            thread = threading.Thread(target=add_interaction)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Then
        final_trust = self.trust_system.calculate_direct_trust(agent_a, agent_b)
        assert isinstance(final_trust, float)
        assert 0 <= final_trust <= 100

    def test_trust_system_error_handling(self):
        """测试信任系统错误处理"""
        # Given
        invalid_interactions = [
            {},  # 空交互
            {'agent_a': 'agent1'},  # 缺少必需字段
            {'agent_a': 'agent1', 'agent_b': 'agent2', 'outcome': 'invalid', 'weight': 1.0},  # 无效结果
            {'agent_a': 'agent1', 'agent_b': 'agent2', 'outcome': 'positive', 'weight': -1.0},  # 无效权重
        ]

        # When & Then - 应该优雅处理无效输入
        for invalid_interaction in invalid_interactions:
            try:
                invalid_interaction['timestamp'] = datetime.now()
                self.trust_system.add_interaction(invalid_interaction)
                # 如果没有抛出异常，应该记录错误但不崩溃
            except Exception as e:
                # 允许抛出验证异常
                assert isinstance(e, (ValueError, KeyError))

    def test_trust_decay_configuration(self):
        """测试信任衰减配置"""
        # Given
        agent_a = "agent1"
        agent_b = "agent2"

        self.trust_system.create_trust_node(agent_a)
        self.trust_system.create_trust_node(agent_b)

        # 创建初始交互
        interaction = {
            'agent_a': agent_a,
            'agent_b': agent_b,
            'outcome': 'positive',
            'weight': 1.0,
            'timestamp': datetime.now() - timedelta(days=10)
        }
        self.trust_system.add_interaction(interaction)

        initial_trust = self.trust_system.calculate_direct_trust(agent_a, agent_b)

        # When - 设置不同的衰减率
        original_decay = self.trust_system.trust_decay_factor
        self.trust_system.trust_decay_factor = 0.1  # 快速衰减

        self.trust_system.apply_trust_decay(days_elapsed=10)

        # Then
        decayed_trust = self.trust_system.calculate_direct_trust(agent_a, agent_b)
        assert decayed_trust < initial_trust

        # 恢复原始衰减率
        self.trust_system.trust_decay_factor = original_decay

    def test_trust_system_boundary_conditions(self):
        """测试信任系统边界条件"""
        # Given
        max_score = 100.0
        min_score = 0.0

        # 测试最大信任分数
        agent_a = "agent1"
        agent_b = "agent2"

        self.trust_system.create_trust_node(agent_a)
        self.trust_system.create_trust_node(agent_b)

        # 添加大量正交互尝试达到上限
        for _ in range(1000):
            interaction = {
                'agent_a': agent_a,
                'agent_b': agent_b,
                'outcome': 'positive',
                'weight': 1.0,
                'timestamp': datetime.now()
            }
            self.trust_system.add_interaction(interaction)

        # When
        final_trust = self.trust_system.calculate_direct_trust(agent_a, agent_b)

        # Then
        assert final_trust <= max_score
        assert final_trust >= min_score

    def test_trust_recommendation_filtering(self):
        """测试信任推荐过滤"""
        # Given
        source_agent = "agent1"
        candidates = ['agent2', 'agent3', 'agent4', 'agent5']

        # 设置不同的信任关系
        self.trust_system.create_trust_node(source_agent)
        for candidate in candidates:
            self.trust_system.create_trust_node(candidate)

        # agent2: 高信任, agent3: 中信任, agent4: 低信任, agent5: 无交互
        high_trust_interactions = [
            {'agent_a': source_agent, 'agent_b': 'agent2', 'outcome': 'positive', 'weight': 1.0}
            for _ in range(20)
        ]

        medium_trust_interactions = [
            {'agent_a': source_agent, 'agent_b': 'agent3', 'outcome': 'positive', 'weight': 1.0}
            for _ in range(10)
        ]

        low_trust_interactions = [
            {'agent_a': source_agent, 'agent_b': 'agent4', 'outcome': 'negative', 'weight': 1.0}
            for _ in range(15)
        ]

        for interactions in [high_trust_interactions, medium_trust_interactions, low_trust_interactions]:
            for interaction in interactions:
                interaction['timestamp'] = datetime.now()
                self.trust_system.add_interaction(interaction)

        # When
        filtered_candidates = self.trust_system.filter_candidates_by_trust(
            source_agent, candidates, min_trust_threshold=50
        )

        # Then
        assert isinstance(filtered_candidates, list)
        assert 'agent2' in filtered_candidates  # 高信任
        assert 'agent3' in filtered_candidates  # 中信任
        assert 'agent4' not in filtered_candidates  # 低信任，被过滤
        assert 'agent5' not in filtered_candidates  # 无交互，低于阈值


if __name__ == "__main__":
    pytest.main([__file__])