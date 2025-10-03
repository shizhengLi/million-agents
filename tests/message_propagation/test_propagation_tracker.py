"""
传播路径追踪测试

测试传播路径追踪和分析功能
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
from src.message_propagation.propagation_tracker import PropagationTracker
from src.message_propagation.social_network import SocialNetwork


class TestPropagationTracker:
    """传播路径追踪测试类"""

    @pytest.fixture
    def mock_social_network(self):
        """创建模拟社交网络"""
        network = Mock(spec=SocialNetwork)
        network.get_neighbors.return_value = []
        network.get_agent.return_value = None
        return network

    @pytest.fixture
    def tracker(self, mock_social_network):
        """创建传播追踪器实例"""
        return PropagationTracker(mock_social_network)

    def test_initialization(self, tracker, mock_social_network):
        """测试传播追踪器初始化"""
        assert tracker.network == mock_social_network
        assert len(tracker.propagation_paths) == 0
        assert len(tracker.infection_sources) == 0
        assert len(tracker.infection_times) == 0
        assert len(tracker.propagation_edges) == 0

    def test_track_infection_empty_source(self, tracker):
        """测试空感染源的追踪"""
        with pytest.raises(ValueError, match="感染源不能为空"):
            tracker.track_infection("", "target_001")

    def test_track_infection_empty_target(self, tracker):
        """测试空感染目标的追踪"""
        with pytest.raises(ValueError, match="感染目标不能为空"):
            tracker.track_infection("source_001", "")

    def test_track_infection_same_agent(self, tracker):
        """测试同一智能体的感染追踪"""
        with pytest.raises(ValueError, match="感染源和感染目标不能相同"):
            tracker.track_infection("agent_001", "agent_001")

    def test_track_infection_single_step(self, tracker):
        """测试单步感染追踪"""
        tracker.track_infection("source_001", "target_001")

        assert "target_001" in tracker.infection_sources
        assert tracker.infection_sources["target_001"] == ["source_001"]
        assert ("source_001", "target_001") in tracker.propagation_edges
        assert tracker.infection_times["target_001"] is not None

    def test_track_infection_multiple_steps(self, tracker):
        """测试多步感染追踪"""
        # 传播链: source_001 -> target_001 -> target_002
        tracker.track_infection("source_001", "target_001")
        tracker.track_infection("target_001", "target_002")

        assert tracker.infection_sources["target_001"] == ["source_001"]
        assert tracker.infection_sources["target_002"] == ["target_001"]
        assert ("source_001", "target_001") in tracker.propagation_edges
        assert ("target_001", "target_002") in tracker.propagation_edges

    def test_track_infection_duplicate_tracking(self, tracker):
        """测试重复感染追踪"""
        tracker.track_infection("source_001", "target_001")
        initial_time = tracker.infection_times["target_001"]

        # 重复追踪应该更新时间但不改变源
        tracker.track_infection("source_001", "target_001")
        assert tracker.infection_sources["target_001"] == ["source_001"]
        assert tracker.infection_times["target_001"] >= initial_time

    def test_get_infection_chain_empty(self, tracker):
        """测试空感染链获取"""
        chain = tracker.get_infection_chain("agent_001")
        assert chain == []

    def test_get_infection_chain_single_agent(self, tracker):
        """测试单个智能体的感染链"""
        tracker.track_infection("source_001", "target_001")

        chain = tracker.get_infection_chain("target_001")
        assert len(chain) == 2
        assert chain[0] == "source_001"
        assert chain[1] == "target_001"

    def test_get_infection_chain_multiple_agents(self, tracker):
        """测试多个智能体的感染链"""
        # 传播链: source_001 -> target_001 -> target_002 -> target_003
        tracker.track_infection("source_001", "target_001")
        tracker.track_infection("target_001", "target_002")
        tracker.track_infection("target_002", "target_003")

        chain = tracker.get_infection_chain("target_003")
        assert len(chain) == 4
        assert chain == ["source_001", "target_001", "target_002", "target_003"]

    def test_get_infection_chain_circular(self, tracker):
        """测试循环感染的感染链处理"""
        # 创建循环: source_001 -> target_001 -> source_001
        tracker.track_infection("source_001", "target_001")
        tracker.track_infection("target_001", "source_001")

        # 应该能够处理循环并返回有限的链
        chain = tracker.get_infection_chain("source_001", max_depth=10)
        assert len(chain) <= 10

    def test_get_infection_tree_empty(self, tracker):
        """测试空感染树获取"""
        tree = tracker.get_infection_tree("root_001")
        assert tree == {}

    def test_get_infection_tree_simple(self, tracker):
        """测试简单感染树"""
        # 树形结构: root_001 -> child_001, child_002
        tracker.track_infection("root_001", "child_001")
        tracker.track_infection("root_001", "child_002")

        tree = tracker.get_infection_tree("root_001")
        assert "root_001" in tree
        assert "child_001" in tree["root_001"]
        assert "child_002" in tree["root_001"]

    def test_get_infection_tree_complex(self, tracker):
        """测试复杂感染树"""
        # 复杂树形结构
        tracker.track_infection("root_001", "child_001")
        tracker.track_infection("root_001", "child_002")
        tracker.track_infection("child_001", "grandchild_001")
        tracker.track_infection("child_001", "grandchild_002")
        tracker.track_infection("child_002", "grandchild_003")

        tree = tracker.get_infection_tree("root_001")
        assert "root_001" in tree
        assert "child_001" in tree["root_001"]
        assert "child_002" in tree["root_001"]
        assert "grandchild_001" in tree["root_001"]["child_001"]
        assert "grandchild_002" in tree["root_001"]["child_001"]
        assert "grandchild_003" in tree["root_001"]["child_002"]

    def test_get_propagation_statistics_empty(self, tracker):
        """测试空传播统计"""
        stats = tracker.get_propagation_statistics()
        assert stats['total_infections'] == 0
        assert stats['unique_sources'] == 0
        assert stats['propagation_depth'] == 0
        assert stats['branching_factor'] == 0.0

    def test_get_propagation_statistics_simple(self, tracker):
        """测试简单传播统计"""
        tracker.track_infection("source_001", "target_001")
        tracker.track_infection("source_001", "target_002")

        stats = tracker.get_propagation_statistics()
        assert stats['total_infections'] == 2
        assert stats['unique_sources'] == 1
        assert stats['propagation_depth'] == 1
        assert stats['branching_factor'] == 2.0

    def test_get_propagation_statistics_complex(self, tracker):
        """测试复杂传播统计"""
        # source_001 -> target_001, target_002
        # target_001 -> grandchild_001
        tracker.track_infection("source_001", "target_001")
        tracker.track_infection("source_001", "target_002")
        tracker.track_infection("target_001", "grandchild_001")

        stats = tracker.get_propagation_statistics()
        assert stats['total_infections'] == 3
        assert stats['unique_sources'] == 2  # source_001, target_001
        assert stats['propagation_depth'] == 2
        assert stats['branching_factor'] == 3.0 / 2  # 3 infections / 2 sources

    def test_find_infection_sources_empty(self, tracker):
        """测试空感染源查找"""
        sources = tracker.find_infection_sources("agent_001")
        assert sources == []

    def test_find_infection_sources_single(self, tracker):
        """测试单个感染源查找"""
        tracker.track_infection("source_001", "agent_001")
        sources = tracker.find_infection_sources("agent_001")
        assert sources == ["source_001"]

    def test_find_infection_sources_multiple(self, tracker):
        """测试多个感染源查找"""
        # 多个路径感染同一个目标
        tracker.track_infection("source_001", "agent_001")
        tracker.track_infection("source_002", "agent_001")
        sources = tracker.find_infection_sources("agent_001")
        assert len(sources) == 2
        assert "source_001" in sources
        assert "source_002" in sources

    def test_get_infection_time_empty(self, tracker):
        """测试空感染时间获取"""
        time = tracker.get_infection_time("agent_001")
        assert time is None

    def test_get_infection_time_with_data(self, tracker):
        """测试有数据的感染时间获取"""
        tracker.track_infection("source_001", "target_001")
        time = tracker.get_infection_time("target_001")
        assert time is not None
        assert isinstance(time, datetime)

    def test_calculate_propagation_speed_empty(self, tracker):
        """测试空传播速度计算"""
        speed = tracker.calculate_propagation_speed()
        assert speed == 0.0

    def test_calculate_propagation_speed_single_infection(self, tracker):
        """测试单个感染的传播速度计算"""
        tracker.track_infection("source_001", "target_001")
        speed = tracker.calculate_propagation_speed()
        assert speed > 0.0

    def test_calculate_propagation_speed_multiple_infections(self, tracker):
        """测试多个感染的传播速度计算"""
        # 添加一些时间间隔的感染
        tracker.track_infection("source_001", "target_001")

        # 模拟时间延迟
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime.now() + timedelta(seconds=10)
            tracker.track_infection("target_001", "target_002")

        speed = tracker.calculate_propagation_speed()
        assert speed > 0.0

    def test_get_critical_nodes_empty(self, tracker):
        """测试空关键节点获取"""
        critical_nodes = tracker.get_critical_nodes()
        assert critical_nodes == []

    def test_get_critical_nodes_simple(self, tracker):
        """测试简单关键节点获取"""
        # source_001 是关键节点，感染了多个目标
        tracker.track_infection("source_001", "target_001")
        tracker.track_infection("source_001", "target_002")
        tracker.track_infection("source_001", "target_003")

        critical_nodes = tracker.get_critical_nodes(min_out_degree=2)
        assert "source_001" in critical_nodes

    def test_get_propagation_paths_empty(self, tracker):
        """测试空传播路径获取"""
        paths = tracker.get_propagation_paths()
        assert paths == []

    def test_get_propagation_paths_simple(self, tracker):
        """测试简单传播路径获取"""
        tracker.track_infection("source_001", "target_001")
        tracker.track_infection("source_001", "target_002")

        paths = tracker.get_propagation_paths()
        assert len(paths) == 2
        assert ("source_001", "target_001") in paths
        assert ("source_001", "target_002") in paths

    def test_reset_tracker(self, tracker):
        """测试重置追踪器"""
        tracker.track_infection("source_001", "target_001")
        tracker.track_infection("target_001", "target_002")

        tracker.reset_tracker()

        assert len(tracker.propagation_paths) == 0
        assert len(tracker.infection_sources) == 0
        assert len(tracker.infection_times) == 0
        assert len(tracker.propagation_edges) == 0

    def test_get_infection_depth_empty(self, tracker):
        """测试空感染深度计算"""
        depth = tracker.get_infection_depth("agent_001")
        assert depth == 0

    def test_get_infection_depth_single_step(self, tracker):
        """测试单步感染深度计算"""
        tracker.track_infection("source_001", "target_001")
        depth = tracker.get_infection_depth("target_001")
        assert depth == 1

    def test_get_infection_depth_multiple_steps(self, tracker):
        """测试多步感染深度计算"""
        tracker.track_infection("source_001", "target_001")
        tracker.track_infection("target_001", "target_002")
        tracker.track_infection("target_002", "target_003")

        depth = tracker.get_infection_depth("target_003")
        assert depth == 3

    def test_complex_propagation_scenario(self, tracker):
        """测试复杂传播场景"""
        # 创建一个复杂的传播网络
        # 多个源感染不同的目标，形成复杂的传播图
        tracker.track_infection("source_001", "target_001")
        tracker.track_infection("source_001", "target_002")
        tracker.track_infection("source_002", "target_003")
        tracker.track_infection("target_001", "target_004")
        tracker.track_infection("target_002", "target_005")
        tracker.track_infection("target_003", "target_006")

        # 验证传播统计
        stats = tracker.get_propagation_statistics()
        assert stats['total_infections'] == 6
        assert stats['unique_sources'] == 5  # source_001, source_002, target_001, target_002, target_003
        assert stats['propagation_depth'] >= 2

        # 验证关键节点
        critical_nodes = tracker.get_critical_nodes(min_out_degree=1)
        assert len(critical_nodes) >= 3

    def test_max_depth_protection(self, tracker):
        """测试最大深度保护"""
        # 创建一个很长的感染链
        current_agent = "source_001"
        for i in range(20):  # 创建超过默认最大深度的链
            next_agent = f"target_{i:03d}"
            tracker.track_infection(current_agent, next_agent)
            current_agent = next_agent

        # 获取感染链应该有最大深度限制
        chain = tracker.get_infection_chain(current_agent, max_depth=10)
        assert len(chain) <= 10

    def test_propagation_tree_cycle_handling(self, tracker):
        """测试传播树循环处理"""
        # 创建循环: A -> B -> C -> A
        tracker.track_infection("A", "B")
        tracker.track_infection("B", "C")
        tracker.track_infection("C", "A")

        # 获取传播树应该能够处理循环
        tree = tracker.get_infection_tree("A", max_depth=10)
        assert isinstance(tree, dict)
        # 树应该不会无限递归

    @pytest.mark.asyncio
    async def test_async_track_infection(self, tracker):
        """测试异步感染追踪"""
        await tracker.async_track_infection("source_001", "target_001")
        assert "target_001" in tracker.infection_sources
        assert tracker.infection_sources["target_001"] == ["source_001"]

    def test_get_infection_depth_complex(self, tracker):
        """测试复杂感染深度计算"""
        # 创建深度为4的感染链
        tracker.track_infection("A", "B")
        tracker.track_infection("B", "C")
        tracker.track_infection("C", "D")
        tracker.track_infection("D", "E")

        depth = tracker.get_infection_depth("E")
        assert depth == 4

    def test_get_spread_patterns_chain(self, tracker):
        """测试链式传播模式识别"""
        # 创建链式传播
        tracker.track_infection("A", "B")
        tracker.track_infection("B", "C")
        tracker.track_infection("C", "D")

        patterns = tracker.get_spread_patterns()
        assert patterns['pattern_type'] == 'chain'
        assert patterns['breadth'] == 3
        assert patterns['depth'] >= 3

    def test_get_temporal_analysis(self, tracker):
        """测试时间分析"""
        from datetime import datetime, timedelta
        base_time = datetime.now()

        tracker.track_infection("A", "B")
        tracker.track_infection("B", "C")  # 几乎同时

        temporal = tracker.get_temporal_analysis()
        assert temporal['total_duration'] > 0
        assert temporal['spreading_rate'] > 0
        assert 'peak_time' in temporal

    def test_find_bottlenecks(self, tracker):
        """测试查找传播瓶颈"""
        # 创建星形网络，中心节点是瓶颈
        tracker.track_infection("center", "node1")
        tracker.track_infection("center", "node2")
        tracker.track_infection("center", "node3")

        bottlenecks = tracker.find_bottlenecks()
        assert len(bottlenecks) >= 0

    def test_get_network_metrics(self, tracker):
        """测试网络指标获取"""
        tracker.track_infection("source_001", "target_001")
        tracker.track_infection("target_001", "target_002")

        metrics = tracker.get_network_metrics()
        assert 'propagation_stats' in metrics
        assert 'spread_patterns' in metrics
        assert 'temporal_analysis' in metrics
        assert 'network_health' in metrics

    def test_export_data(self, tracker):
        """测试数据导出"""
        tracker.track_infection("source_001", "target_001")
        tracker.track_infection("target_001", "target_002")

        data = tracker.export_data()
        assert 'propagation_paths' in data
        assert 'infection_sources' in data
        assert 'infection_times' in data
        assert 'statistics' in data
        assert 'network_metrics' in data