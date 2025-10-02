"""
分布式负载均衡器测试
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch
from src.distributed.load_balancer import LoadBalancer, Node, HealthStatus


class TestLoadBalancer:
    """测试负载均衡器核心功能"""

    def test_load_balancer_initialization(self):
        """测试负载均衡器初始化"""
        # 创建负载均衡器
        balancer = LoadBalancer(strategy="round_robin")

        # 验证初始状态
        assert balancer.strategy == "round_robin"
        assert len(balancer.nodes) == 0
        assert balancer.healthy_nodes == set()
        assert balancer.current_index == 0

    def test_add_node(self):
        """测试添加节点"""
        balancer = LoadBalancer()
        node = Node("node1", "localhost:8001", weight=1)

        # 添加节点
        balancer.add_node(node)

        # 验证节点已添加
        assert len(balancer.nodes) == 1
        assert node in balancer.nodes
        assert node in balancer.healthy_nodes

    def test_remove_node(self):
        """测试移除节点"""
        balancer = LoadBalancer()
        node = Node("node1", "localhost:8001")
        balancer.add_node(node)

        # 移除节点
        balancer.remove_node("node1")

        # 验证节点已移除
        assert len(balancer.nodes) == 0
        assert node not in balancer.nodes
        assert node not in balancer.healthy_nodes

    def test_round_robin_strategy(self):
        """测试轮询负载均衡策略"""
        balancer = LoadBalancer(strategy="round_robin")

        # 添加3个节点
        nodes = [
            Node("node1", "localhost:8001"),
            Node("node2", "localhost:8002"),
            Node("node3", "localhost:8003")
        ]

        for node in nodes:
            balancer.add_node(node)

        # 执行轮询选择
        selections = []
        for _ in range(6):
            selected = balancer.get_next_node()
            selections.append(selected.id)

        # 验证轮询结果
        expected = ["node1", "node2", "node3", "node1", "node2", "node3"]
        assert selections == expected

    def test_weighted_round_robin_strategy(self):
        """测试加权轮询策略"""
        balancer = LoadBalancer(strategy="weighted_round_robin")

        # 添加带权重的节点
        nodes = [
            Node("node1", "localhost:8001", weight=3),
            Node("node2", "localhost:8002", weight=2),
            Node("node3", "localhost:8003", weight=1)
        ]

        for node in nodes:
            balancer.add_node(node)

        # 执行加权轮询选择
        selections = []
        for _ in range(12):  # 总权重为6，循环2次
            selected = balancer.get_next_node()
            selections.append(selected.id)

        # 验证加权轮询结果（基于权重3:2:1）
        # 构建加权列表：node1, node1, node1, node2, node2, node3
        # 循环2次得到：node1, node1, node1, node2, node2, node3, node1, node1, node1, node2, node2, node3
        expected = ["node1", "node1", "node1", "node2", "node2", "node3"] * 2
        assert selections == expected

    def test_least_connections_strategy(self):
        """测试最少连接策略"""
        balancer = LoadBalancer(strategy="least_connections")

        # 添加带不同连接数的节点
        nodes = [
            Node("node1", "localhost:8001", current_connections=10),
            Node("node2", "localhost:8002", current_connections=5),
            Node("node3", "localhost:8003", current_connections=2)
        ]

        for node in nodes:
            balancer.add_node(node)

        # 应该选择连接数最少的节点
        selected = balancer.get_next_node()
        assert selected.id == "node3"

    def test_health_check_passes(self):
        """测试健康检查通过的情况"""
        balancer = LoadBalancer()
        node = Node("node1", "localhost:8001")

        # Mock健康检查返回健康
        with patch.object(balancer, '_check_node_health') as mock_check:
            mock_check.return_value = True

            # 执行健康检查
            is_healthy = balancer.check_node_health(node)

            # 验证结果
            assert is_healthy is True
            mock_check.assert_called_once_with(node)

    def test_health_check_fails(self):
        """测试健康检查失败的情况"""
        balancer = LoadBalancer()
        node = Node("node1", "localhost:8001")

        # Mock健康检查返回不健康
        with patch.object(balancer, '_check_node_health') as mock_check:
            mock_check.return_value = False

            # 执行健康检查
            is_healthy = balancer.check_node_health(node)

            # 验证结果
            assert is_healthy is False
            mock_check.assert_called_once_with(node)

    def test_remove_unhealthy_nodes(self):
        """测试移除不健康节点"""
        balancer = LoadBalancer()

        # 添加节点（node2不健康）
        nodes = [
            Node("node1", "localhost:8001", health_status=HealthStatus.HEALTHY),
            Node("node2", "localhost:8002", health_status=HealthStatus.UNHEALTHY),
            Node("node3", "localhost:8003", health_status=HealthStatus.HEALTHY)
        ]

        for node in nodes:
            balancer.add_node(node)

        # 执行健康检查并移除不健康节点
        removed_count = balancer.remove_unhealthy_nodes()

        # 验证结果
        assert removed_count == 1
        assert len(balancer.healthy_nodes) == 2

        # 检查健康节点包含正确的节点ID
        healthy_node_ids = [node.id for node in balancer.healthy_nodes]
        assert "node1" in healthy_node_ids
        assert "node2" not in healthy_node_ids
        assert "node3" in healthy_node_ids

    def test_no_healthy_nodes(self):
        """测试没有健康节点的情况"""
        balancer = LoadBalancer()

        # 添加不健康的节点
        node = Node("node1", "localhost:8001", health_status=HealthStatus.UNHEALTHY)
        balancer.add_node(node)

        # 应该返回None而不是抛出异常
        selected = balancer.get_next_node()
        assert selected is None

    def test_concurrent_requests(self):
        """测试并发请求处理"""
        balancer = LoadBalancer()
        nodes = [
            Node("node1", "localhost:8001"),
            Node("node2", "localhost:8002"),
            Node("node3", "localhost:8003")
        ]

        for node in nodes:
            balancer.add_node(node)

        # 模拟并发请求
        async def make_request(balancer):
            return balancer.get_next_node()

        async def test_concurrent():
            tasks = [make_request(balancer) for _ in range(100)]
            results = await asyncio.gather(*tasks)

            # 验证所有请求都有结果
            assert all(result is not None for result in results)

            # 验证负载相对均衡（允许一定偏差）
            counts = {}
            for result in results:
                counts[result.id] = counts.get(result.id, 0) + 1

            # 计算标准差
            mean = len(results) / len(nodes)
            variance = sum((count - mean) ** 2 for count in counts.values()) / len(nodes)
            std_dev = variance ** 0.5

            # 标准差应该相对较小
            assert std_dev < 5  # 允许一定偏差

        # 运行异步测试
        asyncio.run(test_concurrent())

    def test_node_weight_validation(self):
        """测试节点权重验证"""
        balancer = LoadBalancer()

        # 测试无效权重
        with pytest.raises(ValueError):
            Node("node1", "localhost:8001", weight=0)

        with pytest.raises(ValueError):
            Node("node2", "localhost:8002", weight=-1)

        # 测试有效权重
        node = Node("node3", "localhost:8003", weight=1)
        assert node.weight == 1

    def test_load_balancer_statistics(self):
        """测试负载均衡器统计信息"""
        balancer = LoadBalancer()

        # 添加带不同连接数的节点
        nodes = [
            Node("node1", "localhost:8001", weight=2, current_connections=10),
            Node("node2", "localhost:8002", weight=1, current_connections=5),
            Node("node3", "localhost:8003", weight=1, current_connections=2)
        ]

        for node in nodes:
            balancer.add_node(node)

        # 获取统计信息
        stats = balancer.get_statistics()

        # 验证统计信息
        assert stats['total_nodes'] == 3
        assert stats['healthy_nodes'] == 3
        assert stats['total_weight'] == 4
        assert stats['total_connections'] == 17
        assert stats['avg_connections'] == 17 / 3

    def test_strategy_validation(self):
        """测试策略验证"""
        # 测试有效策略
        for strategy in ["round_robin", "weighted_round_robin", "least_connections"]:
            balancer = LoadBalancer(strategy=strategy)
            assert balancer.strategy == strategy

        # 测试无效策略
        with pytest.raises(ValueError):
            LoadBalancer(strategy="invalid_strategy")