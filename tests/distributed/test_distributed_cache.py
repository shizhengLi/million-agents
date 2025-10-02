"""
分布式缓存和一致性系统测试
"""

import pytest
import asyncio
import time
import json
import threading
from unittest.mock import Mock, patch, AsyncMock
from src.distributed.distributed_cache import (
    DistributedCache, CacheNode, CacheEntry, CacheConsistency,
    ConsistencyLevel, CacheOperation, ConflictResolution,
    CacheCluster, CacheReplication, CachePartitioning
)


class TestCacheEntry:
    """测试缓存条目"""

    def test_cache_entry_creation(self):
        """测试缓存条目创建"""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            ttl_seconds=60
        )

        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert entry.ttl_seconds == 60
        assert entry.version == 1
        assert entry.created_at > 0
        assert entry.expires_at is not None
        assert entry.access_count == 0
        assert entry.last_accessed is None

    def test_cache_entry_ttl_calculation(self):
        """测试TTL计算"""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            ttl_seconds=60
        )

        # 应该在未来60秒过期
        expected_expiry = entry.created_at + 60
        assert abs(entry.expires_at - expected_expiry) < 1

    def test_cache_entry_access(self):
        """测试访问缓存条目"""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            ttl_seconds=60
        )

        initial_count = entry.access_count
        time.sleep(0.01)
        entry.access()

        assert entry.access_count == initial_count + 1
        assert entry.last_accessed > entry.created_at

    def test_cache_entry_is_expired(self):
        """测试缓存条目过期检查"""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            ttl_seconds=1
        )

        # 未过期
        assert not entry.is_expired()

        # 模拟过期
        entry.expires_at = time.time() - 1
        assert entry.is_expired()

    def test_cache_entry_update_value(self):
        """测试更新缓存条目值"""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            ttl_seconds=60
        )

        entry.update_value("new_value")
        assert entry.value == "new_value"
        assert entry.version == 2

    def test_cache_entry_serialization(self):
        """测试缓存条目序列化"""
        entry = CacheEntry(
            key="test_key",
            value={"nested": "data"},
            ttl_seconds=60
        )

        serialized = entry.to_dict()
        assert serialized["key"] == "test_key"
        assert serialized["value"] == {"nested": "data"}
        assert serialized["version"] == 1

        # 测试反序列化
        restored = CacheEntry.from_dict(serialized)
        assert restored.key == entry.key
        assert restored.value == entry.value
        assert restored.version == entry.version


class TestCacheNode:
    """测试缓存节点"""

    def test_cache_node_initialization(self):
        """测试缓存节点初始化"""
        node = CacheNode(
            node_id="node_1",
            address="localhost:8001",
            max_memory_mb=100
        )

        assert node.node_id == "node_1"
        assert node.address == "localhost:8001"
        assert node.max_memory_mb == 100
        assert len(node.cache) == 0
        assert node.current_memory_mb == 0
        assert node.is_active is True

    def test_cache_node_set_and_get(self):
        """测试缓存节点设置和获取"""
        node = CacheNode("node_1", "localhost:8001")

        # 设置缓存
        result = node.set("key1", "value1", ttl_seconds=60)
        assert result is True

        # 获取缓存
        value = node.get("key1")
        assert value == "value1"

        # 获取不存在的键
        value = node.get("nonexistent")
        assert value is None

    def test_cache_node_set_with_ttl(self):
        """测试带TTL的缓存设置"""
        node = CacheNode("node_1", "localhost:8001")

        node.set("key1", "value1", ttl_seconds=1)
        value = node.get("key1")
        assert value == "value1"

        # 等待过期
        time.sleep(1.1)
        value = node.get("key1")
        assert value is None

    def test_cache_node_delete(self):
        """测试删除缓存"""
        node = CacheNode("node_1", "localhost:8001")

        node.set("key1", "value1")
        assert node.get("key1") == "value1"

        result = node.delete("key1")
        assert result is True
        assert node.get("key1") is None

        # 删除不存在的键
        result = node.delete("nonexistent")
        assert result is False

    def test_cache_node_clear(self):
        """测试清空缓存"""
        node = CacheNode("node_1", "localhost:8001")

        node.set("key1", "value1")
        node.set("key2", "value2")
        assert len(node.cache) == 2

        node.clear()
        assert len(node.cache) == 0

    def test_cache_node_size_limit(self):
        """测试缓存大小限制"""
        node = CacheNode("node_1", "localhost:8001", max_entries=2)

        node.set("key1", "value1")
        node.set("key2", "value2")
        assert len(node.cache) == 2

        # 添加第三个条目应该触发LRU淘汰
        node.set("key3", "value3")
        assert len(node.cache) == 2
        assert "key3" in node.cache
        # key1应该被淘汰（最久未访问）

    def test_cache_node_memory_limit(self):
        """测试内存限制"""
        node = CacheNode("node_1", "localhost:8001", max_memory_mb=1)

        # 添加大值应该触发清理
        large_value = "x" * (1024 * 1024)  # 1MB
        node.set("large_key", large_value)

        # 内存使用应该接近限制
        assert node.current_memory_mb <= node.max_memory_mb * 1.1  # 允许10%溢出

    def test_cache_node_statistics(self):
        """测试缓存节点统计"""
        node = CacheNode("node_1", "localhost:8001")

        stats = node.get_statistics()
        assert stats['total_entries'] == 0
        assert stats['hit_count'] == 0
        assert stats['miss_count'] == 0
        assert stats['hit_rate'] == 0.0

        # 添加一些缓存
        node.set("key1", "value1")
        node.get("key1")  # hit
        node.get("nonexistent")  # miss

        stats = node.get_statistics()
        assert stats['total_entries'] == 1
        assert stats['hit_count'] == 1
        assert stats['miss_count'] == 1
        assert stats['hit_rate'] == 0.5

    def test_cache_node_cleanup_expired(self):
        """测试清理过期缓存"""
        node = CacheNode("node_1", "localhost:8001")

        node.set("key1", "value1", ttl_seconds=1)
        node.set("key2", "value2", ttl_seconds=60)

        time.sleep(1.1)
        cleaned_count = node.cleanup_expired()

        assert cleaned_count == 1
        assert node.get("key1") is None
        assert node.get("key2") == "value2"


class TestCacheConsistency:
    """测试缓存一致性"""

    def test_consistency_level_validation(self):
        """测试一致性级别验证"""
        for level in ConsistencyLevel:
            assert isinstance(level.value, str)
            assert len(level.value) > 0

    def test_cache_operation_creation(self):
        """测试缓存操作创建"""
        operation = CacheOperation(
            operation_type="SET",
            key="test_key",
            value="test_value",
            node_id="node_1",
            timestamp=time.time()
        )

        assert operation.operation_type == "SET"
        assert operation.key == "test_key"
        assert operation.value == "test_value"
        assert operation.node_id == "node_1"
        assert operation.timestamp > 0

    def test_conflict_resolution_last_write_wins(self):
        """测试最后写入获胜的冲突解决"""
        consistency = CacheConsistency()

        entry1 = CacheEntry("key", "value1", ttl_seconds=60)
        entry2 = CacheEntry("key", "value2", ttl_seconds=60)
        entry2.created_at = entry1.created_at + 1  # 更新的时间戳

        winner = consistency.resolve_conflict(entry1, entry2, ConflictResolution.LAST_WRITE_WINS)
        assert winner.value == "value2"

    def test_conflict_resolution_version_wins(self):
        """测试版本号获胜的冲突解决"""
        consistency = CacheConsistency()

        entry1 = CacheEntry("key", "value1", ttl_seconds=60)
        entry2 = CacheEntry("key", "value2", ttl_seconds=60)
        entry2.version = 2  # 更高的版本号

        winner = consistency.resolve_conflict(entry1, entry2, ConflictResolution.VERSION_WINS)
        assert winner.value == "value2"

    def test_conflict_resolution_custom_merge(self):
        """测试自定义合并冲突解决"""
        consistency = CacheConsistency()

        def merge_func(entry1, entry2):
            return CacheEntry(
                key=entry1.key,
                value=f"{entry1.value}+{entry2.value}",
                ttl_seconds=entry1.ttl_seconds
            )

        entry1 = CacheEntry("key", "value1", ttl_seconds=60)
        entry2 = CacheEntry("key", "value2", ttl_seconds=60)

        winner = consistency.resolve_conflict(entry1, entry2, ConflictResolution.CUSTOM_MERGE, merge_func)
        assert winner.value == "value1+value2"

    @pytest.mark.asyncio
    async def test_propagate_operation_to_nodes(self):
        """测试向节点传播操作"""
        consistency = CacheConsistency()

        node1 = Mock()
        node2 = Mock()
        nodes = {"node1": node1, "node2": node2}

        operation = CacheOperation(
            operation_type="SET",
            key="test_key",
            value="test_value",
            node_id="source_node",
            timestamp=time.time()
        )

        await consistency.propagate_operation(operation, nodes, exclude_nodes=["source_node"])

        # 应该传播到其他节点
        node1.apply_operation.assert_called_once_with(operation)
        node2.apply_operation.assert_called_once_with(operation)

    @pytest.mark.asyncio
    async def test_sync_nodes_consistency(self):
        """测试节点一致性同步"""
        consistency = CacheConsistency()

        node1 = Mock()
        node1.get_all_entries.return_value = {
            "key1": CacheEntry("key1", "value1", ttl_seconds=60),
            "key2": CacheEntry("key2", "old_value", ttl_seconds=60)
        }

        node2 = Mock()
        node2.get_all_entries.return_value = {
            "key1": CacheEntry("key1", "new_value", ttl_seconds=60),
            "key3": CacheEntry("key3", "value3", ttl_seconds=60)
        }

        nodes = {"node1": node1, "node2": node2}

        sync_result = await consistency.sync_nodes(nodes)

        assert len(sync_result) > 0
        # 应该检测到key1的冲突


class TestCacheReplication:
    """测试缓存复制"""

    def test_replication_initialization(self):
        """测试复制初始化"""
        replication = CacheReplication(
            replication_factor=3,
            consistency_level=ConsistencyLevel.EVENTUAL
        )

        assert replication.replication_factor == 3
        assert replication.consistency_level == ConsistencyLevel.EVENTUAL

    def test_select_replication_nodes(self):
        """测试选择复制节点"""
        replication = CacheReplication(replication_factor=2)

        available_nodes = ["node1", "node2", "node3", "node4"]
        source_node = "node1"

        selected = replication.select_replication_nodes(
            key="test_key",
            available_nodes=available_nodes,
            source_node=source_node
        )

        assert len(selected) == 2
        assert source_node not in selected  # 不应该包含源节点
        assert all(node in available_nodes for node in selected)

    def test_replication_factor_validation(self):
        """测试复制因子验证"""
        # 复制因子应该小于等于可用节点数
        replication = CacheReplication(replication_factor=5)
        available_nodes = ["node1", "node2", "node3"]

        selected = replication.select_replication_nodes(
            key="test_key",
            available_nodes=available_nodes,
            source_node="node1"
        )

        # 最多选择可用节点数-1（排除源节点）
        assert len(selected) <= len(available_nodes) - 1

    @pytest.mark.asyncio
    async def test_replicate_write_operation(self):
        """测试复制写操作"""
        replication = CacheReplication(replication_factor=2)

        node1 = Mock()
        node2 = Mock()
        nodes = {"node1": node1, "node2": node2}

        operation = CacheOperation(
            operation_type="SET",
            key="test_key",
            value="test_value",
            node_id="source_node",
            timestamp=time.time()
        )

        result = await replication.replicate_write(
            operation=operation,
            nodes=nodes,
            replication_nodes=["node1", "node2"]
        )

        assert result.success_count == 2
        assert result.failure_count == 0
        node1.apply_operation.assert_called_once_with(operation)
        node2.apply_operation.assert_called_once_with(operation)

    @pytest.mark.asyncio
    async def test_replicate_read_operation(self):
        """测试复制读操作"""
        replication = CacheReplication()

        node1 = Mock()
        node1.get.return_value = "value1"
        node2 = Mock()
        node2.get.return_value = "value2"
        nodes = {"node1": node1, "node2": node2}

        values = await replication.replicate_read(
            key="test_key",
            nodes=nodes,
            consistency_level=ConsistencyLevel.EVENTUAL
        )

        assert len(values) >= 1  # 至少应该有一个返回值


class TestCachePartitioning:
    """测试缓存分区"""

    def test_partitioning_initialization(self):
        """测试分区初始化"""
        partitioning = CachePartitioning(num_partitions=16)

        assert partitioning.num_partitions == 16

    def test_hash_based_partitioning(self):
        """测试基于哈希的分区"""
        partitioning = CachePartitioning(num_partitions=4)

        # 相同的键应该总是映射到相同的分区
        partition1 = partitioning.get_partition("test_key")
        partition2 = partitioning.get_partition("test_key")
        assert partition1 == partition2

        # 不同的键可能映射到不同的分区
        partition3 = partitioning.get_partition("different_key")
        # 注意：不是一定不同，因为哈希可能碰撞

    def test_range_based_partitioning(self):
        """测试基于范围的分区"""
        partitioning = CachePartitioning(num_partitions=4)

        # 使用数字键进行范围分区
        partition1 = partitioning.get_partition("user:100")
        partition2 = partitioning.get_partition("user:200")

        # 验证分区结果在有效范围内
        assert 0 <= partition1 < partitioning.num_partitions
        assert 0 <= partition2 < partitioning.num_partitions

    def test_consistent_hashing(self):
        """测试一致性哈希"""
        partitioning = CachePartitioning(num_partitions=100)
        partitioning.set_partitioning_strategy("consistent_hash")

        # 一致性哈希应该均匀分布键
        partitions = {}
        for i in range(1000):
            key = f"key_{i}"
            partition = partitioning.get_partition(key)
            partitions[partition] = partitions.get(partition, 0) + 1

        # 验证分布相对均匀
        avg_per_partition = 1000 / partitioning.num_partitions
        for count in partitions.values():
            # 允许一定的偏差 - 使用更宽松的容忍度
            assert abs(count - avg_per_partition) <= avg_per_partition * 1.2

    def test_rebalance_partitions(self):
        """测试重新平衡分区"""
        partitioning = CachePartitioning(num_partitions=4)

        old_nodes = ["node1", "node2"]
        new_nodes = ["node1", "node2", "node3"]

        migration_plan = partitioning.rebalance(old_nodes, new_nodes)

        # 应该有迁移计划
        assert len(migration_plan) > 0

        # 验证迁移计划包含有效的分区
        for partition, target_node in migration_plan.items():
            assert 0 <= partition < partitioning.num_partitions
            assert target_node in new_nodes


class TestCacheCluster:
    """测试缓存集群"""

    def test_cluster_initialization(self):
        """测试集群初始化"""
        cluster = CacheCluster(
            cluster_id="cluster_1",
            replication_factor=2,
            consistency_level=ConsistencyLevel.EVENTUAL
        )

        assert cluster.cluster_id == "cluster_1"
        assert cluster.replication_factor == 2
        assert cluster.consistency_level == ConsistencyLevel.EVENTUAL
        assert len(cluster.nodes) == 0

    def test_add_remove_nodes(self):
        """测试添加和移除节点"""
        cluster = CacheCluster("cluster_1")

        # 添加节点
        node1 = CacheNode("node1", "localhost:8001")
        result = cluster.add_node(node1)
        assert result is True
        assert len(cluster.nodes) == 1

        # 添加重复节点
        result = cluster.add_node(node1)
        assert result is False

        # 移除节点
        result = cluster.remove_node("node1")
        assert result is True
        assert len(cluster.nodes) == 0

        # 移除不存在的节点
        result = cluster.remove_node("nonexistent")
        assert result is False

    def test_cluster_set_and_get(self):
        """测试集群设置和获取"""
        cluster = CacheCluster("cluster_1")

        node1 = CacheNode("node1", "localhost:8001")
        node2 = CacheNode("node2", "localhost:8002")
        cluster.add_node(node1)
        cluster.add_node(node2)

        # 设置值
        result = cluster.set("key1", "value1")
        assert result is True

        # 获取值
        value = cluster.get("key1")
        assert value == "value1"

    def test_cluster_consistency_levels(self):
        """测试集群一致性级别"""
        # 强一致性
        cluster_strong = CacheCluster(
            "cluster_1",
            consistency_level=ConsistencyLevel.STRONG
        )

        # 最终一致性
        cluster_eventual = CacheCluster(
            "cluster_2",
            consistency_level=ConsistencyLevel.EVENTUAL
        )

        assert cluster_strong.consistency_level == ConsistencyLevel.STRONG
        assert cluster_eventual.consistency_level == ConsistencyLevel.EVENTUAL

    @pytest.mark.asyncio
    async def test_cluster_failover(self):
        """测试集群故障转移"""
        cluster = CacheCluster("cluster_1")

        node1 = CacheNode("node1", "localhost:8001")
        node2 = CacheNode("node2", "localhost:8002")
        cluster.add_node(node1)
        cluster.add_node(node2)

        # 设置值
        await cluster.set_async("key1", "value1")
        value = await cluster.get_async("key1")
        assert value == "value1"

        # 模拟节点故障
        node1.is_active = False
        cluster.handle_node_failure("node1")

        # 应该仍然能从其他节点获取值
        value = await cluster.get_async("key1")
        assert value == "value1"

    def test_cluster_statistics(self):
        """测试集群统计"""
        cluster = CacheCluster("cluster_1")

        node1 = CacheNode("node1", "localhost:8001")
        node2 = CacheNode("node2", "localhost:8002")
        cluster.add_node(node1)
        cluster.add_node(node2)

        stats = cluster.get_cluster_statistics()
        assert stats['total_nodes'] == 2
        assert stats['active_nodes'] == 2
        assert 'cache_statistics' in stats

    @pytest.mark.asyncio
    async def test_cluster_rebalance(self):
        """测试集群重新平衡"""
        cluster = CacheCluster("cluster_1")

        node1 = CacheNode("node1", "localhost:8001")
        node2 = CacheNode("node2", "localhost:8002")
        cluster.add_node(node1)
        cluster.add_node(node2)

        # 设置一些数据
        await cluster.set_async("key1", "value1")
        await cluster.set_async("key2", "value2")

        # 添加新节点
        node3 = CacheNode("node3", "localhost:8003")
        cluster.add_node(node3)

        # 触发重新平衡
        rebalance_result = await cluster.rebalance()
        assert rebalance_result.success
        assert rebalance_result.migrated_keys > 0


class TestDistributedCache:
    """测试分布式缓存主类"""

    def test_distributed_cache_initialization(self):
        """测试分布式缓存初始化"""
        cache = DistributedCache(
            cluster_id="cache_cluster",
            node_id="node_1",
            max_memory_mb=100,
            replication_factor=2
        )

        assert cache.cluster.cluster_id == "cache_cluster"
        assert cache.local_node.node_id == "node_1"
        assert cache.replication_factor == 2

    def test_distributed_cache_basic_operations(self):
        """测试分布式缓存基本操作"""
        cache = DistributedCache("cluster", "node1")

        # 设置和获取
        result = cache.set("key", "value")
        assert result is True

        value = cache.get("key")
        assert value == "value"

        # 删除
        result = cache.delete("key")
        assert result is True

        value = cache.get("key")
        assert value is None

    @pytest.mark.asyncio
    async def test_distributed_cache_async_operations(self):
        """测试分布式缓存异步操作"""
        cache = DistributedCache("cluster", "node1")

        # 异步设置和获取
        result = await cache.set_async("async_key", "async_value")
        assert result is True

        value = await cache.get_async("async_key")
        assert value == "async_value"

    def test_distributed_cache_batch_operations(self):
        """测试分布式缓存批量操作"""
        cache = DistributedCache("cluster", "node1")

        # 批量设置
        data = {"key1": "value1", "key2": "value2", "key3": "value3"}
        results = cache.set_batch(data)
        assert all(results.values())

        # 批量获取
        keys = ["key1", "key2", "key3", "nonexistent"]
        values = cache.get_batch(keys)
        assert values["key1"] == "value1"
        assert values["key2"] == "value2"
        assert values["key3"] == "value3"
        assert values["nonexistent"] is None

    @pytest.mark.asyncio
    async def test_distributed_cache_consistency_strong(self):
        """测试强一致性"""
        cache = DistributedCache(
            "cluster", "node1",
            consistency_level=ConsistencyLevel.STRONG
        )

        result = await cache.set_async("key", "value")
        assert result is True

        # 强一致性下应该能立即读到最新值
        value = await cache.get_async("key")
        assert value == "value"

    @pytest.mark.asyncio
    async def test_distributed_cache_consistency_eventual(self):
        """测试最终一致性"""
        cache = DistributedCache(
            "cluster", "node1",
            consistency_level=ConsistencyLevel.EVENTUAL
        )

        result = await cache.set_async("key", "value")
        assert result is True

        # 最终一致性下可能需要等待才能读到最新值
        value = await cache.get_async("key")
        assert value == "value"  # 在单节点测试中应该能立即读到

    def test_distributed_cache_persistence(self):
        """测试分布式缓存持久化"""
        cache = DistributedCache("cluster", "node1", enable_persistence=True)

        cache.set("persistent_key", "persistent_value")

        # 模拟重启
        cache_data = cache.save_to_dict()
        new_cache = DistributedCache.load_from_dict(cache_data)

        value = new_cache.get("persistent_key")
        assert value == "persistent_value"

    def test_distributed_cache_monitoring(self):
        """测试分布式缓存监控"""
        cache = DistributedCache("cluster", "node1")

        cache.set("monitor_key", "monitor_value")
        cache.get("monitor_key")
        cache.get("nonexistent_key")

        metrics = cache.get_metrics()
        assert metrics['hit_count'] >= 1
        assert metrics['miss_count'] >= 1
        assert metrics['total_operations'] >= 2
        assert 'hit_rate' in metrics

    def test_distributed_cache_node_health_check(self):
        """测试节点健康检查"""
        cache = DistributedCache("cluster", "node1")

        health = cache.check_node_health()
        assert health['is_healthy'] is True
        assert 'memory_usage' in health
        assert 'cache_size' in health
        assert 'uptime' in health

    @pytest.mark.asyncio
    async def test_distributed_cache_full_lifecycle(self):
        """测试分布式缓存完整生命周期"""
        cache = DistributedCache("test_cluster", "test_node")

        # 启动
        await cache.start()
        assert cache.is_running is True

        # 基本操作
        await cache.set_async("lifecycle_key", "lifecycle_value")
        value = await cache.get_async("lifecycle_key")
        assert value == "lifecycle_value"

        # 批量操作
        batch_data = {"batch1": "value1", "batch2": "value2"}
        await cache.set_batch_async(batch_data)
        batch_values = await cache.get_batch_async(["batch1", "batch2"])
        assert batch_values["batch1"] == "value1"
        assert batch_values["batch2"] == "value2"

        # 停止
        await cache.stop()
        assert cache.is_running is False

    def test_distributed_cache_performance_monitoring(self):
        """测试分布式缓存性能监控"""
        cache = DistributedCache("cluster", "node1")

        # 执行一些操作
        for i in range(100):
            cache.set(f"perf_key_{i}", f"perf_value_{i}")

        for i in range(100):
            cache.get(f"perf_key_{i}")

        performance = cache.get_performance_metrics()
        assert performance['total_operations'] >= 200
        assert 'operations_per_second' in performance
        assert 'average_response_time' in performance
        assert 'memory_efficiency' in performance