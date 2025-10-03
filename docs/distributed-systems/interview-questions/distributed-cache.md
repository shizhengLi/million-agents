# 分布式缓存系统面试题详解

## 概述

本文档包含了分布式缓存系统的核心面试题，涵盖缓存原理、架构设计、性能优化和实际应用。这些问题基于我们在百万级智能体社交平台项目中的实践经验。

## 基础概念题

### 1. 什么是分布式缓存？为什么需要分布式缓存？

**回答思路**:
- 定义：跨多个服务器节点的缓存系统
- 与本地缓存的区别
- 分布式缓存的核心价值
- 典型应用场景

**参考答案**:
```
分布式缓存是一种将缓存数据分布在多个服务器节点上的缓存系统，与本地缓存相比具有以下特点：

1. 基本定义
   - 数据分布在多个节点上，形成缓存集群
   - 提供统一的访问接口，对应用透明
   - 支持水平扩展和数据分片

2. 与本地缓存的对比
   - 本地缓存：存储在应用进程内存中，访问速度快但容量有限
   - 分布式缓存：独立的缓存服务，访问相对较慢但容量可扩展

3. 为什么需要分布式缓存
   a) 容量扩展性
   - 单机内存容量有限，通常几十GB
   - 分布式缓存可扩展到TB级别
   - 按需添加节点即可扩容

   b) 高可用性
   - 单点故障风险
   - 数据冗余和故障转移
   - 99.99%可用性保证

   c) 数据一致性
   - 多个应用实例共享缓存数据
   - 避免数据不一致问题
   - 统一的缓存失效策略

   d) 性能优化
   - 减少数据库负载
   - 提高响应速度
   - 支持高并发访问

4. 典型应用场景
   - 网站页面缓存
   - API响应缓存
   - 数据库查询缓存
   - 会话存储
   - 计数器和统计信息
   - 实时排行榜
```

**代码示例**:
```python
class DistributedCache:
    def __init__(self, nodes: List[str]):
        self.nodes = nodes
        self.hash_ring = ConsistentHashRing(nodes)
        self.local_cache = {}  # 本地缓存层
        self.cache_stats = CacheStats()

    def get(self, key: str) -> Optional[Any]:
        """从分布式缓存获取数据"""
        # 1. 先检查本地缓存
        local_value = self.local_cache.get(key)
        if local_value is not None:
            self.cache_stats.hit("local")
            return local_value

        # 2. 从分布式缓存获取
        node = self.hash_ring.get_node(key)
        value = self._get_from_node(node, key)

        if value is not None:
            # 回填本地缓存
            self.local_cache[key] = value
            self.cache_stats.hit("distributed")
        else:
            self.cache_stats.miss()

        return value

    def set(self, key: str, value: Any, ttl: int = 3600):
        """设置分布式缓存"""
        node = self.hash_ring.get_node(key)
        success = self._set_to_node(node, key, value, ttl)

        if success:
            # 同时设置本地缓存
            self.local_cache[key] = value
            self.cache_stats.set_operation()

    def _get_from_node(self, node: str, key: str) -> Optional[Any]:
        """从指定节点获取数据"""
        try:
            # 这里实现具体的网络请求逻辑
            response = requests.get(f"http://{node}/get/{key}")
            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None

    def _set_to_node(self, node: str, key: str, value: Any, ttl: int) -> bool:
        """设置数据到指定节点"""
        try:
            response = requests.post(
                f"http://{node}/set/{key}",
                json={"value": value, "ttl": ttl}
            )
            return response.status_code == 200
        except Exception:
            return False
```

### 2. 分布式缓存有哪些常见的数据分片策略？

**回答思路**:
- 分片的目的和重要性
- 常见分片算法：哈希分片、范围分片、一致性哈希
- 各种策略的优缺点对比
- 适用场景分析

**参考答案**:
```
分布式缓存的数据分片策略决定了数据如何在多个节点间分布，是系统性能和可扩展性的关键：

1. 哈希分片 (Hash Sharding)
   a) 实现原理
   - 使用哈希函数计算key的哈希值
   - 对节点数量取模确定目标节点
   - 公式：node_index = hash(key) % node_count

   b) 优点
   - 数据分布均匀
   - 实现简单，计算高效
   - 访问模式确定

   c) 缺点
   - 节点增减时需要重新分片
   - 数据迁移成本高
   - 可能导致雪崩效应

2. 范围分片 (Range Sharding)
   a) 实现原理
   - 按key的范围分配到不同节点
   - 每个节点负责一个key范围
   - 例如：A-F -> Node1, G-L -> Node2, M-Z -> Node3

   b) 优点
   - 支持范围查询
   - 数据局部性好
   - 易于热点数据管理

   c) 缺点
   - 数据分布可能不均
   - 热点范围可能导致负载不均
   - 需要动态调整范围

3. 一致性哈希 (Consistent Hashing)
   a) 实现原理
   - 将节点和数据都映射到哈希环上
   - 数据顺时针寻找最近的节点
   - 使用虚拟节点解决负载均衡

   b) 优点
   - 节点增减时影响最小
   - 数据分布相对均匀
   - 支持动态扩缩容

   c) 缺点
   - 实现复杂度较高
   - 可能存在数据倾斜
   - 需要维护虚拟节点

4. 自定义分片策略
   a) 基于业务特征的分片
   - 按用户ID分片：user_id % node_count
   - 按地理位置分片：基于IP地址
   - 按数据类型分片：不同业务使用不同策略

   b) 动态分片策略
   - 基于负载的自适应分片
   - 热点数据自动分裂
   - 冷数据自动合并

5. 分片策略选择建议
   - 小规模系统：哈希分片
   - 中等规模：一致性哈希
   - 大规模系统：混合策略
   - 特殊需求：自定义分片
```

**代码示例**:
```python
from typing import List, Dict, Any, Optional
import hashlib
import bisect

class HashSharding:
    """哈希分片策略"""
    def __init__(self, nodes: List[str]):
        self.nodes = nodes

    def get_node(self, key: str) -> str:
        hash_value = self._calculate_hash(key)
        index = hash_value % len(self.nodes)
        return self.nodes[index]

    def _calculate_hash(self, key: str) -> int:
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

class RangeSharding:
    """范围分片策略"""
    def __init__(self, ranges: Dict[str, tuple]):
        """
        ranges: {"node1": ("a", "f"), "node2": ("g", "l"), ...}
        """
        self.ranges = ranges
        self.sorted_ranges = sorted([(start, end, node) for node, (start, end) in ranges.items()])

    def get_node(self, key: str) -> str:
        key_lower = key.lower()
        for start, end, node in self.sorted_ranges:
            if start <= key_lower <= end:
                return node
        return self.sorted_ranges[0][2]  # 默认返回第一个节点

class ConsistentHashSharding:
    """一致性哈希分片策略"""
    def __init__(self, nodes: List[str], virtual_nodes: int = 150):
        self.virtual_nodes = virtual_nodes
        self.ring = {}
        self.sorted_keys = []
        self.nodes = nodes

        for node in nodes:
            self.add_node(node)

    def add_node(self, node: str):
        """添加节点到哈希环"""
        for i in range(self.virtual_nodes):
            key = self._calculate_hash(f"{node}:{i}")
            self.ring[key] = node

        self.sorted_keys = sorted(self.ring.keys())

    def remove_node(self, node: str):
        """从哈希环移除节点"""
        for i in range(self.virtual_nodes):
            key = self._calculate_hash(f"{node}:{i}")
            if key in self.ring:
                del self.ring[key]

        self.sorted_keys = sorted(self.ring.keys())

    def get_node(self, key: str) -> str:
        """获取数据对应的节点"""
        if not self.ring:
            raise Exception("No nodes available")

        hash_value = self._calculate_hash(key)

        # 在环上顺时针查找节点
        for key in self.sorted_keys:
            if key >= hash_value:
                return self.ring[key]

        # 环形结构，返回第一个节点
        return self.ring[self.sorted_keys[0]]

    def _calculate_hash(self, key: str) -> int:
        return int(hashlib.md5(str(key).encode()).hexdigest(), 16)

class HybridSharding:
    """混合分片策略"""
    def __init__(self, strategies: Dict[str, Any]):
        """
        strategies: {
            "user": ConsistentHashSharding(...),
            "session": HashSharding(...),
            "cache": RangeSharding(...)
        }
        """
        self.strategies = strategies
        self.default_strategy = list(strategies.values())[0]

    def get_node(self, key: str, key_type: str = None) -> str:
        """根据key类型选择分片策略"""
        if key_type and key_type in self.strategies:
            strategy = self.strategies[key_type]
            return strategy.get_node(key)

        # 默认策略
        return self.default_strategy.get_node(key)

# 使用示例
if __name__ == "__main__":
    nodes = ["node1:6379", "node2:6379", "node3:6379"]

    # 哈希分片
    hash_sharding = HashSharding(nodes)
    print(f"Hash sharding for 'user:123': {hash_sharding.get_node('user:123')}")

    # 一致性哈希分片
    consistent_sharding = ConsistentHashSharding(nodes)
    print(f"Consistent hash for 'user:123': {consistent_sharding.get_node('user:123')}")

    # 混合分片
    hybrid_sharding = HybridSharding({
        "user": consistent_sharding,
        "session": hash_sharding
    })
    print(f"Hybrid sharding for user key: {hybrid_sharding.get_node('user:123', 'user')}")
    print(f"Hybrid sharding for session key: {hybrid_sharding.get_node('session:abc', 'session')}")
```

## 架构设计题

### 3. 设计一个高性能的分布式缓存系统

**回答思路**:
- 整体架构：分层设计、多级缓存
- 核心组件：数据分片、复制、一致性
- 性能优化：缓存策略、数据压缩
- 高可用：故障检测、自动恢复

**参考答案**:
```
设计一个高性能的分布式缓存系统需要考虑以下关键方面：

1. 整体架构设计
   ```
   ┌─────────────────┐
   │   Application  │
   └─────────┬───────┘
               │
   ┌─────────▼───────┐
   │  Client SDK     │  ← 本地缓存 + 智能路由
   └─────────┬───────┘
               │
   ┌─────────▼───────┐
   │  Load Balancer │  ← 请求分发和故障转移
   └─────────┬───────┘
               │
   ┌─────────▼───────┐
   │  Cache Cluster  │  ← 数据分片 + 复制
   │  ┌───────────┐  │
   │  │   Node 1  │  │
   │  │ (Master)  │  │
   │  └───────────┘  │
   │  ┌───────────┐  │
   │  │   Node 2  │  │
   │  │ (Slave)   │  │
   │  └───────────┘  │
   └─────────────────┘
   ```

2. 核心技术组件

a) 数据分片层
   - 一致性哈希分片算法
   - 动态扩缩容支持
   - 数据均衡迁移

b) 复制层
   - 主从复制架构
   - 异步复制保证性能
   - 半同步复制平衡一致性

c) 缓存策略层
   - 多级缓存 (L1/L2/L3)
   - 智能缓存预热
   - 缓存雪崩保护

d) 一致性层
   - 最终一致性模型
   - 版本向量冲突解决
   - 读写分离策略

3. 性能优化策略

a) 内存优化
   - Slab分配器减少内存碎片
   - 对象池复用内存
   - 压缩算法减少内存占用

b) 网络优化
   - 连接池管理
   - 批量操作支持
   - 数据压缩传输

c) 并发优化
   - 多线程事件循环
   - 无锁数据结构
   - 协程支持

4. 高可用设计

a) 故障检测
   - 心跳检测机制
   - 超时重试策略
   - 快速故障切换

b) 数据恢复
   - 自动故障转移
   - 数据重建机制
   - 增量数据同步

c) 监控告警
   - 性能指标监控
   - 异常检测
   - 自动扩容触发
```

**代码示例**:
```python
import asyncio
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

class CacheNodeRole(Enum):
    MASTER = "master"
    SLAVE = "slave"
    COORDINATOR = "coordinator"

@dataclass
class CacheNode:
    id: str
    address: str
    port: int
    role: CacheNodeRole
    status: str = "healthy"
    last_heartbeat: float = 0
    memory_usage: float = 0.0
    active_connections: int = 0

class HighPerformanceCacheCluster:
    """高性能分布式缓存集群"""

    def __init__(self):
        self.nodes: Dict[str, CacheNode] = {}
        self.consistent_hash = ConsistentHashSharding([])
        self.replication_manager = ReplicationManager()
        self.health_checker = HealthChecker()
        self.cache_manager = CacheManager()
        self.monitoring = MonitoringSystem()

    async def start_cluster(self, initial_nodes: List[CacheNode]):
        """启动缓存集群"""
        # 1. 初始化节点
        for node in initial_nodes:
            await self.add_node(node)

        # 2. 启动健康检查
        await self.health_checker.start(self.nodes)

        # 3. 启动监控系统
        await self.monitoring.start()

        # 4. 启动缓存管理器
        await self.cache_manager.start()

    async def add_node(self, node: CacheNode):
        """添加节点到集群"""
        self.nodes[node.id] = node
        self.consistent_hash.add_node(node.id)

        # 如果是主节点，启动复制
        if node.role == CacheNodeRole.MASTER:
            await self.replication_manager.setup_replication(node)

        logger.info(f"Added node {node.id} to cluster")

    async def remove_node(self, node_id: str):
        """从集群移除节点"""
        if node_id not in self.nodes:
            return

        node = self.nodes[node_id]

        # 1. 停止服务
        await self._stop_node_service(node)

        # 2. 迁移数据
        await self._migrate_data_from_node(node)

        # 3. 从哈希环移除
        self.consistent_hash.remove_node(node_id)

        # 4. 更新复制配置
        await self.replication_manager.remove_replication(node)

        del self.nodes[node_id]
        logger.info(f"Removed node {node_id} from cluster")

    async def get(self, key: str) -> Optional[Any]:
        """获取缓存数据"""
        try:
            # 1. 获取目标节点
            primary_node_id = self.consistent_hash.get_node(key)
            primary_node = self.nodes.get(primary_node_id)

            if not primary_node or primary_node.status != "healthy":
                # 主节点不可用，尝试从从节点读取
                return await self._get_from_slave(key)

            # 2. 从主节点读取
            value = await self._get_from_node(primary_node, key)

            # 3. 更新缓存统计
            await self.monitoring.record_operation("get", key, value is not None)

            return value

        except Exception as e:
            logger.error(f"Get operation failed for key {key}: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """设置缓存数据"""
        try:
            # 1. 序列化数据
            serialized_value = self._serialize_value(value)

            # 2. 压缩数据
            compressed_value = self._compress_data(serialized_value)

            # 3. 获取目标节点
            node_id = self.consistent_hash.get_node(key)
            node = self.nodes.get(node_id)

            if not node:
                logger.error(f"No available node for key {key}")
                return False

            # 4. 写入主节点
            success = await self._set_to_node(node, key, compressed_value, ttl)

            if success:
                # 5. 异步复制到从节点
                asyncio.create_task(
                    self.replication_manager.replicate_write(key, compressed_value, ttl)
                )

                # 6. 更新统计
                await self.monitoring.record_operation("set", key, True)

            return success

        except Exception as e:
            logger.error(f"Set operation failed for key {key}: {e}")
            await self.monitoring.record_operation("set", key, False)
            return False

    async def _get_from_slave(self, key: str) -> Optional[Any]:
        """从从节点读取数据"""
        slave_nodes = [
            node for node in self.nodes.values()
            if node.role == CacheNodeRole.SLAVE and node.status == "healthy"
        ]

        for slave in slave_nodes:
            try:
                value = await self._get_from_node(slave, key)
                if value is not None:
                    return value
            except Exception:
                continue

        return None

    async def _migrate_data_from_node(self, source_node: CacheNode):
        """从指定节点迁移数据"""
        logger.info(f"Starting data migration from node {source_node.id}")

        try:
            # 1. 获取节点上的所有key
            all_keys = await self._get_all_keys_from_node(source_node)

            # 2. 批量迁移数据
            batch_size = 100
            for i in range(0, len(all_keys), batch_size):
                batch_keys = all_keys[i:i + batch_size]
                await self._migrate_batch(source_node, batch_keys)

            logger.info(f"Data migration completed from node {source_node.id}")

        except Exception as e:
            logger.error(f"Data migration failed from node {source_node.id}: {e}")

    async def _migrate_batch(self, source_node: CacheNode, keys: List[str]):
        """批量迁移数据"""
        tasks = []
        for key in keys:
            task = asyncio.create_task(self._migrate_single_key(source_node, key))
            tasks.append(task)

        await asyncio.gather(*tasks, return_exceptions=True)

    async def _migrate_single_key(self, source_node: CacheNode, key: str):
        """迁移单个key"""
        try:
            # 1. 从源节点读取数据
            value = await self._get_from_node(source_node, key)
            if value is None:
                return

            # 2. 写入新节点
            new_node_id = self.consistent_hash.get_node(key)
            new_node = self.nodes.get(new_node_id)

            if new_node and new_node.id != source_node.id:
                await self._set_to_node(new_node, key, value["value"], value["ttl"])

        except Exception as e:
            logger.error(f"Failed to migrate key {key}: {e}")

class ReplicationManager:
    """复制管理器"""

    def __init__(self):
        self.replication_groups = {}  # master_id -> [slave_nodes]
        self.replication_lag = {}      # node_id -> lag_seconds

    async def setup_replication(self, master_node: CacheNode):
        """设置主从复制"""
        master_id = master_node.id
        self.replication_groups[master_id] = []

        # 自动分配从节点
        for node_id, node in self.nodes.items():
            if node.id != master_id and node.role == CacheNodeRole.SLAVE:
                await self._setup_slave_replication(master_node, node)

    async def replicate_write(self, key: str, value: Any, ttl: int):
        """异步复制写操作到从节点"""
        # 获取主节点ID
        master_node_id = self.consistent_hash.get_node(key)

        if master_node_id in self.replication_groups:
            slaves = self.replication_groups[master_node_id]

            # 并行复制到所有从节点
            tasks = []
            for slave in slaves:
                task = asyncio.create_task(
                    self._replicate_to_slave(slave, key, value, ttl)
                )
                tasks.append(task)

            await asyncio.gather(*tasks, return_exceptions=True)

class CacheManager:
    """缓存管理器"""

    def __init__(self):
        self.eviction_policies = {
            "lru": LRUEvictionPolicy(),
            "lfu": LFUEvictionPolicy(),
            "ttl": TTLEVictionPolicy()
        }
        self.memory_manager = MemoryManager()

    async def start(self):
        """启动缓存管理器"""
        # 启动内存监控
        asyncio.create_task(self._memory_monitoring_loop())

        # 启动缓存清理
        asyncio.create_task(self._cleanup_loop())

    async def _memory_monitoring_loop(self):
        """内存监控循环"""
        while True:
            try:
                await self.memory_manager.check_memory_usage()
                await asyncio.sleep(60)
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                await asyncio.sleep(5)

    async def _cleanup_loop(self):
        """缓存清理循环"""
        while True:
            try:
                await self._perform_cleanup()
                await asyncio.sleep(300)  # 每5分钟清理一次
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
                await asyncio.sleep(30)
```

### 4. 如何处理分布式缓存中的数据一致性问题？

**回答思路**:
- 一致性模型的分类
- 缓存一致性的挑战
- 解决方案：失效策略、版本控制、冲突解决
- 实际应用中的权衡

**参考答案**:
```
分布式缓存中的数据一致性是一个复杂的问题，需要根据业务需求选择合适的策略：

1. 一致性模型分类
   a) 强一致性
   - 所有节点同时看到相同数据
   - 性能开销大，延迟高
   - 适用于金融等关键业务

   b) 弱一致性
   - 允许短期数据不一致
   - 性能好，可用性高
   - 适用于社交媒体、内容分发

   c) 最终一致性
   - 短期不一致，最终达到一致
   - 平衡性能和一致性
   - 最常用的缓存一致性模型

2. 缓存一致性的挑战
   a) 写操作不一致
   - 缓存写成功，数据库写失败
   - 数据库写成功，缓存写失败
   - 网络分区导致的部分更新

   b) 读操作不一致
   - 读取到过期数据
   - 副本间数据不一致
   - 缓存与数据库不一致

   c) 并发操作冲突
   - 多个客户端同时更新
   - 读写并发冲突
   - 跨节点操作顺序问题

3. 一致性解决方案

   a) 缓存失效策略
   - 写失效（Write-Invalidate）
   - 写穿透（Write-Through）
   - 写回（Write-Back）
   - 刷新策略（Refresh-Ahead）

   b) 版本控制
   - 时间戳版本控制
   - 向量时钟版本控制
   - 令牌桶版本控制

   c) 冲突解决机制
   - 最后写入获胜（LWW）
   - 基于业务逻辑合并
   - 客户端冲突解决

4. 具体实现策略

   a) 数据库同步失效
   ```
   应用更新数据：
   1. 更新数据库
   2. 发送失效消息到消息队列
   3. 缓存节点消费消息，删除对应缓存
   ```

   b) 双写一致性
   ```
   应用更新数据：
   1. 写数据库（同步）
   2. 写缓存（异步）
   3. 定期一致性检查
   4. 冲突检测和修复
   ```

   c) 懒加载一致性
   ```
   读取数据时：
   1. 先读缓存
   2. 如果缓存不存在，读数据库
   3. 将数据写入缓存
   4. 设置合适的TTL
   ```

5. 一致性保证的最佳实践

   a) 关键数据
   - 使用强一致性策略
   - 同步写入操作
   - 版本控制和冲突检测

   b) 非关键数据
   - 使用最终一致性
   - 异步写入操作
   - 定期数据修复

   c) 性能优化
   - 批量操作
   - 异步处理
   - 智能缓存策略
```

**代码示例**:
```python
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

class ConsistencyLevel(Enum):
    STRONG = "strong"
    EVENTUAL = "eventual"
    WEAK = "weak"

@dataclass
class CacheEntry:
    key: str
    value: Any
    version: int
    timestamp: float
    ttl: int

class ConsistentCacheManager:
    """一致性缓存管理器"""

    def __init__(self, consistency_level: ConsistencyLevel = ConsistencyLevel.EVENTUAL):
        self.consistency_level = consistency_level
        self.cache_store = {}  # key -> CacheEntry
        self.version_clock = VersionClock()
        self.message_broker = MessageBroker()
        self.conflict_resolver = ConflictResolver()

    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """设置缓存值"""
        if self.consistency_level == ConsistencyLevel.STRONG:
            return await self._strong_consistency_set(key, value, ttl)
        else:
            return await self._eventual_consistency_set(key, value, ttl)

    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        if key not in self.cache_store:
            return None

        entry = self.cache_store[key]

        # 检查TTL
        if time.time() - entry.timestamp > entry.ttl:
            del self.cache_store[key]
            return None

        return entry.value

    async def _strong_consistency_set(self, key: str, value: Any, ttl: int) -> bool:
        """强一致性写入"""
        # 1. 获取新版本号
        new_version = self.version_clock.increment(key)

        # 2. 创建缓存条目
        entry = CacheEntry(
            key=key,
            value=value,
            version=new_version,
            timestamp=time.time(),
            ttl=ttl
        )

        # 3. 同步写入所有副本
        success = await self._sync_write_to_replicas(key, entry)

        if success:
            self.cache_store[key] = entry

        return success

    async def _eventual_consistency_set(self, key: str, value: Any, ttl: int) -> bool:
        """最终一致性写入"""
        # 1. 创建本地条目
        entry = CacheEntry(
            key=key,
            value=value,
            version=self.version_clock.increment(key),
            timestamp=time.time(),
            ttl=ttl
        )

        # 2. 立即写入本地缓存
        self.cache_store[key] = entry

        # 3. 异步复制到其他节点
        asyncio.create_task(
            self._async_replicate_to_replicas(key, entry)
        )

        return True

    async def _sync_write_to_replicas(self, key: str, entry: CacheEntry) -> bool:
        """同步写入所有副本"""
        replicas = await self._get_replica_nodes()

        if not replicas:
            return True

        # 写入所有副本
        write_tasks = []
        for replica in replicas:
            task = asyncio.create_task(
                self._write_to_replica(replica, key, entry)
            )
            write_tasks.append(task)

        results = await asyncio.gather(*write_tasks, return_exceptions=True)

        # 检查是否所有副本都写入成功
        success_count = sum(1 for result in results if result is True)
        return success_count == len(replicas)

    async def invalidate_key(self, key: str):
        """失效指定key"""
        if key in self.cache_store:
            del self.cache_store[key]

        # 发送失效消息到其他节点
        await self.message_broker.publish(
            "cache_invalidation",
            {"key": key, "timestamp": time.time()}
        )

class VersionClock:
    """版本时钟管理"""

    def __init__(self):
        self.versions = {}  # key -> version_number

    def increment(self, key: str) -> int:
        """递增版本号"""
        if key not in self.versions:
            self.versions[key] = 0

        self.versions[key] += 1
        return self.versions[key]

    def get_version(self, key: str) -> int:
        """获取当前版本号"""
        return self.versions.get(key, 0)

class ConflictResolver:
    """冲突解决器"""

    async def resolve_conflict(self, key: str, entries: List[CacheEntry]) -> CacheEntry:
        """解决缓存冲突"""
        if len(entries) == 1:
            return entries[0]

        # 按时间戳排序，选择最新的
        sorted_entries = sorted(entries, key=lambda x: x.timestamp, reverse=True)
        latest_entry = sorted_entries[0]

        # 检查是否有版本冲突
        conflicting_entries = [
            entry for entry in entries[1:]
            if entry.version >= latest_entry.version
        ]

        if conflicting_entries:
            # 使用业务逻辑解决冲突
            resolved_value = await self._resolve_by_business_logic(
                latest_entry.value,
                [entry.value for entry in conflicting_entries]
            )

            latest_entry.value = resolved_value

        return latest_entry

    async def _resolve_by_business_logic(self, primary_value: Any, conflict_values: List[Any]) -> Any:
        """基于业务逻辑解决冲突"""
        # 这里可以根据具体业务需求实现
        # 例如：数值相加、字符串拼接、取最大值等

        if isinstance(primary_value, (int, float)):
            # 数值类型：求和
            total = primary_value + sum(conflict_values)
            return total
        elif isinstance(primary_value, str):
            # 字符串类型：拼接
            return primary_value + "|" + "|".join(conflict_values)
        else:
            # 默认：返回最新值
            return primary_value

class CacheInvalidator:
    """缓存失效管理器"""

    def __init__(self, cache_manager: ConsistentCacheManager):
        self.cache_manager = cache_manager
        self.message_broker = MessageBroker()

    async def start(self):
        """启动失效监听"""
        await self.message_broker.subscribe(
            "cache_invalidation",
            self._handle_invalidation_message
        )

    async def _handle_invalidation_message(self, message: Dict[str, Any]):
        """处理失效消息"""
        key = message["key"]
        timestamp = message["timestamp"]

        # 检查消息时效性
        if time.time() - timestamp > 300:  # 5分钟前的消息忽略
            return

        await self.cache_manager.invalidate_key(key)

class CacheSynchronizer:
    """缓存同步器"""

    def __init__(self, cache_manager: ConsistentCacheManager):
        self.cache_manager = cache_manager

    async def start_periodic_sync(self):
        """启动定期同步"""
        while True:
            try:
                await self._perform_sync()
                await asyncio.sleep(600)  # 每10分钟同步一次
            except Exception as e:
                logger.error(f"Cache sync error: {e}")
                await asyncio.sleep(60)

    async def _perform_sync(self):
        """执行同步操作"""
        # 1. 检查数据一致性
        inconsistencies = await self._detect_inconsistencies()

        # 2. 修复不一致数据
        for inconsistency in inconsistencies:
            await self._repair_inconsistency(inconsistency)

    async def _detect_inconsistencies(self) -> List[Dict]:
        """检测数据不一致"""
        inconsistencies = []

        # 比较各节点的数据版本
        for key in self.cache_manager.cache_store.keys():
            versions = await self._get_versions_from_replicas(key)
            if len(set(versions)) > 1:
                inconsistencies.append({
                    "key": key,
                    "versions": versions
                })

        return inconsistencies

    async def _repair_inconsistency(self, inconsistency: Dict):
        """修复不一致数据"""
        key = inconsistency["key"]
        versions = inconsistency["versions"]

        # 选择最新版本
        latest_version = max(versions)
        latest_entry = await self._get_entry_with_version(key, latest_version)

        if latest_entry:
            # 同步到所有副本
            await self.cache_manager._sync_write_to_replicas(key, latest_entry)
```

## 实际应用题

### 5. 在百万级智能体社交平台中，你们是如何设计和优化分布式缓存的？

**回答思路**:
- 业务背景和挑战
- 缓存架构设计
- 关键技术选型
- 性能优化策略
- 实际效果和收益

**参考答案**:
```
在百万级智能体社交平台项目中，我们面临的缓存挑战和解决方案：

1. 业务挑战
   - 智能体行为数据量大：每秒数十万次行为记录
   - 实时性要求高：毫秒级响应
   - 数据关联复杂：智能体间关系网络
   - 热点数据明显：热门智能体访问集中

2. 缓存架构设计
   ```
   ┌─────────────────┐
   │   应用层        │
   │  • 本地缓存      │
   │  • 智能预取      │
   └─────────┬───────┘
             │
   ┌─────────▼───────┐
   │   分布式缓存     │
   │  • Redis Cluster│
   │  • 数据分片      │
   │  • 主从复制      │
   └─────────┬───────┘
             │
   ┌─────────▼───────┐
   │   数据存储层     │
   │  • MySQL       │
   │  • MongoDB     │
   │  • 时序数据库    │
   └─────────────────┘
   ```

3. 关键技术实现

   a) 智能体数据分片策略
   - 基于智能体ID的一致性哈希分片
   - 热点智能体数据多副本存储
   - 冷热数据分离存储

   b) 多级缓存架构
   - L1缓存：应用本地缓存（1KB-1MB）
   - L2缓存：分布式缓存（1MB-1GB）
   - L3缓存：数据库缓存（>1GB）

   c) 缓存预热策略
   - 智能体上线时预加载基础数据
   - 基于行为预测的主动缓存
   - 批量预热减少启动时间

4. 性能优化策略

   a) 数据结构优化
   - 使用Redis的高级数据结构
   - HyperLogLog用于统计
   - Bitmap用于状态标记
   - Sorted Set用于排行榜

   b) 内存优化
   - 对象复用池
   - 数据压缩
   - 内存碎片整理

   c) 网络优化
   - Pipeline批量操作
   - 连接池管理
   - 本地化部署减少延迟

5. 监控和运维
   - 缓存命中率监控
   - 内存使用率监控
   - 延迟和吞吐量监控
   - 自动扩缩容策略

6. 实际效果
   - 响应时间：从50ms降低到5ms
   - 吞吐量：提升10倍
   - 缓存命中率：95%
   - 数据库负载：降低80%
```

**代码示例**:
```python
class IntelligentAgentCacheSystem:
    """智能体专用缓存系统"""

    def __init__(self):
        self.local_cache = LRUCache(maxsize=10000)
        self.redis_cluster = RedisCluster(
            startup_nodes=[{"host": "redis-node1", "port": 6379}],
            decode_responses=True
        )
        self.behavior_predictor = AgentBehaviorPredictor()
        self.cache_warmer = CacheWarmer()
        self.monitoring = CacheMonitoring()

    async def get_agent_data(self, agent_id: str) -> Optional[dict]:
        """获取智能体数据"""
        try:
            # 1. 检查本地缓存
            cache_key = f"agent:{agent_id}:profile"
            local_data = self.local_cache.get(cache_key)
            if local_data:
                self.monitoring.record_hit("local")
                return local_data

            # 2. 检查分布式缓存
            redis_data = await self.redis_cluster.get(cache_key)
            if redis_data:
                data = json.loads(redis_data)
                # 回填本地缓存
                self.local_cache.set(cache_key, data)
                self.monitoring.record_hit("distributed")
                return data

            # 3. 缓存未命中，从数据库加载
            data = await self._load_from_database(agent_id)
            if data:
                await self._cache_agent_data(agent_id, data)
                self.monitoring.record_miss()
                return data

            return None

        except Exception as e:
            logger.error(f"Failed to get agent data for {agent_id}: {e}")
            return None

    async def update_agent_behavior(self, agent_id: str, behavior_data: dict):
        """更新智能体行为数据"""
        try:
            # 1. 更新数据库
            await self._update_database_behavior(agent_id, behavior_data)

            # 2. 更新缓存
            cache_key = f"agent:{agent_id}:behavior"
            await self.redis_cluster.setex(
                cache_key,
                3600,  # 1小时TTL
                json.dumps(behavior_data)
            )

            # 3. 预测下一步行为
            predicted_behavior = await self.behavior_predictor.predict_next_behavior(
                agent_id, behavior_data
            )

            # 4. 预缓存预测的行为数据
            if predicted_behavior:
                await self._pre_cache_predicted_data(agent_id, predicted_behavior)

            return True

        except Exception as e:
            logger.error(f"Failed to update behavior for {agent_id}: {e}")
            return False

    async def _pre_cache_predicted_data(self, agent_id: str, prediction: dict):
        """预缓存预测数据"""
        try:
            # 根据预测结果预加载相关数据
            if prediction["next_action"] == "social_interaction":
                # 预加载社交圈数据
                await self._pre_cache_social_circle(agent_id)
            elif prediction["next_action"] == "content_creation":
                # 预加载创作素材
                await self._pre_cache_creation_materials(agent_id)

        except Exception as e:
            logger.error(f"Failed to pre-cache data for {agent_id}: {e}")

    async def batch_get_agents_data(self, agent_ids: List[str]) -> Dict[str, dict]:
        """批量获取智能体数据"""
        try:
            # 使用Pipeline减少网络开销
            async with self.redis_cluster.pipeline() as pipe:
                for agent_id in agent_ids:
                    cache_key = f"agent:{agent_id}:profile"
                    pipe.get(cache_key)

                results = await pipe.execute()

            # 处理结果
            agent_data = {}
            missing_agents = []

            for i, agent_id in enumerate(agent_ids):
                if results[i]:
                    agent_data[agent_id] = json.loads(results[i])
                else:
                    missing_agents.append(agent_id)

            # 批量加载缺失数据
            if missing_agents:
                batch_data = await self._batch_load_from_database(missing_agents)
                for agent_id, data in batch_data.items():
                    agent_data[agent_id] = data
                    await self._cache_agent_data(agent_id, data)

            return agent_data

        except Exception as e:
            logger.error(f"Failed to batch get agents data: {e}")
            return {}

class AgentBehaviorPredictor:
    """智能体行为预测器"""

    def __init__(self):
        self.model_cache = {}
        self.behavior_history = {}

    async def predict_next_behavior(self, agent_id: str, current_behavior: dict) -> dict:
        """预测智能体下一步行为"""
        try:
            # 1. 加载行为历史
            history = await self._load_behavior_history(agent_id)

            # 2. 特征提取
            features = self._extract_features(history, current_behavior)

            # 3. 预测
            prediction = await self._predict_behavior(agent_id, features)

            return prediction

        except Exception as e:
            logger.error(f"Behavior prediction failed for {agent_id}: {e}")
            return {}

    def _extract_features(self, history: List[dict], current: dict) -> dict:
        """提取行为特征"""
        features = {
            "action_frequency": len([h for h in history if h["action"] == current["action"]]),
            "time_pattern": self._extract_time_pattern(history),
            "social_engagement": current.get("social_score", 0),
            "content_preference": current.get("content_category", "unknown")
        }
        return features

class CacheWarmer:
    """缓存预热器"""

    def __init__(self):
        self.warming_tasks = {}

    async def warm_agent_cache(self, agent_id: str):
        """预热智能体缓存"""
        if agent_id in self.warming_tasks:
            return

        self.warming_tasks[agent_id] = asyncio.create_task(
            self._warm_agent_cache_internal(agent_id)
        )

    async def _warm_agent_cache_internal(self, agent_id: str):
        """内部缓存预热逻辑"""
        try:
            # 1. 加载基础数据
            profile_data = await self._load_agent_profile(agent_id)
            if profile_data:
                await self._cache_agent_data(agent_id, profile_data)

            # 2. 加载行为数据
            behavior_data = await self._load_agent_behavior(agent_id)
            if behavior_data:
                cache_key = f"agent:{agent_id}:behavior"
                await self.redis_cluster.setex(cache_key, 3600, json.dumps(behavior_data))

            # 3. 加载社交圈数据
            social_data = await self._load_social_circle(agent_id)
            if social_data:
                cache_key = f"agent:{agent_id}:social_circle"
                await self.redis_cluster.setex(cache_key, 1800, json.dumps(social_data))

        finally:
            self.warming_tasks.pop(agent_id, None)
```

## 总结

分布式缓存系统的面试题主要考察：

1. **基础知识**: 缓存原理、分片策略、一致性模型
2. **架构设计**: 高性能、高可用的系统设计能力
3. **问题解决**: 一致性、故障处理、性能优化
4. **实践经验**: 结合具体业务场景的优化策略
5. **技术深度**: 底层原理、算法实现、性能调优

在面试中，应该结合实际项目经验，说明技术选型的思考过程，以及如何解决实际遇到的问题，展示对分布式系统原理的深入理解和工程实践能力。

---

**相关阅读**:
- [分布式缓存架构设计](../distributed-cache/architecture.md)
- [CAP理论](../knowledge-base/cap-theory.md)
- [分布式系统核心概念](../knowledge-base/core-concepts.md)