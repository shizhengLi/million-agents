# 分布式系统核心概念

## 概述

分布式系统是由多个通过网络连接的独立计算机组成的系统，这些计算机协同工作以完成共同的目标。理解分布式系统的核心概念是设计和构建可靠、可扩展系统的基础。

## 基础概念

### 1. 分布式 vs 集中式

#### 集中式系统
```
┌─────────────────┐
│   Single       │
│   Server       │
│                 │
│  ┌─────────┐   │
│  │Database │   │
│  └─────────┘   │
│                 │
│  ┌─────────┐   │
│  │App Logic│   │
│  └─────────┘   │
└─────────────────┘
```

**特点**:
- 单点故障风险
- 扩展受限
- 管理简单
- 数据一致性容易保证

#### 分布式系统
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Node 1       │    │   Node 2       │    │   Node 3       │
│                 │    │                 │    │                 │
│  ┌─────────┐   │    │  ┌─────────┐   │    │  ┌─────────┐   │
│  │Local DB │   │    │  │Local DB │   │    │  │Local DB │   │
│  └─────────┘   │    │  └─────────┘   │    │  └─────────┘   │
│                 │    │                 │    │                 │
│  ┌─────────┐   │    │  ┌─────────┐   │    │  ┌─────────┐   │
│  │Service  │   │    │  │Service  │   │    │  │Service  │   │
│  └─────────┘   │    │  └─────────┘   │    │  └─────────┘   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
           │                    │                    │
           └────────────────────┼────────────────────┘
                                │
                        ┌─────────────────┐
                        │ Communication  │
                        │     Network      │
                        └─────────────────┘
```

**特点**:
- 高可用性
- 可扩展性
- 容错能力
- 地理分布
- **挑战**: 网络延迟、数据一致性、故障处理

### 2. 节点与集群

#### 节点 (Node)
节点是分布式系统中的基本计算单元，可以是：
- 物理服务器
- 虚拟机
- 容器
- 进程

```python
class Node:
    """分布式系统节点"""
    def __init__(self, node_id, address, capabilities):
        self.id = node_id
        self.address = address
        self.capabilities = capabilities  # CPU, 内存, 存储, 网络
        self.status = NodeStatus.UNKNOWN
        self.last_heartbeat = time.time()

    async def start(self):
        """启动节点"""
        await self.initialize_services()
        self.status = NodeStatus.RUNNING
        await self.join_cluster()

    async def stop(self):
        """停止节点"""
        await self.leave_cluster()
        await self.shutdown_services()
        self.status = NodeStatus.STOPPED

    async def heartbeat(self):
        """发送心跳"""
        self.last_heartbeat = time.time()
        return await self.cluster_manager.receive_heartbeat(self.id)
```

#### 集群 (Cluster)
集群是一组协同工作的节点集合：

```python
class Cluster:
    """分布式集群"""
    def __init__(self, cluster_id):
        self.id = cluster_id
        self.nodes = {}  # node_id -> Node
        self.leader = None
        self.membership_manager = MembershipManager()
        self.consensus_algorithm = ConsensusAlgorithm()

    async def add_node(self, node):
        """添加节点到集群"""
        self.nodes[node.id] = node

        # 共识算法确保所有节点同意
        await self.consensus_algorithm.propose_membership_change(
            'add', node.id
        )

        # 初始化节点
        await node.join_cluster(self)

    async def remove_node(self, node_id):
        """从集群移除节点"""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            await node.leave_cluster()
            del self.nodes[node_id]

            await self.consensus_algorithm.propose_membership_change(
                'remove', node_id
            )

    def get_leader(self):
        """获取集群领导者"""
        return self.leader

    def is_leader(self, node_id):
        """检查是否是领导者"""
        return self.leader and self.leader.id == node_id
```

## 通信模型

### 1. 消息传递模型

#### 同步通信
```python
class SynchronousCommunication:
    """同步通信模型"""
    def send_request(self, target_node, request):
        """同步发送请求"""
        try:
            # 建立连接
            connection = self.establish_connection(target_node)

            # 发送请求
            connection.send(request)

            # 等待响应
            response = connection.receive()

            # 关闭连接
            connection.close()

            return response

        except TimeoutError:
            raise CommunicationTimeoutError()
        except ConnectionError:
            raise CommunicationFailedError()

    def establish_connection(self, target_node):
        """建立连接"""
        socket = Socket()
        socket.connect(target_node.address)
        return socket
```

#### 异步通信
```python
class AsynchronousCommunication:
    """异步通信模型"""
    def __init__(self):
        self.message_queue = asyncio.Queue()
        self.response_handlers = {}
        self.connection_pool = ConnectionPool()

    async def send_request(self, target_node, request):
        """异步发送请求"""
        # 获取连接
        connection = await self.connection_pool.get_connection(target_node)

        # 生成消息ID
        message_id = generate_message_id()

        # 创建future用于等待响应
        future = asyncio.get_event_loop().create_future()
        self.response_handlers[message_id] = future

        # 发送消息
        message = Message(message_id, request)
        await connection.send(message)

        # 等待响应
        try:
            response = await asyncio.wait_for(future, timeout=30)
            return response
        finally:
            # 清理处理器
            if message_id in self.response_handlers:
                del self.response_handlers[message_id]

    async def handle_response(self, message):
        """处理响应消息"""
        message_id = message.id
        if message_id in self.response_handlers:
            future = self.response_handlers[message_id]
            if not future.done():
                future.set_result(message.payload)
```

### 2. 通信模式

#### 点对点 (Point-to-Point)
```python
class PointToPointCommunication:
    """点对点通信"""
    def __init__(self):
        self.connections = {}  # node_id -> Connection

    async def send_to_node(self, target_node_id, message):
        """发送消息到特定节点"""
        if target_node_id not in self.connections:
            connection = await self.establish_connection(target_node_id)
            self.connections[target_node_id] = connection

        connection = self.connections[target_node_id]
        await connection.send(message)

    async def establish_connection(self, node_id):
        """建立到特定节点的连接"""
        node_address = self.get_node_address(node_id)
        connection = TCPConnection(node_address)
        await connection.connect()
        return connection
```

#### 发布订阅 (Publish-Subscribe)
```python
class PubSubCommunication:
    """发布订阅通信"""
    def __init__(self):
        self.topics = defaultdict(set)  # topic -> subscribers
        self.message_broker = MessageBroker()

    def subscribe(self, topic, subscriber):
        """订阅主题"""
        self.topics[topic].add(subscriber)
        self.message_broker.add_subscriber(topic, subscriber)

    def unsubscribe(self, topic, subscriber):
        """取消订阅"""
        if topic in self.topics:
            self.topics[topic].discard(subscriber)
            self.message_broker.remove_subscriber(topic, subscriber)

    async def publish(self, topic, message):
        """发布消息到主题"""
        if topic in self.topics:
            for subscriber in self.topics[topic]:
                try:
                    await subscriber.receive_message(topic, message)
                except Exception as e:
                    logger.error(f"Failed to deliver message to {subscriber}: {e}")
```

## 时间和顺序

### 1. 物理时钟与逻辑时钟

#### 物理时钟
```python
class PhysicalClock:
    """物理时钟"""
    def __init__(self):
        self.max_drift_rate = 0.001  # 最大漂移率

    def now(self):
        """获取当前时间"""
        return time.time()

    def synchronize(self, reference_time):
        """同步时钟"""
        current_time = self.now()
        drift = current_time - reference_time

        if abs(drift) > self.max_drift_rate:
            # 需要调整时钟
            self.adjust_clock(-drift)

    def adjust_clock(self, adjustment):
        """调整时钟"""
        # 在实际系统中，这会调用系统时钟调整API
        logger.info(f"Adjusting clock by {adjustment} seconds")
```

#### 逻辑时钟
```python
class LamportClock:
    """Lamport逻辑时钟"""
    def __init__(self, node_id):
        self.node_id = node_id
        self.counter = 0

    def tick(self):
        """时钟滴答"""
        self.counter += 1
        return self.counter

    def send_event(self):
        """发送事件"""
        return self.tick()

    def receive_event(self, received_counter):
        """接收事件"""
        self.counter = max(self.counter, received_counter + 1)
        return self.counter

    def get_timestamp(self):
        """获取时间戳"""
        return (self.counter, self.node_id)
```

### 2. 向量时钟

```python
class VectorClock:
    """向量时钟"""
    def __init__(self, node_id):
        self.node_id = node_id
        self.clock = {node_id: 0}  # node_id -> counter

    def tick(self):
        """本地事件"""
        self.clock[self.node_id] += 1

    def merge(self, other_clock):
        """合并另一个向量时钟"""
        for node_id, counter in other_clock.clock.items():
            self.clock[node_id] = max(
                self.clock.get(node_id, 0),
                counter
            )

    def happens_before(self, other_clock):
        """判断是否在另一个时钟之前发生"""
        this_before_other = True
        other_before_this = True

        for node_id in self.clock:
            if self.clock[node_id] > other_clock.clock.get(node_id, 0):
                this_before_other = False
                break

        for node_id in other_clock.clock:
            if other_clock.clock[node_id] > self.clock.get(node_id, 0):
                other_before_this = False
                break

        return this_before_other and not other_before_this

    def concurrent(self, other_clock):
        """判断是否并发"""
        return not self.happens_before(other_clock) and \
               not other_clock.happens_before(self)
```

## 一致性模型

### 1. 强一致性 (Strong Consistency)

```python
class StrongConsistency:
    """强一致性模型"""
    def __init__(self):
        self.lock = DistributedLock()

    async def read(self, key):
        """读取操作"""
        async with self.lock.acquire(key):
            value = await self.storage.get(key)
            return value

    async def write(self, key, value):
        """写入操作"""
        async with self.lock.acquire(key):
            await self.storage.set(key, value)
            # 等待所有副本同步
            await self.replicate_to_all_nodes(key, value)

    async def replicate_to_all_nodes(self, key, value):
        """复制到所有节点"""
        tasks = []
        for node in self.cluster.nodes:
            task = asyncio.create_task(node.set(key, value))
            tasks.append(task)

        await asyncio.gather(*tasks)
```

### 2. 最终一致性 (Eventual Consistency)

```python
class EventualConsistency:
    """最终一致性模型"""
    def __init__(self):
        self.vector_clocks = {}
        self.conflict_resolver = ConflictResolver()
        self.anti_entropy = AntiEntropyManager()

    async def read(self, key):
        """读取操作"""
        # 可以从任意节点读取
        node = self.select_read_node(key)
        value = await node.get(key)
        return value

    async def write(self, key, value):
        """写入操作"""
        # 更新本地向量时钟
        self.vector_clocks[key].tick()

        # 创建带时间戳的条目
        entry = TimestampedEntry(
            key=key,
            value=value,
            timestamp=self.vector_clocks[key].copy()
        )

        # 异步复制到其他节点
        asyncio.create_task(
            self.eventual_replication(entry)
        )

    async def eventual_replication(self, entry):
        """最终一致性复制"""
        for node in self.cluster.nodes:
            if node != self.local_node:
                try:
                    await node.replicate_entry(entry)
                except Exception as e:
                    logger.error(f"Replication failed: {e}")

    async def resolve_conflict(self, key, conflicting_entries):
        """解决冲突"""
        if len(conflicting_entries) <= 1:
            return conflicting_entries[0].value if conflicting_entries else None

        # 按向量时钟排序
        sorted_entries = sorted(
            conflicting_entries,
            key=lambda e: e.timestamp,
            reverse=True
        )

        latest_entry = sorted_entries[0]
        conflicts = []

        for entry in sorted_entries[1:]:
            if not latest_entry.timestamp.happens_before(entry.timestamp):
                conflicts.append(entry)

        if conflicts:
            # 有并发修改，使用冲突解决策略
            resolved = self.conflict_resolver.resolve(
                [latest_entry] + conflicts
            )
            return resolved.value

        return latest_entry.value
```

### 3. 因果一致性 (Causal Consistency)

```python
class CausalConsistency:
    """因果一致性模型"""
    def __init__(self):
        self.dependency_graph = DependencyGraph()
        self.causal_clock = CausalClock()

    async def write(self, key, value, dependencies=None):
        """因果一致性写入"""
        if dependencies is None:
            dependencies = []

        # 创建因果上下文
        causal_context = CausalContext(
            operation_id=generate_operation_id(),
            dependencies=dependencies,
            clock=self.causal_clock.get_timestamp()
        )

        # 记录依赖关系
        self.dependency_graph.add_operation(
            causal_context.operation_id,
            dependencies
        )

        # 执行写入
        await self.storage.set(key, value, causal_context)

        return causal_context

    async def read(self, key):
        """因果一致性读取"""
        entry = await self.storage.get(key)
        if entry:
            # 确保所有依赖操作都已完成
            await self.wait_for_dependencies(entry.causal_context)
            return entry.value
        return None

    async def wait_for_dependencies(self, causal_context):
        """等待依赖操作完成"""
        for dep_id in causal_context.dependencies:
            if not self.dependency_graph.is_completed(dep_id):
                await self.wait_for_operation_completion(dep_id)
```

## 故障模型

### 1. 故障类型

#### 崩溃故障 (Crash Failure)
```python
class CrashFailureDetector:
    """崩溃故障检测"""
    def __init__(self, timeout=30):
        self.timeout = timeout
        self.last_heartbeats = {}
        self.suspected_failures = {}

    def heartbeat(self, node_id, timestamp):
        """处理心跳"""
        self.last_heartbeats[node_id] = timestamp
        if node_id in self.suspected_failures:
            del self.suspected_failures[node_id]

    def is_failed(self, node_id):
        """检查节点是否故障"""
        if node_id not in self.last_heartbeats:
            return False

        time_since_last_heartbeat = (
            time.time() - self.last_heartbeats[node_id]
        )

        return time_since_last_heartbeat > self.timeout

    def get_failed_nodes(self):
        """获取故障节点列表"""
        return [
            node_id for node_id in self.last_heartbeats
            if self.is_failed(node_id)
        ]
```

#### 网络分区 (Network Partition)
```python
class NetworkPartitionHandler:
    """网络分区处理器"""
    def __init__(self):
        self.partition_detection = PartitionDetection()
        self.quorum_manager = QuorumManager()

    async def handle_partition(self, partition_info):
        """处理网络分区"""
        logger.warning(f"Network partition detected: {partition_info}")

        # 检查是否能形成法定人数
        if self.quorum_manager.has_quorum(partition_info.local_nodes):
            # 可以继续提供服务
            await self.continue_service_with_quorum(partition_info)
        else:
            # 需要降级服务
            await self.degrade_service(partition_info)

    async def continue_service_with_quorum(self, partition_info):
        """在法定人数下继续服务"""
        # 只处理法定人数内的节点
        for node in partition_info.local_nodes:
            await node.enable_writes()

        # 禁用分区外的节点
        for node in partition_info.remote_nodes:
            await node.disable_writes()

    async def degrade_service(self, partition_info):
        """降级服务"""
        # 只提供只读服务
        for node in partition_info.local_nodes:
            await node.enable_reads_only()

        logger.warning("Service degraded to read-only mode due to partition")
```

### 2. 容错机制

#### 超时重试 (Timeout and Retry)
```python
class RetryMechanism:
    """重试机制"""
    def __init__(self, max_retries=3, backoff_factor=2):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.jitter = Jitter()

    async def execute_with_retry(self, operation, *args, **kwargs):
        """带重试的执行"""
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                result = await operation(*args, **kwargs)
                return result

            except (TimeoutError, ConnectionError) as e:
                last_exception = e
                if attempt < self.max_retries:
                    # 计算退避时间
                    delay = self.calculate_backoff_delay(attempt)
                    await asyncio.sleep(delay)
                    continue
                raise

        raise OperationFailedError(
            f"Operation failed after {self.max_retries} retries",
            last_exception
        )

    def calculate_backoff_delay(self, attempt):
        """计算退避延迟"""
        base_delay = 1.0  # 基础延迟1秒
        exponential_delay = base_delay * (self.backoff_factor ** attempt)
        jittered_delay = self.jitter.add_jitter(exponential_delay)
        return min(jittered_delay, 60.0)  # 最大60秒
```

#### 熔断器 (Circuit Breaker)
```python
class CircuitBreaker:
    """熔断器模式"""
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN

    async def call(self, operation, *args, **kwargs):
        """熔断器调用"""
        if self.state == 'OPEN':
            if time.time() - self.last_failure_time < self.timeout:
                raise CircuitBreakerOpenError()
            else:
                self.state = 'HALF_OPEN'

        try:
            result = await operation(*args, **kwargs)

            if self.state == 'HALF_OPEN':
                self.state = 'CLOSED'
                self.failure_count = 0

            return result

        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = 'OPEN'

            raise
```

## 总结

分布式系统的核心概念包括：

1. **基础架构**: 节点、集群、通信模型
2. **时间与顺序**: 物理时钟、逻辑时钟、向量时钟
3. **一致性模型**: 强一致性、最终一致性、因果一致性
4. **故障处理**: 故障检测、容错机制、恢复策略

理解这些概念对于设计和实现可靠的分布式系统至关重要。在实际应用中，需要根据具体场景选择合适的模型和策略，在性能、一致性、可用性之间找到平衡。

---

**相关阅读**:
- [CAP理论与BASE理论](./cap-base-theory.md)
- [分布式一致性算法](./consistency-algorithms.md)
- [分布式事务处理](./distributed-transactions.md)