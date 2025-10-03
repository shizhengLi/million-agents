# 声誉引擎与信任系统技术指南

## 目录
1. [系统概述](#系统概述)
2. [架构设计](#架构设计)
3. [核心组件](#核心组件)
4. [技术实现](#技术实现)
5. [扩展策略](#扩展策略)
6. [技术难点](#技术难点)
7. [性能优化](#性能优化)
8. [面试题库](#面试题库)

## 系统概述

### 业务背景
在百万级智能体社交平台中，声誉引擎和信任系统是核心基础设施：
- **声誉引擎**: 基于历史交互行为计算智能体的可信度评分
- **信任系统**: 建立智能体间的信任关系网络，支持信任传播和计算

### 核心价值
1. **风险控制**: 识别恶意行为，降低平台风险
2. **推荐优化**: 基于可信度提升推荐质量
3. **社区治理**: 建立健康的社交生态
4. **商业决策**: 为业务决策提供数据支持

### 系统特性
- **实时性**: 支持实时声誉评分更新
- **准确性**: 多维度、多层次的评分算法
- **扩展性**: 支持百万级智能体的并发访问
- **可靠性**: 99.9%的系统可用性保证

## 架构设计

### 整体架构图

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   智能体A      │────│   声誉引擎      │────│   信任网络      │
│   Agent A       │    │ ReputationEngine│    │ TrustSystem     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │    ┌─────────────────┐ │                       │
         └────│  交互记录存储   │─┘                       │
              │ InteractionDB │                         │
              └─────────────────┘                         │
                       │                                 │
         ┌─────────────────┐    ┌─────────────────┐     │
         │   评分缓存      │────│   网络图存储    │─────┘
         │ ScoreCache      │    │ GraphStorage    │
         └─────────────────┘    └─────────────────┘
```

### 数据流设计

```python
# 交互数据流
Agent Interaction → Reputation Engine → Score Update → Trust Network
        ↓                    ↓                    ↓              ↓
   Event Capture    → Score Calculation → Cache Update → Relationship Update
```

### 分层架构

1. **接入层 (Access Layer)**
   - RESTful API接口
   - 事件驱动接入
   - 限流和熔断保护

2. **业务层 (Business Layer)**
   - 声誉计算引擎
   - 信任关系管理
   - 策略规则引擎

3. **数据层 (Data Layer)**
   - 关系型数据库 (MySQL)
   - 缓存系统 (Redis)
   - 图数据库 (Neo4j)

## 核心组件

### 1. 声誉引擎 (ReputationEngine)

#### 核心功能
```python
class ReputationEngine:
    def __init__(self):
        self.scores: Dict[str, ReputationScore] = {}
        self.interaction_weights = {
            'positive': 1.0,
            'negative': -2.0,
            'neutral': 0.1
        }

    def update_reputation(self, agent_id: str, interaction: Dict):
        """更新智能体声誉"""
        pass

    def calculate_score(self, agent_id: str) -> float:
        """计算声誉分数"""
        pass
```

#### 评分算法
- **基础评分**: 基于历史交互的加权平均
- **时间衰减**: 老旧交互的权重随时间递减
- **异常检测**: 识别异常行为模式
- **多维度评分**: 支持不同场景的差异化评分

### 2. 信任系统 (TrustSystem)

#### 核心功能
```python
class TrustSystem:
    def __init__(self):
        self.network = TrustNetwork()
        self.propagation_cache = {}

    def calculate_direct_trust(self, agent_a: str, agent_b: str) -> float:
        """计算直接信任度"""
        pass

    def calculate_propagated_trust(self, source: str, target: str) -> float:
        """计算传播信任度"""
        pass
```

#### 信任传播算法
- **直接信任**: 基于直接交互历史
- **间接信任**: 通过社交网络传播
- **路径衰减**: 信任度随传播路径长度递减
- **环路检测**: 避免信任传播环路

### 3. 数据模型

#### 声誉评分模型
```python
@dataclass
class ReputationScore:
    agent_id: str
    total_score: float
    interaction_count: int
    last_updated: datetime
    score_components: Dict[str, float]  # 分维度评分

    def calculate_decayed_score(self, decay_factor: float) -> float:
        """计算时间衰减后的分数"""
        time_delta = datetime.now() - self.last_updated
        decay_multiplier = math.exp(-decay_factor * time_delta.days)
        return self.total_score * decay_multiplier
```

#### 信任网络模型
```python
@dataclass
class TrustNode:
    agent_id: str
    trust_incoming: Dict[str, float]  # 入边信任值
    trust_outgoing: Dict[str, float]  # 出边信任值
    last_updated: datetime

class TrustNetwork:
    nodes: Dict[str, TrustNode]
    edges: Dict[Tuple[str, str], float]  # 信任边权重

    def calculate_network_metrics(self) -> Dict:
        """计算网络指标"""
        return {
            'density': self.calculate_density(),
            'clustering_coefficient': self.calculate_clustering(),
            'trust_distribution': self.get_trust_distribution()
        }
```

## 技术实现

### 1. 评分算法实现

#### 基础评分算法
```python
def calculate_base_score(self, interactions: List[Dict]) -> float:
    """
    计算基础声誉分数

    算法公式:
    Score = Σ(weight_i * interaction_i) / Σ(weight_i)
    """
    if not interactions:
        return 50.0  # 中性分数

    total_weighted_score = 0.0
    total_weight = 0.0

    for interaction in interactions:
        weight = self.get_interaction_weight(interaction)
        outcome = self.get_interaction_outcome(interaction)

        # 时间衰减因子
        time_decay = self.calculate_time_decay(interaction['timestamp'])
        adjusted_weight = weight * time_decay

        total_weighted_score += outcome * adjusted_weight
        total_weight += adjusted_weight

    return min(100.0, max(0.0, total_weighted_score / total_weight))
```

#### 异常检测算法
```python
def detect_anomaly(self, agent_id: str, recent_interactions: List[Dict]) -> float:
    """
    异常行为检测

    检测维度:
    1. 交互频率异常
    2. 行为模式突变
    3. 评分分布异常
    """
    anomaly_score = 0.0

    # 频率异常检测
    frequency_score = self.detect_frequency_anomaly(recent_interactions)
    anomaly_score += frequency_score * 0.4

    # 模式异常检测
    pattern_score = self.detect_pattern_anomaly(recent_interactions)
    anomaly_score += pattern_score * 0.3

    # 分布异常检测
    distribution_score = self.detect_distribution_anomaly(recent_interactions)
    anomaly_score += distribution_score * 0.3

    return min(100.0, anomaly_score)
```

### 2. 信任传播算法

#### 最短路径信任传播
```python
def calculate_propagated_trust(self, source: str, target: str) -> float:
    """
    基于最短路径的信任传播

    算法步骤:
    1. 使用Dijkstra算法找到最短信任路径
    2. 计算路径信任度 = Π(edge_trust_i * decay_factor_i)
    3. 返回最大路径信任度
    """
    if source == target:
        return 100.0

    # BFS寻找所有路径
    paths = self.find_all_trust_paths(source, target, max_depth=5)

    if not paths:
        return 0.0

    max_trust = 0.0
    for path in paths:
        path_trust = self.calculate_path_trust(path)
        max_trust = max(max_trust, path_trust)

    return max_trust

def calculate_path_trust(self, path: List[str]) -> float:
    """计算单条路径的信任度"""
    if len(path) < 2:
        return 0.0

    trust = 1.0
    decay_factor = 0.9  # 每跳衰减因子

    for i in range(len(path) - 1):
        edge_trust = self.get_edge_trust(path[i], path[i+1])
        trust *= edge_trust * (decay_factor ** i)

    return trust
```

#### 多路径信任聚合
```python
def aggregate_multi_path_trust(self, source: str, target: str) -> float:
    """
    多路径信任聚合

    聚合策略:
    1. 加权平均: 基于路径长度和信任度
    2. 最大值: 选择最可信的路径
    3. 概率融合: 基于贝叶斯推理
    """
    paths = self.find_all_trust_paths(source, target)

    if not paths:
        return 0.0

    # 计算每条路径的权重
    path_weights = []
    path_trusts = []

    for path in paths:
        path_length = len(path) - 1
        path_trust = self.calculate_path_trust(path)
        path_weight = 1.0 / (path_length + 1)  # 路径越短权重越高

        path_weights.append(path_weight)
        path_trusts.append(path_trust)

    # 加权平均聚合
    total_weight = sum(path_weights)
    weighted_trust = sum(w * t for w, t in zip(path_weights, path_trusts))

    return weighted_trust / total_weight if total_weight > 0 else 0.0
```

### 3. 并发控制实现

#### 读写锁机制
```python
import threading
from contextlib import contextmanager

class ConcurrentTrustSystem:
    def __init__(self):
        self._lock = threading.RLock()  # 可重入锁
        self._read_count = 0
        self._write_lock = threading.Lock()

    @contextmanager
    def read_lock(self):
        """读锁上下文管理器"""
        with self._write_lock:
            self._read_count += 1
        try:
            yield
        finally:
            with self._write_lock:
                self._read_count -= 1

    @contextmanager
    def write_lock(self):
        """写锁上下文管理器"""
        self._write_lock.acquire()
        while self._read_count > 0:
            # 等待所有读操作完成
            pass
        try:
            yield
        finally:
            self._write_lock.release()

    def update_trust(self, agent_a: str, agent_b: str, delta: float):
        """线程安全的信任更新"""
        with self.write_lock():
            current_trust = self.get_trust(agent_a, agent_b)
            new_trust = max(0.0, min(100.0, current_trust + delta))
            self.set_trust(agent_a, agent_b, new_trust)
```

## 扩展策略

### 1. 水平扩展

#### 分片策略 (Sharding)
```python
class ShardedReputationSystem:
    def __init__(self, shard_count: int):
        self.shard_count = shard_count
        self.shards = [ReputationEngine() for _ in range(shard_count)]

    def get_shard(self, agent_id: str) -> ReputationEngine:
        """根据智能体ID计算分片"""
        shard_index = hash(agent_id) % self.shard_count
        return self.shards[shard_index]

    def update_reputation(self, agent_id: str, interaction: Dict):
        """分片更新声誉"""
        shard = self.get_shard(agent_id)
        shard.update_reputation(agent_id, interaction)
```

#### 缓存分层策略
```python
class TieredCacheSystem:
    def __init__(self):
        # L1: 本地内存缓存 (最快)
        self.l1_cache = TTLCache(maxsize=1000, ttl=60)

        # L2: Redis分布式缓存
        self.l2_cache = RedisCache()

        # L3: 数据库 (最慢但最准确)
        self.l3_storage = DatabaseStorage()

    def get_score(self, agent_id: str) -> Optional[float]:
        """分层缓存获取"""
        # L1缓存
        score = self.l1_cache.get(agent_id)
        if score is not None:
            return score

        # L2缓存
        score = self.l2_cache.get(agent_id)
        if score is not None:
            self.l1_cache[agent_id] = score
            return score

        # L3存储
        score = self.l3_storage.get_score(agent_id)
        if score is not None:
            self.l2_cache.set(agent_id, score, ttl=300)
            self.l1_cache[agent_id] = score

        return score
```

### 2. 垂直扩展

#### 微服务拆分
```yaml
# 微服务架构
services:
  reputation-service:
    description: "声誉计算服务"
    responsibilities:
      - 声誉分数计算
      - 异常检测
      - 历史数据管理

  trust-service:
    description: "信任关系服务"
    responsibilities:
      - 信任网络构建
      - 信任传播计算
      - 关系查询

  analytics-service:
    description: "分析统计服务"
    responsibilities:
      - 趋势分析
      - 报表生成
      - 数据挖掘
```

#### 事件驱动架构
```python
class ReputationEventProcessor:
    def __init__(self):
        self.event_handlers = {
            'interaction_created': self.handle_interaction,
            'trust_updated': self.handle_trust_update,
            'anomaly_detected': self.handle_anomaly
        }

    async def process_event(self, event: Dict):
        """异步事件处理"""
        event_type = event['type']
        handler = self.event_handlers.get(event_type)

        if handler:
            await handler(event)

            # 发布下游事件
            await self.publish_downstream_event(event)

    async def handle_interaction(self, event: Dict):
        """处理交互事件"""
        agent_id = event['agent_id']
        interaction = event['interaction']

        # 更新声誉
        await self.reputation_service.update_score(agent_id, interaction)

        # 更新信任关系
        if 'target_agent' in interaction:
            await self.trust_service.update_trust(
                agent_id,
                interaction['target_agent'],
                interaction['outcome']
            )
```

### 3. 数据扩展

#### 冷热数据分离
```python
class DataTierManager:
    def __init__(self):
        # 热数据: 最近7天的数据
        self.hot_storage = RedisCluster()

        # 温数据: 最近30天的数据
        self.warm_storage = PostgreSQL()

        # 冷数据: 30天前的数据
        self.cold_storage = S3Storage()

    def get_interaction_history(self, agent_id: str, days: int = 30):
        """按时间范围获取交互历史"""
        if days <= 7:
            return self.hot_storage.get_interactions(agent_id)
        elif days <= 30:
            return self.warm_storage.get_interactions(agent_id, days)
        else:
            # 从冷数据恢复
            cold_data = self.cold_storage.get_interactions(agent_id, days)
            warm_data = self.warm_storage.get_interactions(agent_id, 30)
            return cold_data + warm_data

    def archive_old_data(self):
        """数据归档任务"""
        # 将30天前的数据从温存储迁移到冷存储
        cutoff_date = datetime.now() - timedelta(days=30)

        old_interactions = self.warm_storage.get_interactions_before(cutoff_date)
        if old_interactions:
            self.cold_storage.archive_interactions(old_interactions)
            self.warm_storage.delete_interactions_before(cutoff_date)
```

## 技术难点

### 1. 大规模信任网络计算

#### 难点描述
- 百万级节点和千万级边的图计算
- 实时信任传播计算的复杂度为O(n²)
- 内存占用过大，计算延迟高

#### 解决方案

**算法优化:**
```python
class OptimizedTrustPropagation:
    def __init__(self):
        self.max_propagation_depth = 3  # 限制传播深度
        self.min_trust_threshold = 10.0  # 最小信任阈值
        self.cache_enabled = True

    def calculate_trust_with_pruning(self, source: str, target: str) -> float:
        """带剪枝的信任传播"""
        # 1. 早期剪枝: 排除低信任边
        pruned_graph = self.prune_low_trust_edges()

        # 2. 路径剪枝: 限制搜索深度
        paths = self.find_limited_paths(pruned_graph, source, target)

        # 3. 结果缓存: 缓存常用路径
        cache_key = f"{source}->{target}"
        if cache_key in self.propagation_cache:
            return self.propagation_cache[cache_key]

        result = self.aggregate_path_trust(paths)

        if self.cache_enabled:
            self.propagation_cache[cache_key] = result

        return result

    def prune_low_trust_edges(self, threshold: float = 10.0) -> Dict:
        """剪枝低信任边"""
        pruned_edges = {}
        for (u, v), trust in self.network.edges.items():
            if trust >= threshold:
                pruned_edges[(u, v)] = trust
        return pruned_edges
```

**分布式计算:**
```python
class DistributedTrustCalculator:
    def __init__(self, worker_count: int = 10):
        self.worker_count = worker_count
        self.task_queue = Queue()
        self.result_queue = Queue()
        self.workers = []

    def calculate_batch_trust(self, agent_pairs: List[Tuple[str, str]]) -> Dict:
        """批量分布式计算信任度"""
        # 分发任务
        for pair in agent_pairs:
            self.task_queue.put(pair)

        # 启动工作进程
        for i in range(self.worker_count):
            worker = Process(target=self.worker_process)
            worker.start()
            self.workers.append(worker)

        # 收集结果
        results = {}
        for _ in range(len(agent_pairs)):
            result = self.result_queue.get()
            results[result['pair']] = result['trust']

        # 清理工作进程
        for worker in self.workers:
            worker.terminate()

        return results

    def worker_process(self):
        """工作进程"""
        while True:
            try:
                source, target = self.task_queue.get(timeout=1)
                trust = self.calculate_single_trust(source, target)
                self.result_queue.put({
                    'pair': (source, target),
                    'trust': trust
                })
            except:
                break
```

### 2. 实时性与一致性平衡

#### 难点描述
- 声誉分数需要实时更新，但计算复杂
- 分布式环境下数据一致性难以保证
- 高并发写入导致的性能瓶颈

#### 解决方案

**最终一致性设计:**
```python
class EventuallyConsistentReputation:
    def __init__(self):
        self.write_ahead_log = WAL()
        self.score_cache = RedisCache()
        self.batch_processor = BatchProcessor()

    def update_score_async(self, agent_id: str, interaction: Dict):
        """异步更新声誉分数"""
        # 1. 写入预写日志
        self.write_ahead_log.append({
            'agent_id': agent_id,
            'interaction': interaction,
            'timestamp': datetime.now()
        })

        # 2. 更新缓存 (最终一致)
        self.update_cache_immediately(agent_id, interaction)

        # 3. 异步批量处理
        self.batch_processor.schedule_update(agent_id, interaction)

    def update_cache_immediately(self, agent_id: str, interaction: Dict):
        """立即更新缓存"""
        current_score = self.score_cache.get(agent_id, 50.0)
        delta = self.calculate_interaction_delta(interaction)
        new_score = max(0.0, min(100.0, current_score + delta))

        self.score_cache.set(agent_id, new_score, ttl=300)

    async def batch_process_updates(self):
        """批量处理更新"""
        while True:
            batch = await self.batch_processor.get_batch(size=100)
            if batch:
                await self.process_batch(batch)
            await asyncio.sleep(1)  # 每秒处理一次
```

### 3. 异常行为检测

#### 难点描述
- 恶意用户行为模式复杂多变
- 误报率和漏报率的平衡
- 实时检测与性能的权衡

#### 解决方案

**机器学习检测:**
```python
class MLDetectionSystem:
    def __init__(self):
        self.isolation_forest = IsolationForest()
        self.lstm_detector = LSTMDetector()
        self.rule_engine = RuleEngine()

    def detect_anomaly(self, agent_id: str, recent_interactions: List[Dict]) -> Dict:
        """多层次异常检测"""
        anomaly_indicators = {}

        # 1. 统计异常检测
        statistical_score = self.detect_statistical_anomaly(recent_interactions)
        anomaly_indicators['statistical'] = statistical_score

        # 2. 时序异常检测
        time_series_score = self.detect_time_series_anomaly(recent_interactions)
        anomaly_indicators['time_series'] = time_series_score

        # 3. 规则引擎检测
        rule_violations = self.rule_engine.check_violations(recent_interactions)
        anomaly_indicators['rule_violations'] = rule_violations

        # 4. 综合评分
        final_score = self.aggregate_anomaly_scores(anomaly_indicators)

        return {
            'anomaly_score': final_score,
            'indicators': anomaly_indicators,
            'is_anomaly': final_score > 0.7
        }

    def detect_statistical_anomaly(self, interactions: List[Dict]) -> float:
        """统计异常检测"""
        features = self.extract_statistical_features(interactions)
        anomaly_score = self.isolation_forest.decision_function([features])[0]
        return max(0.0, min(1.0, -anomaly_score))  # 归一化到[0,1]
```

## 性能优化

### 1. 缓存策略

#### 多级缓存设计
```python
class MultiLevelCache:
    def __init__(self):
        # L1: 进程内缓存 (毫秒级)
        self.process_cache = LRUCache(maxsize=10000)

        # L2: Redis缓存 (10毫秒级)
        self.redis_cache = RedisCluster()

        # L3: 本地SSD缓存 (100毫秒级)
        self.ssd_cache = SSDCache()

        # 预热策略
        self.warmup_strategy = WarmupStrategy()

    def get_with_fallback(self, key: str) -> Optional[Any]:
        """多级缓存获取"""
        # L1缓存
        value = self.process_cache.get(key)
        if value is not None:
            return value

        # L2缓存
        value = self.redis_cache.get(key)
        if value is not None:
            self.process_cache[key] = value  # 回填L1
            return value

        # L3缓存
        value = self.ssd_cache.get(key)
        if value is not None:
            self.redis_cache.set(key, value, ttl=300)  # 回填L2
            self.process_cache[key] = value  # 回填L1
            return value

        return None

    async def warmup_hot_data(self):
        """预热热点数据"""
        hot_keys = await self.identify_hot_keys()

        # 并行预热
        tasks = [self.warmup_key(key) for key in hot_keys]
        await asyncio.gather(*tasks)
```

#### 智能缓存更新
```python
class SmartCacheUpdater:
    def __init__(self):
        self.update_frequency = {}
        self.access_patterns = {}

    def should_update_cache(self, key: str, data_age: timedelta) -> bool:
        """智能判断是否需要更新缓存"""
        # 基于访问频率
        access_freq = self.update_frequency.get(key, 0)

        # 基于数据变化模式
        change_pattern = self.access_patterns.get(key, 'stable')

        if change_pattern == 'frequent':
            # 频繁变化的数据，缓存时间短
            max_age = timedelta(minutes=5)
        elif change_pattern == 'stable':
            # 稳定的数据，缓存时间长
            max_age = timedelta(hours=1)
        else:
            # 默认缓存时间
            max_age = timedelta(minutes=30)

        return data_age > max_age

    def adaptive_cache_ttl(self, key: str) -> int:
        """自适应缓存TTL"""
        access_count = self.update_frequency.get(key, 0)

        if access_count > 1000:  # 高频访问
            return 3600  # 1小时
        elif access_count > 100:  # 中频访问
            return 1800  # 30分钟
        else:  # 低频访问
            return 300   # 5分钟
```

### 2. 数据库优化

#### 索引策略
```sql
-- 交互记录表优化
CREATE TABLE interactions (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    agent_id VARCHAR(64) NOT NULL,
    target_agent VARCHAR(64),
    interaction_type ENUM('positive', 'negative', 'neutral') NOT NULL,
    score DECIMAL(5,2) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- 复合索引优化查询
    INDEX idx_agent_time (agent_id, created_at),
    INDEX idx_target_time (target_agent, created_at),
    INDEX idx_type_time (interaction_type, created_at),

    -- 分区表优化
    PARTITION BY RANGE (YEAR(created_at)) (
        PARTITION p2023 VALUES LESS THAN (2024),
        PARTITION p2024 VALUES LESS THAN (2025),
        PARTITION p_future VALUES LESS THAN MAXVALUE
    )
);

-- 声誉分数表优化
CREATE TABLE reputation_scores (
    agent_id VARCHAR(64) PRIMARY KEY,
    total_score DECIMAL(6,2) NOT NULL,
    interaction_count INT DEFAULT 0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    -- 支持高并发读取
    INDEX idx_score_desc (total_score DESC),
    INDEX idx_updated_desc (last_updated DESC)
) ENGINE=InnoDB;
```

#### 批量操作优化
```python
class BatchOperationOptimizer:
    def __init__(self, batch_size: int = 1000):
        self.batch_size = batch_size
        self.pending_updates = {}
        self.update_timer = None

    def schedule_update(self, agent_id: str, delta: float):
        """调度更新操作"""
        if agent_id in self.pending_updates:
            self.pending_updates[agent_id] += delta
        else:
            self.pending_updates[agent_id] = delta

        # 达到批次大小或定时触发
        if len(self.pending_updates) >= self.batch_size:
            self.flush_updates()
        elif self.update_timer is None:
            self.update_timer = asyncio.create_task(self.delayed_flush())

    async def delayed_flush(self):
        """延迟刷新"""
        await asyncio.sleep(5)  # 5秒延迟
        self.flush_updates()

    def flush_updates(self):
        """批量刷新到数据库"""
        if not self.pending_updates:
            return

        # 构建批量更新SQL
        batch_updates = []
        for agent_id, delta in self.pending_updates.items():
            batch_updates.append(f"('{agent_id}', {delta})")

        sql = f"""
            INSERT INTO reputation_updates (agent_id, score_delta)
            VALUES {','.join(batch_updates)}
            ON DUPLICATE KEY UPDATE
            total_score = total_score + VALUES(score_delta),
            interaction_count = interaction_count + 1
        """

        # 执行批量更新
        database.execute_batch(sql)

        # 清空待更新队列
        self.pending_updates.clear()

        # 取消定时器
        if self.update_timer:
            self.update_timer.cancel()
            self.update_timer = None
```

### 3. 网络通信优化

#### 连接池管理
```python
class ConnectionPoolManager:
    def __init__(self):
        self.pools = {
            'database': self.create_db_pool(),
            'redis': self.create_redis_pool(),
            'neo4j': self.create_neo4j_pool()
        }

    def create_db_pool(self):
        """创建数据库连接池"""
        return aiomysql.create_pool(
            host='db-master',
            port=3306,
            user='reputation_user',
            password='secure_password',
            db='reputation_db',
            minsize=5,
            maxsize=20,
            connect_timeout=10,
            pool_recycle=3600
        )

    async def execute_query(self, query: str, params: tuple = None):
        """执行查询"""
        async with self.pools['database'].acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(query, params)
                return await cursor.fetchall()

    async def execute_transaction(self, operations: List[Dict]):
        """执行事务"""
        async with self.pools['database'].acquire() as conn:
            async with conn.begin() as trans:
                try:
                    for op in operations:
                        await conn.execute(op['sql'], op['params'])
                    await trans.commit()
                except Exception:
                    await trans.rollback()
                    raise
```

#### 数据压缩传输
```python
class CompressedDataTransfer:
    def __init__(self):
        self.compressor = zlib.compressobj(level=6)
        self.decompressor = zlib.decompressobj()

    def compress_trust_data(self, trust_data: Dict) -> bytes:
        """压缩信任数据"""
        json_data = json.dumps(trust_data, separators=(',', ':'))
        compressed = self.compressor.compress(json_data.encode())
        compressed += self.compressor.flush()
        return compressed

    def decompress_trust_data(self, compressed_data: bytes) -> Dict:
        """解压信任数据"""
        decompressed = self.decompressor.decompress(compressed_data)
        return json.loads(decompressed.decode())

    async def send_compressed_data(self, destination: str, data: Dict):
        """发送压缩数据"""
        compressed = self.compress_trust_data(data)

        # 使用消息队列发送
        await self.message_queue.send(
            destination=destination,
            data=compressed,
            content_type='application/json',
            compression='gzip'
        )
```

## 面试题库

### 基础题目

#### Q1: 什么是声誉系统？它和信任系统有什么区别？
**A:**
- **声誉系统**: 基于历史行为对单个智能体进行评分，反映其可信度
- **信任系统**: 描述智能体之间的信任关系，形成信任网络
- **区别**: 声誉是单个节点的属性，信任是节点间的关系

#### Q2: 解释信任传播的原理和算法
**A:**
信任传播是指信任关系在社交网络中的传递过程。常用算法：
- **最短路径传播**: 信任度沿最短路径传递，每跳乘以衰减因子
- **多路径聚合**: 综合多条路径的信任度，使用加权平均或最大值
- **贝叶斯推理**: 基于概率模型计算信任传播

### 进阶题目

#### Q3: 如何设计一个支持百万级用户的声誉系统？
**A:**
关键设计要点：
1. **分片策略**: 按用户ID哈希分片，支持水平扩展
2. **缓存设计**: 多级缓存，热点数据预加载
3. **异步处理**: 最终一致性，批量更新
4. **数据分层**: 冷热数据分离，历史数据归档
5. **监控告警**: 实时监控系统健康状态

#### Q4: 声誉系统中的时间衰减算法如何实现？
**A:**
时间衰减算法实现：
```python
def calculate_time_decay(self, timestamp: datetime, decay_factor: float = 0.1) -> float:
    """计算时间衰减因子"""
    time_delta = datetime.now() - timestamp
    days_passed = time_delta.days

    # 指数衰减
    decay = math.exp(-decay_factor * days_passed)
    return max(0.1, decay)  # 最小保留10%权重
```

#### Q5: 如何处理声誉系统中的恶意行为？
**A:**
恶意行为处理策略：
1. **异常检测**: 使用统计方法和机器学习识别异常模式
2. **惩罚机制**: 对恶意行为实施声誉惩罚
3. **举报系统**: 用户举报+自动审核机制
4. **行为限制**: 低声誉用户限制某些功能
5. **恢复机制**: 允许用户通过良好行为恢复声誉

### 架构题目

#### Q6: 设计一个高可用的声誉系统架构
**A:**
架构设计要点：
1. **多活部署**: 跨地域多活，避免单点故障
2. **数据同步**: 基于事件溯源的数据同步机制
3. **故障转移**: 自动故障检测和切换
4. **负载均衡**: 智能路由和流量分发
5. **监控体系**: 全链路监控和告警

#### Q7: 声誉系统如何保证数据一致性？
**A:**
一致性保证策略：
1. **强一致性**: 关键操作使用分布式事务
2. **最终一致性**: 非关键操作使用消息队列异步处理
3. **冲突解决**: 向量时钟或CRDT算法解决冲突
4. **数据校验**: 定期数据一致性检查和修复

### 算法题目

#### Q8: 实现一个信任传播算法
**A:**
```python
def trust_propagation(network: Dict, source: str, target: str, max_depth: int = 3) -> float:
    """信任传播算法实现"""
    from collections import deque

    queue = deque([(source, 1.0, 0)])  # (node, trust, depth)
    visited = set()
    max_trust = 0.0

    while queue:
        current, current_trust, depth = queue.popleft()

        if current == target:
            max_trust = max(max_trust, current_trust)
            continue

        if depth >= max_depth or current in visited:
            continue

        visited.add(current)

        for neighbor, edge_trust in network.get(current, {}).items():
            if neighbor not in visited:
                propagated_trust = current_trust * edge_trust * 0.8  # 衰减因子
                queue.append((neighbor, propagated_trust, depth + 1))

    return max_trust
```

#### Q9: 设计一个实时异常检测算法
**A:**
```python
class RealTimeAnomalyDetector:
    def __init__(self, window_size: int = 100, threshold: float = 2.0):
        self.window_size = window_size
        self.threshold = threshold
        self.data_window = deque(maxlen=window_size)

    def detect_anomaly(self, new_value: float) -> bool:
        """实时异常检测"""
        self.data_window.append(new_value)

        if len(self.data_window) < self.window_size:
            return False

        # 计算滑动窗口统计量
        mean = sum(self.data_window) / len(self.data_window)
        variance = sum((x - mean) ** 2 for x in self.data_window) / len(self.data_window)
        std_dev = math.sqrt(variance)

        # Z-score异常检测
        z_score = abs(new_value - mean) / std_dev if std_dev > 0 else 0

        return z_score > self.threshold
```

### 场景题目

#### Q10: 声誉系统突然出现大量低分，如何排查和解决？
**A:**
排查步骤：
1. **监控检查**: 查看系统监控指标，确定异常范围
2. **日志分析**: 检查错误日志和业务日志
3. **数据验证**: 验证计算逻辑和数据正确性
4. **回滚操作**: 必要时回滚到稳定版本
5. **根因分析**: 深入分析问题根因
6. **预防措施**: 制定预防类似问题的措施

#### Q11: 如何设计声誉系统的A/B测试框架？
**A:**
A/B测试框架设计：
1. **用户分桶**: 基于用户ID哈希分桶，确保一致性
2. **算法配置**: 支持多算法并行运行
3. **指标收集**: 收集各种业务和技术指标
4. **统计分析**: 使用统计方法验证显著性
5. **流量控制**: 渐进式流量切换
6. **监控告警**: 实时监控实验效果

### 性能题目

#### Q12: 声誉系统QPS突然下降，如何排查？
**A:**
排查思路：
1. **系统资源**: 检查CPU、内存、磁盘、网络使用率
2. **数据库性能**: 检查慢查询、连接池、锁等待
3. **缓存状态**: 检查缓存命中率和响应时间
4. **外部依赖**: 检查依赖服务的响应时间
5. **业务逻辑**: 检查是否有性能回归的代码变更
6. **流量模式**: 检查流量洪峰或异常流量

#### Q13: 如何优化声誉系统的计算性能？
**A:**
性能优化策略：
1. **算法优化**: 使用近似算法和剪枝策略
2. **缓存优化**: 多级缓存和智能预加载
3. **并发优化**: 异步处理和批量操作
4. **数据结构优化**: 选择合适的数据结构
5. **硬件优化**: SSD存储和更多内存
6. **架构优化**: 微服务拆分和负载均衡

---

## 总结

声誉引擎和信任系统是大规模社交平台的核心基础设施，需要综合考虑算法设计、系统架构、性能优化等多个方面。通过合理的技术选型和架构设计，可以构建出高可用、高性能、可扩展的声誉信任系统。