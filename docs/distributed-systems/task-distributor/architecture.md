# 任务分发器架构设计

## 概述

任务分发器是百万级智能体社交平台的核心组件，负责将大规模任务智能地分发到不同的工作节点，实现高效的任务并行处理和资源利用。本文档详细介绍了任务分发器的架构设计、实现原理以及在大规模环境下的优化策略。

## 核心架构

### 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                    Task Distributor System                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐│
│  │   Task          │    │   Task          │    │   Task          ││
│  │   Generator     │    │   Scheduler     │    │   Dispatcher    ││
│  │                 │    │                 │    │                 ││
│  │ • Batch Tasks   │    │ • Priority Queue│    │ • Load Balance  ││
│  │ • Real-time     │    │ • Resource Alloc│    │ • Failover      ││
│  │ • Scheduled     │    │ • Dependency    │    │ • Retry         ││
│  └─────────────────┘    └─────────────────┘    └─────────────────┘│
│           │                       │                       │       │
│           └───────────────────────┼───────────────────────┘       │
│                                   │                               │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐│
│  │   Task          │    │   Resource      │    │   Monitoring    ││
│  │   Queue         │    │   Manager       │    │   System        ││
│  │                 │    │                 │    │                 ││
│  │ • Priority      │    │ • Node Registry │    │ • Performance   ││
│  │ • FIFO          │    │ • Load Monitor  │    │ • Health Check  ││
│  │ • Delayed       │    │ • Capacity Plan │    │ • Alert System  ││
│  └─────────────────┘    └─────────────────┘    └─────────────────┘│
│                                   │                               │
│                                   ▼                               │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                    Worker Nodes Pool                          ││
│  │                                                                 ││
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            ││
│  │  │   Worker 1  │  │   Worker 2  │  │   Worker N  │            ││
│  │  │             │  │             │  │             │            ││
│  │  │ • Task Exec │  │ • Task Exec  │  │ • Task Exec  │            ││
│  │  │ • Resource  │  │ • Resource   │  │ • Resource   │            ││
│  │  │ • Health    │  │ • Health     │  │ • Health     │            ││
│  │  └─────────────┘  └─────────────┘  └─────────────┘            ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## 核心组件实现

### 1. 任务调度器 (Task Scheduler)

```python
import asyncio
import heapq
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

class TaskPriority(Enum):
    LOW = 3
    NORMAL = 2
    HIGH = 1
    CRITICAL = 0

class TaskStatus(Enum):
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class Task:
    id: str
    task_type: str
    payload: Dict[str, Any]
    priority: TaskPriority = TaskPriority.NORMAL
    delay_until: float = field(default_factory=time.time)
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other):
        """用于优先队列排序"""
        if self.delay_until != other.delay_until:
            return self.delay_until < other.delay_until
        return self.priority.value < other.priority.value

class TaskScheduler:
    """高级任务调度器"""

    def __init__(self):
        self.priority_queue = []
        self.delayed_queue = []
        self.running_tasks = {}
        self.completed_tasks = {}
        self.task_dependencies = {}
        self.lock = asyncio.Lock()
        self.scheduler_running = False

    async def start(self):
        """启动调度器"""
        self.scheduler_running = True
        asyncio.create_task(self._schedule_loop())
        asyncio.create_task(self._delayed_task_checker())

    async def stop(self):
        """停止调度器"""
        self.scheduler_running = False

    async def submit_task(self, task: Task) -> bool:
        """提交任务"""
        async with self.lock:
            # 检查依赖
            if not await self._check_dependencies(task):
                return False

            # 添加到相应队列
            if task.delay_until <= time.time():
                heapq.heappush(self.priority_queue, task)
            else:
                heapq.heappush(self.delayed_queue, task)

            # 记录依赖关系
            if task.dependencies:
                self.task_dependencies[task.id] = task.dependencies

            return True

    async def _check_dependencies(self, task: Task) -> bool:
        """检查任务依赖是否满足"""
        if not task.dependencies:
            return True

        for dep_id in task.dependencies:
            if dep_id not in self.completed_tasks:
                return False
            if self.completed_tasks[dep_id].status != TaskStatus.COMPLETED:
                return False

        return True

    async def _schedule_loop(self):
        """主调度循环"""
        while self.scheduler_running:
            try:
                # 获取可执行任务
                task = await self._get_next_task()
                if task:
                    await self._execute_task(task)
                else:
                    await asyncio.sleep(0.1)  # 没有任务时短暂休息

            except Exception as e:
                print(f"Scheduler error: {e}")
                await asyncio.sleep(1)

    async def _delayed_task_checker(self):
        """延迟任务检查器"""
        while self.scheduler_running:
            try:
                current_time = time.time()
                moved_tasks = []

                # 移动到期的延迟任务
                while self.delayed_queue and self.delayed_queue[0].delay_until <= current_time:
                    task = heapq.heappop(self.delayed_queue)
                    heapq.heappush(self.priority_queue, task)
                    moved_tasks.append(task.id)

                if moved_tasks:
                    print(f"Moved {len(moved_tasks)} delayed tasks to priority queue")

                await asyncio.sleep(1)  # 每秒检查一次

            except Exception as e:
                print(f"Delayed task checker error: {e}")
                await asyncio.sleep(5)

    async def _get_next_task(self) -> Optional[Task]:
        """获取下一个可执行任务"""
        async with self.lock:
            while self.priority_queue:
                task = heapq.heappop(self.priority_queue)

                # 再次检查依赖（可能在上次检查后有变化）
                if await self._check_dependencies(task):
                    return task
                else:
                    # 依赖未满足，重新加入队列
                    heapq.heappush(self.priority_queue, task)

            return None

    async def _execute_task(self, task: Task):
        """执行任务"""
        task.status = TaskStatus.RUNNING
        self.running_tasks[task.id] = task

        try:
            # 分发任务到工作节点
            dispatcher = TaskDispatcher()
            success = await dispatcher.dispatch_task(task)

            if success:
                print(f"Task {task.id} dispatched successfully")
            else:
                await self._handle_task_failure(task, "Dispatch failed")

        except Exception as e:
            await self._handle_task_failure(task, str(e))

    async def _handle_task_failure(self, task: Task, error: str):
        """处理任务失败"""
        task.status = TaskStatus.FAILED
        task.retry_count += 1

        # 检查是否需要重试
        if task.retry_count < task.max_retries:
            # 指数退避重试
            delay = min(2 ** task.retry_count, 300)  # 最大5分钟
            task.delay_until = time.time() + delay

            # 重新加入队列
            heapq.heappush(self.delayed_queue, task)
            print(f"Task {task.id} scheduled for retry {task.retry_count}/{task.max_retries}")
        else:
            print(f"Task {task.id} failed permanently: {error}")

        # 从运行任务中移除
        if task.id in self.running_tasks:
            del self.running_tasks[task.id]

    async def complete_task(self, task_id: str, result: Any = None):
        """标记任务完成"""
        async with self.lock:
            if task_id in self.running_tasks:
                task = self.running_tasks[task_id]
                task.status = TaskStatus.COMPLETED
                self.completed_tasks[task_id] = task
                del self.running_tasks[task_id]

                # 检查是否有依赖此任务的其他任务
                await self._check_dependent_tasks(task_id)

    async def _check_dependent_tasks(self, completed_task_id: str):
        """检查并激活依赖此任务的其他任务"""
        for task_id, dependencies in self.task_dependencies.items():
            if completed_task_id in dependencies:
                # 从依赖列表中移除已完成任务
                dependencies.remove(completed_task_id)

                # 如果所有依赖都完成了，激活任务
                if not dependencies:
                    del self.task_dependencies[task_id]
                    # 这里可以触发重新调度逻辑

    def get_queue_status(self) -> Dict[str, int]:
        """获取队列状态"""
        return {
            "priority_queue": len(self.priority_queue),
            "delayed_queue": len(self.delayed_queue),
            "running_tasks": len(self.running_tasks),
            "completed_tasks": len(self.completed_tasks)
        }
```

### 2. 资源管理器 (Resource Manager)

```python
class ResourceMonitor:
    """资源监控器"""

    def __init__(self):
        self.node_resources = {}
        self.resource_history = {}
        self.alert_thresholds = {
            "cpu_usage": 0.8,
            "memory_usage": 0.85,
            "disk_usage": 0.9,
            "network_usage": 0.8
        }

    async def register_node(self, node_id: str, capabilities: Dict[str, Any]):
        """注册节点资源"""
        self.node_resources[node_id] = {
            "capabilities": capabilities,
            "current_load": {
                "cpu": 0.0,
                "memory": 0.0,
                "disk": 0.0,
                "network": 0.0,
                "active_tasks": 0
            },
            "last_update": time.time(),
            "status": "healthy"
        }

    async def update_node_load(self, node_id: str, load_info: Dict[str, float]):
        """更新节点负载信息"""
        if node_id in self.node_resources:
            self.node_resources[node_id]["current_load"].update(load_info)
            self.node_resources[node_id]["last_update"] = time.time()

            # 检查告警阈值
            await self._check_alerts(node_id)

    async def _check_alerts(self, node_id: str):
        """检查资源告警"""
        if node_id not in self.node_resources:
            return

        node_info = self.node_resources[node_id]
        current_load = node_info["current_load"]

        for resource, threshold in self.alert_thresholds.items():
            current_value = current_load.get(resource, 0)
            if current_value > threshold:
                await self._send_alert(node_id, resource, current_value, threshold)

    async def _send_alert(self, node_id: str, resource: str, current: float, threshold: float):
        """发送资源告警"""
        print(f"ALERT: Node {node_id} {resource} usage {current:.2%} exceeds threshold {threshold:.2%}")

    async def get_best_node(self, task_requirements: Dict[str, Any]) -> Optional[str]:
        """根据任务需求选择最佳节点"""
        available_nodes = []

        for node_id, node_info in self.node_resources.items():
            if node_info["status"] != "healthy":
                continue

            # 检查节点是否满足任务需求
            if await self._can_handle_task(node_id, task_requirements):
                score = self._calculate_node_score(node_id, task_requirements)
                available_nodes.append((score, node_id))

        if not available_nodes:
            return None

        # 选择得分最高的节点
        available_nodes.sort(reverse=True)
        return available_nodes[0][1]

    async def _can_handle_task(self, node_id: str, task_requirements: Dict[str, Any]) -> bool:
        """检查节点是否能处理任务"""
        node_info = self.node_resources[node_id]
        capabilities = node_info["capabilities"]
        current_load = node_info["current_load"]

        # 检查CPU需求
        required_cpu = task_requirements.get("cpu", 0.1)
        if current_load["cpu"] + required_cpu > capabilities.get("cpu", 1.0):
            return False

        # 检查内存需求
        required_memory = task_requirements.get("memory", 0.1)
        if current_load["memory"] + required_memory > capabilities.get("memory", 1.0):
            return False

        return True

    def _calculate_node_score(self, node_id: str, task_requirements: Dict[str, Any]) -> float:
        """计算节点得分"""
        node_info = self.node_resources[node_id]
        current_load = node_info["current_load"]

        # 综合考虑负载、延迟等因素
        cpu_score = 1.0 - current_load["cpu"]
        memory_score = 1.0 - current_load["memory"]
        task_score = 1.0 / (1.0 + current_load["active_tasks"])

        # 加权平均
        total_score = (cpu_score * 0.4 + memory_score * 0.4 + task_score * 0.2)

        return total_score
```

### 3. 任务分发器 (Task Dispatcher)

```python
class TaskDispatcher:
    """任务分发器"""

    def __init__(self):
        self.worker_connections = {}
        self.dispatch_history = {}
        self.circuit_breaker = CircuitBreaker()
        self.load_balancer = LoadBalancer()

    async def dispatch_task(self, task: Task) -> bool:
        """分发任务到工作节点"""
        try:
            # 选择工作节点
            worker_node = await self._select_worker_node(task)
            if not worker_node:
                print(f"No available worker for task {task.id}")
                return False

            # 建立连接
            connection = await self._get_worker_connection(worker_node)
            if not connection:
                return False

            # 发送任务
            success = await self._send_task_to_worker(connection, task)
            if success:
                self.dispatch_history[task.id] = {
                    "worker": worker_node,
                    "dispatch_time": time.time(),
                    "status": "dispatched"
                }

            return success

        except Exception as e:
            print(f"Failed to dispatch task {task.id}: {e}")
            return False

    async def _select_worker_node(self, task: Task) -> Optional[str]:
        """选择工作节点"""
        # 获取任务资源需求
        task_requirements = self._extract_task_requirements(task)

        # 使用资源管理器选择最佳节点
        resource_manager = ResourceMonitor()
        best_node = await resource_manager.get_best_node(task_requirements)

        return best_node

    def _extract_task_requirements(self, task: Task) -> Dict[str, Any]:
        """提取任务资源需求"""
        requirements = {
            "cpu": 0.1,  # 默认CPU需求
            "memory": 0.1,  # 默认内存需求
            "disk": 0.01,  # 默认磁盘需求
            "network": 0.05  # 默认网络需求
        }

        # 根据任务类型调整需求
        if task.task_type == "cpu_intensive":
            requirements["cpu"] = 0.8
        elif task.task_type == "memory_intensive":
            requirements["memory"] = 0.7
        elif task.task_type == "io_intensive":
            requirements["disk"] = 0.5
            requirements["network"] = 0.3

        # 从任务元数据中获取自定义需求
        if "resource_requirements" in task.metadata:
            custom_req = task.metadata["resource_requirements"]
            requirements.update(custom_req)

        return requirements

    async def _get_worker_connection(self, worker_node: str):
        """获取工作节点连接"""
        if worker_node not in self.worker_connections:
            # 建立新连接
            connection = await self._establish_connection(worker_node)
            if connection:
                self.worker_connections[worker_node] = connection

        return self.worker_connections.get(worker_node)

    async def _establish_connection(self, worker_node: str):
        """建立到工作节点的连接"""
        try:
            # 这里实现具体的连接逻辑
            # 可能是HTTP、gRPC、TCP等连接
            print(f"Establishing connection to worker {worker_node}")
            return MockWorkerConnection(worker_node)

        except Exception as e:
            print(f"Failed to connect to worker {worker_node}: {e}")
            return None

    async def _send_task_to_worker(self, connection, task: Task) -> bool:
        """发送任务到工作节点"""
        try:
            # 使用熔断器保护
            return await self.circuit_breaker.call(
                connection.send_task, task
            )

        except Exception as e:
            print(f"Failed to send task to worker: {e}")
            return False

class CircuitBreaker:
    """熔断器实现"""

    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    async def call(self, func, *args, **kwargs):
        """熔断器调用"""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time < self.timeout:
                raise Exception("Circuit breaker is OPEN")
            else:
                self.state = "HALF_OPEN"

        try:
            result = await func(*args, **kwargs)

            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0

            return result

        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"

            raise

class MockWorkerConnection:
    """模拟工作节点连接"""

    def __init__(self, worker_id: str):
        self.worker_id = worker_id

    async def send_task(self, task: Task) -> bool:
        """发送任务"""
        print(f"Sending task {task.id} to worker {self.worker_id}")
        # 模拟任务处理时间
        await asyncio.sleep(0.1)
        return True
```

## 大规模优化策略

### 1. 分层调度

```python
class HierarchicalScheduler:
    """分层调度器"""

    def __init__(self):
        self.local_schedulers = {}  # 本地调度器
        self.global_scheduler = TaskScheduler()  # 全局调度器
        self.node_affinity = {}  # 节点亲和性

    async def submit_task(self, task: Task, preferred_region: str = None):
        """提交任务到分层调度器"""
        if preferred_region and preferred_region in self.local_schedulers:
            # 优先使用本地调度器
            local_scheduler = self.local_schedulers[preferred_region]
            success = await local_scheduler.submit_task(task)

            if success:
                return True

        # 本地调度失败，使用全局调度器
        return await self.global_scheduler.submit_task(task)

    async def add_region(self, region: str, scheduler: TaskScheduler):
        """添加区域调度器"""
        self.local_schedulers[region] = scheduler
```

### 2. 批量处理优化

```python
class BatchTaskProcessor:
    """批量任务处理器"""

    def __init__(self, batch_size=100, flush_interval=5):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.pending_tasks = []
        self.last_flush = time.time()

    async def add_task(self, task: Task):
        """添加任务到批次"""
        self.pending_tasks.append(task)

        # 检查是否需要刷新
        if len(self.pending_tasks) >= self.batch_size or \
           time.time() - self.last_flush >= self.flush_interval:
            await self.flush_batch()

    async def flush_batch(self):
        """刷新任务批次"""
        if not self.pending_tasks:
            return

        batch = self.pending_tasks.copy()
        self.pending_tasks.clear()
        self.last_flush = time.time()

        # 批量分发任务
        await self._dispatch_batch(batch)

    async def _dispatch_batch(self, tasks: List[Task]):
        """批量分发任务"""
        print(f"Dispatching batch of {len(tasks)} tasks")

        # 按任务类型分组
        task_groups = {}
        for task in tasks:
            task_type = task.task_type
            if task_type not in task_groups:
                task_groups[task_type] = []
            task_groups[task_type].append(task)

        # 并行分发各组任务
        dispatch_tasks = []
        for task_type, group_tasks in task_groups.items():
            dispatch_tasks.append(
                self._dispatch_task_group(task_type, group_tasks)
            )

        await asyncio.gather(*dispatch_tasks)

    async def _dispatch_task_group(self, task_type: str, tasks: List[Task]):
        """分发同类型任务组"""
        # 为同类型任务选择最优的工作节点
        for task in tasks:
            dispatcher = TaskDispatcher()
            await dispatcher.dispatch_task(task)
```

## 监控和可观测性

### 1. 性能监控

```python
class TaskDistributorMetrics:
    """任务分发器指标"""

    def __init__(self):
        self.metrics = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "avg_task_duration": 0,
            "throughput": 0,
            "error_rate": 0
        }
        self.task_durations = []

    def record_task_submitted(self):
        """记录任务提交"""
        self.metrics["tasks_submitted"] += 1

    def record_task_completed(self, duration: float):
        """记录任务完成"""
        self.metrics["tasks_completed"] += 1
        self.task_durations.append(duration)

        # 保持最近1000个任务的持续时间
        if len(self.task_durations) > 1000:
            self.task_durations.pop(0)

        self._update_avg_duration()

    def record_task_failed(self):
        """记录任务失败"""
        self.metrics["tasks_failed"] += 1

    def _update_avg_duration(self):
        """更新平均任务持续时间"""
        if self.task_durations:
            self.metrics["avg_task_duration"] = sum(self.task_durations) / len(self.task_durations)

    def calculate_throughput(self, time_window: float = 60) -> float:
        """计算吞吐量（任务/秒）"""
        recent_tasks = len([d for d in self.task_durations if d <= time_window])
        return recent_tasks / time_window

    def calculate_error_rate(self) -> float:
        """计算错误率"""
        total = self.metrics["tasks_completed"] + self.metrics["tasks_failed"]
        if total == 0:
            return 0
        return self.metrics["tasks_failed"] / total
```

## 总结

任务分发器作为分布式系统的核心组件，需要考虑：

1. **智能调度**: 基于优先级、依赖关系和资源需求的智能任务调度
2. **资源管理**: 实时监控节点负载，动态选择最优执行节点
3. **容错机制**: 熔断器、重试机制和故障转移
4. **可扩展性**: 分层调度和批量处理支持大规模部署
5. **监控可观测**: 完善的指标监控和性能分析

这种架构设计能够有效支撑百万级智能体社交平台的任务处理需求，确保系统的高性能、高可用性和可扩展性。

---

**相关阅读**:
- [负载均衡器设计原理](../load-balancer/design-principles.md)
- [服务发现架构](../service-discovery/architecture.md)
- [分布式缓存架构](../distributed-cache/architecture.md)