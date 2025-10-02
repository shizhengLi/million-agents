"""
分布式任务分发器
"""

import asyncio
import time
import uuid
import threading
import heapq
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """任务优先级"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


class WorkerStatus(Enum):
    """工作节点状态"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    BUSY = "busy"
    FAILED = "failed"


@dataclass
class Task:
    """任务数据结构"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_type: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3

    # 状态和时间戳
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    assigned_worker_id: Optional[str] = None

    # 结果
    result: Optional['TaskResult'] = None
    error_message: Optional[str] = None


@dataclass
class TaskResult:
    """任务结果"""
    task_id: str
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    execution_time: float = 0.0
    completed_at: float = field(default_factory=time.time)


@dataclass
class WorkerNode:
    """工作节点"""
    id: str
    address: str
    max_concurrent_tasks: int = 5
    current_tasks: int = 0
    status: WorkerStatus = WorkerStatus.ACTIVE

    # 统计信息
    completed_tasks: int = 0
    failed_tasks: int = 0
    total_execution_time: float = 0.0
    last_heartbeat: float = field(default_factory=time.time)

    @property
    def available_capacity(self) -> int:
        """可用容量"""
        return max(0, self.max_concurrent_tasks - self.current_tasks)

    @property
    def utilization(self) -> float:
        """利用率"""
        if self.max_concurrent_tasks == 0:
            return 0.0
        return self.current_tasks / self.max_concurrent_tasks

    @property
    def success_rate(self) -> float:
        """成功率"""
        total = self.completed_tasks + self.failed_tasks
        if total == 0:
            return 1.0
        return self.completed_tasks / total

    def can_accept_task(self) -> bool:
        """是否能接受新任务"""
        return (self.status == WorkerStatus.ACTIVE and
                self.current_tasks < self.max_concurrent_tasks)


class TaskDistributor:
    """分布式任务分发器"""

    def __init__(self,
                 max_concurrent_tasks: int = 100,
                 distribution_interval: float = 1.0,
                 task_timeout: float = 300.0):
        """初始化任务分发器

        Args:
            max_concurrent_tasks: 最大并发任务数
            distribution_interval: 分发间隔（秒）
            task_timeout: 默认任务超时时间（秒）
        """
        self.max_concurrent_tasks = max_concurrent_tasks
        self.distribution_interval = distribution_interval
        self.task_timeout = task_timeout

        # 任务存储
        self.pending_tasks: Dict[str, Task] = {}
        self.running_tasks: Dict[str, Task] = {}
        self.completed_tasks: Dict[str, Task] = {}

        # 工作节点
        self.worker_nodes: Dict[str, WorkerNode] = {}

        # 分发控制
        self.is_running = False
        self.distribution_task: Optional[asyncio.Task] = None
        self._lock = threading.RLock()

        # 任务执行器
        self.task_executors: Dict[str, Callable] = {}

        logger.info(f"TaskDistributor initialized with max_concurrent_tasks={max_concurrent_tasks}")

    def add_worker_node(self, worker: WorkerNode) -> None:
        """添加工作节点

        Args:
            worker: 工作节点
        """
        with self._lock:
            self.worker_nodes[worker.id] = worker
            logger.info(f"Added worker node {worker.id} at {worker.address}")

    def remove_worker_node(self, worker_id: str) -> bool:
        """移除工作节点

        Args:
            worker_id: 工作节点ID

        Returns:
            是否成功移除
        """
        with self._lock:
            if worker_id in self.worker_nodes:
                worker = self.worker_nodes[worker_id]

                # 将该节点上的任务重新分配
                tasks_to_reassign = []
                for task_id, task in list(self.running_tasks.items()):
                    if task.assigned_worker_id == worker_id:
                        tasks_to_reassign.append(task)

                # 重新分配任务
                for task in tasks_to_reassign:
                    self._reassign_task(task)

                # 移除节点
                del self.worker_nodes[worker_id]
                logger.info(f"Removed worker node {worker_id}")
                return True
            return False

    def submit_task(self, task: Task) -> str:
        """提交任务

        Args:
            task: 任务对象

        Returns:
            任务ID
        """
        with self._lock:
            # 检查并发任务限制
            current_total = len(self.pending_tasks) + len(self.running_tasks)
            if current_total >= self.max_concurrent_tasks:
                raise Exception(f"Task queue is full (max: {self.max_concurrent_tasks})")

            # 设置超时
            if task.timeout is None:
                task.timeout = self.task_timeout

            # 添加到待处理队列
            self.pending_tasks[task.id] = task

            # 使用优先级队列
            self._update_priority_queue()

            logger.debug(f"Submitted task {task.id} of type {task.task_type}")
            return task.id

    def get_next_task(self) -> Optional[Task]:
        """获取下一个待执行任务

        Returns:
            下一个任务，如果没有则返回None
        """
        with self._lock:
            if not self.pending_tasks:
                return None

            # 按优先级和时间排序，获取第一个任务
            sorted_tasks = sorted(
                self.pending_tasks.values(),
                key=lambda t: (-t.priority.value, t.created_at)
            )

            return sorted_tasks[0] if sorted_tasks else None

    def assign_task_to_worker(self, task_id: str, worker_id: str) -> bool:
        """分配任务给工作节点

        Args:
            task_id: 任务ID
            worker_id: 工作节点ID

        Returns:
            是否分配成功
        """
        with self._lock:
            if task_id not in self.pending_tasks:
                return False

            if worker_id not in self.worker_nodes:
                return False

            worker = self.worker_nodes[worker_id]
            task = self.pending_tasks[task_id]

            if not worker.can_accept_task():
                return False

            # 更新任务状态
            task.status = TaskStatus.RUNNING
            task.started_at = time.time()
            task.assigned_worker_id = worker_id

            # 移动到运行队列
            del self.pending_tasks[task_id]
            self.running_tasks[task_id] = task

            # 更新工作节点状态
            worker.current_tasks += 1

            logger.debug(f"Assigned task {task_id} to worker {worker_id}")
            return True

    def complete_task(self, task_id: str, result: TaskResult) -> bool:
        """完成任务

        Args:
            task_id: 任务ID
            result: 任务结果

        Returns:
            是否成功完成
        """
        with self._lock:
            if task_id not in self.running_tasks:
                return False

            task = self.running_tasks[task_id]
            worker_id = task.assigned_worker_id

            # 更新任务状态
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()
            task.result = result

            # 移动到完成队列
            del self.running_tasks[task_id]
            self.completed_tasks[task_id] = task

            # 更新工作节点统计
            if worker_id and worker_id in self.worker_nodes:
                worker = self.worker_nodes[worker_id]
                worker.current_tasks = max(0, worker.current_tasks - 1)
                worker.completed_tasks += 1
                worker.total_execution_time += result.execution_time

            logger.debug(f"Completed task {task_id}")
            return True

    def fail_task(self, task_id: str, error_message: str) -> bool:
        """任务失败

        Args:
            task_id: 任务ID
            error_message: 错误信息

        Returns:
            是否成功标记为失败
        """
        with self._lock:
            if task_id not in self.running_tasks:
                return False

            task = self.running_tasks[task_id]
            worker_id = task.assigned_worker_id

            # 检查是否需要重试
            task.retry_count += 1
            if task.retry_count <= task.max_retries:
                # 重新放入待处理队列
                task.status = TaskStatus.PENDING
                task.started_at = None
                task.assigned_worker_id = None
                task.error_message = error_message

                del self.running_tasks[task_id]
                self.pending_tasks[task_id] = task

                # 更新工作节点状态
                if worker_id and worker_id in self.worker_nodes:
                    worker = self.worker_nodes[worker_id]
                    worker.current_tasks = max(0, worker.current_tasks - 1)

                logger.warning(f"Retrying task {task_id} (attempt {task.retry_count}/{task.max_retries})")
                return True
            else:
                # 标记为最终失败
                task.status = TaskStatus.FAILED
                task.completed_at = time.time()
                task.error_message = error_message

                del self.running_tasks[task_id]
                self.completed_tasks[task_id] = task

                # 更新工作节点统计
                if worker_id and worker_id in self.worker_nodes:
                    worker = self.worker_nodes[worker_id]
                    worker.current_tasks = max(0, worker.current_tasks - 1)
                    worker.failed_tasks += 1

                logger.error(f"Task {task_id} failed after {task.retry_count} retries: {error_message}")
                return True

    def get_available_worker(self) -> Optional[WorkerNode]:
        """获取可用工作节点

        Returns:
            可用的工作节点，如果没有则返回None
        """
        with self._lock:
            available_workers = [
                worker for worker in self.worker_nodes.values()
                if worker.can_accept_task()
            ]

            if not available_workers:
                return None

            # 选择利用率最低的节点
            return min(available_workers, key=lambda w: w.utilization)

    async def execute_task_on_worker(self, task: Task, worker: WorkerNode) -> TaskResult:
        """在工作节点上执行任务

        Args:
            task: 任务
            worker: 工作节点

        Returns:
            任务结果
        """
        start_time = time.time()

        try:
            # 获取任务执行器
            executor = self.task_executors.get(task.task_type)
            if not executor:
                raise Exception(f"No executor found for task type: {task.task_type}")

            # 执行任务
            if asyncio.iscoroutinefunction(executor):
                result_data = await executor(task.data)
            else:
                result_data = executor(task.data)

            execution_time = time.time() - start_time

            return TaskResult(
                task_id=task.id,
                success=True,
                data=result_data,
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Task {task.id} execution failed: {e}")

            return TaskResult(
                task_id=task.id,
                success=False,
                error=str(e),
                execution_time=execution_time
            )

    def register_task_executor(self, task_type: str, executor: Callable) -> None:
        """注册任务执行器

        Args:
            task_type: 任务类型
            executor: 执行器函数
        """
        self.task_executors[task_type] = executor
        logger.info(f"Registered executor for task type: {task_type}")

    async def _distribution_loop(self) -> None:
        """任务分发循环"""
        while self.is_running:
            try:
                await self._distribute_pending_tasks()
                await asyncio.sleep(self.distribution_interval)
            except Exception as e:
                logger.error(f"Error in distribution loop: {e}")
                await asyncio.sleep(self.distribution_interval)

    async def _distribute_pending_tasks(self) -> None:
        """分发待处理任务"""
        while True:
            # 获取可用工作节点
            worker = self.get_available_worker()
            if not worker:
                break

            # 获取下一个任务
            task = self.get_next_task()
            if not task:
                break

            # 分配任务
            if self.assign_task_to_worker(task.id, worker.id):
                # 异步执行任务
                asyncio.create_task(self._execute_and_complete_task(task, worker))
            else:
                break

    async def _execute_and_complete_task(self, task: Task, worker: WorkerNode) -> None:
        """执行任务并处理结果"""
        try:
            # 执行任务
            result = await self.execute_task_on_worker(task, worker)

            # 处理结果
            if result.success:
                self.complete_task(task.id, result)
            else:
                self.fail_task(task.id, result.error or "Task execution failed")

        except asyncio.TimeoutError:
            self.fail_task(task.id, f"Task timeout after {task.timeout} seconds")
        except Exception as e:
            self.fail_task(task.id, f"Unexpected error: {str(e)}")

    def start_task_distribution(self) -> None:
        """启动任务分发"""
        if self.is_running:
            return

        self.is_running = True
        self.distribution_task = asyncio.create_task(self._distribution_loop())
        logger.info("Task distribution started")

    def stop_task_distribution(self) -> None:
        """停止任务分发"""
        if not self.is_running:
            return

        self.is_running = False
        if self.distribution_task:
            self.distribution_task.cancel()
            self.distribution_task = None

        logger.info("Task distribution stopped")

    def _reassign_task(self, task: Task) -> None:
        """重新分配任务"""
        task.status = TaskStatus.PENDING
        task.started_at = None
        task.assigned_worker_id = None

        if task.id in self.running_tasks:
            del self.running_tasks[task.id]
            self.pending_tasks[task.id] = task

    def _update_priority_queue(self) -> None:
        """更新优先级队列（占位符，实际使用时可以实现更复杂的队列）"""
        pass

    def get_task_statistics(self) -> Dict[str, Any]:
        """获取任务统计信息

        Returns:
            统计信息字典
        """
        with self._lock:
            total_tasks = len(self.pending_tasks) + len(self.running_tasks) + len(self.completed_tasks)
            completed_count = sum(1 for t in self.completed_tasks.values() if t.status == TaskStatus.COMPLETED)
            failed_count = sum(1 for t in self.completed_tasks.values() if t.status == TaskStatus.FAILED)

            success_rate = completed_count / (completed_count + failed_count) if (completed_count + failed_count) > 0 else 0.0

            return {
                'total_tasks': total_tasks,
                'pending_tasks': len(self.pending_tasks),
                'running_tasks': len(self.running_tasks),
                'completed_tasks': completed_count,
                'failed_tasks': failed_count,
                'success_rate': success_rate,
                'worker_nodes': len(self.worker_nodes)
            }

    def get_worker_statistics(self) -> Dict[str, Dict[str, Any]]:
        """获取工作节点统计信息

        Returns:
            工作节点统计信息字典
        """
        with self._lock:
            stats = {}
            for worker_id, worker in self.worker_nodes.items():
                stats[worker_id] = {
                    'id': worker.id,
                    'address': worker.address,
                    'status': worker.status.value,
                    'current_tasks': worker.current_tasks,
                    'max_concurrent_tasks': worker.max_concurrent_tasks,
                    'available_capacity': worker.available_capacity,
                    'utilization': worker.utilization,
                    'completed_tasks': worker.completed_tasks,
                    'failed_tasks': worker.failed_tasks,
                    'success_rate': worker.success_rate,
                    'total_execution_time': worker.total_execution_time
                }
            return stats

    def cleanup_completed_tasks(self, max_age_seconds: float = 3600) -> int:
        """清理已完成的任务

        Args:
            max_age_seconds: 最大保留时间（秒）

        Returns:
            清理的任务数量
        """
        with self._lock:
            current_time = time.time()
            tasks_to_remove = []

            for task_id, task in self.completed_tasks.items():
                if task.completed_at and (current_time - task.completed_at) > max_age_seconds:
                    tasks_to_remove.append(task_id)

            for task_id in tasks_to_remove:
                del self.completed_tasks[task_id]

            if tasks_to_remove:
                logger.info(f"Cleaned up {len(tasks_to_remove)} completed tasks")

            return len(tasks_to_remove)