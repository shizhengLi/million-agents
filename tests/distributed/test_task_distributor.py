"""
分布式任务分发器测试
"""

import pytest
import asyncio
import time
import uuid
from unittest.mock import Mock, patch, AsyncMock
from src.distributed.task_distributor import (
    TaskDistributor, Task, TaskStatus, TaskPriority,
    TaskResult, WorkerNode, WorkerStatus
)


class TestTaskDistributor:
    """测试任务分发器核心功能"""

    def test_task_distributor_initialization(self):
        """测试任务分发器初始化"""
        distributor = TaskDistributor(max_concurrent_tasks=100)

        # 验证初始状态
        assert distributor.max_concurrent_tasks == 100
        assert len(distributor.pending_tasks) == 0
        assert len(distributor.running_tasks) == 0
        assert len(distributor.completed_tasks) == 0
        assert len(distributor.worker_nodes) == 0
        assert distributor.is_running == False

    def test_task_creation(self):
        """测试任务创建"""
        task = Task(
            id="task_1",
            task_type="compute",
            data={"input": "test_data"},
            priority=TaskPriority.HIGH,
            timeout=30
        )

        # 验证任务属性
        assert task.id == "task_1"
        assert task.task_type == "compute"
        assert task.data == {"input": "test_data"}
        assert task.priority == TaskPriority.HIGH
        assert task.timeout == 30
        assert task.status == TaskStatus.PENDING
        assert task.created_at > 0
        assert task.started_at is None
        assert task.completed_at is None
        assert task.result is None
        assert task.error_message is None

    def test_worker_node_creation(self):
        """测试工作节点创建"""
        worker = WorkerNode(
            id="worker_1",
            address="localhost:8001",
            max_concurrent_tasks=3,
            current_tasks=0
        )

        # 验证工作节点属性
        assert worker.id == "worker_1"
        assert worker.address == "localhost:8001"
        assert worker.max_concurrent_tasks == 3
        assert worker.current_tasks == 0
        assert worker.available_capacity == 3
        assert worker.utilization == 0.0

    def test_task_submission(self):
        """测试任务提交"""
        distributor = TaskDistributor()
        worker = WorkerNode("worker1", "localhost:8001")
        distributor.add_worker_node(worker)

        # 创建并提交任务
        task = Task(
            id="task_1",
            task_type="test_task",
            data={"message": "hello"},
            priority=TaskPriority.NORMAL
        )

        task_id = distributor.submit_task(task)

        # 验证任务已提交
        assert task_id == task.id
        assert len(distributor.pending_tasks) == 1
        assert task_id in distributor.pending_tasks
        assert distributor.pending_tasks[task_id].status == TaskStatus.PENDING

    def test_task_priority_queue(self):
        """测试任务优先级队列"""
        distributor = TaskDistributor()
        worker = WorkerNode("worker1", "localhost:8001")
        distributor.add_worker_node(worker)

        # 创建不同优先级的任务
        low_task = Task(
            id="low",
            task_type="normal_task",
            data={},
            priority=TaskPriority.LOW
        )
        normal_task = Task(
            id="normal",
            task_type="normal_task",
            data={},
            priority=TaskPriority.NORMAL
        )
        high_task = Task(
            id="high",
            task_type="high_task",
            data={},
            priority=TaskPriority.HIGH
        )
        urgent_task = Task(
            id="urgent",
            task_type="urgent_task",
            data={},
            priority=TaskPriority.URGENT
        )

        # 按顺序提交
        distributor.submit_task(low_task)
        distributor.submit_task(normal_task)
        distributor.submit_task(high_task)
        distributor.submit_task(urgent_task)

        # 验证任务按优先级排序
        pending_tasks = list(distributor.pending_tasks.values())
        # 按优先级和时间排序
        sorted_tasks = sorted(
            pending_tasks,
            key=lambda t: (-t.priority.value, t.created_at)
        )

        assert sorted_tasks[0].priority == TaskPriority.URGENT
        assert sorted_tasks[1].priority == TaskPriority.HIGH
        assert sorted_tasks[2].priority == TaskPriority.NORMAL
        assert sorted_tasks[3].priority == TaskPriority.LOW

    def test_add_worker_node(self):
        """测试添加工作节点"""
        distributor = TaskDistributor()
        worker = WorkerNode(
            id="worker1",
            address="localhost:8001",
            max_concurrent_tasks=5
        )

        # 添加工作节点
        distributor.add_worker_node(worker)

        # 验证节点已添加
        assert len(distributor.worker_nodes) == 1
        assert worker.id in distributor.worker_nodes
        assert distributor.worker_nodes[worker.id].status == WorkerStatus.ACTIVE

    def test_remove_worker_node(self):
        """测试移除工作节点"""
        distributor = TaskDistributor()
        worker = WorkerNode("worker1", "localhost:8001")
        distributor.add_worker_node(worker)

        # 移除工作节点
        result = distributor.remove_worker_node("worker1")

        # 验证节点已移除
        assert result is True
        assert len(distributor.worker_nodes) == 0
        assert worker.id not in distributor.worker_nodes

    def test_task_assignment_to_worker(self):
        """测试分配任务给工作节点"""
        distributor = TaskDistributor()
        worker = WorkerNode("worker1", "localhost:8001")
        distributor.add_worker_node(worker)

        # 创建并提交任务
        task = Task("task_1", "test_task", {})
        distributor.submit_task(task)

        # 分配任务给工作节点
        success = distributor.assign_task_to_worker(task.id, worker.id)

        # 验证任务分配成功
        assert success is True
        assert task.status == TaskStatus.RUNNING
        assert task.assigned_worker_id == worker.id
        assert task.id in distributor.running_tasks
        assert task.id not in distributor.pending_tasks

        # 验证工作节点任务计数
        assert worker.current_tasks == 1

    def test_task_completion(self):
        """测试任务完成"""
        distributor = TaskDistributor()
        worker = WorkerNode("worker1", "localhost:8001")
        distributor.add_worker_node(worker)

        # 创建、提交并分配任务
        task = Task("task_1", "test_task", {})
        distributor.submit_task(task)
        distributor.assign_task_to_worker(task.id, worker.id)

        # 完成任务
        result = TaskResult(
            task_id=task.id,
            success=True,
            data={"output": "success"},
            execution_time=1.5
        )

        success = distributor.complete_task(task.id, result)

        # 验证任务完成
        assert success is True
        assert task.status == TaskStatus.COMPLETED
        assert task.result == result
        assert task.id in distributor.completed_tasks
        assert task.id not in distributor.running_tasks

        # 验证工作节点任务计数
        assert worker.current_tasks == 0
        assert worker.completed_tasks == 1

    def test_task_failure(self):
        """测试任务失败"""
        distributor = TaskDistributor()
        worker = WorkerNode("worker1", "localhost:8001")
        distributor.add_worker_node(worker)

        # 创建、提交并分配任务（不重试）
        task = Task("task_1", "test_task", {}, max_retries=0)
        distributor.submit_task(task)
        distributor.assign_task_to_worker(task.id, worker.id)

        # 任务失败
        error_msg = "Task execution failed"
        success = distributor.fail_task(task.id, error_msg)

        # 验证任务失败
        assert success is True
        assert task.status == TaskStatus.FAILED
        assert task.error_message == error_msg
        assert task.id in distributor.completed_tasks
        assert task.id not in distributor.running_tasks

        # 验证工作节点任务计数
        assert worker.current_tasks == 0
        assert worker.failed_tasks == 1

    def test_get_available_worker(self):
        """测试获取可用工作节点"""
        distributor = TaskDistributor()

        # 添加工作节点
        worker1 = WorkerNode("worker1", "localhost:8001", max_concurrent_tasks=2)
        worker2 = WorkerNode("worker2", "localhost:8002", max_concurrent_tasks=1)
        distributor.add_worker_node(worker1)
        distributor.add_worker_node(worker2)

        # 初始状态：两个节点都可用
        available_worker = distributor.get_available_worker()
        assert available_worker is not None
        assert available_worker.id in ["worker1", "worker2"]

        # worker1满载
        worker1.current_tasks = 2
        available_worker = distributor.get_available_worker()
        assert available_worker.id == "worker2"

        # 所有节点都满载
        worker2.current_tasks = 1
        available_worker = distributor.get_available_worker()
        assert available_worker is None

    @pytest.mark.asyncio
    async def test_start_task_distribution(self):
        """测试启动任务分发"""
        distributor = TaskDistributor()

        # 启动分发器
        distributor.start_task_distribution()

        # 验证分发器已启动
        assert distributor.is_running is True

        # 停止分发器
        distributor.stop_task_distribution()
        assert distributor.is_running is False

    @pytest.mark.asyncio
    async def test_automatic_task_distribution(self):
        """测试自动任务分发"""
        distributor = TaskDistributor(distribution_interval=0.1)

        # Mock工作节点
        worker = WorkerNode("worker1", "localhost:8001")
        distributor.add_worker_node(worker)

        # Mock任务执行
        with patch.object(distributor, 'execute_task_on_worker', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = TaskResult("test_task", True, {})

            # 启动分发器
            distributor.start_task_distribution()

            # 提交任务
            task = Task(str(uuid.uuid4()), "test_task", {})
            distributor.submit_task(task)

            # 等待任务被分发
            await asyncio.sleep(0.2)

            # 验证任务被分发
            assert mock_execute.called
            assert task.status in [TaskStatus.RUNNING, TaskStatus.COMPLETED]

            # 停止分发器
            distributor.stop_task_distribution()

    def test_task_statistics(self):
        """测试任务统计"""
        distributor = TaskDistributor()
        worker = WorkerNode("worker1", "localhost:8001")
        distributor.add_worker_node(worker)

        # 创建任务
        completed_task = Task(str(uuid.uuid4()), "completed_task", {}, max_retries=0)
        failed_task = Task(str(uuid.uuid4()), "failed_task", {}, max_retries=0)
        pending_task = Task(str(uuid.uuid4()), "pending_task", {})

        # 提交任务
        distributor.submit_task(completed_task)
        distributor.submit_task(failed_task)
        distributor.submit_task(pending_task)

        # 完成和失败任务
        distributor.assign_task_to_worker(completed_task.id, worker.id)
        distributor.complete_task(completed_task.id, TaskResult(completed_task.id, True, {}))

        distributor.assign_task_to_worker(failed_task.id, worker.id)
        distributor.fail_task(failed_task.id, "Failed")

        # 获取统计信息
        stats = distributor.get_task_statistics()

        # 验证统计信息
        assert stats['total_tasks'] == 3
        assert stats['pending_tasks'] == 1
        assert stats['running_tasks'] == 0
        assert stats['completed_tasks'] == 1
        assert stats['failed_tasks'] == 1
        assert stats['success_rate'] == 0.5

    def test_worker_statistics(self):
        """测试工作节点统计"""
        distributor = TaskDistributor()

        # 添加工作节点
        worker1 = WorkerNode("worker1", "localhost:8001")
        worker2 = WorkerNode("worker2", "localhost:8002")
        distributor.add_worker_node(worker1)
        distributor.add_worker_node(worker2)

        # 模拟任务执行
        task1 = Task(str(uuid.uuid4()), "task1", {}, max_retries=0)
        task2 = Task(str(uuid.uuid4()), "task2", {}, max_retries=0)

        distributor.submit_task(task1)
        distributor.submit_task(task2)

        distributor.assign_task_to_worker(task1.id, worker1.id)
        distributor.complete_task(task1.id, TaskResult(task1.id, True, {}))

        distributor.assign_task_to_worker(task2.id, worker2.id)
        distributor.fail_task(task2.id, "Failed")

        # 获取工作节点统计
        stats = distributor.get_worker_statistics()

        # 验证统计信息
        assert len(stats) == 2
        assert stats['worker1']['completed_tasks'] == 1
        assert stats['worker1']['success_rate'] == 1.0
        assert stats['worker2']['failed_tasks'] == 1
        assert stats['worker2']['success_rate'] == 0.0

    def test_cleanup_completed_tasks(self):
        """测试清理已完成任务"""
        distributor = TaskDistributor()
        worker = WorkerNode("worker1", "localhost:8001")
        distributor.add_worker_node(worker)

        # 创建并完成任务
        task = Task(str(uuid.uuid4()), "test_task", {})
        distributor.submit_task(task)
        distributor.assign_task_to_worker(task.id, worker.id)
        distributor.complete_task(task.id, TaskResult(task.id, True, {}))

        # 验证任务在完成列表中
        assert len(distributor.completed_tasks) == 1

        # 清理超过1秒的已完成任务
        task.completed_at = time.time() - 2  # 设置为2秒前完成
        cleaned_count = distributor.cleanup_completed_tasks(max_age_seconds=1)

        # 验证任务被清理
        assert cleaned_count == 1
        assert len(distributor.completed_tasks) == 0

    def test_task_priority_validation(self):
        """测试任务优先级验证"""
        # 测试有效优先级
        valid_priorities = [TaskPriority.LOW, TaskPriority.NORMAL, TaskPriority.HIGH, TaskPriority.URGENT]
        for priority in valid_priorities:
            task = Task(str(uuid.uuid4()), "test_task", {}, priority=priority)
            assert task.priority == priority

    def test_task_status_transitions(self):
        """测试任务状态转换"""
        distributor = TaskDistributor()
        worker = WorkerNode("worker1", "localhost:8001")
        distributor.add_worker_node(worker)

        task = Task(str(uuid.uuid4()), "test_task", {})

        # 初始状态
        assert task.status == TaskStatus.PENDING

        # 分配任务
        distributor.submit_task(task)
        distributor.assign_task_to_worker(task.id, worker.id)
        assert task.status == TaskStatus.RUNNING

        # 完成任务
        distributor.complete_task(task.id, TaskResult(task.id, True, {}))
        assert task.status == TaskStatus.COMPLETED

    def test_concurrent_task_submission(self):
        """测试并发任务提交"""
        distributor = TaskDistributor()
        worker = WorkerNode("worker1", "localhost:8001", max_concurrent_tasks=10)
        distributor.add_worker_node(worker)

        # 并发提交多个任务
        tasks = []
        for i in range(5):
            task = Task(str(uuid.uuid4()), f"task_{i}", {"index": i})
            task_id = distributor.submit_task(task)
            tasks.append(task_id)

        # 验证所有任务都已提交
        assert len(tasks) == 5
        assert len(distributor.pending_tasks) == 5
        assert all(task_id in distributor.pending_tasks for task_id in tasks)

    def test_max_concurrent_tasks_limit(self):
        """测试最大并发任务限制"""
        distributor = TaskDistributor(max_concurrent_tasks=2)

        # 创建多个任务
        for i in range(2):
            task = Task(str(uuid.uuid4()), f"task_{i}", {})
            distributor.submit_task(task)

        # 验证2个任务被提交
        assert len(distributor.pending_tasks) == 2

        # 尝试提交第3个任务应该失败
        extra_task = Task(str(uuid.uuid4()), "extra_task", {})
        with pytest.raises(Exception, match="Task queue is full"):
            distributor.submit_task(extra_task)

    def test_task_retry_mechanism(self):
        """测试任务重试机制"""
        distributor = TaskDistributor()
        worker = WorkerNode("worker1", "localhost:8001")
        distributor.add_worker_node(worker)

        # 创建可重试的任务
        task = Task("retry_task", "test_task", {}, max_retries=2)
        distributor.submit_task(task)
        distributor.assign_task_to_worker(task.id, worker.id)

        # 第一次失败，应该重试
        distributor.fail_task(task.id, "First failure")
        assert task.status == TaskStatus.PENDING
        assert task.retry_count == 1

        # 第二次失败，应该重试
        distributor.assign_task_to_worker(task.id, worker.id)
        distributor.fail_task(task.id, "Second failure")
        assert task.status == TaskStatus.PENDING
        assert task.retry_count == 2

        # 第三次失败，应该最终失败
        distributor.assign_task_to_worker(task.id, worker.id)
        distributor.fail_task(task.id, "Final failure")
        assert task.status == TaskStatus.FAILED
        assert task.retry_count == 3

    def test_worker_can_accept_task(self):
        """测试工作节点是否能接受任务"""
        # 活跃且有容量
        worker1 = WorkerNode("worker1", "localhost:8001", max_concurrent_tasks=3, current_tasks=1)
        assert worker1.can_accept_task() == True

        # 活跃但满载
        worker2 = WorkerNode("worker2", "localhost:8002", max_concurrent_tasks=2, current_tasks=2)
        assert worker2.can_accept_task() == False

        # 非活跃状态
        worker3 = WorkerNode("worker3", "localhost:8003", max_concurrent_tasks=3, current_tasks=0)
        worker3.status = WorkerStatus.INACTIVE
        assert worker3.can_accept_task() == False

    def test_task_result_creation(self):
        """测试任务结果创建"""
        result = TaskResult(
            task_id="test_task",
            success=True,
            data={"output": "result"},
            execution_time=1.5
        )

        assert result.task_id == "test_task"
        assert result.success == True
        assert result.data == {"output": "result"}
        assert result.execution_time == 1.5
        assert result.completed_at > 0

    def test_worker_utilization_zero_capacity(self):
        """测试工作节点零容量利用率计算"""
        worker = WorkerNode("worker1", "localhost:8001", max_concurrent_tasks=0)
        assert worker.utilization == 0.0

    def test_worker_success_rate_no_tasks(self):
        """测试工作节点无任务时成功率计算"""
        worker = WorkerNode("worker1", "localhost:8001")
        assert worker.success_rate == 1.0

    def test_remove_worker_node_with_running_tasks(self):
        """测试移除有运行任务的工作节点"""
        distributor = TaskDistributor()
        worker = WorkerNode("worker1", "localhost:8001")
        distributor.add_worker_node(worker)

        # 创建并分配任务
        task = Task("task_1", "test_task", {})
        distributor.submit_task(task)
        distributor.assign_task_to_worker(task.id, worker.id)

        # 移除工作节点应该重新分配任务
        result = distributor.remove_worker_node("worker1")
        assert result is True
        assert len(distributor.worker_nodes) == 0
        assert task.status == TaskStatus.PENDING
        assert task.assigned_worker_id is None

    def test_remove_nonexistent_worker_node(self):
        """测试移除不存在的工作节点"""
        distributor = TaskDistributor()
        result = distributor.remove_worker_node("nonexistent")
        assert result is False

    def test_assign_task_to_nonexistent_task(self):
        """测试分配不存在的任务"""
        distributor = TaskDistributor()
        worker = WorkerNode("worker1", "localhost:8001")
        distributor.add_worker_node(worker)

        result = distributor.assign_task_to_worker("nonexistent_task", "worker1")
        assert result is False

    def test_assign_task_to_nonexistent_worker(self):
        """测试任务分配给不存在的工作节点"""
        distributor = TaskDistributor()
        task = Task("task_1", "test_task", {})
        distributor.submit_task(task)

        result = distributor.assign_task_to_worker("task_1", "nonexistent_worker")
        assert result is False

    def test_assign_task_to_full_worker(self):
        """测试任务分配给满载工作节点"""
        distributor = TaskDistributor()
        worker = WorkerNode("worker1", "localhost:8001", max_concurrent_tasks=1)
        distributor.add_worker_node(worker)

        # 分配第一个任务
        task1 = Task("task_1", "test_task", {})
        distributor.submit_task(task1)
        distributor.assign_task_to_worker(task1.id, worker.id)

        # 尝试分配第二个任务应该失败
        task2 = Task("task_2", "test_task", {})
        distributor.submit_task(task2)
        result = distributor.assign_task_to_worker(task2.id, worker.id)
        assert result is False

    def test_fail_nonexistent_task(self):
        """测试失败不存在的任务"""
        distributor = TaskDistributor()
        result = distributor.fail_task("nonexistent_task", "Not found")
        assert result is False

    @pytest.mark.asyncio
    async def test_execute_task_with_sync_executor(self):
        """测试使用同步执行器执行任务"""
        distributor = TaskDistributor()
        worker = WorkerNode("worker1", "localhost:8001")

        # 注册同步执行器
        def sync_executor(data):
            return {"result": f"sync_{data.get('input', '')}"}

        distributor.register_task_executor("sync_task", sync_executor)

        task = Task("task_1", "sync_task", {"input": "test"})
        result = await distributor.execute_task_on_worker(task, worker)

        assert result.success is True
        assert result.data == {"result": "sync_test"}

    @pytest.mark.asyncio
    async def test_execute_task_with_async_executor(self):
        """测试使用异步执行器执行任务"""
        distributor = TaskDistributor()
        worker = WorkerNode("worker1", "localhost:8001")

        # 注册异步执行器
        async def async_executor(data):
            await asyncio.sleep(0.01)  # 模拟异步操作
            return {"result": f"async_{data.get('input', '')}"}

        distributor.register_task_executor("async_task", async_executor)

        task = Task("task_1", "async_task", {"input": "test"})
        result = await distributor.execute_task_on_worker(task, worker)

        assert result.success is True
        assert result.data == {"result": "async_test"}

    @pytest.mark.asyncio
    async def test_execute_task_no_executor_found(self):
        """测试任务执行器未找到"""
        distributor = TaskDistributor()
        worker = WorkerNode("worker1", "localhost:8001")

        task = Task("task_1", "unknown_task", {})
        result = await distributor.execute_task_on_worker(task, worker)

        assert result.success is False
        assert "No executor found" in result.error

    @pytest.mark.asyncio
    async def test_execute_task_executor_exception(self):
        """测试任务执行器抛出异常"""
        distributor = TaskDistributor()
        worker = WorkerNode("worker1", "localhost:8001")

        # 注册会抛出异常的执行器
        def failing_executor(data):
            raise ValueError("Test error")

        distributor.register_task_executor("failing_task", failing_executor)

        task = Task("task_1", "failing_task", {})
        result = await distributor.execute_task_on_worker(task, worker)

        assert result.success is False
        assert "Test error" in result.error

    def test_register_task_executor(self):
        """测试注册任务执行器"""
        distributor = TaskDistributor()

        def test_executor(data):
            return {"result": "test"}

        distributor.register_task_executor("test_type", test_executor)
        assert "test_type" in distributor.task_executors
        assert distributor.task_executors["test_type"] == test_executor

    @pytest.mark.asyncio
    async def test_distribution_loop_error_handling(self):
        """测试分发循环错误处理"""
        distributor = TaskDistributor(distribution_interval=0.01)

        # 启动分发器
        distributor.start_task_distribution()

        # 等待一小段时间让分发循环运行
        await asyncio.sleep(0.05)

        # 停止分发器
        distributor.stop_task_distribution()
        assert distributor.is_running is False

    @pytest.mark.asyncio
    async def test_distribute_pending_tasks_exits_when_no_worker(self):
        """测试无可用工作节点时退出分发"""
        distributor = TaskDistributor()

        # 添加任务但不添加工作节点
        task = Task("task_1", "test_task", {})
        distributor.submit_task(task)

        # 调用分发方法应该直接返回
        await distributor._distribute_pending_tasks()
        assert len(distributor.pending_tasks) == 1

    @pytest.mark.asyncio
    async def test_distribute_pending_tasks_exits_when_no_task(self):
        """测试无待处理任务时退出分发"""
        distributor = TaskDistributor()
        worker = WorkerNode("worker1", "localhost:8001")
        distributor.add_worker_node(worker)

        # 调用分发方法应该直接返回
        await distributor._distribute_pending_tasks()
        assert len(distributor.worker_nodes) == 1

    @pytest.mark.asyncio
    async def test_execute_and_complete_task_success(self):
        """测试任务执行并完成成功流程"""
        distributor = TaskDistributor()
        worker = WorkerNode("worker1", "localhost:8001")
        distributor.add_worker_node(worker)

        # Mock执行器
        async def mock_executor(data):
            return {"result": "success"}

        distributor.register_task_executor("test_task", mock_executor)

        # 首先提交并分配任务
        task = Task("task_1", "test_task", {})
        distributor.submit_task(task)
        distributor.assign_task_to_worker(task.id, worker.id)

        # 获取已分配的任务对象
        assigned_task = distributor.running_tasks[task.id]

        await distributor._execute_and_complete_task(assigned_task, worker)

        assert assigned_task.status == TaskStatus.COMPLETED
        assert assigned_task.result is not None
        assert assigned_task.result.success is True

    @pytest.mark.asyncio
    async def test_execute_and_complete_task_failure(self):
        """测试任务执行并处理失败流程"""
        distributor = TaskDistributor()
        worker = WorkerNode("worker1", "localhost:8001")
        distributor.add_worker_node(worker)

        # Mock失败的执行器
        async def mock_executor(data):
            raise Exception("Execution failed")

        distributor.register_task_executor("test_task", mock_executor)

        # 首先提交并分配任务
        task = Task("task_1", "test_task", {}, max_retries=0)
        distributor.submit_task(task)
        distributor.assign_task_to_worker(task.id, worker.id)

        # 获取已分配的任务对象
        assigned_task = distributor.running_tasks[task.id]

        await distributor._execute_and_complete_task(assigned_task, worker)

        assert assigned_task.status == TaskStatus.FAILED
        assert "Execution failed" in assigned_task.error_message

    def test_reassign_task_not_in_running(self):
        """测试重新分配不在运行队列中的任务"""
        distributor = TaskDistributor()

        task = Task("task_1", "test_task", {})
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
        task.assigned_worker_id = "worker1"

        # 不手动添加到运行队列
        assert task.id not in distributor.running_tasks

        # 重新分配任务应该不会出错
        distributor._reassign_task(task)

        assert task.status == TaskStatus.PENDING
        assert task.started_at is None
        assert task.assigned_worker_id is None

    def test_start_task_distribution_when_already_running(self):
        """测试已运行时启动任务分发"""
        distributor = TaskDistributor()

        # 手动设置为运行状态
        distributor.is_running = True
        distributor.start_task_distribution()

        # 应该没有创建新的任务
        assert distributor.is_running is True

    def test_stop_task_distribution_when_not_running(self):
        """测试未运行时停止任务分发"""
        distributor = TaskDistributor()

        # 停止未运行的分发器
        distributor.stop_task_distribution()

        # 应该保持未运行状态
        assert distributor.is_running is False

    def test_reassign_task_removes_from_running(self):
        """测试重新分配任务时从运行队列移除"""
        distributor = TaskDistributor()

        task = Task("task_1", "test_task", {})
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
        task.assigned_worker_id = "worker1"

        # 手动添加到运行队列
        distributor.running_tasks[task.id] = task

        # 重新分配任务
        distributor._reassign_task(task)

        # 验证任务已从运行队列移除并添加到待处理队列
        assert task.id not in distributor.running_tasks
        assert task.id in distributor.pending_tasks
        assert task.status == TaskStatus.PENDING
        assert task.started_at is None
        assert task.assigned_worker_id is None