"""
分布式负载均衡器
"""

import asyncio
import time
import random
from enum import Enum
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field
import threading
import aiohttp
import logging

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """节点健康状态"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass(frozen=True)  # frozen=True使Node可哈希
class Node:
    """负载均衡节点"""
    id: str
    address: str
    weight: int = 1
    current_connections: int = 0
    health_status: HealthStatus = HealthStatus.HEALTHY  # 默认健康状态
    last_health_check: float = field(default_factory=time.time)
    total_requests: int = 0
    failed_requests: int = 0

    def __post_init__(self):
        """验证节点参数"""
        if self.weight <= 0:
            raise ValueError(f"Node weight must be positive, got {self.weight}")

    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.total_requests == 0:
            return 1.0
        return (self.total_requests - self.failed_requests) / self.total_requests

    @property
    def failure_rate(self) -> float:
        """失败率"""
        return 1.0 - self.success_rate


class LoadBalancer:
    """分布式负载均衡器"""

    def __init__(self, strategy: str = "round_robin"):
        """初始化负载均衡器

        Args:
            strategy: 负载均衡策略
                - round_robin: 轮询
                - weighted_round_robin: 加权轮询
                - least_connections: 最少连接
        """
        # 验证策略
        valid_strategies = ["round_robin", "weighted_round_robin", "least_connections"]
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy: {strategy}. Valid strategies: {valid_strategies}")

        self.strategy = strategy
        self.nodes: List[Node] = []
        self.healthy_nodes: Set[Node] = set()  # 存储Node对象而不是节点ID
        self.current_index: int = 0
        self._lock = threading.RLock()

        logger.info(f"LoadBalancer initialized with strategy: {strategy}")

    def add_node(self, node: Node) -> None:
        """添加节点

        Args:
            node: 要添加的节点
        """
        with self._lock:
            # 检查节点是否已存在
            if node.id in [n.id for n in self.nodes]:
                raise ValueError(f"Node {node.id} already exists")

            self.nodes.append(node)

            # 如果节点健康，添加到健康节点集合
            if node.health_status == HealthStatus.HEALTHY:
                self.healthy_nodes.add(node)

            logger.info(f"Added node {node.id} at {node.address}")

    def remove_node(self, node_id: str) -> bool:
        """移除节点

        Args:
            node_id: 节点ID

        Returns:
            是否成功移除
        """
        with self._lock:
            for i, node in enumerate(self.nodes):
                if node.id == node_id:
                    self.nodes.pop(i)
                    self.healthy_nodes.discard(node)  # 移除Node对象而不是节点ID
                    logger.info(f"Removed node {node_id}")
                    return True
            return False

    def get_next_node(self) -> Optional[Node]:
        """获取下一个可用节点

        Returns:
            选中的节点，如果没有健康节点则返回None
        """
        with self._lock:
            # 过滤健康节点
            healthy_nodes = [node for node in self.nodes
                           if node.health_status == HealthStatus.HEALTHY]

            if not healthy_nodes:
                logger.warning("No healthy nodes available")
                return None

            # 根据策略选择节点
            if self.strategy == "round_robin":
                selected = self._round_robin_select(healthy_nodes)
            elif self.strategy == "weighted_round_robin":
                selected = self._weighted_round_robin_select(healthy_nodes)
            elif self.strategy == "least_connections":
                selected = self._least_connections_select(healthy_nodes)
            else:
                # 默认使用轮询
                selected = self._round_robin_select(healthy_nodes)

            if selected:
                # 创建新的Node对象，更新连接数和请求数
                updated_node = Node(
                    id=selected.id,
                    address=selected.address,
                    weight=selected.weight,
                    current_connections=selected.current_connections + 1,
                    health_status=selected.health_status,
                    last_health_check=selected.last_health_check,
                    total_requests=selected.total_requests + 1,
                    failed_requests=selected.failed_requests
                )

                # 更新nodes列表中的节点
                for i, node in enumerate(self.nodes):
                    if node.id == selected.id:
                        self.nodes[i] = updated_node
                        # 同时更新healthy_nodes集合
                        if node in self.healthy_nodes:
                            self.healthy_nodes.remove(node)
                            self.healthy_nodes.add(updated_node)
                        break

                return updated_node

            return None

    def _round_robin_select(self, healthy_nodes: List[Node]) -> Node:
        """轮询选择节点"""
        if not healthy_nodes:
            return None

        selected = healthy_nodes[self.current_index % len(healthy_nodes)]
        self.current_index += 1
        return selected

    def _weighted_round_robin_select(self, healthy_nodes: List[Node]) -> Node:
        """加权轮询选择节点"""
        if not healthy_nodes:
            return None

        # 构建加权列表
        weighted_nodes = []
        for node in healthy_nodes:
            weighted_nodes.extend([node] * node.weight)

        if not weighted_nodes:
            return healthy_nodes[0]

        # 使用加权索引选择
        weighted_index = self.current_index % len(weighted_nodes)
        selected = weighted_nodes[weighted_index]
        self.current_index += 1

        return selected

    def _least_connections_select(self, healthy_nodes: List[Node]) -> Node:
        """最少连接选择节点"""
        if not healthy_nodes:
            return None

        # 找到连接数最少的节点
        min_connections = min(node.current_connections for node in healthy_nodes)
        least_connected_nodes = [node for node in healthy_nodes
                               if node.current_connections == min_connections]

        # 如果有多个节点连接数相同，随机选择
        return random.choice(least_connected_nodes)

    def release_connection(self, node_id: str) -> bool:
        """释放节点连接

        Args:
            node_id: 节点ID

        Returns:
            是否成功释放
        """
        with self._lock:
            for i, node in enumerate(self.nodes):
                if node.id == node_id:
                    if node.current_connections > 0:
                        # 创建新的Node对象，减少连接数
                        updated_node = Node(
                            id=node.id,
                            address=node.address,
                            weight=node.weight,
                            current_connections=node.current_connections - 1,
                            health_status=node.health_status,
                            last_health_check=node.last_health_check,
                            total_requests=node.total_requests,
                            failed_requests=node.failed_requests
                        )
                        self.nodes[i] = updated_node

                        # 更新healthy_nodes集合
                        if node in self.healthy_nodes:
                            self.healthy_nodes.remove(node)
                            self.healthy_nodes.add(updated_node)
                    return True
            return False

    def _check_node_health(self, node: Node) -> bool:
        """内部健康检查方法（用于测试）

        Args:
            node: 要检查的节点

        Returns:
            节点是否健康
        """
        # 简单的健康检查逻辑，用于测试
        # 在实际环境中，这里会进行网络检查
        return node.health_status == HealthStatus.HEALTHY

    def check_node_health(self, node: Node) -> bool:
        """检查节点健康状态（同步版本，用于测试）

        Args:
            node: 要检查的节点

        Returns:
            节点是否健康
        """
        is_healthy = self._check_node_health(node)

        # 创建新的Node对象，更新健康状态
        updated_node = Node(
            id=node.id,
            address=node.address,
            weight=node.weight,
            current_connections=node.current_connections,
            health_status=HealthStatus.HEALTHY if is_healthy else HealthStatus.UNHEALTHY,
            last_health_check=time.time(),
            total_requests=node.total_requests,
            failed_requests=node.failed_requests
        )

        # 更新nodes列表中的节点
        for i, old_node in enumerate(self.nodes):
            if old_node.id == node.id:
                self.nodes[i] = updated_node
                # 更新healthy_nodes集合
                if old_node in self.healthy_nodes:
                    self.healthy_nodes.remove(old_node)
                if is_healthy:
                    self.healthy_nodes.add(updated_node)
                break

        return is_healthy

    async def check_node_health_async(self, node: Node) -> bool:
        """检查单个节点的健康状态

        Args:
            node: 要检查的节点

        Returns:
            节点是否健康
        """
        try:
            # 这里应该实现实际的健康检查逻辑
            # 比如发送HTTP请求到节点的健康检查端点
            # 现在我们使用模拟的健康检查

            # 模拟网络请求
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # 假设节点有 /health 端点
                health_url = f"http://{node.address}/health"
                async with session.get(health_url) as response:
                    is_healthy = response.status == 200

                    # 更新节点健康状态
                    node.health_status = HealthStatus.HEALTHY if is_healthy else HealthStatus.UNHEALTHY
                    node.last_health_check = time.time()

                    if is_healthy:
                        self.healthy_nodes.add(node)  # 添加Node对象
                    else:
                        self.healthy_nodes.discard(node)  # 移除Node对象

                    logger.debug(f"Health check for {node.id}: {'healthy' if is_healthy else 'unhealthy'}")
                    return is_healthy

        except Exception as e:
            # 健康检查失败，标记为不健康
            node.health_status = HealthStatus.UNHEALTHY
            node.last_health_check = time.time()
            self.healthy_nodes.discard(node)  # 移除Node对象

            # 增加失败计数
            node.failed_requests += 1

            logger.warning(f"Health check failed for {node.id}: {e}")
            return False

    def remove_unhealthy_nodes(self) -> int:
        """移除所有不健康的节点

        Returns:
            移除的节点数量
        """
        with self._lock:
            unhealthy_nodes = [node for node in self.nodes
                             if node.health_status != HealthStatus.HEALTHY]

            removed_count = 0
            for node in unhealthy_nodes:
                if self.remove_node(node.id):
                    removed_count += 1

            if removed_count > 0:
                logger.info(f"Removed {removed_count} unhealthy nodes")

            return removed_count

    def get_statistics(self) -> Dict[str, any]:
        """获取负载均衡器统计信息

        Returns:
            统计信息字典
        """
        with self._lock:
            total_requests = sum(node.total_requests for node in self.nodes)
            total_connections = sum(node.current_connections for node in self.nodes)
            total_weight = sum(node.weight for node in self.nodes)

            healthy_count = len([node for node in self.nodes
                               if node.health_status == HealthStatus.HEALTHY])

            avg_connections = total_connections / len(self.nodes) if self.nodes else 0

            # 计算平均成功率
            if self.nodes:
                avg_success_rate = sum(node.success_rate for node in self.nodes) / len(self.nodes)
            else:
                avg_success_rate = 0.0

            return {
                'total_nodes': len(self.nodes),
                'healthy_nodes': healthy_count,
                'total_weight': total_weight,
                'total_connections': total_connections,
                'avg_connections': avg_connections,
                'total_requests': total_requests,
                'avg_success_rate': avg_success_rate,
                'strategy': self.strategy,
                'current_index': self.current_index
            }

    def get_node_by_id(self, node_id: str) -> Optional[Node]:
        """根据ID获取节点

        Args:
            node_id: 节点ID

        Returns:
            节点对象或None
        """
        with self._lock:
            for node in self.nodes:
                if node.id == node_id:
                    return node
            return None

    def update_node_weight(self, node_id: str, weight: int) -> bool:
        """更新节点权重

        Args:
            node_id: 节点ID
            weight: 新权重

        Returns:
            是否成功更新
        """
        if weight <= 0:
            raise ValueError(f"Weight must be positive, got {weight}")

        with self._lock:
            node = self.get_node_by_id(node_id)
            if node:
                node.weight = weight
                logger.info(f"Updated node {node_id} weight to {weight}")
                return True
            return False

    def reset_statistics(self) -> None:
        """重置所有统计信息"""
        with self._lock:
            for node in self.nodes:
                node.total_requests = 0
                node.failed_requests = 0
                node.current_connections = 0
            self.current_index = 0
            logger.info("Reset all statistics")