"""
影响力最大化算法

实现贪心算法、度启发式和CELF算法用于找到网络中最有影响力的节点集合
"""

import random
import asyncio
import heapq
from typing import Dict, Set, List, Tuple, Optional
from .social_network import SocialNetwork


class InfluenceMaximization:
    """影响力最大化算法类"""

    def __init__(
        self,
        network: SocialNetwork,
        propagation_probability: float = 0.1,
        simulation_rounds: int = 100
    ):
        """
        初始化影响力最大化算法

        Args:
            network: 社交网络实例
            propagation_probability: 传播概率 (0-1)
            simulation_rounds: 模拟轮数

        Raises:
            ValueError: 当概率参数不在0-1范围内或模拟轮数无效时
        """
        if not 0 <= propagation_probability <= 1:
            raise ValueError("传播概率必须在0和1之间")
        if simulation_rounds <= 0:
            raise ValueError("模拟轮数必须大于0")

        self.network = network
        self.propagation_probability = propagation_probability
        self.simulation_rounds = simulation_rounds

        # 结果存储
        self.selected_seeds: List[str] = []
        self.estimated_influence: int = 0
        self.marginal_gains: List[float] = []

    def greedy_algorithm(self, seed_count: int) -> List[str]:
        """
        贪心算法选择种子节点

        Args:
            seed_count: 种子节点数量

        Returns:
            List[str]: 选中的种子节点ID列表

        Raises:
            ValueError: 当种子数量无效时
        """
        if seed_count <= 0:
            raise ValueError("种子数量必须大于0")

        all_agents = self.network.get_all_agents()
        if seed_count > len(all_agents):
            raise ValueError("种子数量不能超过总智能体数量")

        selected_seeds = []
        current_influence = 0

        for _ in range(seed_count):
            best_seed = None
            best_marginal_gain = 0

            # 尝试每个未选中的节点
            for agent in all_agents:
                if agent not in selected_seeds:
                    # 计算边际增益
                    test_seeds = selected_seeds + [agent]
                    influence = self.estimate_influence(test_seeds)
                    marginal_gain = influence - current_influence

                    if marginal_gain > best_marginal_gain:
                        best_marginal_gain = marginal_gain
                        best_seed = agent

            if best_seed is not None:
                selected_seeds.append(best_seed)
                current_influence += best_marginal_gain
                self.marginal_gains.append(best_marginal_gain)

        self.selected_seeds = selected_seeds
        self.estimated_influence = current_influence

        return selected_seeds

    def estimate_influence(self, seed_set: List[str]) -> int:
        """
        估计给定种子集合的影响力

        Args:
            seed_set: 种子节点集合

        Returns:
            int: 平均影响力大小
        """
        if not seed_set:
            return 0

        total_influenced = 0

        for _ in range(self.simulation_rounds):
            influenced = self.simulate_propagation(seed_set)
            total_influenced += len(influenced)

        return total_influenced // self.simulation_rounds

    def simulate_propagation(self, seeds: List[str]) -> Set[str]:
        """
        模拟传播过程

        Args:
            seeds: 种子节点列表

        Returns:
            Set[str]: 被影响的节点集合
        """
        if not seeds:
            return set()

        influenced = set(seeds)
        active = set(seeds)

        while active:
            new_active = set()

            for current_agent in active:
                neighbors = self.network.get_neighbors(current_agent)

                for neighbor in neighbors:
                    if neighbor not in influenced:
                        # 根据传播概率决定是否激活
                        if random.random() < self.propagation_probability:
                            new_active.add(neighbor)

            influenced.update(new_active)
            active = new_active

        return influenced

    def degree_heuristic(self, seed_count: int) -> List[str]:
        """
        度启发式算法选择种子节点

        Args:
            seed_count: 种子节点数量

        Returns:
            List[str]: 选中的种子节点ID列表
        """
        all_agents = self.network.get_all_agents()
        agent_degrees = []

        # 计算每个节点的度
        for agent in all_agents:
            neighbors = self.network.get_neighbors(agent)
            degree = len(neighbors)
            agent_degrees.append((-degree, agent))  # 使用负值用于最大堆

        # 选择度最高的节点
        heapq.heapify(agent_degrees)
        selected_seeds = []

        for _ in range(min(seed_count, len(agent_degrees))):
            if agent_degrees:
                _, agent = heapq.heappop(agent_degrees)
                selected_seeds.append(agent)

        self.selected_seeds = selected_seeds
        self.estimated_influence = self.estimate_influence(selected_seeds)

        return selected_seeds

    def celf_algorithm(self, seed_count: int) -> List[str]:
        """
        CELF (Cost-Effective Lazy Forward) 算法

        Args:
            seed_count: 种子节点数量

        Returns:
            List[str]: 选中的种子节点ID列表
        """
        all_agents = self.network.get_all_agents()
        selected_seeds = []
        current_influence = 0

        # 初始化优先队列
        priority_queue = []
        for agent in all_agents:
            influence = self.estimate_influence([agent])
            heapq.heappush(priority_queue, (-influence, agent, 0))  # (负影响力, 节点, 轮次)

        iteration = 1

        while len(selected_seeds) < seed_count and priority_queue:
            # 获取优先级最高的节点
            neg_influence, agent, last_round = heapq.heappop(priority_queue)
            current_influence_estimate = -neg_influence

            # 如果这个节点在之前的轮次中计算过，需要重新计算
            if last_round < iteration - 1:
                # 重新计算边际增益
                new_influence = self.estimate_influence(selected_seeds + [agent])
                marginal_gain = new_influence - current_influence
                heapq.heappush(priority_queue, (-new_influence, agent, iteration))
                continue

            # 选择这个节点
            selected_seeds.append(agent)
            current_influence = current_influence_estimate
            self.marginal_gains.append(current_influence_estimate - current_influence + self.estimate_influence([agent]))
            iteration += 1

        self.selected_seeds = selected_seeds
        self.estimated_influence = current_influence

        return selected_seeds

    def calculate_marginal_gain(self, agent: str, current_seed_set: List[str]) -> float:
        """
        计算添加一个节点的边际增益

        Args:
            agent: 要添加的节点
            current_seed_set: 当前种子集合

        Returns:
            float: 边际增益
        """
        current_influence = self.estimate_influence(current_seed_set)
        new_influence = self.estimate_influence(current_seed_set + [agent])
        return new_influence - current_influence

    def get_influence_statistics(self) -> Dict:
        """
        获取影响力统计信息

        Returns:
            Dict: 包含各种统计指标的字典
        """
        avg_marginal_gain = sum(self.marginal_gains) / len(self.marginal_gains) if self.marginal_gains else 0.0

        return {
            'total_seeds': len(self.selected_seeds),
            'estimated_influence': self.estimated_influence,
            'average_marginal_gain': avg_marginal_gain,
            'marginal_gains': self.marginal_gains.copy(),
            'seed_set': self.selected_seeds.copy()
        }

    def reset_influence_model(self) -> None:
        """重置影响力模型"""
        self.selected_seeds.clear()
        self.estimated_influence = 0
        self.marginal_gains.clear()

    def get_degree_centrality(self) -> Dict[str, float]:
        """
        计算度中心性

        Returns:
            Dict[str, float]: 每个节点的度中心性
        """
        all_agents = self.network.get_all_agents()
        total_agents = len(all_agents)
        degree_centrality = {}

        for agent in all_agents:
            neighbors = self.network.get_neighbors(agent)
            degree = len(neighbors)
            degree_centrality[agent] = degree / (total_agents - 1) if total_agents > 1 else 0.0

        return degree_centrality

    def get_betweenness_centrality(self) -> Dict[str, float]:
        """
        计算介数中心性（简化版）

        Returns:
            Dict[str, float]: 每个节点的介数中心性
        """
        all_agents = self.network.get_all_agents()
        betweenness = {agent: 0.0 for agent in all_agents}

        # 简化的介数中心性计算：计算节点作为最短路径中介的次数
        for source in all_agents:
            for target in all_agents:
                if source != target:
                    # 简单的最短路径（使用BFS）
                    path = self.find_shortest_path(source, target)
                    if len(path) > 2:
                        # 路径中间的节点获得介数加分
                        for node in path[1:-1]:
                            betweenness[node] += 1

        # 归一化
        total_pairs = len(all_agents) * (len(all_agents) - 1)
        if total_pairs > 0:
            for node in betweenness:
                betweenness[node] /= total_pairs

        return betweenness

    def find_shortest_path(self, source: str, target: str) -> List[str]:
        """
        查找最短路径（BFS）

        Args:
            source: 源节点
            target: 目标节点

        Returns:
            List[str]: 最短路径上的节点列表
        """
        if source == target:
            return [source]

        from collections import deque

        queue = deque([(source, [source])])
        visited = {source}

        while queue:
            current, path = queue.popleft()

            neighbors = self.network.get_neighbors(current)
            for neighbor in neighbors:
                if neighbor == target:
                    return path + [neighbor]

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return []  # 没有路径

    def compare_algorithms(self, seed_count: int) -> Dict[str, Dict]:
        """
        比较不同算法的性能

        Args:
            seed_count: 种子节点数量

        Returns:
            Dict[str, Dict]: 各算法的性能比较结果
        """
        results = {}

        # 贪心算法
        import time
        start_time = time.time()
        greedy_seeds = self.greedy_algorithm(seed_count)
        greedy_time = time.time() - start_time
        greedy_influence = self.estimated_influence

        results['greedy'] = {
            'seeds': greedy_seeds,
            'influence': greedy_influence,
            'time': greedy_time
        }

        # 度启发式
        self.reset_influence_model()
        start_time = time.time()
        degree_seeds = self.degree_heuristic(seed_count)
        degree_time = time.time() - start_time
        degree_influence = self.estimated_influence

        results['degree'] = {
            'seeds': degree_seeds,
            'influence': degree_influence,
            'time': degree_time
        }

        # CELF算法
        self.reset_influence_model()
        start_time = time.time()
        celf_seeds = self.celf_algorithm(seed_count)
        celf_time = time.time() - start_time
        celf_influence = self.estimated_influence

        results['celf'] = {
            'seeds': celf_seeds,
            'influence': celf_influence,
            'time': celf_time
        }

        # 添加比较信息
        results['comparison'] = {
            'best_influence': max(greedy_influence, degree_influence, celf_influence),
            'fastest_algorithm': min([
                ('greedy', greedy_time),
                ('degree', degree_time),
                ('celf', celf_time)
            ], key=lambda x: x[1])[0]
        }

        return results

    async def async_greedy_algorithm(self, seed_count: int) -> List[str]:
        """
        异步贪心算法

        Args:
            seed_count: 种子节点数量

        Returns:
            List[str]: 选中的种子节点ID列表
        """
        return await asyncio.get_event_loop().run_in_executor(
            None, self.greedy_algorithm, seed_count
        )

    def validate_seed_set(self, seeds: List[str]) -> bool:
        """
        验证种子集合的有效性

        Args:
            seeds: 种子节点列表

        Returns:
            bool: 是否有效
        """
        if not seeds:
            return False

        all_agents = set(self.network.get_all_agents())
        seed_set = set(seeds)

        # 检查所有种子节点是否都存在于网络中
        return seed_set.issubset(all_agents) and len(seeds) == len(seed_set)