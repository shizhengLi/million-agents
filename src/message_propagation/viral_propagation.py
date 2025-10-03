"""
病毒式传播模型

实现经典的SIR (Susceptible-Infected-Recovered) 病毒传播模型
用于模拟信息在社交网络中的病毒式传播过程
"""

import random
import asyncio
from datetime import datetime
from typing import Dict, Set, List, Tuple, Optional
from .social_network import SocialNetwork


class ViralPropagationModel:
    """病毒式传播模型类"""

    def __init__(
        self,
        network: SocialNetwork,
        infection_probability: float = 0.1,
        recovery_probability: float = 0.05,
        max_iterations: int = 100
    ):
        """
        初始化病毒式传播模型

        Args:
            network: 社交网络实例
            infection_probability: 感染概率 (0-1)
            recovery_probability: 恢复概率 (0-1)
            max_iterations: 最大迭代次数

        Raises:
            ValueError: 当概率参数不在0-1范围内或迭代次数无效时
        """
        if not 0 <= infection_probability <= 1:
            raise ValueError("感染概率必须在0和1之间")
        if not 0 <= recovery_probability <= 1:
            raise ValueError("恢复概率必须在0和1之间")
        if max_iterations <= 0:
            raise ValueError("最大迭代次数必须大于0")

        self.network = network
        self.infection_probability = infection_probability
        self.recovery_probability = recovery_probability
        self.max_iterations = max_iterations

        # 传播状态集合
        self.infected_agents: Set[str] = set()
        self.recovered_agents: Set[str] = set()
        self.susceptible_agents: Set[str] = set()

        # 传播记录
        self.infection_time: Dict[str, datetime] = {}
        self.recovery_time: Dict[str, datetime] = {}
        self.infection_source: Dict[str, str] = {}  # 感染源映射
        self.propagation_history: List[Dict] = []

        # 统计信息
        self.peak_infection = 0
        self.total_infected_count = 0

    def set_initial_infected(self, initial_agents: List[str]) -> None:
        """
        设置初始感染智能体

        Args:
            initial_agents: 初始感染智能体ID列表

        Raises:
            ValueError: 当初始智能体列表为空时
        """
        if not initial_agents:
            raise ValueError("至少需要指定一个初始感染智能体")

        current_time = datetime.now()

        for agent_id in set(initial_agents):  # 去重
            self.infected_agents.add(agent_id)
            self.infection_time[agent_id] = current_time
            self.infection_source[agent_id] = None  # 初始感染没有感染源

        self.peak_infection = len(self.infected_agents)

    def propagate_step(self) -> int:
        """
        执行一步传播

        Returns:
            int: 新感染的智能体数量

        Raises:
            ValueError: 当未设置初始感染智能体时
        """
        if not self.infected_agents and not self.infection_time:
            raise ValueError("需要先设置初始感染智能体")

        newly_infected = set()
        current_time = datetime.now()

        # 1. 感染阶段
        for infected_agent in list(self.infected_agents):
            neighbors = self.network.get_neighbors(infected_agent)

            for neighbor in neighbors:
                # 只感染易感智能体
                if (neighbor not in self.infected_agents and
                    neighbor not in self.recovered_agents and
                    neighbor not in newly_infected):

                    # 根据感染概率决定是否感染
                    if random.random() < self.infection_probability:
                        newly_infected.add(neighbor)
                        self.infection_time[neighbor] = current_time
                        self.infection_source[neighbor] = infected_agent

        # 2. 恢复阶段
        recovered = set()
        for infected_agent in list(self.infected_agents):
            # 根据恢复概率决定是否恢复
            if random.random() < self.recovery_probability:
                recovered.add(infected_agent)
                self.recovery_time[infected_agent] = current_time

        # 3. 更新状态集合
        self.infected_agents.update(newly_infected)
        self.infected_agents -= recovered
        self.recovered_agents.update(recovered)

        # 4. 更新统计信息
        self.total_infected_count += len(newly_infected)
        self.peak_infection = max(self.peak_infection, len(self.infected_agents))

        # 5. 记录传播历史
        history_record = {
            'timestamp': current_time,
            'infected': len(self.infected_agents),
            'recovered': len(self.recovered_agents),
            'new_infections': len(newly_infected),
            'new_recoveries': len(recovered)
        }
        self.propagation_history.append(history_record)

        return len(newly_infected)

    def propagate_full_simulation(self) -> List[Dict]:
        """
        执行完整的传播模拟

        Returns:
            List[Dict]: 传播历史记录列表
        """
        if not self.infected_agents and not self.infection_time:
            raise ValueError("需要先设置初始感染智能体")

        self.propagation_history = []

        for iteration in range(self.max_iterations):
            new_infections = self.propagate_step()

            # 如果没有新的感染且没有当前感染，提前结束
            if new_infections == 0 and len(self.infected_agents) == 0:
                break

        return self.propagation_history

    def get_infection_statistics(self) -> Dict:
        """
        获取感染统计信息

        Returns:
            Dict: 包含各种统计指标的字典
        """
        # 使用总感染数作为分母
        total_agents = max(self.total_infected_count, 1)

        return {
            'total_infected': len(self.infected_agents),
            'total_recovered': len(self.recovered_agents),
            'total_affected': self.total_infected_count,
            'peak_infection': self.peak_infection,
            'infection_rate': len(self.infected_agents) / total_agents,
            'recovery_rate': len(self.recovered_agents) / max(self.total_infected_count, 1),
            'propagation_steps': len(self.propagation_history)
        }

    def reset_simulation(self) -> None:
        """重置模拟状态"""
        self.infected_agents.clear()
        self.recovered_agents.clear()
        self.susceptible_agents.clear()
        self.infection_time.clear()
        self.recovery_time.clear()
        self.infection_source.clear()
        self.propagation_history.clear()
        self.peak_infection = 0
        self.total_infected_count = 0

    def get_propagation_tree(self) -> Dict:
        """
        获取传播树结构

        Returns:
            Dict: 传播树，键为智能体ID，值为直接感染的智能体列表
        """
        tree = {}

        for target, source in self.infection_source.items():
            if source is not None:  # 排除初始感染
                if source not in tree:
                    tree[source] = []
                tree[source].append(target)

        return tree

    def calculate_reproduction_number(self) -> float:
        """
        计算基本再生数 (R0)

        Returns:
            float: 平均每个感染智能体感染的智能体数量
        """
        if not self.infection_source:
            return 0.0

        # 统计每个感染源感染的智能体数量
        source_counts = {}
        for target, source in self.infection_source.items():
            if source is not None:
                source_counts[source] = source_counts.get(source, 0) + 1

        if not source_counts:
            return 0.0

        # 计算平均值
        return sum(source_counts.values()) / len(source_counts)

    def get_infection_time_series(self) -> List[Tuple[datetime, int]]:
        """
        获取感染时间序列

        Returns:
            List[Tuple[datetime, int]]: (时间, 感染数量) 的列表
        """
        return sorted(self.infection_time.items(), key=lambda x: x[1])

    def get_superspreaders(self, threshold: int = 5) -> List[str]:
        """
        获取超级传播者

        Args:
            threshold: 超级传播者阈值（感染的智能体数量）

        Returns:
            List[str]: 超级传播者ID列表
        """
        source_counts = {}
        for target, source in self.infection_source.items():
            if source is not None:
                source_counts[source] = source_counts.get(source, 0) + 1

        return [source for source, count in source_counts.items() if count >= threshold]

    async def async_propagate_simulation(self) -> List[Dict]:
        """
        异步执行完整的传播模拟

        Returns:
            List[Dict]: 传播历史记录列表
        """
        return await asyncio.get_event_loop().run_in_executor(
            None, self.propagate_full_simulation
        )

    def get_network_metrics(self) -> Dict:
        """
        获取网络传播指标

        Returns:
            Dict: 网络传播相关指标
        """
        if not self.propagation_history:
            return {
                'outbreak_size': 0,
                'outbreak_duration': 0,
                'peak_time': None,
                'growth_rate': 0.0,
                'reproduction_number': 0.0
            }

        # 计算爆发规模
        outbreak_size = self.total_infected_count

        # 计算爆发持续时间
        outbreak_duration = len(self.propagation_history)

        # 计算峰值时间
        peak_record = max(self.propagation_history, key=lambda x: x['infected'])
        peak_time = peak_record['timestamp']

        # 计算增长率
        if len(self.propagation_history) > 1:
            initial_infected = self.propagation_history[0]['new_infections']
            final_infected = self.total_infected_count
            growth_rate = (final_infected - initial_infected) / max(initial_infected, 1)
        else:
            growth_rate = 0.0

        return {
            'outbreak_size': outbreak_size,
            'outbreak_duration': outbreak_duration,
            'peak_time': peak_time,
            'growth_rate': growth_rate,
            'reproduction_number': self.calculate_reproduction_number()
        }