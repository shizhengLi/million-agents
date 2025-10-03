"""
信息扩散预测模型

实现独立级联模型(Independent Cascade)和线性阈值模型(Linear Threshold)
用于预测信息和创新在社交网络中的扩散过程
"""

import random
import asyncio
from datetime import datetime
from typing import Dict, Set, List, Tuple, Optional
from .social_network import SocialNetwork


class InformationDiffusionModel:
    """信息扩散预测模型类"""

    def __init__(
        self,
        network: SocialNetwork,
        adoption_probability: float = 0.1,
        abandon_probability: float = 0.05,
        max_time_steps: int = 100
    ):
        """
        初始化信息扩散模型

        Args:
            network: 社交网络实例
            adoption_probability: 采用概率 (0-1)
            abandon_probability: 放弃概率 (0-1)
            max_time_steps: 最大时间步数

        Raises:
            ValueError: 当概率参数不在0-1范围内或时间步数无效时
        """
        if not 0 <= adoption_probability <= 1:
            raise ValueError("采用概率必须在0和1之间")
        if not 0 <= abandon_probability <= 1:
            raise ValueError("放弃概率必须在0和1之间")
        if max_time_steps <= 0:
            raise ValueError("最大时间步数必须大于0")

        self.network = network
        self.adoption_probability = adoption_probability
        self.abandon_probability = abandon_probability
        self.max_time_steps = max_time_steps

        # 扩散状态集合
        self.adopted_agents: Set[str] = set()
        self.abandoned_agents: Set[str] = set()
        self.unaware_agents: Set[str] = set()

        # 扩散记录
        self.adoption_time: Dict[str, datetime] = {}
        self.abandon_time: Dict[str, datetime] = {}
        self.adoption_source: Dict[str, str] = {}  # 采用源映射
        self.diffusion_history: List[Dict] = []

        # 统计信息
        self.peak_adoption = 0
        self.total_adoption_count = 0
        self.influence_success: Dict[str, int] = {}  # 影响成功次数
        self.influence_attempts: Dict[str, int] = {}  # 影响尝试次数

    def set_initial_adopters(self, initial_adopters: List[str]) -> None:
        """
        设置初始采用者

        Args:
            initial_adopters: 初始采用者ID列表

        Raises:
            ValueError: 当初始采用者列表为空时
        """
        if not initial_adopters:
            raise ValueError("至少需要指定一个初始采用者")

        current_time = datetime.now()

        for agent_id in set(initial_adopters):  # 去重
            self.adopted_agents.add(agent_id)
            self.adoption_time[agent_id] = current_time
            self.adoption_source[agent_id] = None  # 初始采用者没有采用源

        self.peak_adoption = len(self.adopted_agents)

    def diffuse_step(self) -> int:
        """
        执行一步扩散

        Returns:
            int: 新采用的智能体数量

        Raises:
            ValueError: 当未设置初始采用者时
        """
        if not self.adopted_agents and not self.adoption_time:
            raise ValueError("需要先设置初始采用者")

        newly_adopted = set()
        current_time = datetime.now()

        # 1. 采用阶段
        for adopted_agent in list(self.adopted_agents):
            neighbors = self.network.get_neighbors(adopted_agent)

            for neighbor in neighbors:
                # 只影响未感知和未放弃的智能体
                if (neighbor not in self.adopted_agents and
                    neighbor not in self.abandoned_agents and
                    neighbor not in newly_adopted):

                    # 记录影响尝试
                    self.influence_attempts[adopted_agent] = \
                        self.influence_attempts.get(adopted_agent, 0) + 1

                    # 根据采用概率决定是否采用
                    if random.random() < self.adoption_probability:
                        newly_adopted.add(neighbor)
                        self.adoption_time[neighbor] = current_time
                        self.adoption_source[neighbor] = adopted_agent

                        # 记录影响成功
                        self.influence_success[adopted_agent] = \
                            self.influence_success.get(adopted_agent, 0) + 1

        # 2. 放弃阶段
        abandoned = set()
        for adopted_agent in list(self.adopted_agents):
            # 根据放弃概率决定是否放弃
            if random.random() < self.abandon_probability:
                abandoned.add(adopted_agent)
                self.abandon_time[adopted_agent] = current_time

        # 3. 更新状态集合
        self.adopted_agents.update(newly_adopted)
        self.adopted_agents -= abandoned
        self.abandoned_agents.update(abandoned)

        # 4. 更新统计信息
        self.total_adoption_count += len(newly_adopted)
        self.peak_adoption = max(self.peak_adoption, len(self.adopted_agents))

        # 5. 记录扩散历史
        history_record = {
            'timestamp': current_time,
            'adopted': len(self.adopted_agents),
            'abandoned': len(self.abandoned_agents),
            'new_adoptions': len(newly_adopted),
            'new_abandonments': len(abandoned),
            'cumulative_adoptions': self.total_adoption_count
        }
        self.diffusion_history.append(history_record)

        return len(newly_adopted)

    def predict_diffusion(self) -> List[Dict]:
        """
        预测完整的扩散过程

        Returns:
            List[Dict]: 扩散历史记录列表
        """
        if not self.adopted_agents and not self.adoption_time:
            raise ValueError("需要先设置初始采用者")

        self.diffusion_history = []

        for time_step in range(self.max_time_steps):
            new_adoptions = self.diffuse_step()

            # 如果没有新的采用且没有当前采用者，提前结束
            if new_adoptions == 0 and len(self.adopted_agents) == 0:
                break

        return self.diffusion_history

    def calculate_adoption_rate(self) -> float:
        """
        计算当前采用率

        Returns:
            float: 采用率 (0-1)
        """
        total_agents = self.network.get_agent_count()
        if total_agents == 0:
            return 0.0
        return len(self.adopted_agents) / total_agents

    def get_diffusion_speed(self) -> float:
        """
        计算扩散速度

        Returns:
            float: 平均每步新增采用者数量
        """
        if not self.diffusion_history:
            return 0.0

        total_new_adoptions = sum(record['new_adoptions'] for record in self.diffusion_history)
        return total_new_adoptions / len(self.diffusion_history)

    def predict_time_to_saturation(self, saturation_threshold: float = 0.95) -> int:
        """
        预测达到饱和的时间

        Args:
            saturation_threshold: 饱和阈值 (0-1)

        Returns:
            int: 达到饱和的时间步数，0表示未达到饱和
        """
        total_agents = self.network.get_agent_count()
        if total_agents == 0:
            return 0

        saturation_count = int(total_agents * saturation_threshold)

        for i, record in enumerate(self.diffusion_history):
            if record['cumulative_adoptions'] >= saturation_count:
                return i + 1

        return 0

    def get_influence_probability(self, agent_id: str) -> float:
        """
        获取智能体的影响力概率

        Args:
            agent_id: 智能体ID

        Returns:
            float: 影响力概率 (0-1)
        """
        attempts = self.influence_attempts.get(agent_id, 0)
        if attempts == 0:
            return 0.0

        successes = self.influence_success.get(agent_id, 0)
        return successes / attempts

    def get_diffusion_statistics(self) -> Dict:
        """
        获取扩散统计信息

        Returns:
            Dict: 包含各种统计指标的字典
        """
        return {
            'total_adopted': len(self.adopted_agents),
            'total_abandoned': len(self.abandoned_agents),
            'total_adoption_count': self.total_adoption_count,
            'peak_adoption': self.peak_adoption,
            'adoption_rate': self.calculate_adoption_rate(),
            'abandonment_rate': len(self.abandoned_agents) / max(self.total_adoption_count, 1),
            'diffusion_speed': self.get_diffusion_speed(),
            'diffusion_steps': len(self.diffusion_history),
            'time_to_saturation': self.predict_time_to_saturation()
        }

    def reset_diffusion(self) -> None:
        """重置扩散状态"""
        self.adopted_agents.clear()
        self.abandoned_agents.clear()
        self.unaware_agents.clear()
        self.adoption_time.clear()
        self.abandon_time.clear()
        self.adoption_source.clear()
        self.diffusion_history.clear()
        self.peak_adoption = 0
        self.total_adoption_count = 0
        self.influence_success.clear()
        self.influence_attempts.clear()

    def get_critical_mass(self) -> int:
        """
        计算关键规模

        Returns:
            int: 关键规模（当前采用者数量）
        """
        return len(self.adopted_agents)

    def get_adoption_time_series(self) -> List[Tuple[datetime, int]]:
        """
        获取采用时间序列

        Returns:
            List[Tuple[datetime, int]]: (时间, 采用数量) 的列表
        """
        return sorted(self.adoption_time.items(), key=lambda x: x[1])

    def get_influential_agents(self, threshold: float = 0.5) -> List[str]:
        """
        获取有影响力的智能体

        Args:
            threshold: 影响力阈值

        Returns:
            List[str]: 有影响力的智能体ID列表
        """
        influential = []
        for agent_id in self.influence_attempts:
            influence_prob = self.get_influence_probability(agent_id)
            if influence_prob >= threshold:
                influential.append(agent_id)

        return influential

    def calculate_diffusion_threshold(self) -> float:
        """
        计算扩散阈值

        Returns:
            float: 扩散阈值
        """
        if not self.diffusion_history:
            return 0.0

        # 简单的阈值计算：峰值采用率
        max_adopted = max(record['adopted'] for record in self.diffusion_history)
        total_agents = self.network.get_agent_count()

        return max_adopted / max(total_agents, 1)

    async def async_predict_diffusion(self) -> List[Dict]:
        """
        异步预测完整的扩散过程

        Returns:
            List[Dict]: 扩散历史记录列表
        """
        return await asyncio.get_event_loop().run_in_executor(
            None, self.predict_diffusion
        )

    def get_diffusion_network_metrics(self) -> Dict:
        """
        获取扩散网络指标

        Returns:
            Dict: 扩散网络相关指标
        """
        if not self.diffusion_history:
            return {
                'cascade_size': 0,
                'cascade_depth': 0,
                'branching_factor': 0.0,
                'influence_entropy': 0.0,
                'diffusion_threshold': 0.0,
                'critical_mass': 0
            }

        # 计算级联规模
        cascade_size = self.total_adoption_count

        # 计算级联深度
        cascade_depth = len(self.diffusion_history)

        # 计算分支因子
        total_influences = sum(self.influence_attempts.values())
        unique_influencers = len(self.influence_attempts)
        branching_factor = total_influences / max(unique_influencers, 1)

        # 计算影响熵（衡量影响分布的均匀性）
        if self.influence_success:
            import math
            total_successes = sum(self.influence_success.values())
            if total_successes > 0:
                entropy = -sum(
                    (count / total_successes) * math.log2(count / total_successes)
                    for count in self.influence_success.values()
                )
            else:
                entropy = 0.0
        else:
            entropy = 0.0

        return {
            'cascade_size': cascade_size,
            'cascade_depth': cascade_depth,
            'branching_factor': branching_factor,
            'influence_entropy': entropy,
            'diffusion_threshold': self.calculate_diffusion_threshold(),
            'critical_mass': self.get_critical_mass()
        }