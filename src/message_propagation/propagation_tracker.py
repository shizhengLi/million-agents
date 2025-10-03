"""
传播路径追踪器

用于追踪和分析消息在社交网络中的传播路径、
关键节点和传播模式
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Set, List, Tuple, Optional
from collections import defaultdict, deque
from .social_network import SocialNetwork


class PropagationTracker:
    """传播路径追踪器类"""

    def __init__(self, network: SocialNetwork):
        """
        初始化传播追踪器

        Args:
            network: 社交网络实例
        """
        self.network = network

        # 追踪数据
        self.propagation_paths: List[Tuple[str, str, datetime]] = []  # (源, 目标, 时间)
        self.infection_sources: Dict[str, List[str]] = defaultdict(list)  # 目标 -> 源列表
        self.infection_times: Dict[str, datetime] = {}  # 智能体 -> 感染时间
        self.propagation_edges: Set[Tuple[str, str]] = set()  # 传播边集合

        # 统计信息
        self.propagation_depth: Dict[str, int] = {}  # 智能体 -> 传播深度
        self.out_degrees: Dict[str, int] = defaultdict(int)  # 出度统计
        self.in_degrees: Dict[str, int] = defaultdict(int)  # 入度统计

    def track_infection(self, source: str, target: str) -> None:
        """
        追踪感染事件

        Args:
            source: 感染源智能体ID
            target: 被感染智能体ID

        Raises:
            ValueError: 当源或目标为空或相同时
        """
        if not source:
            raise ValueError("感染源不能为空")
        if not target:
            raise ValueError("感染目标不能为空")
        if source == target:
            raise ValueError("感染源和感染目标不能相同")

        current_time = datetime.now()

        # 记录感染路径
        self.propagation_paths.append((source, target, current_time))
        if source not in self.infection_sources[target]:
            self.infection_sources[target].append(source)
        self.infection_times[target] = current_time
        self.propagation_edges.add((source, target))

        # 更新统计信息
        self.out_degrees[source] += 1
        self.in_degrees[target] += 1

        # 计算传播深度
        source_depth = self.propagation_depth.get(source, 0)
        self.propagation_depth[target] = source_depth + 1

    def get_infection_chain(self, target: str, max_depth: int = 50) -> List[str]:
        """
        获取感染链

        Args:
            target: 目标智能体ID
            max_depth: 最大深度限制

        Returns:
            List[str]: 感染链上的智能体ID列表
        """
        if target not in self.infection_sources or not self.infection_sources[target]:
            return []

        chain = []
        current = target
        visited = set()
        depth = 0

        while current and current not in visited and depth < max_depth:
            chain.append(current)
            visited.add(current)
            sources = self.infection_sources.get(current)
            current = sources[0] if sources else None  # 取第一个感染源
            depth += 1

        return list(reversed(chain))

    def get_infection_tree(self, root: str, max_depth: int = 50) -> Dict:
        """
        获取感染树

        Args:
            root: 根节点ID
            max_depth: 最大深度限制

        Returns:
            Dict: 感染树结构
        """
        tree = {}
        visited = set()

        def build_tree(node: str, depth: int) -> Dict:
            if depth >= max_depth or node in visited:
                return {}

            visited.add(node)
            children = {}

            # 找到所有由node感染的节点
            for source, target in self.propagation_edges:
                if source == node and target not in visited:
                    children[target] = build_tree(target, depth + 1)

            # 只有当节点有子节点时才将其添加到树中
            if children:
                tree[node] = children

            return children

        build_tree(root, 0)
        return tree

    def get_propagation_statistics(self) -> Dict:
        """
        获取传播统计信息

        Returns:
            Dict: 包含各种统计指标的字典
        """
        total_infections = len(self.infection_sources)
        unique_sources = len(set(source for sources in self.infection_sources.values() for source in sources))
        max_depth = max(self.propagation_depth.values()) if self.propagation_depth else 0
        avg_out_degree = sum(self.out_degrees.values()) / max(len(self.out_degrees), 1)

        return {
            'total_infections': total_infections,
            'unique_sources': unique_sources,
            'propagation_depth': max_depth,
            'branching_factor': avg_out_degree,
            'max_out_degree': max(self.out_degrees.values()) if self.out_degrees else 0,
            'max_in_degree': max(self.in_degrees.values()) if self.in_degrees else 0
        }

    def find_infection_sources(self, target: str) -> List[str]:
        """
        找到感染目标的所有可能源

        Args:
            target: 目标智能体ID

        Returns:
            List[str]: 感染源ID列表
        """
        return self.infection_sources.get(target, [])

    def get_infection_time(self, agent: str) -> Optional[datetime]:
        """
        获取智能体的感染时间

        Args:
            agent: 智能体ID

        Returns:
            Optional[datetime]: 感染时间，如果未感染则返回None
        """
        return self.infection_times.get(agent)

    def calculate_propagation_speed(self) -> float:
        """
        计算传播速度（感染次数/秒）

        Returns:
            float: 传播速度
        """
        if len(self.propagation_paths) == 0:
            return 0.0

        # 计算时间范围
        times = [path[2] for path in self.propagation_paths]
        time_range = (max(times) - min(times)).total_seconds()

        if time_range == 0:
            # 如果所有感染同时发生，返回一个很高的速度表示瞬时传播
            return float('inf') if len(self.propagation_paths) > 0 else 0.0

        return len(self.propagation_paths) / time_range

    def get_critical_nodes(self, min_out_degree: int = 2) -> List[str]:
        """
        获取关键节点（传播枢纽）

        Args:
            min_out_degree: 最小出度阈值

        Returns:
            List[str]: 关键节点ID列表
        """
        critical_nodes = []
        for node, out_degree in self.out_degrees.items():
            if out_degree >= min_out_degree:
                critical_nodes.append(node)

        return sorted(critical_nodes, key=lambda x: self.out_degrees[x], reverse=True)

    def get_propagation_paths(self) -> List[Tuple[str, str]]:
        """
        获取所有传播路径

        Returns:
            List[Tuple[str, str]]: 传播路径列表
        """
        return [(source, target) for source, target, _ in self.propagation_paths]

    def reset_tracker(self) -> None:
        """重置追踪器状态"""
        self.propagation_paths.clear()
        self.infection_sources.clear()
        self.infection_times.clear()
        self.propagation_edges.clear()
        self.propagation_depth.clear()
        self.out_degrees.clear()
        self.in_degrees.clear()

    def get_infection_depth(self, agent: str) -> int:
        """
        获取智能体的感染深度

        Args:
            agent: 智能体ID

        Returns:
            int: 感染深度，如果未感染则返回0
        """
        return self.propagation_depth.get(agent, 0)

    def get_spread_patterns(self) -> Dict:
        """
        分析传播模式

        Returns:
            Dict: 传播模式分析结果
        """
        if not self.propagation_paths:
            return {
                'pattern_type': 'none',
                'breadth': 0,
                'depth': 0,
                'density': 0.0
            }

        # 计算传播广度（不同层级的节点数）
        depth_counts = defaultdict(int)
        for agent, depth in self.propagation_depth.items():
            depth_counts[depth] += 1

        breadth = len(depth_counts)
        max_depth = max(depth_counts.keys()) if depth_counts else 0

        # 计算网络密度
        total_possible_edges = len(self.infection_sources) * (len(self.infection_sources) - 1) / 2
        density = len(self.propagation_edges) / max(total_possible_edges, 1)

        # 判断传播模式
        # 检查是否为链式传播（每层只有一个节点）
        is_chain = all(count == 1 for count in depth_counts.values())
        if is_chain and max_depth >= 3:
            pattern_type = 'chain'  # 链式传播
        elif breadth >= 3 and max_depth <= 3:
            pattern_type = 'broadcast'  # 广播式传播
        elif breadth >= 2 and max_depth >= 3:
            pattern_type = 'cascade'  # 级联传播
        else:
            pattern_type = 'mixed'  # 混合传播

        return {
            'pattern_type': pattern_type,
            'breadth': breadth,
            'depth': max_depth,
            'density': density,
            'depth_distribution': dict(depth_counts)
        }

    def get_temporal_analysis(self) -> Dict:
        """
        获取时间分析

        Returns:
            Dict: 时间分析结果
        """
        if not self.propagation_paths:
            return {
                'total_duration': 0,
                'peak_time': None,
                'spreading_rate': 0.0
            }

        times = [path[2] for path in self.propagation_paths]
        start_time = min(times)
        end_time = max(times)

        # 计算每分钟的感染数量
        minute_counts = defaultdict(int)
        for _, _, time in self.propagation_paths:
            minute_key = time.replace(second=0, microsecond=0)
            minute_counts[minute_key] += 1

        # 找到峰值时间
        peak_time = max(minute_counts, key=minute_counts.get) if minute_counts else None
        peak_infections = minute_counts[peak_time] if peak_time else 0

        # 计算传播速率
        total_duration = (end_time - start_time).total_seconds()
        spreading_rate = len(self.propagation_paths) / max(total_duration, 1)

        return {
            'total_duration': total_duration,
            'start_time': start_time,
            'end_time': end_time,
            'peak_time': peak_time,
            'peak_infections': peak_infections,
            'spreading_rate': spreading_rate,
            'minute_distribution': dict(minute_counts)
        }

    def find_bottlenecks(self) -> List[str]:
        """
        找到传播瓶颈节点

        Returns:
            List[str]: 瓶颈节点ID列表
        """
        bottlenecks = []

        # 计算每个节点的介数中心性（简化版）
        betweenness = defaultdict(int)
        all_agents = set(self.infection_sources.keys()) | set(source for sources in self.infection_sources.values() for source in sources)

        for source in all_agents:
            for target in all_agents:
                if source != target:
                    path = self.get_infection_chain(target)
                    if len(path) > 2:
                        # 路径中间的节点
                        for node in path[1:-1]:
                            betweenness[node] += 1

        # 找到高介数中心性的节点
        if betweenness:
            avg_betweenness = sum(betweenness.values()) / len(betweenness)
            threshold = avg_betweenness * 1.5  # 高于平均值1.5倍
            bottlenecks = [node for node, score in betweenness.items() if score >= threshold]

        return sorted(bottlenecks, key=lambda x: betweenness[x], reverse=True)

    def get_network_metrics(self) -> Dict:
        """
        获取网络传播指标

        Returns:
            Dict: 网络传播指标
        """
        stats = self.get_propagation_statistics()
        patterns = self.get_spread_patterns()
        temporal = self.get_temporal_analysis()

        return {
            'propagation_stats': stats,
            'spread_patterns': patterns,
            'temporal_analysis': temporal,
            'critical_nodes': self.get_critical_nodes(),
            'bottlenecks': self.find_bottlenecks(),
            'network_health': self._calculate_network_health()
        }

    def _calculate_network_health(self) -> Dict:
        """
        计算网络健康状态

        Returns:
            Dict: 网络健康指标
        """
        if not self.propagation_paths:
            return {
                'status': 'inactive',
                'connectivity': 0.0,
                'efficiency': 0.0,
                'robustness': 0.0
            }

        # 连通性
        total_agents = len(set(self.infection_sources.keys()) | set(source for sources in self.infection_sources.values() for source in sources))
        connectivity = len(self.propagation_edges) / max(total_agents * (total_agents - 1) / 2, 1)

        # 效率（平均路径长度的倒数）
        total_path_length = sum(len(self.get_infection_chain(target)) for target in self.infection_sources.keys())
        avg_path_length = total_path_length / max(len(self.infection_sources), 1)
        efficiency = 1.0 / max(avg_path_length, 1)

        # 鲁棒性（关键节点比例）
        critical_count = len(self.get_critical_nodes())
        robustness = 1.0 - (critical_count / max(total_agents, 1))

        # 整体状态
        if connectivity > 0.7 and efficiency > 0.5:
            status = 'healthy'
        elif connectivity > 0.3 and efficiency > 0.2:
            status = 'moderate'
        else:
            status = 'fragile'

        return {
            'status': status,
            'connectivity': connectivity,
            'efficiency': efficiency,
            'robustness': robustness
        }

    async def async_track_infection(self, source: str, target: str) -> None:
        """
        异步追踪感染

        Args:
            source: 感染源智能体ID
            target: 被感染智能体ID
        """
        await asyncio.get_event_loop().run_in_executor(
            None, self.track_infection, source, target
        )

    def export_data(self) -> Dict:
        """
        导出追踪数据

        Returns:
            Dict: 完整的追踪数据
        """
        return {
            'propagation_paths': [
                (source, target, time.isoformat()) for source, target, time in self.propagation_paths
            ],
            'infection_sources': self.infection_sources,
            'infection_times': {k: v.isoformat() for k, v in self.infection_times.items()},
            'propagation_edges': list(self.propagation_edges),
            'propagation_depth': self.propagation_depth,
            'out_degrees': dict(self.out_degrees),
            'in_degrees': dict(self.in_degrees),
            'statistics': self.get_propagation_statistics(),
            'network_metrics': self.get_network_metrics()
        }