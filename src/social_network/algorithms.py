"""
社交网络算法模块
"""

from typing import Dict, List, Tuple, Optional, Set
import numpy as np
import networkx as nx
from .graph import SocialNetworkGraph


class PageRankCalculator:
    """PageRank算法计算器"""

    def __init__(self, damping_factor: float = 0.85, max_iterations: int = 100, tolerance: float = 1e-6):
        """
        初始化PageRank计算器

        Args:
            damping_factor: 阻尼因子，通常为0.85
            max_iterations: 最大迭代次数
            tolerance: 收敛容差
        """
        self.damping_factor = damping_factor
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def calculate_pagerank(self,
                          graph: SocialNetworkGraph,
                          damping_factor: Optional[float] = None,
                          max_iterations: Optional[int] = None,
                          tolerance: Optional[float] = None) -> Dict[int, float]:
        """
        计算图中所有节点的PageRank值

        Args:
            graph: 社交网络图
            damping_factor: 阻尼因子（可选，覆盖初始化值）
            max_iterations: 最大迭代次数（可选，覆盖初始化值）
            tolerance: 收敛容差（可选，覆盖初始化值）

        Returns:
            Dict[int, float]: 节点ID到PageRank值的映射
        """
        # 使用参数或默认值
        alpha = damping_factor if damping_factor is not None else self.damping_factor
        max_iter = max_iterations if max_iterations is not None else self.max_iterations
        tol = tolerance if tolerance is not None else self.tolerance

        # 如果图为空，返回空字典
        if graph.get_agent_count() == 0:
            return {}

        # 如果只有一个节点，返回PageRank为1
        if graph.get_agent_count() == 1:
            only_agent = next(iter(graph.agents.keys()))
            return {only_agent: 1.0}

        # 获取图的邻接矩阵
        nodes = list(graph.agents.keys())
        n = len(nodes)
        node_index = {node: i for i, node in enumerate(nodes)}

        # 构建转移矩阵
        transition_matrix = self._build_transition_matrix(graph, nodes, node_index)

        # 初始化PageRank值（均匀分布）
        pagerank = np.ones(n) / n

        # 迭代计算PageRank
        for iteration in range(max_iter):
            old_pagerank = pagerank.copy()

            # PageRank迭代公式: PR = α * M * PR + (1-α) * e/n
            pagerank = alpha * transition_matrix @ pagerank + (1 - alpha) / n

            # 检查收敛
            if np.linalg.norm(pagerank - old_pagerank, 1) < tol:
                break

        # 转换为字典格式
        result = {nodes[i]: pagerank[i] for i in range(n)}

        return result

    def _build_transition_matrix(self,
                               graph: SocialNetworkGraph,
                               nodes: List[int],
                               node_index: Dict[int, int]) -> np.ndarray:
        """
        构建转移矩阵

        Args:
            graph: 社交网络图
            nodes: 节点列表
            node_index: 节点到索引的映射

        Returns:
            np.ndarray: 转移矩阵
        """
        n = len(nodes)
        transition_matrix = np.zeros((n, n))

        # 构建转移矩阵
        for i, node in enumerate(nodes):
            neighbors = graph.get_agent_friends(node)

            if not neighbors:
                # 悬挂节点：均匀转移到所有节点
                transition_matrix[:, i] = 1.0 / n
            else:
                # 正常节点：按权重分配到邻居
                total_weight = 0.0
                neighbor_weights = []

                for neighbor in neighbors:
                    weight = graph.get_friendship_strength(node, neighbor) or 1.0
                    neighbor_weights.append((neighbor, weight))
                    total_weight += weight

                # 分配转移概率
                for neighbor, weight in neighbor_weights:
                    j = node_index[neighbor]
                    transition_matrix[j, i] = weight / total_weight

        return transition_matrix

    def get_top_influential_agents(self,
                                 graph: SocialNetworkGraph,
                                 top_k: int = 10) -> List[Tuple[int, float]]:
        """
        获取影响力最大的Agent

        Args:
            graph: 社交网络图
            top_k: 返回前k个Agent

        Returns:
            List[Tuple[int, float]]: (agent_id, pagerank_score) 列表，按PageRank降序排列
        """
        rankings = self.calculate_pagerank(graph)

        # 按PageRank值降序排序
        sorted_rankings = sorted(rankings.items(), key=lambda x: x[1], reverse=True)

        return sorted_rankings[:top_k]

    def get_agent_influence_rank(self,
                               graph: SocialNetworkGraph,
                               agent_id: int) -> Tuple[int, float]:
        """
        获取特定Agent的影响力排名

        Args:
            graph: 社交网络图
            agent_id: Agent ID

        Returns:
            Tuple[int, float]: (排名, PageRank值)
        """
        if not graph.has_agent(agent_id):
            raise ValueError(f"Agent {agent_id} does not exist")

        rankings = self.calculate_pagerank(graph)

        # 按PageRank值降序排序
        sorted_rankings = sorted(rankings.items(), key=lambda x: x[1], reverse=True)

        # 找到指定Agent的排名
        for rank, (node_id, score) in enumerate(sorted_rankings, 1):
            if node_id == agent_id:
                return rank, score

        return 0, 0.0  # 理论上不应该到达这里


class CommunityDetector:
    """社区发现算法"""

    def __init__(self, resolution: float = 1.0):
        """
        初始化社区发现器

        Args:
            resolution: Louvain方法的分辨率参数，控制社区大小
        """
        self.resolution = resolution

    def detect_communities(self,
                          graph: SocialNetworkGraph,
                          resolution: Optional[float] = None) -> List[Set[int]]:
        """
        使用Louvain方法检测社区

        Args:
            graph: 社交网络图
            resolution: 分辨率参数（可选，覆盖初始化值）

        Returns:
            List[Set[int]]: 社区列表，每个社区是一个节点ID集合
        """
        if graph.get_agent_count() == 0:
            return []

        if graph.get_agent_count() == 1:
            only_agent = next(iter(graph.agents.keys()))
            return [{only_agent}]

        # 使用参数或默认值
        res = resolution if resolution is not None else self.resolution

        try:
            # 尝试使用networkx的community检测
            import networkx.algorithms.community as nx_community

            # 转换为networkx图进行社区检测
            communities = nx_community.louvain_communities(
                graph.graph,
                resolution=res,
                seed=42
            )

            return [set(community) for community in communities]

        except ImportError:
            # 如果networkx community模块不可用，使用简单的连通分量算法
            return self._detect_connected_components(graph)

    def _detect_connected_components(self, graph: SocialNetworkGraph) -> List[Set[int]]:
        """
        使用连通分量作为社区（备用方法）

        Args:
            graph: 社交网络图

        Returns:
            List[Set[int]]: 连通分量列表
        """
        return graph.get_connected_components()

    def get_community_statistics(self,
                                graph: SocialNetworkGraph,
                                communities: List[Set[int]]) -> Dict[str, any]:
        """
        获取社区统计信息

        Args:
            graph: 社交网络图
            communities: 社区列表

        Returns:
            Dict[str, any]: 统计信息
        """
        if not communities:
            return {
                'num_communities': 0,
                'community_sizes': [],
                'largest_community_size': 0,
                'smallest_community_size': 0,
                'average_community_size': 0.0
            }

        community_sizes = [len(community) for community in communities]

        return {
            'num_communities': len(communities),
            'community_sizes': community_sizes,
            'largest_community_size': max(community_sizes),
            'smallest_community_size': min(community_sizes),
            'average_community_size': sum(community_sizes) / len(communities)
        }

    def get_agent_community_assignment(self, communities: List[Set[int]]) -> Dict[int, int]:
        """
        获取每个Agent的社区分配

        Args:
            communities: 社区列表

        Returns:
            Dict[int, int]: Agent ID到社区ID的映射
        """
        assignment = {}
        for community_id, community in enumerate(communities):
            for agent_id in community:
                assignment[agent_id] = community_id
        return assignment


class ShortestPathCalculator:
    """最短路径算法"""

    def calculate_shortest_path(self,
                                graph: SocialNetworkGraph,
                                start_agent: int,
                                end_agent: int,
                                use_weights: bool = False) -> Optional[List[int]]:
        """
        计算两个Agent之间的最短路径

        Args:
            graph: 社交网络图
            start_agent: 起始Agent ID
            end_agent: 目标Agent ID
            use_weights: 是否使用权重（关系强度）

        Returns:
            Optional[List[int]]: 最短路径（节点ID列表），如果不存在路径则返回None
        """
        # 检查节点是否存在
        if not graph.has_agent(start_agent) or not graph.has_agent(end_agent):
            return None

        # 如果起点和终点相同
        if start_agent == end_agent:
            return [start_agent]

        try:
            if use_weights:
                # 使用Dijkstra算法计算加权最短路径
                # 注意：在社交网络中，高权重意味着强关系，我们想要强关系的路径
                # 所以使用 1/weight 作为距离
                length, path = nx.single_source_dijkstra(
                    graph.graph,
                    start_agent,
                    target=end_agent,
                    weight=lambda u, v, d: 1.0 / d.get('weight', 1.0)
                )
                return path
            else:
                # 使用BFS计算无权重最短路径
                return nx.shortest_path(graph.graph, start_agent, end_agent)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def get_all_shortest_paths(self,
                               graph: SocialNetworkGraph,
                               use_weights: bool = False) -> Dict[Tuple[int, int], List[int]]:
        """
        获取所有节点对之间的最短路径

        Args:
            graph: 社交网络图
            use_weights: 是否使用权重

        Returns:
            Dict[Tuple[int, int], List[int]]: 所有节点对的最短路径
        """
        all_paths = {}
        nodes = list(graph.agents.keys())

        for i, start_node in enumerate(nodes):
            for end_node in nodes[i:]:
                if start_node != end_node:
                    path = self.calculate_shortest_path(
                        graph,
                        start_node,
                        end_node,
                        use_weights
                    )
                    if path:
                        all_paths[(start_node, end_node)] = path
                        all_paths[(end_node, start_node)] = path[::-1]

        return all_paths

    def get_path_length(self, path: List[int]) -> int:
        """
        获取路径长度（边的数量）

        Args:
            path: 路径（节点ID列表）

        Returns:
            int: 路径长度
        """
        if not path or len(path) <= 1:
            return 0
        return len(path) - 1

    def get_path_weight(self,
                        graph: SocialNetworkGraph,
                        path: List[int]) -> float:
        """
        获取路径的总权重

        Args:
            graph: 社交网络图
            path: 路径（节点ID列表）

        Returns:
            float: 路径总权重
        """
        if not path or len(path) <= 1:
            return 0.0

        total_weight = 0.0
        for i in range(len(path) - 1):
            weight = graph.get_friendship_strength(path[i], path[i + 1])
            total_weight += weight or 1.0

        return total_weight

    def calculate_average_path_length(self, graph: SocialNetworkGraph) -> float:
        """
        计算图的平均路径长度

        Args:
            graph: 社交网络图

        Returns:
            float: 平均路径长度
        """
        if graph.get_agent_count() <= 1:
            return 0.0

        try:
            # 使用networkx计算平均路径长度
            if nx.is_connected(graph.graph):
                return nx.average_shortest_path_length(graph.graph, weight='weight')
            else:
                # 对于不连通的图，计算各个连通分量的平均值
                components = list(nx.connected_components(graph.graph))
                total_length = 0.0
                total_pairs = 0

                for component in components:
                    if len(component) > 1:
                        subgraph = graph.graph.subgraph(component)
                        avg_length = nx.average_shortest_path_length(subgraph)
                        total_length += avg_length * (len(component) * (len(component) - 1))
                        total_pairs += len(component) * (len(component) - 1)

                return total_length / total_pairs if total_pairs > 0 else 0.0

        except (nx.NetworkXError, ZeroDivisionError):
            return 0.0

    def get_diameter(self, graph: SocialNetworkGraph) -> int:
        """
        获取图的直径（最长最短路径的长度）

        Args:
            graph: 社交网络图

        Returns:
            int: 图的直径
        """
        if graph.get_agent_count() <= 1:
            return 0

        try:
            # 使用networkx计算直径
            if nx.is_connected(graph.graph):
                return nx.diameter(graph.graph)
            else:
                # 对于不连通的图，计算各个连通分量的最大直径
                components = list(nx.connected_components(graph.graph))
                max_diameter = 0

                for component in components:
                    if len(component) > 1:
                        subgraph = graph.graph.subgraph(component)
                        diameter = nx.diameter(subgraph)
                        max_diameter = max(max_diameter, diameter)

                return max_diameter

        except (nx.NetworkXError):
            return 0

    def get_centrality_measures(self,
                                graph: SocialNetworkGraph) -> Dict[int, Dict[str, float]]:
        """
        获取节点的中心性度量

        Args:
            graph: 社交网络图

        Returns:
            Dict[int, Dict[str, float]]: 每个节点的中心性度量
        """
        centrality_measures = {}

        try:
            # 度中心性
            degree_centrality = nx.degree_centrality(graph.graph)

            # 接近中心性
            closeness_centrality = nx.closeness_centrality(graph.graph)

            # 介数中心性
            betweenness_centrality = nx.betweenness_centrality(graph.graph)

            # 特征向量中心性
            try:
                eigenvector_centrality = nx.eigenvector_centrality(graph.graph)
            except nx.NetworkXError:
                eigenvector_centrality = {node: 0.0 for node in graph.graph.nodes()}

            # 组合所有中心性度量
            for node in graph.graph.nodes():
                centrality_measures[node] = {
                    'degree_centrality': degree_centrality[node],
                    'closeness_centrality': closeness_centrality[node],
                    'betweenness_centrality': betweenness_centrality[node],
                    'eigenvector_centrality': eigenvector_centrality[node]
                }

        except Exception:
            # 如果计算失败，返回默认值
            for node in graph.graph.nodes():
                centrality_measures[node] = {
                    'degree_centrality': 0.0,
                    'closeness_centrality': 0.0,
                    'betweenness_centrality': 0.0,
                    'eigenvector_centrality': 0.0
                }

        return centrality_measures