"""
社交推荐引擎实现

基于社交关系和用户影响力进行推荐的核心算法实现
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Set
import time
from collections import defaultdict, deque

from .models import (
    RecommendationResult, RecommendationItem
)


class SocialRecommendationEngine:
    """社交推荐引擎"""

    def __init__(self):
        """初始化社交推荐引擎"""
        self.social_network: Dict[str, Dict[str, float]] = {}
        self.user_influence: Dict[str, float] = {}
        self.trust_scores: Dict[Tuple[str, str], float] = {}
        self.recommendation_cache: Dict[str, RecommendationResult] = {}

    def add_social_connection(self, user_a: str, user_b: str, strength: float):
        """
        添加社交连接

        Args:
            user_a: 用户A的ID
            user_b: 用户B的ID
            strength: 连接强度 (0-1)

        Raises:
            ValueError: 当输入无效时
        """
        if not user_a or not user_b:
            raise ValueError("用户ID不能为空")

        if user_a == user_b:
            raise ValueError("用户不能与自己建立连接")

        if strength < 0 or strength > 1:
            raise ValueError(f"连接强度必须在0-1之间，当前值: {strength}")

        # 初始化用户网络（如果不存在）
        if user_a not in self.social_network:
            self.social_network[user_a] = {}
        if user_b not in self.social_network:
            self.social_network[user_b] = {}

        # 建立双向连接
        self.social_network[user_a][user_b] = strength
        self.social_network[user_b][user_a] = strength

        # 清空相关缓存
        self._clear_trust_cache_for_user(user_a)
        self._clear_trust_cache_for_user(user_b)

    def remove_social_connection(self, user_a: str, user_b: str):
        """
        移除社交连接

        Args:
            user_a: 用户A的ID
            user_b: 用户B的ID
        """
        if user_a in self.social_network and user_b in self.social_network[user_a]:
            del self.social_network[user_a][user_b]
        if user_b in self.social_network and user_a in self.social_network[user_b]:
            del self.social_network[user_b][user_a]

        # 清空相关缓存
        self._clear_trust_cache_for_user(user_a)
        self._clear_trust_cache_for_user(user_b)

    def update_user_influence(self, user_id: str, influence_score: float):
        """
        更新用户影响力

        Args:
            user_id: 用户ID
            influence_score: 影响力分数 (0-1)

        Raises:
            ValueError: 当分数无效时
        """
        if influence_score < 0 or influence_score > 1:
            raise ValueError(f"影响力分数必须在0-1之间，当前值: {influence_score}")

        self.user_influence[user_id] = influence_score
        # 清空相关推荐缓存
        if user_id in self.recommendation_cache:
            del self.recommendation_cache[user_id]

    def _calculate_social_influence(self, source_user: str, target_user: str,
                                  max_depth: int = 3) -> float:
        """
        计算社交影响力

        Args:
            source_user: 源用户
            target_user: 目标用户
            max_depth: 最大搜索深度

        Returns:
            float: 影响力分数
        """
        if source_user == target_user:
            return 1.0

        if source_user not in self.social_network:
            return 0.0

        # 使用BFS查找路径并计算影响力
        visited = set()
        queue = deque([(source_user, 1.0, 0)])  # (user, influence, depth)

        while queue:
            current_user, current_influence, depth = queue.popleft()

            if current_user in visited or depth >= max_depth:
                continue

            visited.add(current_user)

            if current_user == target_user:
                return current_influence

            if current_user in self.social_network:
                source_influence = self.user_influence.get(current_user, 0.5)

                for friend, connection_strength in self.social_network[current_user].items():
                    if friend not in visited:
                        # 影响力随距离衰减
                        new_influence = current_influence * connection_strength * source_influence
                        queue.append((friend, new_influence, depth + 1))

        return 0.0

    def _calculate_trust_score(self, user_a: str, user_b: str,
                             max_depth: int = 4) -> float:
        """
        计算信任分数

        Args:
            user_a: 用户A
            user_b: 用户B
            max_depth: 最大搜索深度

        Returns:
            float: 信任分数
        """
        if user_a == user_b:
            return 1.0

        # 检查缓存
        cache_key = tuple(sorted([user_a, user_b]))
        if cache_key in self.trust_scores:
            return self.trust_scores[cache_key]

        if user_a not in self.social_network:
            trust_score = 0.0
            self.trust_scores[cache_key] = trust_score
            return trust_score

        # 直接连接
        if user_b in self.social_network[user_a]:
            trust_score = self.social_network[user_a][user_b]
            self.trust_scores[cache_key] = trust_score
            return trust_score

        # 间接连接的信任度计算
        trust_score = 0.0
        visited = set()
        queue = deque([(user_a, 1.0, 0)])  # (user, trust, depth)

        while queue:
            current_user, current_trust, depth = queue.popleft()

            if current_user in visited or depth >= max_depth:
                continue

            visited.add(current_user)

            if current_user == user_b:
                trust_score = max(trust_score, current_trust)
                break

            if current_user in self.social_network:
                for friend, connection_strength in self.social_network[current_user].items():
                    if friend not in visited:
                        # 信任度随距离衰减
                        new_trust = current_trust * connection_strength * 0.8  # 衰减因子
                        queue.append((friend, new_trust, depth + 1))

        self.trust_scores[cache_key] = trust_score
        return trust_score

    def _get_friends_recommendations(self, user_id: str,
                                   friends_activities: Dict[str, Dict[str, float]],
                                   k: int) -> List[Tuple[str, float]]:
        """
        基于朋友行为获取推荐

        Args:
            user_id: 用户ID
            friends_activities: 朋友活动数据
            k: 推荐数量

        Returns:
            List[Tuple[str, float]]: 推荐列表
        """
        if user_id not in self.social_network:
            return []

        item_scores = defaultdict(float)
        total_weight = 0.0

        # 计算每个朋友的推荐权重
        for friend_id in self.social_network[user_id]:
            if friend_id in friends_activities:
                # 基于连接强度和朋友影响力计算权重
                connection_strength = self.social_network[user_id][friend_id]
                friend_influence = self.user_influence.get(friend_id, 0.5)
                weight = connection_strength * friend_influence

                total_weight += weight

                # 累加朋友评分的加权分数
                for item_id, rating in friends_activities[friend_id].items():
                    item_scores[item_id] += weight * rating

        # 归一化分数
        if total_weight > 0:
            for item_id in item_scores:
                item_scores[item_id] /= total_weight

        # 按分数排序
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:k]

    def generate_recommendations(self, user_id: str,
                               user_activities: Dict[str, Dict[str, float]],
                               k: int = 10) -> RecommendationResult:
        """
        为用户生成推荐

        Args:
            user_id: 用户ID
            user_activities: 用户活动数据 {user_id: {item_id: rating}}
            k: 推荐数量

        Returns:
            RecommendationResult: 推荐结果
        """
        # 简化缓存逻辑，避免哈希问题
        cache_key = f"{user_id}_{len(user_activities)}"
        if cache_key in self.recommendation_cache:
            cached_result = self.recommendation_cache[cache_key]
            # 返回缓存结果的前k个
            return RecommendationResult(
                user_id, "social_based", cached_result.items[:k]
            )

        # 获取用户已交互的物品
        user_items = set(user_activities.get(user_id, {}).keys())

        # 获取朋友推荐
        friends_recommendations = self._get_friends_recommendations(
            user_id, user_activities, k * 2  # 获取更多候选
        )

        # 获取基于影响力的推荐
        influence_recommendations = self._get_influence_based_recommendations(
            user_id, user_activities, k * 2
        )

        # 合并推荐结果
        combined_scores = defaultdict(float)

        # 处理朋友推荐
        for item_id, score in friends_recommendations:
            if item_id not in user_items:
                combined_scores[item_id] += score * 0.6  # 朋友推荐权重

        # 处理影响力推荐
        for item_id, score in influence_recommendations:
            if item_id not in user_items:
                combined_scores[item_id] += score * 0.4  # 影响力推荐权重

        # 生成推荐项
        recommendations = []
        for item_id, score in combined_scores.items():
            if score > 0:
                # 归一化到0-1范围
                normalized_score = min(score / 5.0, 1.0)
                recommendations.append(RecommendationItem(item_id, normalized_score))

        # 按分数排序
        recommendations.sort(key=lambda x: x.score, reverse=True)

        # 创建推荐结果
        result = RecommendationResult(user_id, "social_based", recommendations[:k])

        # 缓存结果
        self.recommendation_cache[cache_key] = result

        return result

    def _get_influence_based_recommendations(self, user_id: str,
                                           user_activities: Dict[str, Dict[str, float]],
                                           k: int) -> List[Tuple[str, float]]:
        """
        基于影响力获取推荐

        Args:
            user_id: 用户ID
            user_activities: 用户活动数据
            k: 推荐数量

        Returns:
            List[Tuple[str, float]]: 推荐列表
        """
        user_items = set(user_activities.get(user_id, {}).keys())
        item_scores = defaultdict(float)

        # 找到网络中有影响力的用户
        influential_users = [
            (uid, influence) for uid, influence in self.user_influence.items()
            if uid != user_id and influence > 0.5
        ]

        # 按影响力排序
        influential_users.sort(key=lambda x: x[1], reverse=True)

        # 基于影响力用户的交互推荐物品
        for inf_user_id, influence in influential_users[:10]:  # 只考虑前10个
            if inf_user_id in user_activities:
                social_influence = self._calculate_social_influence(user_id, inf_user_id)
                if social_influence > 0:
                    for item_id, rating in user_activities[inf_user_id].items():
                        if item_id not in user_items:
                            weight = influence * social_influence * rating
                            item_scores[item_id] += weight

        # 按分数排序
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:k]

    def get_influential_users(self, k: int = 10) -> List[Tuple[str, float]]:
        """
        获取有影响力的用户

        Args:
            k: 返回的用户数量

        Returns:
            List[Tuple[str, float]]: 有影响力用户列表
        """
        sorted_users = sorted(
            self.user_influence.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_users[:k]

    def get_user_social_network(self, user_id: str) -> Dict[str, float]:
        """
        获取用户社交网络

        Args:
            user_id: 用户ID

        Returns:
            Dict[str, float]: 用户的朋友列表和连接强度
        """
        return self.social_network.get(user_id, {}).copy()

    def clear_cache(self):
        """清空所有缓存"""
        self.trust_scores.clear()
        self.recommendation_cache.clear()

    def _clear_trust_cache_for_user(self, user_id: str):
        """清空特定用户的信任缓存"""
        keys_to_remove = [
            key for key in self.trust_scores.keys()
            if user_id in key
        ]
        for key in keys_to_remove:
            del self.trust_scores[key]

    def get_engine_stats(self) -> Dict[str, Any]:
        """获取引擎统计信息"""
        return {
            "num_users": len(self.social_network),
            "num_connections": sum(len(connections) for connections in self.social_network.values()) // 2,
            "num_influence_scores": len(self.user_influence),
            "cache_size": len(self.recommendation_cache) + len(self.trust_scores),
            "avg_connections_per_user": np.mean([
                len(connections) for connections in self.social_network.values()
            ]) if self.social_network else 0
        }

    def _get_social_paths(self, source_user: str, target_user: str,
                         max_depth: int = 4) -> List[List[str]]:
        """
        获取两个用户之间的社交路径

        Args:
            source_user: 源用户
            target_user: 目标用户
            max_depth: 最大路径深度

        Returns:
            List[List[str]]: 路径列表
        """
        if source_user == target_user:
            return [[]]

        if source_user not in self.social_network:
            return []

        paths = []
        visited_global = set()
        queue = deque([(source_user, [source_user])])  # (current_user, path)

        while queue:
            current_user, path = queue.popleft()

            if current_user in visited_global or len(path) > max_depth + 1:
                continue

            # 为每个路径维护独立的visited集合
            visited_path = set(path)

            if current_user == target_user:
                paths.append(path[1:])  # 排除源用户
                continue

            visited_global.add(current_user)

            if current_user in self.social_network:
                for friend in self.social_network[current_user]:
                    if friend not in visited_path:  # 避免循环
                        queue.append((friend, path + [friend]))

        return paths