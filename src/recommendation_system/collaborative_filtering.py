"""
协同过滤推荐引擎实现

基于用户行为数据进行推荐的核心算法实现
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import time
from collections import defaultdict

from .models import (
    UserItemMatrix, RecommendationResult, RecommendationItem,
    Interaction
)


class CollaborativeFilteringEngine:
    """协同过滤推荐引擎"""

    def __init__(self):
        """初始化协同过滤引擎"""
        self.user_item_matrix: Optional[UserItemMatrix] = None
        self.similarity_matrix: Optional[np.ndarray] = None
        self.user_similarity_cache: Dict[Tuple[str, str], float] = {}
        self.item_similarity_cache: Dict[Tuple[str, str], float] = {}

    def build_user_item_matrix(self, interactions: List[Tuple[str, str, float]]) -> UserItemMatrix:
        """
        构建用户-物品矩阵

        Args:
            interactions: 交互数据列表，格式为 (user_id, item_id, rating)

        Returns:
            UserItemMatrix: 构建好的用户-物品矩阵

        Raises:
            ValueError: 当交互数据为空时
        """
        if not interactions:
            raise ValueError("不能从空数据构建矩阵")

        # 验证评分范围
        for user_id, item_id, rating in interactions:
            if rating < 1.0 or rating > 5.0:
                raise ValueError(f"评分必须在1.0-5.0范围内，当前值: {rating}")

        # 收集所有用户和物品
        user_ids = list(sorted(set(user_id for user_id, _, _ in interactions)))
        item_ids = list(sorted(set(item_id for _, item_id, _ in interactions)))

        # 创建用户-物品映射
        user_index_map = {user_id: idx for idx, user_id in enumerate(user_ids)}
        item_index_map = {item_id: idx for idx, item_id in enumerate(item_ids)}

        # 初始化矩阵
        matrix_data = np.zeros((len(user_ids), len(item_ids)))

        # 填充矩阵数据（重复评分使用最新值）
        for user_id, item_id, rating in interactions:
            user_idx = user_index_map[user_id]
            item_idx = item_index_map[item_id]
            matrix_data[user_idx, item_idx] = rating

        self.user_item_matrix = UserItemMatrix(matrix_data, user_ids, item_ids)
        return self.user_item_matrix

    def build_user_similarity_matrix(self, matrix: UserItemMatrix,
                                  method: str = "cosine") -> np.ndarray:
        """
        构建用户相似度矩阵

        Args:
            matrix: 用户-物品矩阵
            method: 相似度计算方法 ("cosine" 或 "pearson")

        Returns:
            np.ndarray: 用户相似度矩阵
        """
        num_users = len(matrix.user_ids)
        similarity_matrix = np.eye(num_users)  # 对角线为1

        # 计算用户之间的相似度
        for i in range(num_users):
            for j in range(i + 1, num_users):
                user_i_vector = matrix.get_user_vector(matrix.user_ids[i])
                user_j_vector = matrix.get_user_vector(matrix.user_ids[j])

                if method == "cosine":
                    similarity = self._calculate_cosine_similarity(user_i_vector, user_j_vector)
                elif method == "pearson":
                    similarity = self._calculate_pearson_correlation(user_i_vector, user_j_vector)
                else:
                    raise ValueError(f"不支持的相似度计算方法: {method}")

                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity

        self.similarity_matrix = similarity_matrix
        return similarity_matrix

    def _calculate_cosine_similarity(self, vector_a: np.ndarray, vector_b: np.ndarray) -> float:
        """计算余弦相似度"""
        # 找到两个向量都非零的位置
        mask = (vector_a > 0) & (vector_b > 0)

        if not np.any(mask):
            return 0.0  # 没有共同评分，相似度为0

        # 只在有共同评分的位置计算相似度
        a_common = vector_a[mask]
        b_common = vector_b[mask]

        # 计算余弦相似度
        dot_product = np.dot(a_common, b_common)
        norm_a = np.linalg.norm(a_common)
        norm_b = np.linalg.norm(b_common)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def _calculate_pearson_correlation(self, vector_a: np.ndarray, vector_b: np.ndarray) -> float:
        """计算皮尔逊相关系数"""
        # 找到两个向量都非零的位置
        mask = (vector_a > 0) & (vector_b > 0)

        if not np.any(mask):
            return 0.0  # 没有共同评分，相关系数为0

        if len(np.where(mask)[0]) < 2:  # 至少需要2个共同点
            return 0.0

        # 只在有共同评分的位置计算相关系数
        a_common = vector_a[mask]
        b_common = vector_b[mask]

        # 计算皮尔逊相关系数
        mean_a = np.mean(a_common)
        mean_b = np.mean(b_common)

        numerator = np.sum((a_common - mean_a) * (b_common - mean_b))
        denominator = np.sqrt(np.sum((a_common - mean_a) ** 2)) * np.sqrt(np.sum((b_common - mean_b) ** 2))

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def find_similar_users(self, target_user: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        找到与目标用户最相似的用户

        Args:
            target_user: 目标用户ID
            k: 返回的相似用户数量

        Returns:
            List[Tuple[str, float]]: 相似用户列表，格式为 (user_id, similarity_score)

        Raises:
            ValueError: 当用户不存在或相似度矩阵未构建时
        """
        if self.user_item_matrix is None:
            raise ValueError("用户-物品矩阵未构建")

        if self.similarity_matrix is None:
            raise ValueError("用户相似度矩阵未构建")

        if target_user not in self.user_item_matrix.user_ids:
            raise ValueError(f"用户 {target_user} 不存在")

        target_user_idx = self.user_item_matrix.get_user_index(target_user)
        similarities = self.similarity_matrix[target_user_idx, :]

        # 获取相似度排序（排除自己）
        similar_indices = np.argsort(similarities)[::-1]

        similar_users = []
        for idx in similar_indices:
            if idx != target_user_idx:  # 排除自己
                user_id = self.user_item_matrix.user_ids[idx]
                similarity = similarities[idx]
                similar_users.append((user_id, similarity))

                if len(similar_users) >= k:
                    break

        return similar_users

    def generate_recommendations(self, user_id: str, method: str = "user_based",
                               k: int = 10) -> RecommendationResult:
        """
        为用户生成推荐

        Args:
            user_id: 用户ID
            method: 推荐方法 ("user_based" 或 "item_based")
            k: 推荐数量

        Returns:
            RecommendationResult: 推荐结果

        Raises:
            ValueError: 当用户不存在或方法不支持时
        """
        if self.user_item_matrix is None:
            raise ValueError("用户-物品矩阵未构建")

        if user_id not in self.user_item_matrix.user_ids:
            raise ValueError(f"用户 {user_id} 不存在")

        if method == "user_based":
            return self._generate_user_based_recommendations(user_id, k)
        elif method == "item_based":
            return self._generate_item_based_recommendations(user_id, k)
        else:
            raise ValueError(f"不支持的推荐方法: {method}")

    def _generate_user_based_recommendations(self, user_id: str, k: int) -> RecommendationResult:
        """生成基于用户的推荐"""
        if self.similarity_matrix is None:
            self.build_user_similarity_matrix(self.user_item_matrix)

        # 找到相似用户
        similar_users = self.find_similar_users(user_id, k=min(20, len(self.user_item_matrix.user_ids)))

        if not similar_users:
            return RecommendationResult(user_id, "user_based", [])

        # 获取用户已评分的物品
        user_rated_items = set(self.user_item_matrix.get_user_rated_items(user_id))

        # 计算候选物品的推荐分数
        item_scores = defaultdict(float)
        item_weights = defaultdict(float)

        for similar_user_id, similarity in similar_users:
            if similarity <= 0:
                continue  # 跳过不相似的用户

            similar_user_rated_items = self.user_item_matrix.get_user_rated_items(similar_user_id)

            for item_id in similar_user_rated_items:
                if item_id not in user_rated_items:  # 只推荐未评分的物品
                    rating = self.user_item_matrix.get_rating(similar_user_id, item_id)
                    item_scores[item_id] += similarity * rating
                    item_weights[item_id] += abs(similarity)

        # 计算最终推荐分数
        recommendations = []
        for item_id, score in item_scores.items():
            if item_weights[item_id] > 0:
                final_score = score / item_weights[item_id]
                # 归一化到0-1范围
                final_score = min(max(final_score / 5.0, 0.0), 1.0)
                recommendations.append(RecommendationItem(item_id, final_score))

        # 按分数排序并返回前k个
        recommendations.sort(key=lambda x: x.score, reverse=True)
        return RecommendationResult(user_id, "user_based", recommendations[:k])

    def _generate_item_based_recommendations(self, user_id: str, k: int) -> RecommendationResult:
        """生成基于物品的推荐"""
        # 这里简化实现，实际中需要构建物品相似度矩阵
        user_rated_items = self.user_item_matrix.get_user_rated_items(user_id)

        if not user_rated_items:
            return RecommendationResult(user_id, "item_based", [])

        # 基于用户已评分物品找相似物品
        recommendations = []
        user_rated_items_set = set(user_rated_items)

        # 找到与用户已评分物品相似的其他物品
        for rated_item in user_rated_items:
            similar_items = self._find_similar_items(rated_item, k=min(10, len(self.user_item_matrix.item_ids)))

            for similar_item, similarity in similar_items:
                if similar_item not in user_rated_items_set:
                    # 使用用户对原物品的评分和相似度计算推荐分数
                    user_rating = self.user_item_matrix.get_rating(user_id, rated_item)
                    score = (user_rating / 5.0) * similarity  # 归一化
                    recommendations.append(RecommendationItem(similar_item, score))

        # 去重并排序
        unique_recommendations = {}
        for rec in recommendations:
            if rec.item_id not in unique_recommendations or rec.score > unique_recommendations[rec.item_id].score:
                unique_recommendations[rec.item_id] = rec

        final_recommendations = sorted(unique_recommendations.values(), key=lambda x: x.score, reverse=True)
        return RecommendationResult(user_id, "item_based", final_recommendations[:k])

    def _find_similar_items(self, target_item: str, k: int) -> List[Tuple[str, float]]:
        """找到与目标物品相似的物品"""
        if self.user_item_matrix is None:
            return []

        target_item_idx = self.user_item_matrix.get_item_index(target_item)
        target_vector = self.user_item_matrix.get_item_vector(target_item)

        similarities = []
        for idx, item_id in enumerate(self.user_item_matrix.item_ids):
            if item_id != target_item:
                item_vector = self.user_item_matrix.get_item_vector(item_id)
                similarity = self._calculate_cosine_similarity(target_vector, item_vector)
                similarities.append((item_id, similarity))

        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

    def clear_cache(self):
        """清空缓存"""
        self.user_similarity_cache.clear()
        self.item_similarity_cache.clear()

    def get_engine_stats(self) -> Dict[str, Any]:
        """获取引擎统计信息"""
        if self.user_item_matrix is None:
            return {"status": "未初始化"}

        num_users, num_items = self.user_item_matrix.shape
        total_interactions = np.sum(self.user_item_matrix.data > 0)
        sparsity = 1 - (total_interactions / (num_users * num_items))

        return {
            "status": "已初始化",
            "num_users": num_users,
            "num_items": num_items,
            "total_interactions": int(total_interactions),
            "sparsity": float(sparsity),
            "similarity_matrix_built": self.similarity_matrix is not None,
            "cache_size": len(self.user_similarity_cache) + len(self.item_similarity_cache)
        }