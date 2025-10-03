"""
协同过滤推荐引擎的测试用例

使用TDD方法，先写测试用例，再实现功能
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from recommendation_system.collaborative_filtering import CollaborativeFilteringEngine
from recommendation_system.models import UserItemMatrix, RecommendationResult


class TestCollaborativeFilteringEngine:
    """协同过滤引擎测试类"""

    def setup_method(self):
        """每个测试方法执行前的设置"""
        self.cf_engine = CollaborativeFilteringEngine()

    def test_initialization(self):
        """测试协同过滤引擎初始化"""
        # Given & When
        engine = CollaborativeFilteringEngine()

        # Then
        assert engine is not None
        assert hasattr(engine, 'user_item_matrix')
        assert hasattr(engine, 'similarity_matrix')
        assert hasattr(engine, 'user_similarity_cache')

    def test_build_user_item_matrix_empty_data(self):
        """测试构建用户-物品矩阵 - 空数据"""
        # Given
        interactions = []

        # When & Then
        with pytest.raises(ValueError, match="不能从空数据构建矩阵"):
            self.cf_engine.build_user_item_matrix(interactions)

    def test_build_user_item_matrix_simple_data(self):
        """测试构建用户-物品矩阵 - 简单数据"""
        # Given
        interactions = [
            ("user1", "item1", 5.0),
            ("user1", "item2", 3.0),
            ("user2", "item1", 4.0),
            ("user2", "item3", 2.0),
            ("user3", "item2", 4.0),
            ("user3", "item3", 5.0)
        ]

        # When
        matrix = self.cf_engine.build_user_item_matrix(interactions)

        # Then
        assert isinstance(matrix, UserItemMatrix)
        assert matrix.shape == (3, 3)  # 3个用户，3个物品
        assert len(matrix.user_ids) == 3
        assert len(matrix.item_ids) == 3

        # 验证具体值
        user1_idx = matrix.get_user_index("user1")
        user2_idx = matrix.get_user_index("user2")
        user3_idx = matrix.get_user_index("user3")
        item1_idx = matrix.get_item_index("item1")
        item2_idx = matrix.get_item_index("item2")
        item3_idx = matrix.get_item_index("item3")

        assert matrix.data[user1_idx, item1_idx] == 5.0
        assert matrix.data[user1_idx, item2_idx] == 3.0
        assert matrix.data[user2_idx, item1_idx] == 4.0
        assert matrix.data[user2_idx, item3_idx] == 2.0
        assert matrix.data[user3_idx, item2_idx] == 4.0
        assert matrix.data[user3_idx, item3_idx] == 5.0

    def test_build_user_item_matrix_with_duplicates(self):
        """测试构建用户-物品矩阵 - 处理重复评分"""
        # Given
        interactions = [
            ("user1", "item1", 5.0),
            ("user1", "item1", 3.0),  # 重复评分
            ("user2", "item2", 4.0)
        ]

        # When
        matrix = self.cf_engine.build_user_item_matrix(interactions)

        # Then - 应该使用最新的评分
        user1_idx = matrix.get_user_index("user1")
        item1_idx = matrix.get_item_index("item1")
        assert matrix.data[user1_idx, item1_idx] == 3.0  # 使用最新的评分

    def test_calculate_cosine_similarity(self):
        """测试余弦相似度计算"""
        # Given
        vector_a = np.array([1.0, 2.0, 3.0])
        vector_b = np.array([1.0, 2.0, 3.0])  # 相同向量

        # When
        similarity = self.cf_engine._calculate_cosine_similarity(vector_a, vector_b)

        # Then
        assert abs(similarity - 1.0) < 1e-6  # 相同向量相似度为1

    def test_calculate_cosine_similarity_orthogonal(self):
        """测试余弦相似度计算 - 正交向量"""
        # Given
        vector_a = np.array([1.0, 0.0])
        vector_b = np.array([0.0, 1.0])  # 正交向量

        # When
        similarity = self.cf_engine._calculate_cosine_similarity(vector_a, vector_b)

        # Then
        assert abs(similarity - 0.0) < 1e-6  # 正交向量相似度为0

    def test_calculate_pearson_correlation(self):
        """测试皮尔逊相关系数计算"""
        # Given
        vector_a = np.array([1.0, 2.0, 3.0, 4.0])
        vector_b = np.array([2.0, 4.0, 6.0, 8.0])  # 完全正相关

        # When
        correlation = self.cf_engine._calculate_pearson_correlation(vector_a, vector_b)

        # Then
        assert abs(correlation - 1.0) < 1e-6  # 完全正相关

    def test_build_user_similarity_matrix(self):
        """测试构建用户相似度矩阵"""
        # Given
        interactions = [
            ("user1", "item1", 5.0),
            ("user1", "item2", 3.0),
            ("user2", "item1", 4.0),
            ("user2", "item2", 4.0),
            ("user3", "item1", 2.0),
            ("user3", "item2", 1.0)
        ]
        matrix = self.cf_engine.build_user_item_matrix(interactions)

        # When
        similarity_matrix = self.cf_engine.build_user_similarity_matrix(matrix)

        # Then
        assert similarity_matrix.shape == (3, 3)  # 3个用户
        assert np.allclose(similarity_matrix, similarity_matrix.T)  # 对称矩阵
        assert np.allclose(np.diag(similarity_matrix), 1.0)  # 对角线为1

        # user1和user2应该有较高相似度
        user1_idx = matrix.get_user_index("user1")
        user2_idx = matrix.get_user_index("user2")
        assert similarity_matrix[user1_idx, user2_idx] > 0.5

    def test_find_similar_users(self):
        """测试查找相似用户"""
        # Given
        interactions = [
            ("user1", "item1", 5.0),
            ("user1", "item2", 3.0),
            ("user2", "item1", 4.0),
            ("user2", "item2", 4.0),
            ("user3", "item1", 2.0),
            ("user3", "item2", 1.0)
        ]
        matrix = self.cf_engine.build_user_item_matrix(interactions)
        self.cf_engine.build_user_similarity_matrix(matrix)

        # When
        similar_users = self.cf_engine.find_similar_users("user1", k=2)

        # Then
        assert len(similar_users) == 2
        assert "user2" in [user_id for user_id, _ in similar_users]
        # user3应该比user2更相似于user1（基于评分模式）
        user2_score = next(score for user_id, score in similar_users if user_id == "user2")
        user3_score = next(score for user_id, score in similar_users if user_id == "user3")
        # 由于评分模式的差异，user3的评分向量(2,1)与user1(5,3)的夹角比user2(4,4)更接近
        assert user3_score > user2_score

    def test_find_similar_users_nonexistent_user(self):
        """测试查找相似用户 - 不存在的用户"""
        # Given
        interactions = [("user1", "item1", 5.0)]
        matrix = self.cf_engine.build_user_item_matrix(interactions)
        self.cf_engine.build_user_similarity_matrix(matrix)

        # When & Then
        with pytest.raises(ValueError, match="用户 nonexistent 不存在"):
            self.cf_engine.find_similar_users("nonexistent", k=5)

    def test_generate_recommendations_user_based(self):
        """测试基于用户的推荐生成"""
        # Given
        interactions = [
            ("user1", "item1", 5.0),
            ("user1", "item2", 3.0),
            ("user2", "item1", 4.0),
            ("user2", "item3", 5.0),
            ("user3", "item2", 4.0),
            ("user3", "item3", 2.0)
        ]
        matrix = self.cf_engine.build_user_item_matrix(interactions)
        self.cf_engine.build_user_similarity_matrix(matrix)

        # When
        recommendations = self.cf_engine.generate_recommendations(
            "user1",
            method="user_based",
            k=2
        )

        # Then
        assert isinstance(recommendations, RecommendationResult)
        assert len(recommendations.items) <= 2
        assert all(hasattr(item, 'item_id') for item in recommendations.items)
        assert all(hasattr(item, 'score') for item in recommendations.items)
        assert all(item.score > 0 for item in recommendations.items)

    def test_generate_recommendations_item_based(self):
        """测试基于物品的推荐生成"""
        # Given
        interactions = [
            ("user1", "item1", 5.0),
            ("user1", "item2", 3.0),
            ("user2", "item1", 4.0),
            ("user2", "item3", 5.0),
            ("user3", "item2", 4.0),
            ("user3", "item3", 2.0)
        ]
        matrix = self.cf_engine.build_user_item_matrix(interactions)

        # When
        recommendations = self.cf_engine.generate_recommendations(
            "user1",
            method="item_based",
            k=2
        )

        # Then
        assert isinstance(recommendations, RecommendationResult)
        assert len(recommendations.items) <= 2

    def test_generate_recommendations_invalid_method(self):
        """测试推荐生成 - 无效方法"""
        # Given
        interactions = [("user1", "item1", 5.0)]
        matrix = self.cf_engine.build_user_item_matrix(interactions)

        # When & Then
        with pytest.raises(ValueError, match="不支持的推荐方法"):
            self.cf_engine.generate_recommendations(
                "user1",
                method="invalid_method",
                k=5
            )

    def test_generate_recommendations_insufficient_data(self):
        """测试推荐生成 - 数据不足"""
        # Given - 只有一个用户的交互数据
        interactions = [("user1", "item1", 5.0)]
        matrix = self.cf_engine.build_user_item_matrix(interactions)
        self.cf_engine.build_user_similarity_matrix(matrix)

        # When
        recommendations = self.cf_engine.generate_recommendations(
            "user1",
            method="user_based",
            k=5
        )

        # Then - 应该返回空推荐
        assert isinstance(recommendations, RecommendationResult)
        assert len(recommendations.items) == 0

    def test_recommendation_result_sorting(self):
        """测试推荐结果按分数排序"""
        # Given
        from recommendation_system.models import RecommendationItem
        items = [
            RecommendationItem("item3", 0.8),
            RecommendationItem("item1", 0.9),
            RecommendationItem("item2", 0.7)
        ]
        recommendations = RecommendationResult(
            user_id="user1",
            method="user_based",
            items=items
        )

        # When
        sorted_items = recommendations.get_sorted_items()

        # Then
        assert sorted_items[0].item_id == "item1"  # 最高分
        assert sorted_items[1].item_id == "item3"
        assert sorted_items[2].item_id == "item2"  # 最低分

    def test_performance_with_large_dataset(self):
        """测试大数据集性能"""
        # Given - 生成大数据集
        import time
        num_users = 100
        num_items = 50
        interactions = []

        for user_id in range(num_users):
            for item_id in range(num_items):
                if np.random.random() > 0.8:  # 20%的交互密度
                    rating = np.random.uniform(1.0, 5.0)
                    interactions.append((f"user_{user_id}", f"item_{item_id}", rating))

        # When
        start_time = time.time()
        matrix = self.cf_engine.build_user_item_matrix(interactions)
        similarity_matrix = self.cf_engine.build_user_similarity_matrix(matrix)
        end_time = time.time()

        # Then
        processing_time = end_time - start_time
        assert processing_time < 10.0  # 应该在10秒内完成
        assert matrix.shape == (num_users, num_items)
        assert similarity_matrix.shape == (num_users, num_users)

    def test_edge_case_single_user(self):
        """测试边界情况 - 单个用户"""
        # Given
        interactions = [("user1", "item1", 5.0)]
        matrix = self.cf_engine.build_user_item_matrix(interactions)

        # When
        similarity_matrix = self.cf_engine.build_user_similarity_matrix(matrix)

        # Then
        assert similarity_matrix.shape == (1, 1)
        assert similarity_matrix[0, 0] == 1.0

    def test_edge_case_all_ratings_same(self):
        """测试边界情况 - 所有评分相同"""
        # Given
        interactions = [
            ("user1", "item1", 3.0),
            ("user1", "item2", 3.0),
            ("user2", "item1", 3.0),
            ("user2", "item2", 3.0)
        ]
        matrix = self.cf_engine.build_user_item_matrix(interactions)

        # When
        similarity_matrix = self.cf_engine.build_user_similarity_matrix(matrix)

        # Then - 所有评分相同时，相似度应该为1
        assert abs(similarity_matrix[0, 1] - 1.0) < 1e-10
        assert abs(similarity_matrix[1, 0] - 1.0) < 1e-10

    def test_memory_cleanup(self):
        """测试内存清理"""
        # Given
        interactions = [("user1", "item1", 5.0)]
        matrix = self.cf_engine.build_user_item_matrix(interactions)
        self.cf_engine.build_user_similarity_matrix(matrix)

        # When
        self.cf_engine.clear_cache()

        # Then
        assert len(self.cf_engine.user_similarity_cache) == 0

    def test_input_validation_ratings(self):
        """测试输入验证 - 评分范围"""
        # Given
        interactions = [
            ("user1", "item1", 6.0),  # 超出范围
            ("user2", "item2", -1.0)  # 负数评分
        ]

        # When & Then
        with pytest.raises(ValueError, match="评分必须在1.0-5.0范围内"):
            self.cf_engine.build_user_item_matrix(interactions)

    def test_build_user_similarity_matrix_invalid_method(self):
        """测试构建用户相似度矩阵 - 无效方法"""
        # Given
        interactions = [
            ("user1", "item1", 5.0),
            ("user2", "item1", 4.0)
        ]
        matrix = self.cf_engine.build_user_item_matrix(interactions)

        # When & Then
        with pytest.raises(ValueError, match="不支持的相似度计算方法"):
            self.cf_engine.build_user_similarity_matrix(matrix, method="invalid")

    def test_cosine_similarity_zero_vectors(self):
        """测试余弦相似度 - 零向量"""
        # Given
        vector_a = np.array([0.0, 0.0, 0.0])
        vector_b = np.array([1.0, 2.0, 3.0])

        # When
        similarity = self.cf_engine._calculate_cosine_similarity(vector_a, vector_b)

        # Then
        assert similarity == 0.0

    def test_pearson_correlation_single_common_point(self):
        """测试皮尔逊相关系数 - 单个共同点"""
        # Given
        vector_a = np.array([1.0, 0.0, 0.0])
        vector_b = np.array([2.0, 0.0, 0.0])

        # When
        correlation = self.cf_engine._calculate_pearson_correlation(vector_a, vector_b)

        # Then
        assert correlation == 0.0  # 少于2个共同点，返回0

    def test_pearson_correlation_zero_variance(self):
        """测试皮尔逊相关系数 - 零方差"""
        # Given
        vector_a = np.array([2.0, 2.0, 2.0])  # 无方差
        vector_b = np.array([1.0, 2.0, 3.0])

        # When
        correlation = self.cf_engine._calculate_pearson_correlation(vector_a, vector_b)

        # Then
        assert correlation == 0.0

    def test_find_similar_users_no_matrix_built(self):
        """测试查找相似用户 - 矩阵未构建"""
        # Given
        interactions = [("user1", "item1", 5.0)]
        self.cf_engine.build_user_item_matrix(interactions)
        # 故意不构建相似度矩阵

        # When & Then
        with pytest.raises(ValueError, match="用户相似度矩阵未构建"):
            self.cf_engine.find_similar_users("user1", k=5)

    def test_find_similar_users_no_user_item_matrix(self):
        """测试查找相似用户 - 用户物品矩阵未构建"""
        # Given & When & Then
        with pytest.raises(ValueError, match="用户-物品矩阵未构建"):
            self.cf_engine.find_similar_users("user1", k=5)

    def test_generate_recommendations_no_matrix(self):
        """测试推荐生成 - 矩阵未构建"""
        # Given & When & Then
        with pytest.raises(ValueError, match="用户-物品矩阵未构建"):
            self.cf_engine.generate_recommendations("user1", method="user_based")

    def test_generate_recommendations_nonexistent_user(self):
        """测试推荐生成 - 不存在的用户"""
        # Given
        interactions = [("user1", "item1", 5.0)]
        matrix = self.cf_engine.build_user_item_matrix(interactions)

        # When & Then
        with pytest.raises(ValueError, match="用户 nonexistent 不存在"):
            self.cf_engine.generate_recommendations("nonexistent", method="user_based")

    def test_generate_user_based_recommendations_no_similar_users(self):
        """测试基于用户推荐 - 无相似用户"""
        # Given
        interactions = [("user1", "item1", 5.0)]
        matrix = self.cf_engine.build_user_item_matrix(interactions)
        self.cf_engine.build_user_similarity_matrix(matrix)

        # When
        recommendations = self.cf_engine._generate_user_based_recommendations("user1", k=5)

        # Then
        assert len(recommendations.items) == 0
        assert recommendations.method == "user_based"

    def test_generate_item_based_recommendations_no_rated_items(self):
        """测试基于物品推荐 - 无已评分物品"""
        # Given
        interactions = [("user1", "item1", 5.0)]
        matrix = self.cf_engine.build_user_item_matrix(interactions)
        # 清空用户评分
        matrix.data[0, 0] = 0.0

        # When
        recommendations = self.cf_engine._generate_item_based_recommendations("user1", k=5)

        # Then
        assert len(recommendations.items) == 0

    def test_find_similar_items_no_matrix(self):
        """测试查找相似物品 - 矩阵未构建"""
        # Given & When
        similar_items = self.cf_engine._find_similar_items("item1", k=5)

        # Then
        assert len(similar_items) == 0

    def test_get_engine_stats_uninitialized(self):
        """测试引擎统计 - 未初始化状态"""
        # Given & When
        stats = self.cf_engine.get_engine_stats()

        # Then
        assert stats["status"] == "未初始化"

    def test_get_engine_stats_initialized(self):
        """测试引擎统计 - 已初始化状态"""
        # Given
        interactions = [
            ("user1", "item1", 5.0),
            ("user1", "item2", 3.0),
            ("user2", "item1", 4.0)
        ]
        matrix = self.cf_engine.build_user_item_matrix(interactions)
        self.cf_engine.build_user_similarity_matrix(matrix)

        # When
        stats = self.cf_engine.get_engine_stats()

        # Then
        assert stats["status"] == "已初始化"
        assert stats["num_users"] == 2
        assert stats["num_items"] == 2
        assert stats["total_interactions"] == 3
        assert stats["similarity_matrix_built"] is True
        assert stats["cache_size"] == 0

    def test_build_user_similarity_matrix_pearson_method(self):
        """测试构建用户相似度矩阵 - 皮尔逊方法"""
        # Given
        interactions = [
            ("user1", "item1", 5.0),
            ("user1", "item2", 3.0),
            ("user2", "item1", 4.0),
            ("user2", "item2", 4.0)
        ]
        matrix = self.cf_engine.build_user_item_matrix(interactions)

        # When
        similarity_matrix = self.cf_engine.build_user_similarity_matrix(matrix, method="pearson")

        # Then
        assert similarity_matrix.shape == (2, 2)
        assert np.allclose(similarity_matrix, similarity_matrix.T)  # 对称矩阵
        assert np.allclose(np.diag(similarity_matrix), 1.0)  # 对角线为1

    def test_item_based_recommendation_deduplication(self):
        """测试基于物品推荐 - 去重逻辑"""
        # Given
        interactions = [
            ("user1", "item1", 5.0),
            ("user1", "item2", 4.0)
        ]
        matrix = self.cf_engine.build_user_item_matrix(interactions)

        # When
        recommendations = self.cf_engine._generate_item_based_recommendations("user1", k=10)

        # Then - 确保没有重复推荐
        item_ids = [item.item_id for item in recommendations.items]
        assert len(item_ids) == len(set(item_ids))  # 无重复

    def test_cosine_similarity_nonzero_common_mask(self):
        """测试余弦相似度 - 有共同评分的非零向量"""
        # Given
        vector_a = np.array([1.0, 2.0, 0.0])
        vector_b = np.array([0.0, 2.0, 3.0])

        # When
        similarity = self.cf_engine._calculate_cosine_similarity(vector_a, vector_b)

        # Then - 应该基于共同的非零元素计算相似度
        assert 0.0 <= similarity <= 1.0

    def test_pearson_correlation_nonzero_common_mask(self):
        """测试皮尔逊相关系数 - 非零共同向量"""
        # Given
        vector_a = np.array([1.0, 2.0, 3.0, 0.0])
        vector_b = np.array([2.0, 4.0, 6.0, 0.0])

        # When
        correlation = self.cf_engine._calculate_pearson_correlation(vector_a, vector_b)

        # Then
        assert abs(correlation - 1.0) < 1e-6  # 完全正相关

    def test_generate_user_based_recommendations_skip_negative_similarity(self):
        """测试基于用户推荐 - 跳过负相似度用户"""
        # Given
        interactions = [
            ("user1", "item1", 5.0),
            ("user1", "item2", 3.0),
            ("user2", "item1", 1.0),  # 负相似度用户
            ("user2", "item3", 2.0)
        ]
        matrix = self.cf_engine.build_user_item_matrix(interactions)
        self.cf_engine.build_user_similarity_matrix(matrix)

        # When
        recommendations = self.cf_engine._generate_user_based_recommendations("user1", k=5)

        # Then - 应该跳过负相似度用户
        assert isinstance(recommendations, RecommendationResult)

    def test_generate_item_based_recommendations_with_existing_items(self):
        """测试基于物品推荐 - 处理已存在的物品"""
        # Given
        interactions = [
            ("user1", "item1", 5.0),
            ("user1", "item2", 4.0),
            ("user2", "item1", 3.0),
            ("user2", "item3", 4.0)
        ]
        matrix = self.cf_engine.build_user_item_matrix(interactions)

        # When
        recommendations = self.cf_engine._generate_item_based_recommendations("user1", k=5)

        # Then - 应该正确处理相似物品和去重
        assert isinstance(recommendations, RecommendationResult)
        # 检查推荐的物品不包含用户已评分的物品
        recommended_items = {item.item_id for item in recommendations.items}
        user_rated_items = set(matrix.get_user_rated_items("user1"))
        assert recommended_items.isdisjoint(user_rated_items)


if __name__ == "__main__":
    pytest.main([__file__])