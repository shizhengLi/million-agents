"""
推荐系统模型测试

测试推荐系统的数据模型和工具类
"""

import pytest
import numpy as np
from datetime import datetime
from src.recommendation_system.models import (
    RecommendationItem,
    RecommendationResult,
    UserItemMatrix,
    UserProfile,
    ContentFeature,
    Interaction,
    SimilarityResult,
    RecommendationMetrics
)


class TestRecommendationItem:
    """测试推荐物品项"""

    def test_valid_recommendation_item(self):
        """测试有效的推荐物品项"""
        item = RecommendationItem(item_id="item1", score=0.8, reason="高质量内容")
        assert item.item_id == "item1"
        assert item.score == 0.8
        assert item.reason == "高质量内容"

    def test_recommendation_item_without_reason(self):
        """测试无推荐原因的物品项"""
        item = RecommendationItem(item_id="item2", score=0.6)
        assert item.item_id == "item2"
        assert item.score == 0.6
        assert item.reason is None

    def test_recommendation_item_score_boundary_low(self):
        """测试分数边界值 - 最小值"""
        item = RecommendationItem(item_id="item3", score=0.0)
        assert item.score == 0.0

    def test_recommendation_item_score_boundary_high(self):
        """测试分数边界值 - 最大值"""
        item = RecommendationItem(item_id="item4", score=1.0)
        assert item.score == 1.0

    def test_recommendation_item_invalid_score_negative(self):
        """测试无效分数 - 负数"""
        with pytest.raises(ValueError, match="推荐分数必须在0-1之间"):
            RecommendationItem(item_id="item5", score=-0.1)

    def test_recommendation_item_invalid_score_too_high(self):
        """测试无效分数 - 超过1"""
        with pytest.raises(ValueError, match="推荐分数必须在0-1之间"):
            RecommendationItem(item_id="item6", score=1.1)


class TestRecommendationResult:
    """测试推荐结果"""

    def test_recommendation_result_creation(self):
        """测试推荐结果创建"""
        items = [
            RecommendationItem("item1", 0.9),
            RecommendationItem("item2", 0.7),
            RecommendationItem("item3", 0.5)
        ]
        result = RecommendationResult("user1", "collaborative", items)

        assert result.user_id == "user1"
        assert result.method == "collaborative"
        assert len(result.items) == 3
        assert isinstance(result.generated_at, datetime)

    def test_get_sorted_items(self):
        """测试获取排序后的推荐项"""
        items = [
            RecommendationItem("item1", 0.5),
            RecommendationItem("item2", 0.9),
            RecommendationItem("item3", 0.7)
        ]
        result = RecommendationResult("user1", "test", items)

        sorted_items = result.get_sorted_items()
        assert sorted_items[0].item_id == "item2"  # 最高分数
        assert sorted_items[1].item_id == "item3"
        assert sorted_items[2].item_id == "item1"  # 最低分数

    def test_get_top_k_normal_case(self):
        """测试获取前k个推荐项 - 正常情况"""
        items = [
            RecommendationItem("item1", 0.9),
            RecommendationItem("item2", 0.8),
            RecommendationItem("item3", 0.7),
            RecommendationItem("item4", 0.6)
        ]
        result = RecommendationResult("user1", "test", items)

        top_k = result.get_top_k(2)
        assert len(top_k) == 2
        assert top_k[0].item_id == "item1"
        assert top_k[1].item_id == "item2"

    def test_get_top_k_larger_than_available(self):
        """测试获取前k个推荐项 - k大于可用数量"""
        items = [
            RecommendationItem("item1", 0.9),
            RecommendationItem("item2", 0.8)
        ]
        result = RecommendationResult("user1", "test", items)

        top_k = result.get_top_k(5)
        assert len(top_k) == 2  # 只返回可用的数量

    def test_get_top_k_zero(self):
        """测试获取前k个推荐项 - k为0"""
        items = [
            RecommendationItem("item1", 0.9),
            RecommendationItem("item2", 0.8)
        ]
        result = RecommendationResult("user1", "test", items)

        top_k = result.get_top_k(0)
        assert len(top_k) == 0

    def test_get_top_k_empty_items(self):
        """测试获取前k个推荐项 - 空物品列表"""
        result = RecommendationResult("user1", "test", [])

        top_k = result.get_top_k(3)
        assert len(top_k) == 0


class TestUserItemMatrix:
    """测试用户-物品矩阵"""

    def test_user_item_matrix_creation(self):
        """测试矩阵创建"""
        data = np.array([[5.0, 3.0, 0.0],
                        [4.0, 0.0, 2.0],
                        [1.0, 2.0, 3.0]])
        user_ids = ["user1", "user2", "user3"]
        item_ids = ["item1", "item2", "item3"]

        matrix = UserItemMatrix(data, user_ids, item_ids)

        assert matrix.shape == (3, 3)
        assert len(matrix.user_ids) == 3
        assert len(matrix.item_ids) == 3
        np.testing.assert_array_equal(matrix.data, data)

    def test_user_item_matrix_dimension_mismatch(self):
        """测试矩阵维度不匹配"""
        data = np.array([[5.0, 3.0],
                        [4.0, 0.0]])
        user_ids = ["user1", "user2", "user3"]  # 3个用户
        item_ids = ["item1", "item2"]           # 2个物品

        with pytest.raises(ValueError, match="矩阵维度与用户/物品数量不匹配"):
            UserItemMatrix(data, user_ids, item_ids)

    def test_get_user_index_valid(self):
        """测试获取用户索引 - 有效用户"""
        data = np.array([[5.0, 3.0],
                        [4.0, 0.0]])
        matrix = UserItemMatrix(data, ["user1", "user2"], ["item1", "item2"])

        assert matrix.get_user_index("user1") == 0
        assert matrix.get_user_index("user2") == 1

    def test_get_user_index_invalid(self):
        """测试获取用户索引 - 无效用户"""
        data = np.array([[5.0, 3.0]])
        matrix = UserItemMatrix(data, ["user1"], ["item1", "item2"])

        with pytest.raises(ValueError, match="用户 user2 不存在"):
            matrix.get_user_index("user2")

    def test_get_item_index_valid(self):
        """测试获取物品索引 - 有效物品"""
        data = np.array([[5.0, 3.0, 1.0]])
        matrix = UserItemMatrix(data, ["user1"], ["item1", "item2", "item3"])

        assert matrix.get_item_index("item1") == 0
        assert matrix.get_item_index("item2") == 1
        assert matrix.get_item_index("item3") == 2

    def test_get_item_index_invalid(self):
        """测试获取物品索引 - 无效物品"""
        data = np.array([[5.0, 3.0]])
        matrix = UserItemMatrix(data, ["user1"], ["item1", "item2"])

        with pytest.raises(ValueError, match="物品 item3 不存在"):
            matrix.get_item_index("item3")

    def test_get_user_vector(self):
        """测试获取用户评分向量"""
        data = np.array([[5.0, 3.0, 0.0],
                        [4.0, 0.0, 2.0]])
        matrix = UserItemMatrix(data, ["user1", "user2"], ["item1", "item2", "item3"])

        user_vector = matrix.get_user_vector("user1")
        expected = np.array([5.0, 3.0, 0.0])
        np.testing.assert_array_equal(user_vector, expected)

    def test_get_item_vector(self):
        """测试获取物品评分向量"""
        data = np.array([[5.0, 3.0],
                        [4.0, 0.0],
                        [1.0, 2.0]])
        matrix = UserItemMatrix(data, ["user1", "user2", "user3"], ["item1", "item2"])

        item_vector = matrix.get_item_vector("item1")
        expected = np.array([5.0, 4.0, 1.0])
        np.testing.assert_array_equal(item_vector, expected)

    def test_get_rating(self):
        """测试获取评分"""
        data = np.array([[5.0, 3.0],
                        [4.0, 0.0]])
        matrix = UserItemMatrix(data, ["user1", "user2"], ["item1", "item2"])

        rating = matrix.get_rating("user1", "item2")
        assert rating == 3.0

    def test_set_rating_valid(self):
        """测试设置评分 - 有效评分"""
        data = np.array([[5.0, 3.0],
                        [4.0, 0.0]])
        matrix = UserItemMatrix(data, ["user1", "user2"], ["item1", "item2"])

        matrix.set_rating("user1", "item2", 4.5)
        assert matrix.get_rating("user1", "item2") == 4.5

    def test_set_rating_boundary_low(self):
        """测试设置评分 - 边界最小值"""
        data = np.array([[5.0, 3.0]])
        matrix = UserItemMatrix(data, ["user1"], ["item1", "item2"])

        matrix.set_rating("user1", "item1", 1.0)
        assert matrix.get_rating("user1", "item1") == 1.0

    def test_set_rating_boundary_high(self):
        """测试设置评分 - 边界最大值"""
        data = np.array([[5.0, 3.0]])
        matrix = UserItemMatrix(data, ["user1"], ["item1", "item2"])

        matrix.set_rating("user1", "item1", 5.0)
        assert matrix.get_rating("user1", "item1") == 5.0

    def test_set_rating_too_low(self):
        """测试设置评分 - 过低评分"""
        data = np.array([[5.0, 3.0]])
        matrix = UserItemMatrix(data, ["user1"], ["item1", "item2"])

        with pytest.raises(ValueError, match="评分必须在1.0-5.0范围内"):
            matrix.set_rating("user1", "item1", 0.9)

    def test_set_rating_too_high(self):
        """测试设置评分 - 过高评分"""
        data = np.array([[5.0, 3.0]])
        matrix = UserItemMatrix(data, ["user1"], ["item1", "item2"])

        with pytest.raises(ValueError, match="评分必须在1.0-5.0范围内"):
            matrix.set_rating("user1", "item1", 5.1)


class TestMatrixAdvancedFeatures:
    """测试矩阵高级功能"""

    def test_matrix_with_zeros(self):
        """测试包含零值的矩阵"""
        data = np.array([[0.0, 0.0, 0.0],
                        [1.0, 0.0, 3.0]])
        matrix = UserItemMatrix(data, ["user1", "user2"], ["item1", "item2", "item3"])

        # 测试零值评分
        assert matrix.get_rating("user1", "item1") == 0.0

        # 测试零值向量
        user_vector = matrix.get_user_vector("user1")
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_equal(user_vector, expected)

    def test_get_user_rated_items(self):
        """测试获取用户已评分的物品"""
        data = np.array([[5.0, 0.0, 3.0, 0.0],
                        [0.0, 4.0, 0.0, 2.0]])
        matrix = UserItemMatrix(data, ["user1", "user2"], ["item1", "item2", "item3", "item4"])

        rated_items = matrix.get_user_rated_items("user1")
        assert set(rated_items) == {"item1", "item3"}

        rated_items_2 = matrix.get_user_rated_items("user2")
        assert set(rated_items_2) == {"item2", "item4"}

    def test_get_user_rated_items_none(self):
        """测试获取用户已评分物品 - 无评分用户"""
        data = np.array([[0.0, 0.0, 0.0],
                        [5.0, 3.0, 1.0]])
        matrix = UserItemMatrix(data, ["user1", "user2"], ["item1", "item2", "item3"])

        rated_items = matrix.get_user_rated_items("user1")
        assert len(rated_items) == 0

    def test_get_item_raters(self):
        """测试获取物品评分者"""
        data = np.array([[5.0, 0.0, 3.0],
                        [0.0, 4.0, 0.0],
                        [2.0, 0.0, 1.0]])
        matrix = UserItemMatrix(data, ["user1", "user2", "user3"], ["item1", "item2", "item3"])

        item1_raters = matrix.get_item_raters("item1")
        assert set(item1_raters) == {"user1", "user3"}

        item2_raters = matrix.get_item_raters("item2")
        assert item2_raters == ["user2"]

        item3_raters = matrix.get_item_raters("item3")
        assert set(item3_raters) == {"user1", "user3"}

    def test_get_item_raters_none(self):
        """测试获取物品评分者 - 无评分者"""
        data = np.array([[5.0, 0.0],
                        [3.0, 0.0]])
        matrix = UserItemMatrix(data, ["user1", "user2"], ["item1", "item2"])

        item2_raters = matrix.get_item_raters("item2")
        assert len(item2_raters) == 0

    def test_get_common_items(self):
        """测试获取用户共同评分物品"""
        data = np.array([[5.0, 3.0, 0.0, 2.0],
                        [4.0, 0.0, 3.0, 2.0],
                        [0.0, 3.0, 3.0, 0.0]])
        matrix = UserItemMatrix(data, ["user1", "user2", "user3"], ["item1", "item2", "item3", "item4"])

        common_12 = matrix.get_common_items("user1", "user2")
        assert set(common_12) == {"item1", "item4"}

        common_13 = matrix.get_common_items("user1", "user3")
        assert common_13 == ["item2"]

        common_23 = matrix.get_common_items("user2", "user3")
        assert common_23 == ["item3"]

    def test_get_common_items_none(self):
        """测试获取用户共同评分物品 - 无共同物品"""
        data = np.array([[5.0, 3.0, 0.0],
                        [0.0, 0.0, 4.0]])
        matrix = UserItemMatrix(data, ["user1", "user2"], ["item1", "item2", "item3"])

        common = matrix.get_common_items("user1", "user2")
        assert len(common) == 0

    def test_get_user_stats_with_ratings(self):
        """测试获取用户统计信息 - 有评分"""
        data = np.array([[5.0, 3.0, 1.0, 0.0]])
        matrix = UserItemMatrix(data, ["user1"], ["item1", "item2", "item3", "item4"])

        stats = matrix.get_user_stats("user1")
        assert stats["rated_count"] == 3
        assert stats["avg_rating"] == 3.0  # (5+3+1)/3
        assert stats["min_rating"] == 1.0
        assert stats["max_rating"] == 5.0

    def test_get_user_stats_no_ratings(self):
        """测试获取用户统计信息 - 无评分"""
        data = np.array([[0.0, 0.0, 0.0]])
        matrix = UserItemMatrix(data, ["user1"], ["item1", "item2", "item3"])

        stats = matrix.get_user_stats("user1")
        assert stats["rated_count"] == 0
        assert stats["avg_rating"] == 0.0
        assert stats["min_rating"] == 0.0
        assert stats["max_rating"] == 0.0

    def test_large_matrix_performance(self):
        """测试大矩阵性能"""
        # 创建100x100的矩阵
        n_users, n_items = 100, 100
        data = np.random.rand(n_users, n_items) * 4 + 1  # 1-5之间的随机评分
        user_ids = [f"user{i}" for i in range(n_users)]
        item_ids = [f"item{i}" for i in range(n_items)]

        matrix = UserItemMatrix(data, user_ids, item_ids)

        # 测试各种操作的性能
        assert matrix.shape == (100, 100)

        # 测试随机访问
        user_vector = matrix.get_user_vector("user50")
        assert len(user_vector) == 100

        item_vector = matrix.get_item_vector("item50")
        assert len(item_vector) == 100

        # 测试设置评分
        matrix.set_rating("user99", "item99", 5.0)
        assert matrix.get_rating("user99", "item99") == 5.0


class TestUserProfile:
    """测试用户画像"""

    def test_user_profile_creation(self):
        """测试用户画像创建"""
        profile = UserProfile("user1")

        assert profile.user_id == "user1"
        assert profile.demographics == {}
        assert profile.preferences == {}
        assert profile.behavior_features == {}
        assert isinstance(profile.created_at, datetime)
        assert isinstance(profile.updated_at, datetime)

    def test_update_preference_valid(self):
        """测试更新偏好 - 有效分数"""
        profile = UserProfile("user1")

        profile.update_preference("sports", 0.8)
        profile.update_preference("music", 0.6)

        assert profile.preferences["sports"] == 0.8
        assert profile.preferences["music"] == 0.6
        assert profile.updated_at > profile.created_at

    def test_update_preference_boundary_low(self):
        """测试更新偏好 - 边界最小值"""
        profile = UserProfile("user1")

        profile.update_preference("category", 0.0)
        assert profile.preferences["category"] == 0.0

    def test_update_preference_boundary_high(self):
        """测试更新偏好 - 边界最大值"""
        profile = UserProfile("user1")

        profile.update_preference("category", 1.0)
        assert profile.preferences["category"] == 1.0

    def test_update_preference_invalid_negative(self):
        """测试更新偏好 - 无效负数"""
        profile = UserProfile("user1")

        with pytest.raises(ValueError, match="偏好分数必须在0-1之间"):
            profile.update_preference("category", -0.1)

    def test_update_preference_invalid_too_high(self):
        """测试更新偏好 - 无效高数"""
        profile = UserProfile("user1")

        with pytest.raises(ValueError, match="偏好分数必须在0-1之间"):
            profile.update_preference("category", 1.1)

    def test_update_behavior_feature(self):
        """测试更新行为特征"""
        profile = UserProfile("user1")

        profile.update_behavior_feature("login_frequency", 5.2)
        profile.update_behavior_feature("avg_session_time", 30.5)

        assert profile.behavior_features["login_frequency"] == 5.2
        assert profile.behavior_features["avg_session_time"] == 30.5
        assert profile.updated_at > profile.created_at

    def test_update_demographics(self):
        """测试更新人口统计信息"""
        profile = UserProfile("user1")

        profile.demographics["age"] = 25
        profile.demographics["city"] = "北京"
        profile.demographics["gender"] = "male"

        assert profile.demographics["age"] == 25
        assert profile.demographics["city"] == "北京"
        assert profile.demographics["gender"] == "male"

    def test_profile_timestamp_update(self):
        """测试时间戳更新"""
        import time

        profile = UserProfile("user1")
        original_time = profile.updated_at

        time.sleep(0.01)  # 确保时间差异
        profile.update_preference("test", 0.5)

        assert profile.updated_at > original_time

        time.sleep(0.01)
        profile.update_behavior_feature("test_feature", 1.0)

        assert profile.updated_at > original_time


class TestContentFeature:
    """测试内容特征"""

    def test_content_feature_creation(self):
        """测试内容特征创建"""
        feature = ContentFeature("item1", "electronics")

        assert feature.item_id == "item1"
        assert feature.category == "electronics"
        assert feature.features == {}
        assert feature.tags == []
        assert isinstance(feature.created_at, datetime)

    def test_add_feature_valid(self):
        """测试添加特征 - 有效值"""
        feature = ContentFeature("item1", "books")

        feature.add_feature("price", 29.99)
        feature.add_feature("quality", 4.5)
        feature.add_feature("popularity", 0.8)

        assert feature.features["price"] == 29.99
        assert feature.features["quality"] == 4.5
        assert feature.features["popularity"] == 0.8

    def test_add_feature_zero(self):
        """测试添加特征 - 零值"""
        feature = ContentFeature("item1", "food")
        feature.add_feature("calories", 0.0)
        assert feature.features["calories"] == 0.0

    def test_add_feature_negative(self):
        """测试添加特征 - 负数（无效）"""
        feature = ContentFeature("item1", "test")

        with pytest.raises(ValueError, match="特征值不能为负数"):
            feature.add_feature("invalid_feature", -1.0)

    def test_add_tag_new(self):
        """测试添加标签 - 新标签"""
        feature = ContentFeature("item1", "music")
        feature.add_tag("rock")
        feature.add_tag("pop")
        feature.add_tag("classic")

        assert "rock" in feature.tags
        assert "pop" in feature.tags
        assert "classic" in feature.tags
        assert len(feature.tags) == 3

    def test_add_tag_duplicate(self):
        """测试添加标签 - 重复标签"""
        feature = ContentFeature("item1", "movie")
        feature.add_tag("action")
        feature.add_tag("action")  # 重复添加
        feature.add_tag("thriller")

        assert feature.tags.count("action") == 1  # 只有一个
        assert "thriller" in feature.tags
        assert len(feature.tags) == 2

    def test_get_feature_vector_complete(self):
        """测试获取特征向量 - 完整特征"""
        feature = ContentFeature("item1", "test")
        feature.add_feature("f1", 1.0)
        feature.add_feature("f2", 2.0)
        feature.add_feature("f3", 3.0)

        vector = feature.get_feature_vector(["f1", "f2", "f3"])
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(vector, expected)

    def test_get_feature_vector_partial(self):
        """测试获取特征向量 - 部分特征缺失"""
        feature = ContentFeature("item1", "test")
        feature.add_feature("f1", 1.0)
        feature.add_feature("f3", 3.0)

        vector = feature.get_feature_vector(["f1", "f2", "f3", "f4"])
        expected = np.array([1.0, 0.0, 3.0, 0.0])  # 缺失的用0.0填充
        np.testing.assert_array_equal(vector, expected)

    def test_get_feature_vector_empty(self):
        """测试获取特征向量 - 空特征列表"""
        feature = ContentFeature("item1", "test")
        feature.add_feature("f1", 1.0)

        vector = feature.get_feature_vector([])
        expected = np.array([])
        np.testing.assert_array_equal(vector, expected)


class TestInteraction:
    """测试用户交互记录"""

    def test_interaction_creation_default_type(self):
        """测试交互记录创建 - 默认类型"""
        interaction = Interaction("user1", "item1", 4.5)

        assert interaction.user_id == "user1"
        assert interaction.item_id == "item1"
        assert interaction.rating == 4.5
        assert interaction.interaction_type == "rating"
        assert interaction.context == {}
        assert isinstance(interaction.timestamp, datetime)

    def test_interaction_creation_custom_type(self):
        """测试交互记录创建 - 自定义类型"""
        interaction = Interaction("user1", "item1", 5.0, "like")

        assert interaction.interaction_type == "like"
        assert interaction.rating == 5.0

    def test_interaction_valid_rating_boundary(self):
        """测试有效评分边界值"""
        # 最小值
        interaction1 = Interaction("user1", "item1", 1.0)
        assert interaction1.rating == 1.0

        # 最大值
        interaction2 = Interaction("user1", "item1", 5.0)
        assert interaction2.rating == 5.0

    def test_interaction_invalid_rating_too_low(self):
        """测试无效评分 - 过低"""
        with pytest.raises(ValueError, match="评分必须在1.0-5.0范围内"):
            Interaction("user1", "item1", 0.9)

    def test_interaction_invalid_rating_too_high(self):
        """测试无效评分 - 过高"""
        with pytest.raises(ValueError, match="评分必须在1.0-5.0范围内"):
            Interaction("user1", "item1", 5.1)

    def test_interaction_context(self):
        """测试交互上下文"""
        interaction = Interaction("user1", "item1", 3.5)

        interaction.context["device"] = "mobile"
        interaction.context["location"] = "home"
        interaction.context["time_of_day"] = "evening"

        assert interaction.context["device"] == "mobile"
        assert interaction.context["location"] == "home"
        assert interaction.context["time_of_day"] == "evening"


class TestSimilarityResult:
    """测试相似度结果"""

    def test_similarity_result_creation(self):
        """测试相似度结果创建"""
        result = SimilarityResult("item1", "item2", 0.8, "cosine")

        assert result.item_a == "item1"
        assert result.item_b == "item2"
        assert result.similarity == 0.8
        assert result.method == "cosine"
        assert isinstance(result.calculated_at, datetime)

    def test_similarity_boundary_values(self):
        """测试相似度边界值"""
        # 最小值
        result1 = SimilarityResult("item1", "item2", -1.0, "pearson")
        assert result1.similarity == -1.0

        # 最大值
        result2 = SimilarityResult("item1", "item2", 1.0, "euclidean")
        assert result2.similarity == 1.0

        # 零值
        result3 = SimilarityResult("item1", "item2", 0.0, "jaccard")
        assert result3.similarity == 0.0

    def test_similarity_invalid_too_low(self):
        """测试无效相似度 - 过低"""
        with pytest.raises(ValueError, match="相似度必须在-1到1之间"):
            SimilarityResult("item1", "item2", -1.1, "test")

    def test_similarity_invalid_too_high(self):
        """测试无效相似度 - 过高"""
        with pytest.raises(ValueError, match="相似度必须在-1到1之间"):
            SimilarityResult("item1", "item2", 1.1, "test")


class TestRecommendationMetrics:
    """测试推荐系统性能指标"""

    def test_metrics_creation_normal(self):
        """测试指标创建 - 正常情况"""
        metrics = RecommendationMetrics(
            precision=0.8,
            recall=0.6,
            coverage=0.9,
            diversity=0.7,
            novelty=0.5
        )

        assert metrics.precision == 0.8
        assert metrics.recall == 0.6
        assert metrics.coverage == 0.9
        assert metrics.diversity == 0.7
        assert metrics.novelty == 0.5
        assert isinstance(metrics.calculated_at, datetime)

        # F1分数计算: 2 * (0.8 * 0.6) / (0.8 + 0.6) = 0.68
        expected_f1 = 2 * (0.8 * 0.6) / (0.8 + 0.6)
        assert abs(metrics.f1_score - expected_f1) < 0.001

    def test_metrics_f1_zero_precision(self):
        """测试F1计算 - 精确度为0"""
        metrics = RecommendationMetrics(
            precision=0.0,
            recall=0.8,
            coverage=0.5,
            diversity=0.6,
            novelty=0.4
        )

        assert metrics.f1_score == 0.0

    def test_metrics_f1_zero_recall(self):
        """测试F1计算 - 召回率为0"""
        metrics = RecommendationMetrics(
            precision=0.8,
            recall=0.0,
            coverage=0.5,
            diversity=0.6,
            novelty=0.4
        )

        assert metrics.f1_score == 0.0

    def test_metrics_f1_both_zero(self):
        """测试F1计算 - 精确度和召回率都为0"""
        metrics = RecommendationMetrics(
            precision=0.0,
            recall=0.0,
            coverage=0.1,
            diversity=0.2,
            novelty=0.3
        )

        assert metrics.f1_score == 0.0

    def test_metrics_boundary_values(self):
        """测试指标边界值"""
        metrics = RecommendationMetrics(
            precision=1.0,
            recall=1.0,
            coverage=1.0,
            diversity=1.0,
            novelty=1.0
        )

        assert metrics.f1_score == 1.0

    def test_metrics_small_values(self):
        """测试小数值"""
        metrics = RecommendationMetrics(
            precision=0.1,
            recall=0.1,
            coverage=0.1,
            diversity=0.1,
            novelty=0.1
        )

        # F1分数计算: 2 * (0.1 * 0.1) / (0.1 + 0.1) = 0.1
        expected_f1 = 2 * (0.1 * 0.1) / (0.1 + 0.1)
        assert abs(metrics.f1_score - expected_f1) < 0.001


if __name__ == "__main__":
    pytest.main([__file__])