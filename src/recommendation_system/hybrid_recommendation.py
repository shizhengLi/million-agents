"""
混合推荐引擎实现

结合协同过滤、内容推荐和社交推荐的混合算法实现
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Set
import time
from collections import defaultdict
import logging

from .collaborative_filtering import CollaborativeFilteringEngine
from .content_based import ContentBasedEngine
from .social_recommendation import SocialRecommendationEngine
from .models import (
    RecommendationResult, RecommendationItem
)

logger = logging.getLogger(__name__)


class HybridRecommendationEngine:
    """混合推荐引擎"""

    def __init__(self,
                 collaborative_engine: Optional[CollaborativeFilteringEngine] = None,
                 content_engine: Optional[ContentBasedEngine] = None,
                 social_engine: Optional[SocialRecommendationEngine] = None):
        """
        初始化混合推荐引擎

        Args:
            collaborative_engine: 协同过滤引擎
            content_engine: 内容推荐引擎
            social_engine: 社交推荐引擎
        """
        self.collaborative_engine = collaborative_engine or CollaborativeFilteringEngine()
        self.content_engine = content_engine or ContentBasedEngine()
        self.social_engine = social_engine or SocialRecommendationEngine()

        # 推荐权重配置
        self.recommendation_weights = {
            "collaborative": 0.5,  # 协同过滤权重
            "content": 0.3,        # 内容推荐权重
            "social": 0.2          # 社交推荐权重
        }

        # 推荐缓存
        self.recommendation_cache: Dict[str, RecommendationResult] = {}

        # 用户个性化权重
        self.personalized_weights: Dict[str, Dict[str, float]] = {}

    def update_recommendation_weights(self, weights: Dict[str, float]):
        """
        更新推荐权重

        Args:
            weights: 新的权重配置

        Raises:
            ValueError: 当权重无效时
        """
        # 验证权重总和
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"推荐权重总和必须等于1.0，当前值: {total_weight}")

        # 验证必需的键
        required_keys = {"collaborative", "content", "social"}
        missing_keys = required_keys - set(weights.keys())
        if missing_keys:
            raise ValueError(f"必须包含所有必需的推荐引擎权重: {missing_keys}")

        self.recommendation_weights = weights.copy()
        logger.info(f"推荐权重已更新: {weights}")

    def _calculate_hybrid_score(self, cf_score: Optional[float],
                              content_score: Optional[float],
                              social_score: Optional[float]) -> float:
        """
        计算混合推荐分数

        Args:
            cf_score: 协同过滤分数
            content_score: 内容推荐分数
            social_score: 社交推荐分数

        Returns:
            float: 混合分数
        """
        scores = []
        weights = []

        if cf_score is not None:
            scores.append(cf_score)
            weights.append(self.recommendation_weights["collaborative"])

        if content_score is not None:
            scores.append(content_score)
            weights.append(self.recommendation_weights["content"])

        if social_score is not None:
            scores.append(social_score)
            weights.append(self.recommendation_weights["social"])

        if not scores:
            return 0.0

        # 如果有部分引擎无结果，重新分配权重
        total_weight = sum(weights)
        if total_weight > 0:
            normalized_weights = [w / total_weight for w in weights]
            hybrid_score = sum(score * weight for score, weight in zip(scores, normalized_weights))
            return hybrid_score

        return 0.0

    def generate_recommendations(self, user_id: str,
                               interactions: List[Tuple[str, str, float]],
                               user_activities: Dict[str, Dict[str, float]],
                               k: int = 10,
                               enhance_diversity: bool = False,
                               criteria: Optional[Dict[str, Any]] = None) -> RecommendationResult:
        """
        生成混合推荐

        Args:
            user_id: 用户ID
            interactions: 用户交互数据 [(user_id, item_id, rating)]
            user_activities: 用户活动数据
            k: 推荐数量
            enhance_diversity: 是否增强多样性
            criteria: 推荐标准约束

        Returns:
            RecommendationResult: 推荐结果
        """
        # 检查缓存
        cache_key = f"{user_id}_{hash(tuple(sorted(interactions)))}_{k}"
        if cache_key in self.recommendation_cache:
            return self.recommendation_cache[cache_key]

        # 获取各引擎推荐结果
        engine_results = {}

        try:
            # 协同过滤推荐
            cf_result = self.collaborative_engine.generate_recommendations(
                user_id, method="user_based", k=k*2
            )
            engine_results["collaborative"] = cf_result
        except Exception as e:
            logger.warning(f"协同过滤引擎失败: {e}")
            engine_results["collaborative"] = None

        try:
            # 内容推荐
            content_result = self.content_engine.generate_recommendations(user_id, k=k*2)
            engine_results["content"] = content_result
        except Exception as e:
            logger.warning(f"内容推荐引擎失败: {e}")
            engine_results["content"] = None

        try:
            # 社交推荐
            social_result = self.social_engine.generate_recommendations(
                user_id, user_activities, k=k*2
            )
            engine_results["social"] = social_result
        except Exception as e:
            logger.warning(f"社交推荐引擎失败: {e}")
            engine_results["social"] = None

        # 合并推荐结果
        combined_recommendations = self._combine_recommendations(
            engine_results, user_id, k
        )

        # 多样性增强
        if enhance_diversity:
            combined_recommendations = self._enhance_diversity(
                combined_recommendations, k
            )

        # 应用标准约束
        if criteria:
            combined_recommendations = self._apply_criteria(
                combined_recommendations, criteria, user_activities, user_id
            )

        result = RecommendationResult(user_id, "hybrid", combined_recommendations[:k])

        # 缓存结果
        self.recommendation_cache[cache_key] = result

        return result

    def _combine_recommendations(self, engine_results: Dict[str, Optional[RecommendationResult]],
                               user_id: str, k: int) -> List[RecommendationItem]:
        """
        合并多个引擎的推荐结果

        Args:
            engine_results: 各引擎推荐结果
            user_id: 用户ID
            k: 推荐数量

        Returns:
            List[RecommendationItem]: 合并后的推荐列表
        """
        # 收集所有推荐物品及其分数
        item_scores = defaultdict(dict)  # {item_id: {engine: score}}

        for engine_name, result in engine_results.items():
            if result is None:
                continue

            for item in result.items:
                item_scores[item.item_id][engine_name] = item.score

        # 计算混合分数
        hybrid_items = []
        for item_id, engine_scores in item_scores.items():
            cf_score = engine_scores.get("collaborative")
            content_score = engine_scores.get("content")
            social_score = engine_scores.get("social")

            hybrid_score = self._calculate_hybrid_score(
                cf_score, content_score, social_score
            )

            if hybrid_score > 0:
                hybrid_items.append(RecommendationItem(item_id, hybrid_score))

        # 按分数排序
        hybrid_items.sort(key=lambda x: x.score, reverse=True)
        return hybrid_items[:k]

    def _enhance_diversity(self, recommendations: List[RecommendationItem],
                          k: int) -> List[RecommendationItem]:
        """
        增强推荐多样性

        Args:
            recommendations: 原始推荐列表
            k: 推荐数量

        Returns:
            List[RecommendationItem]: 多样化后的推荐列表
        """
        if len(recommendations) <= k:
            return recommendations

        # 简单的多样性策略：确保不同类别/来源的推荐
        diverse_recommendations = []
        used_sources = set()

        # 按分数排序，优先选择高分数的
        sorted_recs = sorted(recommendations, key=lambda x: x.score, reverse=True)

        for rec in sorted_recs:
            if len(diverse_recommendations) >= k:
                break

            # 简单的多样性判断（基于物品ID的模式）
            item_category = rec.item_id.split("_")[0] if "_" in rec.item_id else "other"

            if item_category not in used_sources or len(diverse_recommendations) < k // 2:
                diverse_recommendations.append(rec)
                used_sources.add(item_category)

        # 如果还需要更多推荐，从剩余的按分数补充
        remaining_items = [rec for rec in sorted_recs if rec not in diverse_recommendations]
        diverse_recommendations.extend(remaining_items[:k - len(diverse_recommendations)])

        return diverse_recommendations

    def _apply_criteria(self, recommendations: List[RecommendationItem],
                       criteria: Dict[str, Any],
                       user_activities: Dict[str, Dict[str, float]],
                       user_id: Optional[str] = None) -> List[RecommendationItem]:
        """
        应用推荐标准约束

        Args:
            recommendations: 原始推荐列表
            criteria: 约束条件
            user_activities: 用户活动数据

        Returns:
            List[RecommendationItem]: 过滤后的推荐列表
        """
        filtered_recommendations = []

        for rec in recommendations:
            include_item = True

            # 类别过滤
            if "category" in criteria:
                item_category = rec.item_id.split("_")[0] if "_" in rec.item_id else "other"
                if item_category != criteria["category"]:
                    include_item = False

            # 排除已见过的物品
            if criteria.get("exclude_seen", False):
                # 获取当前用户的已见物品（需要传入正确的user_id）
                current_user_items = set()
                for uid, activities in user_activities.items():
                    if uid == user_id or user_id in str(uid):  # 兼容不同的用户ID格式
                        current_user_items = set(activities.keys())
                        break
                if rec.item_id in current_user_items:
                    include_item = False

            # 最小评分过滤（如果可用）
            if "min_rating" in criteria and hasattr(rec, 'rating'):
                if rec.rating < criteria["min_rating"]:
                    include_item = False

            if include_item:
                filtered_recommendations.append(rec)

        return filtered_recommendations

    def get_recommendation_explanation(self, user_id: str, item_id: str,
                                    interactions: List[Tuple[str, str, float]],
                                    user_activities: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        获取推荐解释

        Args:
            user_id: 用户ID
            item_id: 物品ID
            interactions: 用户交互数据
            user_activities: 用户活动数据

        Returns:
            Dict[str, Any]: 推荐解释
        """
        engine_contributions = {}

        # 获取各引擎对该物品的推荐分数
        try:
            cf_result = self.collaborative_engine.generate_recommendations(
                user_id, method="user_based", k=100
            )
            cf_score = next((item.score for item in cf_result.items if item.item_id == item_id), None)
            if cf_score:
                engine_contributions["collaborative"] = cf_score
        except Exception:
            pass

        try:
            content_result = self.content_engine.generate_recommendations(user_id, k=100)
            content_score = next((item.score for item in content_result.items if item.item_id == item_id), None)
            if content_score:
                engine_contributions["content"] = content_score
        except Exception:
            pass

        try:
            social_result = self.social_engine.generate_recommendations(
                user_id, user_activities, k=100
            )
            social_score = next((item.score for item in social_result.items if item.item_id == item_id), None)
            if social_score:
                engine_contributions["social"] = social_score
        except Exception:
            pass

        # 计算最终分数
        final_score = self._calculate_hybrid_score(
            engine_contributions.get("collaborative"),
            engine_contributions.get("content"),
            engine_contributions.get("social")
        )

        return {
            "item_id": item_id,
            "engines": engine_contributions,
            "final_score": final_score,
            "weighting": self.recommendation_weights.copy()
        }

    def adaptive_weight_adjustment(self, user_id: str,
                                 performance_metrics: Dict[str, Dict[str, float]]):
        """
        自适应权重调整

        Args:
            user_id: 用户ID
            performance_metrics: 各引擎性能指标
        """
        # 基于性能指标调整权重
        engine_scores = {}

        for engine_name, metrics in performance_metrics.items():
            if engine_name in self.recommendation_weights:
                # 使用精确率和召回率的F1分数
                precision = metrics.get("precision", 0)
                recall = metrics.get("recall", 0)

                if precision + recall > 0:
                    f1_score = 2 * (precision * recall) / (precision + recall)
                else:
                    f1_score = 0

                engine_scores[engine_name] = f1_score

        # 归一化分数作为新权重
        total_score = sum(engine_scores.values())
        if total_score > 0:
            new_weights = {}
            for engine_name in self.recommendation_weights:
                new_weights[engine_name] = engine_scores.get(engine_name, 0) / total_score

            self.personalized_weights[user_id] = new_weights
            logger.info(f"用户 {user_id} 的权重已自适应调整: {new_weights}")
        else:
            # 即使所有性能指标都为零，也要存储默认权重
            default_weights = self.recommendation_weights.copy()
            self.personalized_weights[user_id] = default_weights
            logger.info(f"用户 {user_id} 使用默认权重: {default_weights}")

    def learn_personalized_weights(self, user_id: str,
                                 historical_feedback: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        学习个性化权重

        Args:
            user_id: 用户ID
            historical_feedback: 历史反馈数据

        Returns:
            Dict[str, float]: 学习到的权重
        """
        # 统计各引擎的贡献
        engine_performance = defaultdict(list)

        for feedback in historical_feedback:
            rating = feedback.get("rating", 0)
            source = feedback.get("source", "unknown")

            if source in self.recommendation_weights:
                # 将评分转换为性能分数
                performance_score = rating / 5.0  # 假设评分范围1-5
                engine_performance[source].append(performance_score)

        # 计算平均性能作为权重
        learned_weights = {}
        total_performance = 0

        for engine_name in self.recommendation_weights:
            performances = engine_performance.get(engine_name, [0.1])  # 默认低性能
            avg_performance = np.mean(performances)
            learned_weights[engine_name] = avg_performance
            total_performance += avg_performance

        # 归一化权重
        if total_performance > 0:
            for engine_name in learned_weights:
                learned_weights[engine_name] /= total_performance

        self.personalized_weights[user_id] = learned_weights
        return learned_weights

    def clear_cache(self):
        """清空推荐缓存"""
        self.recommendation_cache.clear()

    def get_engine_stats(self) -> Dict[str, Any]:
        """获取引擎统计信息"""
        return {
            "collaborative_engine": {
                "type": "CollaborativeFilteringEngine",
                "initialized": self.collaborative_engine is not None
            },
            "content_engine": {
                "type": "ContentBasedEngine",
                "initialized": self.content_engine is not None
            },
            "social_engine": {
                "type": "SocialRecommendationEngine",
                "initialized": self.social_engine is not None
            },
            "hybrid_weights": self.recommendation_weights.copy(),
            "cache_size": len(self.recommendation_cache),
            "personalized_weights_count": len(self.personalized_weights)
        }