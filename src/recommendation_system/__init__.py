"""
推荐系统模块

提供基于协同过滤、内容分析、社交关系的智能体推荐功能
"""

from .collaborative_filtering import CollaborativeFilteringEngine
from .content_based import ContentBasedEngine
from .social_recommendation import SocialRecommendationEngine
from .models import (
    UserItemMatrix,
    RecommendationResult,
    UserProfile,
    ContentFeature
)

__all__ = [
    "CollaborativeFilteringEngine",
    "ContentBasedEngine",
    "SocialRecommendationEngine",
    "UserItemMatrix",
    "RecommendationResult",
    "UserProfile",
    "ContentFeature"
]

__version__ = "1.0.0"
__author__ = "Million Agents Team"