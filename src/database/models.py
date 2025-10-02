"""
Database models for million-agent social platform
"""

from .config import Base, metadata

# Import all models here for proper initialization
from .agent import Agent
from .social_agent import SocialAgent
from .interaction import Interaction
from .community import Community
from .friendship import Friendship
from .community_membership import CommunityMembership
# from .interest import Interest

__all__ = ['Base', 'Agent', 'SocialAgent', 'Interaction', 'Community', 'Friendship', 'CommunityMembership']