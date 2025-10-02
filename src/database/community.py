"""
Community model for million-agent social platform
"""

from datetime import datetime, timedelta
from typing import List, Optional
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, JSON, UniqueConstraint, CheckConstraint
from sqlalchemy.orm import relationship
from .config import Base


class Community(Base):
    """Community model representing agent communities and groups"""

    __tablename__ = "communities"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, unique=True, index=True)
    description = Column(Text, nullable=True)

    # Community classification
    community_type = Column(String(50), nullable=False, default="general", index=True)
    privacy_level = Column(String(20), nullable=False, default="public", index=True)

    # Membership and capacity
    max_members = Column(Integer, nullable=False, default=10000)
    member_count = Column(Integer, nullable=False, default=0)

    # Community status
    is_active = Column(Boolean, nullable=False, default=True, index=True)

    # Activity tracking
    last_activity = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    # Tags and categorization
    tags = Column(JSON, nullable=False, default=list)

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    members = relationship("CommunityMembership", back_populates="community")

    __table_args__ = (
        # Validate max_members range
        CheckConstraint('max_members > 0 AND max_members <= 10000000', name='ck_max_members_range'),
        # Validate member_count range
        CheckConstraint('member_count >= 0', name='ck_member_count_range'),
        # Validate privacy level
        CheckConstraint("privacy_level IN ('public', 'private', 'secret')", name='ck_privacy_level'),
        {"sqlite_autoincrement": True}
    )

    def __init__(self, name: str, **kwargs):
        """Initialize community with validation"""
        self.name = name

        # Validate community type
        valid_types = [
            "general", "academic", "professional", "hobby", "support",
            "research", "development", "education", "entertainment", "business"
        ]
        community_type = kwargs.get('community_type', 'general')
        if community_type not in valid_types:
            raise ValueError(f"Invalid community_type: {community_type}. Must be one of: {valid_types}")
        self.community_type = community_type

        # Validate privacy level
        valid_privacy = ["public", "private", "secret"]
        privacy_level = kwargs.get('privacy_level', 'public')
        if privacy_level not in valid_privacy:
            raise ValueError(f"Invalid privacy_level: {privacy_level}. Must be one of: {valid_privacy}")
        self.privacy_level = privacy_level

        # Validate and set max_members
        max_members = kwargs.get('max_members', 10000)
        if not 1 <= max_members <= 10000000:
            raise ValueError(f"max_members must be between 1 and 10,000,000")
        self.max_members = max_members

        # Set other attributes
        self.description = kwargs.get('description')
        self.member_count = kwargs.get('member_count', 0)
        self.is_active = kwargs.get('is_active', True)
        self.tags = kwargs.get('tags', [])

        # Set custom last_activity if provided
        last_activity = kwargs.get('last_activity')
        if last_activity:
            self.last_activity = last_activity

    def __str__(self):
        """String representation of community"""
        return f"Community(name='{self.name}', type='{self.community_type}', members={self.member_count})"

    def __repr__(self):
        """Detailed representation of community"""
        return (f"Community(id={self.id}, name='{self.name}', "
                f"type='{self.community_type}', privacy='{self.privacy_level}', "
                f"members={self.member_count}/{self.max_members}, active={self.is_active})")

    def update_member_count(self, delta: int):
        """Update member count with validation"""
        new_count = self.member_count + delta

        # Prevent negative count
        if new_count < 0:
            self.member_count = 0
        else:
            # Don't exceed max_members
            self.member_count = min(new_count, self.max_members)

        self.updated_at = datetime.utcnow()

    def is_full(self) -> bool:
        """Check if community is at maximum capacity"""
        return self.member_count >= self.max_members

    def can_join(self) -> bool:
        """Check if agent can join this community"""
        return (self.is_active and
                not self.is_full() and
                self.privacy_level in ["public", "private"])

    def deactivate(self):
        """Deactivate the community"""
        self.is_active = False
        self.updated_at = datetime.utcnow()

    def activate(self):
        """Activate the community"""
        self.is_active = True
        self.updated_at = datetime.utcnow()

    def add_tag(self, tag: str):
        """Add a tag to the community"""
        if tag and tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = datetime.utcnow()

    def remove_tag(self, tag: str):
        """Remove a tag from the community"""
        if tag in self.tags:
            self.tags.remove(tag)
            self.updated_at = datetime.utcnow()

    def has_tag(self, tag: str) -> bool:
        """Check if community has a specific tag"""
        return tag in self.tags

    def record_activity(self):
        """Record community activity"""
        self.last_activity = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def is_recently_active(self, days: int = 7) -> bool:
        """Check if community has been active within specified days"""
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        return self.last_activity >= cutoff_time

    def get_popularity_score(self) -> float:
        """Calculate popularity score based on various factors"""
        # Normalize member count (0-1 scale)
        member_score = min(1.0, self.member_count / self.max_members)

        # Activity score based on last activity
        days_since_activity = (datetime.utcnow() - self.last_activity).days
        activity_score = max(0.0, 1.0 - (days_since_activity / 30.0))  # Decay over 30 days

        # Age score (newer communities get slight boost)
        days_since_creation = (datetime.utcnow() - self.created_at).days
        age_score = max(0.3, 1.0 - (days_since_creation / 365.0))  # Decay over 1 year

        # Weighted average
        popularity_score = (
            0.5 * member_score +      # 50% weight on membership
            0.3 * activity_score +    # 30% weight on recent activity
            0.2 * age_score          # 20% weight on community age
        )

        return round(popularity_score, 3)

    def get_popularity_level(self) -> str:
        """Get popularity level based on popularity score"""
        score = self.get_popularity_score()

        if score >= 0.8:
            return "very_high"
        elif score >= 0.6:
            return "high"
        elif score >= 0.3:
            return "medium"
        else:
            return "low"

    def is_trending(self, days: int = 7) -> bool:
        """Check if community is trending (high recent activity)"""
        return (self.is_recently_active(days) and
                self.get_popularity_score() >= 0.6 and
                self.member_count >= 10)

    @staticmethod
    def search_by_name(session, query: str, limit: int = 20) -> List['Community']:
        """Search communities by name"""
        return session.query(Community).filter(
            Community.name.ilike(f"%{query}%")
        ).limit(limit).all()

    @staticmethod
    def search_by_tags(session, tags: List[str], limit: int = 20) -> List['Community']:
        """Search communities by tags"""
        if not tags:
            return []

        # Build JSON query for tags
        tag_conditions = []
        for tag in tags:
            tag_conditions.append(Community.tags.contains([tag]))

        # Combine conditions with OR
        from sqlalchemy import or_
        query_filter = or_(*tag_conditions)

        return session.query(Community).filter(
            query_filter,
            Community.is_active == True
        ).limit(limit).all()

    @staticmethod
    def get_communities_by_type(session, community_type: str, active_only: bool = True) -> List['Community']:
        """Get communities by type"""
        query = session.query(Community).filter(
            Community.community_type == community_type
        )

        if active_only:
            query = query.filter(Community.is_active == True)

        return query.order_by(Community.member_count.desc()).all()

    @staticmethod
    def get_trending_communities(session, days: int = 7, limit: int = 10) -> List['Community']:
        """Get trending communities"""
        cutoff_time = datetime.utcnow() - timedelta(days=days)

        return session.query(Community).filter(
            Community.last_activity >= cutoff_time,
            Community.is_active == True,
            Community.member_count >= 10
        ).order_by(Community.member_count.desc()).limit(limit).all()

    def get_community_summary(self) -> dict:
        """Get comprehensive community summary"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'community_type': self.community_type,
            'privacy_level': self.privacy_level,
            'member_count': self.member_count,
            'max_members': self.max_members,
            'is_full': self.is_full(),
            'is_active': self.is_active,
            'can_join': self.can_join(),
            'popularity_score': self.get_popularity_score(),
            'popularity_level': self.get_popularity_level(),
            'is_trending': self.is_trending(),
            'last_activity': self.last_activity,
            'is_recently_active': self.is_recently_active(),
            'tags': self.tags,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }

    @staticmethod
    def get_valid_community_types() -> List[str]:
        """Get list of valid community types"""
        return [
            "general", "academic", "professional", "hobby", "support",
            "research", "development", "education", "entertainment", "business"
        ]

    @staticmethod
    def get_valid_privacy_levels() -> List[str]:
        """Get list of valid privacy levels"""
        return ["public", "private", "secret"]