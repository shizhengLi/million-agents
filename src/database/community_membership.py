"""
CommunityMembership model for million-agent social platform
"""

from datetime import datetime, timedelta
from typing import List, Optional
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, UniqueConstraint, CheckConstraint
from sqlalchemy.orm import relationship
from .config import Base


class CommunityMembership(Base):
    """CommunityMembership model representing agent membership in communities"""

    __tablename__ = "community_memberships"

    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=False, index=True)
    community_id = Column(Integer, ForeignKey("communities.id"), nullable=False, index=True)

    # Membership properties
    role = Column(String(20), nullable=False, default="member", index=True)
    is_active = Column(Boolean, nullable=False, default=True, index=True)
    contribution_score = Column(Float, nullable=False, default=0.0, index=True)

    # Activity tracking
    last_activity = Column(DateTime, nullable=True)
    joined_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    agent = relationship("Agent", back_populates="community_memberships")
    community = relationship("Community", back_populates="members")

    __table_args__ = (
        # Prevent duplicate memberships
        UniqueConstraint('agent_id', 'community_id', name='uq_agent_community'),
        # Validate role
        CheckConstraint("role IN ('member', 'moderator', 'admin')", name='ck_membership_role'),
        # Validate contribution score
        CheckConstraint('contribution_score >= 0', name='ck_contribution_score_range'),
        {"sqlite_autoincrement": True}
    )

    def __init__(self, agent_id: int, community_id: int, **kwargs):
        """Initialize community membership with validation"""
        self.agent_id = agent_id
        self.community_id = community_id

        # Validate role
        valid_roles = ["member", "moderator", "admin"]
        role = kwargs.get('role', 'member')
        if role not in valid_roles:
            raise ValueError(f"Invalid role: {role}. Must be one of: {valid_roles}")
        self.role = role

        # Set other attributes
        self.is_active = kwargs.get('is_active', True)
        self.contribution_score = kwargs.get('contribution_score', 0.0)
        self.last_activity = kwargs.get('last_activity')

        # Validate contribution score
        if self.contribution_score < 0:
            raise ValueError("contribution_score cannot be negative")

    def __str__(self):
        """String representation of community membership"""
        agent_name = getattr(self.agent, 'name', str(self.agent_id))
        community_name = getattr(self.community, 'name', str(self.community_id))
        return f"CommunityMembership({agent_name} in {community_name}, role='{self.role}', contribution={self.contribution_score})"

    def __repr__(self):
        """Detailed representation of community membership"""
        return (f"CommunityMembership(id={self.id}, agent={self.agent_id}, "
                f"community={self.community_id}, role='{self.role}', "
                f"active={self.is_active}, contribution={self.contribution_score})")

    def update_role(self, new_role: str):
        """Update membership role with validation"""
        valid_roles = ["member", "moderator", "admin"]
        if new_role not in valid_roles:
            raise ValueError(f"Invalid role: {new_role}. Must be one of: {valid_roles}")

        self.role = new_role
        self.updated_at = datetime.utcnow()

    def activate(self):
        """Activate membership"""
        self.is_active = True
        self.updated_at = datetime.utcnow()
        self.record_activity()

    def deactivate(self):
        """Deactivate membership"""
        self.is_active = False
        self.updated_at = datetime.utcnow()

    def add_contribution(self, score: float):
        """Add contribution score with validation"""
        if score < 0:
            raise ValueError("Contribution score to add cannot be negative")

        self.contribution_score += score
        self.updated_at = datetime.utcnow()
        self.record_activity()

    def remove_contribution(self, score: float):
        """Remove contribution score with validation"""
        if score < 0:
            raise ValueError("Contribution score to remove cannot be negative")

        new_score = self.contribution_score - score
        self.contribution_score = max(0, new_score)  # Prevent negative score
        self.updated_at = datetime.utcnow()

    def record_activity(self):
        """Record member activity"""
        self.last_activity = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def is_active_member(self) -> bool:
        """Check if member is active"""
        return self.is_active

    def is_contributor(self, threshold: float = 20.0) -> bool:
        """Check if member is a contributor based on contribution score"""
        return self.contribution_score >= threshold

    def is_moderator(self) -> bool:
        """Check if member is a moderator"""
        return self.role in ["moderator", "admin"]

    def is_admin(self) -> bool:
        """Check if member is an admin"""
        return self.role == "admin"

    def get_membership_duration(self) -> timedelta:
        """Get duration of membership"""
        return datetime.utcnow() - self.joined_at

    def get_days_since_last_activity(self) -> int:
        """Get days since last activity"""
        if self.last_activity is None:
            return (datetime.utcnow() - self.joined_at).days
        return (datetime.utcnow() - self.last_activity).days

    def is_inactive_member(self, days: int = 30) -> bool:
        """Check if member is inactive (no recent activity)"""
        return not self.is_active or self.get_days_since_last_activity() > days

    def can_moderate(self) -> bool:
        """Check if member can moderate the community"""
        return self.is_active and self.is_moderator()

    def can_admin(self) -> bool:
        """Check if member can administrate the community"""
        return self.is_active and self.is_admin()

    def get_membership_level(self) -> str:
        """Get membership level based on contribution and role"""
        if self.is_admin():
            return "admin"
        elif self.is_moderator():
            return "moderator"
        elif self.is_contributor(50.0):
            return "top_contributor"
        elif self.is_contributor(20.0):
            return "contributor"
        elif self.is_contributor(5.0):
            return "active_member"
        else:
            return "member"

    def get_membership_summary(self) -> dict:
        """Get comprehensive membership summary"""
        return {
            'id': self.id,
            'agent_id': self.agent_id,
            'community_id': self.community_id,
            'role': self.role,
            'is_active': self.is_active,
            'contribution_score': self.contribution_score,
            'membership_level': self.get_membership_level(),
            'is_contributor': self.is_contributor(),
            'is_moderator': self.is_moderator(),
            'is_admin': self.is_admin(),
            'can_moderate': self.can_moderate(),
            'can_admin': self.can_admin(),
            'last_activity': self.last_activity,
            'days_since_last_activity': self.get_days_since_last_activity(),
            'is_inactive_member': self.is_inactive_member(),
            'joined_at': self.joined_at,
            'membership_duration_days': self.get_membership_duration().days,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }

    @staticmethod
    def get_valid_roles() -> List[str]:
        """Get list of valid membership roles"""
        return ["member", "moderator", "admin"]

    @staticmethod
    def get_active_members(session, community_id: int, limit: int = 50) -> List['CommunityMembership']:
        """Get active members of a community"""
        return session.query(CommunityMembership).filter(
            CommunityMembership.community_id == community_id,
            CommunityMembership.is_active == True
        ).limit(limit).all()

    @staticmethod
    def get_members_by_role(session, community_id: int, role: str, limit: int = 50) -> List['CommunityMembership']:
        """Get members of a community by role"""
        return session.query(CommunityMembership).filter(
            CommunityMembership.community_id == community_id,
            CommunityMembership.role == role,
            CommunityMembership.is_active == True
        ).limit(limit).all()

    @staticmethod
    def get_top_contributors(session, community_id: int, limit: int = 10) -> List['CommunityMembership']:
        """Get top contributors of a community"""
        return session.query(CommunityMembership).filter(
            CommunityMembership.community_id == community_id,
            CommunityMembership.is_active == True
        ).order_by(CommunityMembership.contribution_score.desc()).limit(limit).all()

    @staticmethod
    def get_inactive_members(session, community_id: int, days: int = 30, limit: int = 50) -> List['CommunityMembership']:
        """Get inactive members of a community"""
        cutoff_time = datetime.utcnow() - timedelta(days=days)

        return session.query(CommunityMembership).filter(
            CommunityMembership.community_id == community_id,
            CommunityMembership.is_active == True,
            (CommunityMembership.last_activity < cutoff_time) |
            (CommunityMembership.last_activity.is_(None) & (CommunityMembership.joined_at < cutoff_time))
        ).limit(limit).all()

    @staticmethod
    def get_membership_for_agent(session, agent_id: int, community_id: int) -> Optional['CommunityMembership']:
        """Get membership for a specific agent in a community"""
        return session.query(CommunityMembership).filter(
            CommunityMembership.agent_id == agent_id,
            CommunityMembership.community_id == community_id
        ).first()

    @staticmethod
    def get_agent_memberships(session, agent_id: int, active_only: bool = True, limit: int = 50) -> List['CommunityMembership']:
        """Get all community memberships for an agent"""
        query = session.query(CommunityMembership).filter(
            CommunityMembership.agent_id == agent_id
        )

        if active_only:
            query = query.filter(CommunityMembership.is_active == True)

        return query.limit(limit).all()

    def promote_to_moderator(self):
        """Promote member to moderator"""
        if self.role == "admin":
            raise ValueError("Admin cannot be promoted to moderator")
        self.update_role("moderator")

    def promote_to_admin(self):
        """Promote member to admin"""
        self.update_role("admin")

    def demote_to_member(self):
        """Demote member to regular member"""
        self.update_role("member")

    def reset_contribution_score(self):
        """Reset contribution score to zero"""
        self.contribution_score = 0.0
        self.updated_at = datetime.utcnow()