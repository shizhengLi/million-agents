"""
Friendship Repository implementation
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc, func
from datetime import datetime, timedelta

from .base_repository import BaseRepository
from ..database.models import Friendship


class FriendshipRepository(BaseRepository[Friendship]):
    """Repository for Friendship model operations"""

    def __init__(self, session: Session):
        super().__init__(session, Friendship)

    def get_friendships_for_agent(self, agent_id: int, active_only: bool = False) -> List[Friendship]:
        """Get all friendships for a specific agent"""
        query = self.session.query(Friendship).filter(
            or_(
                Friendship.initiator_id == agent_id,
                Friendship.recipient_id == agent_id
            )
        )

        if active_only:
            query = query.filter(Friendship.friendship_status == "accepted")

        return query.order_by(desc(Friendship.strength_level)).all()

    def get_active_friendships(self, limit: int = 50) -> List[Friendship]:
        """Get all active (accepted) friendships"""
        return self.session.query(Friendship).filter(
            Friendship.friendship_status == "accepted"
        ).order_by(desc(Friendship.strength_level)).limit(limit).all()

    def get_by_status(self, status: str, limit: int = 50) -> List[Friendship]:
        """Get friendships by status"""
        return self.session.query(Friendship).filter(
            Friendship.friendship_status == status
        ).order_by(desc(Friendship.created_at)).limit(limit).all()

    def update_status(self, friendship_id: int, new_status: str) -> Optional[Friendship]:
        """Update friendship status"""
        friendship = self.get_by_id(friendship_id)
        if not friendship:
            return None

        friendship.update_status(new_status)
        self.session.commit()
        self.session.refresh(friendship)
        return friendship

    def update_strength_level(self, friendship_id: int, new_level: float) -> Optional[Friendship]:
        """Update friendship strength level"""
        friendship = self.get_by_id(friendship_id)
        if not friendship:
            return None

        friendship.update_strength_level(new_level)
        self.session.commit()
        self.session.refresh(friendship)
        return friendship

    def get_friendship_between_agents(self, agent1_id: int, agent2_id: int) -> Optional[Friendship]:
        """Get friendship between two specific agents (either direction)"""
        return self.session.query(Friendship).filter(
            or_(
                and_(Friendship.initiator_id == agent1_id, Friendship.recipient_id == agent2_id),
                and_(Friendship.initiator_id == agent2_id, Friendship.recipient_id == agent1_id)
            )
        ).first()

    def get_strong_friendships(self, threshold: float = 0.7, limit: int = 50) -> List[Friendship]:
        """Get strong friendships above threshold"""
        return self.session.query(Friendship).filter(
            and_(
                Friendship.friendship_status == "accepted",
                Friendship.strength_level >= threshold
            )
        ).order_by(desc(Friendship.strength_level)).limit(limit).all()

    def get_pending_friendships_for_agent(self, agent_id: int, limit: int = 50) -> List[Friendship]:
        """Get pending friendships where agent is recipient"""
        return self.session.query(Friendship).filter(
            and_(
                Friendship.recipient_id == agent_id,
                Friendship.friendship_status == "pending"
            )
        ).order_by(desc(Friendship.created_at)).limit(limit).all()

    def record_interaction(self, friendship_id: int) -> bool:
        """Record interaction for a friendship"""
        friendship = self.get_by_id(friendship_id)
        if not friendship:
            return False

        friendship.record_interaction()
        self.session.commit()
        return True

    def get_friendships_by_strength_range(self, min_strength: float, max_strength: float, limit: int = 50) -> List[Friendship]:
        """Get friendships within strength range"""
        return self.session.query(Friendship).filter(
            and_(
                Friendship.friendship_status == "accepted",
                Friendship.strength_level >= min_strength,
                Friendship.strength_level <= max_strength
            )
        ).order_by(desc(Friendship.strength_level)).limit(limit).all()

    def get_stale_friendships(self, days: int = 30, limit: int = 50) -> List[Friendship]:
        """Get stale friendships (no recent interaction)"""
        cutoff_time = datetime.utcnow() - timedelta(days=days)

        return self.session.query(Friendship).filter(
            and_(
                Friendship.friendship_status == "accepted",
                or_(
                    Friendship.last_interaction < cutoff_time,
                    Friendship.last_interaction.is_(None)
                )
            )
        ).order_by(asc(Friendship.last_interaction)).limit(limit).all()

    def get_recently_created_friendships(self, days: int = 7, limit: int = 50) -> List[Friendship]:
        """Get friendships created recently"""
        cutoff_time = datetime.utcnow() - timedelta(days=days)

        return self.session.query(Friendship).filter(
            Friendship.created_at >= cutoff_time
        ).order_by(desc(Friendship.created_at)).limit(limit).all()

    def get_friendships_by_interaction_count_range(self, min_interactions: int, max_interactions: int, limit: int = 50) -> List[Friendship]:
        """Get friendships within interaction count range"""
        return self.session.query(Friendship).filter(
            and_(
                Friendship.friendship_status == "accepted",
                Friendship.interaction_count >= min_interactions,
                Friendship.interaction_count <= max_interactions
            )
        ).order_by(desc(Friendship.interaction_count)).limit(limit).all()

    def get_top_interacted_friendships(self, limit: int = 50) -> List[Friendship]:
        """Get friendships with highest interaction counts"""
        return self.session.query(Friendship).filter(
            Friendship.friendship_status == "accepted"
        ).order_by(desc(Friendship.interaction_count)).limit(limit).all()

    def get_friendship_statistics(self) -> Dict[str, Any]:
        """Get friendship statistics"""
        total_friendships = self.count()

        # Status distribution
        status_counts = self.session.query(
            Friendship.friendship_status,
            Friendship.id.count().label('count')
        ).group_by(Friendship.friendship_status).all()

        status_distribution = {row.friendship_status: row.count for row in status_counts}

        # Average strength level for active friendships
        avg_strength = self.session.query(
            Friendship.id,
            func.avg(Friendship.strength_level).label('avg_strength')
        ).filter(Friendship.friendship_status == "accepted").first()

        # Average interaction count
        avg_interactions = self.session.query(
            func.avg(Friendship.interaction_count).label('avg_interactions')
        ).filter(Friendship.friendship_status == "accepted").scalar() or 0

        # Strong friendships count
        strong_count = self.session.query(Friendship).filter(
            and_(
                Friendship.friendship_status == "accepted",
                Friendship.strength_level >= 0.7
            )
        ).count()

        return {
            'total_friendships': total_friendships,
            'status_distribution': status_distribution,
            'average_strength': round(avg_strength.avg_strength or 0, 3) if avg_strength else 0,
            'average_interactions': round(avg_interactions, 2),
            'strong_friendships_count': strong_count,
            'strong_friendships_percentage': round((strong_count / max(total_friendships, 1)) * 100, 2)
        }

    def get_agent_friendship_statistics(self, agent_id: int) -> Dict[str, Any]:
        """Get friendship statistics for a specific agent"""
        agent_friendships = self.get_friendships_for_agent(agent_id)

        total_friendships = len(agent_friendships)
        active_friendships = len([f for f in agent_friendships if f.is_active()])
        initiated_count = len([f for f in agent_friendships if f.initiator_id == agent_id])
        received_count = len([f for f in agent_friendships if f.recipient_id == agent_id])

        if active_friendships > 0:
            avg_strength = sum(f.strength_level for f in agent_friendships if f.is_active()) / active_friendships
            total_interactions = sum(f.interaction_count for f in agent_friendships if f.is_active())
            avg_interactions = total_interactions / active_friendships
        else:
            avg_strength = 0
            avg_interactions = 0

        strong_friendships = len([f for f in agent_friendships if f.is_active() and f.is_strong()])

        return {
            'agent_id': agent_id,
            'total_friendships': total_friendships,
            'active_friendships': active_friendships,
            'initiated_friendships': initiated_count,
            'received_friendships': received_count,
            'average_strength': round(avg_strength, 3),
            'average_interactions': round(avg_interactions, 2),
            'strong_friendships': strong_friendships,
            'friendship_activity_ratio': round(active_friendships / max(total_friendships, 1), 2)
        }

    def find_potential_friends(self, agent_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Find potential friends for an agent based on mutual connections"""
        from ..database.models import Agent

        # Get current friends
        current_friend_ids = set()
        for friendship in self.get_friendships_for_agent(agent_id):
            current_friend_ids.add(friendship.initiator_id if friendship.recipient_id == agent_id else friendship.recipient_id)

        # Find agents who are friends with current friends (friends of friends)
        potential_candidates = {}
        for friend_id in current_friend_ids:
            # Get friends of this friend
            friend_friendships = self.get_friendships_for_agent(friend_id, active_only=True)

            for friendship in friend_friendships:
                # Get the other agent in the friendship
                potential_friend_id = friendship.initiator_id if friendship.recipient_id == friend_id else friendship.recipient_id

                # Skip if it's the original agent or already a friend
                if potential_friend_id == agent_id or potential_friend_id in current_friend_ids:
                    continue

                # Count mutual connections
                if potential_friend_id not in potential_candidates:
                    potential_candidates[potential_friend_id] = 0
                potential_candidates[potential_friend_id] += 1

        # Sort by number of mutual connections
        sorted_candidates = sorted(potential_candidates.items(), key=lambda x: x[1], reverse=True)

        # Get full agent objects
        result = []
        for candidate_id, mutual_count in sorted_candidates[:limit]:
            candidate_agent = self.session.query(Agent).filter(Agent.id == candidate_id).first()
            if candidate_agent:
                result.append({
                    'agent': candidate_agent,
                    'mutual_connections': mutual_count
                })

        return result

    def decay_strength_for_inactive_friendships(self, decay_rate: float = 0.1, max_decay: float = 0.3) -> int:
        """Apply strength decay to inactive friendships"""
        stale_friendships = self.get_stale_friendships(days=30, limit=1000)
        updated_count = 0

        for friendship in stale_friendships:
            friendship.decay_strength(decay_rate, max_decay)
            updated_count += 1

        if updated_count > 0:
            self.session.commit()

        return updated_count

    def create(self, data: Dict[str, Any]) -> Friendship:
        """Override create to add validation and prevent self-friendship"""
        # Prevent self-friendship
        if data.get('initiator_id') == data.get('recipient_id'):
            raise ValueError("Agent cannot be friends with themselves")

        # Validate friendship status
        if 'friendship_status' in data:
            valid_statuses = Friendship.get_valid_statuses()
            if data['friendship_status'] not in valid_statuses:
                raise ValueError(f"Invalid friendship_status: {data['friendship_status']}. "
                               f"Must be one of: {valid_statuses}")

        # Validate strength level
        if 'strength_level' in data:
            strength_level = data['strength_level']
            if not 0.0 <= strength_level <= 1.0:
                raise ValueError("strength_level must be between 0.0 and 1.0")

        return super().create(data)

    def update(self, record_id: int, data: Dict[str, Any]) -> Optional[Friendship]:
        """Override update to add validation"""
        # Apply same validation as create for updated fields
        if 'friendship_status' in data:
            valid_statuses = Friendship.get_valid_statuses()
            if data['friendship_status'] not in valid_statuses:
                raise ValueError(f"Invalid friendship_status: {data['friendship_status']}. "
                               f"Must be one of: {valid_statuses}")

        if 'strength_level' in data:
            strength_level = data['strength_level']
            if not 0.0 <= strength_level <= 1.0:
                raise ValueError("strength_level must be between 0.0 and 1.0")

        return super().update(record_id, data)

    def get_friendships_created_after(self, cutoff_time: datetime) -> List[Friendship]:
        """Get friendships created after cutoff time"""
        return self.session.query(Friendship).filter(
            Friendship.created_at >= cutoff_time
        ).order_by(desc(Friendship.created_at)).all()