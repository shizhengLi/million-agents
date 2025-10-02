"""
Community Repository implementation
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc, func
from datetime import datetime, timedelta

from .base_repository import BaseRepository
from ..database.models import Community


class CommunityRepository(BaseRepository[Community]):
    """Repository for Community model operations"""

    def __init__(self, session: Session):
        super().__init__(session, Community)

    def get_by_name(self, name: str) -> Optional[Community]:
        """Get community by name"""
        return self.session.query(Community).filter(
            Community.name == name
        ).first()

    def get_by_type(self, community_type: str, active_only: bool = True) -> List[Community]:
        """Get communities by type"""
        query = self.session.query(Community).filter(
            Community.community_type == community_type
        )

        if active_only:
            query = query.filter(Community.is_active == True)

        return query.order_by(desc(Community.member_count)).all()

    def get_by_privacy_level(self, privacy_level: str, active_only: bool = True) -> List[Community]:
        """Get communities by privacy level"""
        query = self.session.query(Community).filter(
            Community.privacy_level == privacy_level
        )

        if active_only:
            query = query.filter(Community.is_active == True)

        return query.order_by(desc(Community.member_count)).all()

    def search_by_name(self, query: str, limit: int = 50) -> List[Community]:
        """Search communities by name"""
        return self.session.query(Community).filter(
            Community.name.ilike(f"%{query}%")
        ).limit(limit).all()

    def get_active_communities(self, limit: int = 50) -> List[Community]:
        """Get active communities"""
        return self.session.query(Community).filter(
            Community.is_active == True
        ).order_by(desc(Community.member_count)).limit(limit).all()

    def get_trending_communities(self, days: int = 7, limit: int = 10) -> List[Community]:
        """Get trending communities (recently active with good member count)"""
        cutoff_time = datetime.utcnow() - timedelta(days=days)

        return self.session.query(Community).filter(
            and_(
                Community.is_active == True,
                Community.last_activity >= cutoff_time,
                Community.member_count >= 10
            )
        ).order_by(desc(Community.member_count)).limit(limit).all()

    def search_by_tags(self, tags: List[str], limit: int = 20) -> List[Community]:
        """Search communities by tags"""
        if not tags:
            return []

        # Build conditions for each tag
        conditions = []
        for tag in tags:
            conditions.append(Community.tags.contains([tag]))

        # Use OR condition (communities matching any tag)
        query_filter = or_(*conditions)

        return self.session.query(Community).filter(
            and_(
                query_filter,
                Community.is_active == True
            )
        ).limit(limit).all()

    def get_by_member_count_range(self, min_members: int, max_members: int) -> List[Community]:
        """Get communities within member count range"""
        return self.session.query(Community).filter(
            and_(
                Community.member_count >= min_members,
                Community.member_count <= max_members,
                Community.is_active == True
            )
        ).order_by(desc(Community.member_count)).all()

    def get_created_after(self, cutoff_time: datetime) -> List[Community]:
        """Get communities created after cutoff time"""
        return self.session.query(Community).filter(
            Community.created_at >= cutoff_time
        ).order_by(desc(Community.created_at)).all()

    def get_communities_with_popularity_score(self, min_score: float = 0.0, limit: int = 50) -> List[Community]:
        """Get communities with minimum popularity score"""
        communities = self.get_active_communities(limit * 2)  # Get more to filter

        result = []
        for community in communities:
            if community.get_popularity_score() >= min_score:
                result.append(community)

        # Sort by popularity score
        result.sort(key=lambda x: x.get_popularity_score(), reverse=True)
        return result[:limit]

    def get_communities_by_popularity_level(self, level: str, limit: int = 50) -> List[Community]:
        """Get communities by popularity level"""
        communities = self.get_active_communities(limit * 2)

        result = []
        for community in communities:
            if community.get_popularity_level() == level:
                result.append(community)

        # Sort by member count as secondary sort
        result.sort(key=lambda x: x.member_count, reverse=True)
        return result[:limit]

    def update_member_count(self, community_id: int, delta: int) -> bool:
        """Update community member count"""
        community = self.get_by_id(community_id)
        if not community:
            return False

        community.update_member_count(delta)
        self.session.commit()
        return True

    def get_communities_needing_moderators(self, min_members: int = 50, limit: int = 20) -> List[Community]:
        """Get communities that might need moderators (large but may lack admin structure)"""
        return self.session.query(Community).filter(
            and_(
                Community.is_active == True,
                Community.member_count >= min_members,
                Community.community_type.in_(['general', 'academic', 'professional'])
            )
        ).order_by(desc(Community.member_count)).limit(limit).all()

    def get_communities_with_statistics(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get communities with detailed statistics"""
        communities = self.get_active_communities(limit * 2)

        result = []
        for community in communities:
            result.append({
                'community': community,
                'popularity_score': community.get_popularity_score(),
                'popularity_level': community.get_popularity_level(),
                'is_trending': community.is_trending(),
                'is_full': community.is_full(),
                'can_join': community.can_join(),
                'days_since_last_activity': (datetime.utcnow() - community.last_activity).days,
                'membership_ratio': community.member_count / community.max_members if community.max_members > 0 else 0,
                'age_days': (datetime.utcnow() - community.created_at).days
            })

        # Sort by popularity score
        result.sort(key=lambda x: x['popularity_score'], reverse=True)
        return result[:limit]

    def get_community_types_distribution(self) -> Dict[str, int]:
        """Get distribution of community types"""
        result = self.session.query(
            Community.community_type,
            func.count(Community.id).label('count')
        ).filter(Community.is_active == True).group_by(Community.community_type).all()

        return {row.community_type: row.count for row in result}

    def get_privacy_levels_distribution(self) -> Dict[str, int]:
        """Get distribution of privacy levels"""
        result = self.session.query(
            Community.privacy_level,
            func.count(Community.id).label('count')
        ).filter(Community.is_active == True).group_by(Community.privacy_level).all()

        return {row.privacy_level: row.count for row in result}

    def get_growth_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get community growth statistics"""
        cutoff_time = datetime.utcnow() - timedelta(days=days)

        # Communities created in period
        new_communities = self.session.query(Community).filter(
            Community.created_at >= cutoff_time
        ).count()

        # Total communities
        total_communities = self.count()

        # Average member count for new vs old communities
        new_avg = self.session.query(
            func.avg(Community.member_count)
        ).filter(Community.created_at >= cutoff_time).scalar() or 0

        old_avg = self.session.query(
            func.avg(Community.member_count)
        ).filter(Community.created_at < cutoff_time).scalar() or 0

        return {
            'period_days': days,
            'new_communities': new_communities,
            'total_communities': total_communities,
            'new_avg_members': round(new_avg, 2),
            'established_avg_members': round(old_avg, 2),
            'growth_rate': round((new_communities / max(total_communities - new_communities, 1)) * 100, 2)
        }

    def find_similar_communities(self, community_id: int, limit: int = 5) -> List[Community]:
        """Find communities similar to the given community"""
        target_community = self.get_by_id(community_id)
        if not target_community:
            return []

        # Find communities with same type
        same_type = self.session.query(Community).filter(
            and_(
                Community.id != community_id,
                Community.community_type == target_community.community_type,
                Community.is_active == True
            )
        ).all()

        # If enough same type communities, return top by member count
        if len(same_type) >= limit:
            return sorted(same_type, key=lambda x: x.member_count, reverse=True)[:limit]

        # Otherwise, find communities with similar tags
        similar_by_tags = []
        if target_community.tags:
            for community in self.get_active_communities(limit * 3):
                if community.id != community_id:
                    # Calculate tag similarity
                    common_tags = set(target_community.tags) & set(community.tags)
                    if common_tags:
                        similarity_score = len(common_tags) / max(len(target_community.tags), len(community.tags))
                        if similarity_score >= 0.3:  # At least 30% tag overlap
                            similar_by_tags.append((community, similarity_score))

        # Sort by similarity
        similar_by_tags.sort(key=lambda x: x[1], reverse=True)

        # Combine results
        result = same_type.copy()
        for community, _ in similar_by_tags:
            if community not in result and len(result) < limit:
                result.append(community)

        return result[:limit]

    def create(self, data: Dict[str, Any]) -> Community:
        """Override create to add validation"""
        # Validate community type
        if 'community_type' in data:
            valid_types = Community.get_valid_community_types()
            if data['community_type'] not in valid_types:
                raise ValueError(f"Invalid community_type: {data['community_type']}. "
                               f"Must be one of: {valid_types}")

        # Validate privacy level
        if 'privacy_level' in data:
            valid_privacy = Community.get_valid_privacy_levels()
            if data['privacy_level'] not in valid_privacy:
                raise ValueError(f"Invalid privacy_level: {data['privacy_level']}. "
                               f"Must be one of: {valid_privacy}")

        # Validate max_members
        if 'max_members' in data:
            max_members = data['max_members']
            if not 1 <= max_members <= 10000000:
                raise ValueError("max_members must be between 1 and 10,000,000")

        return super().create(data)

    def update(self, record_id: int, data: Dict[str, Any]) -> Optional[Community]:
        """Override update to add validation"""
        # Apply same validation as create for updated fields
        if 'community_type' in data:
            valid_types = Community.get_valid_community_types()
            if data['community_type'] not in valid_types:
                raise ValueError(f"Invalid community_type: {data['community_type']}. "
                               f"Must be one of: {valid_types}")

        if 'privacy_level' in data:
            valid_privacy = Community.get_valid_privacy_levels()
            if data['privacy_level'] not in valid_privacy:
                raise ValueError(f"Invalid privacy_level: {data['privacy_level']}. "
                               f"Must be one of: {valid_privacy}")

        if 'max_members' in data:
            max_members = data['max_members']
            if not 1 <= max_members <= 10000000:
                raise ValueError("max_members must be between 1 and 10,000,000")

        return super().update(record_id, data)