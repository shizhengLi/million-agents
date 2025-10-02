"""
Agent Repository implementation
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc
from datetime import datetime

from .base_repository import BaseRepository
from ..database.models import Agent


class AgentRepository(BaseRepository[Agent]):
    """Repository for Agent model operations"""

    def __init__(self, session: Session):
        super().__init__(session, Agent)

    def get_by_name(self, name: str) -> Optional[Agent]:
        """Get agent by name"""
        return self.session.query(Agent).filter(
            Agent.name == name
        ).first()

    def get_by_personality_type(self, personality_type: str) -> List[Agent]:
        """Get agents by personality type"""
        return self.session.query(Agent).filter(
            Agent.personality_type == personality_type
        ).all()

    def search_by_name(self, query: str, limit: int = 50) -> List[Agent]:
        """Search agents by name"""
        return self.session.query(Agent).filter(
            Agent.name.ilike(f"%{query}%")
        ).limit(limit).all()

    def get_by_trait_range(self, trait_name: str, min_value: float, max_value: float) -> List[Agent]:
        """Get agents by personality trait range"""
        if not hasattr(Agent, trait_name):
            return []

        trait_column = getattr(Agent, trait_name)
        return self.session.query(Agent).filter(
            and_(trait_column >= min_value, trait_column <= max_value)
        ).all()

    def get_created_after(self, cutoff_time: datetime) -> List[Agent]:
        """Get agents created after cutoff time"""
        return self.session.query(Agent).filter(
            Agent.created_at >= cutoff_time
        ).order_by(desc(Agent.created_at)).all()

    def get_explorers(self, limit: int = 50) -> List[Agent]:
        """Get agents with explorer personality type and high openness"""
        return self.session.query(Agent).filter(
            and_(
                Agent.personality_type == "explorer",
                Agent.openness >= 0.7
            )
        ).order_by(desc(Agent.openness)).limit(limit).all()

    def get_leaders(self, limit: int = 50) -> List[Agent]:
        """Get agents with leader personality type and high extraversion"""
        return self.session.query(Agent).filter(
            and_(
                Agent.personality_type == "leader",
                Agent.extraversion >= 0.7,
                Agent.conscientiousness >= 0.6
            )
        ).order_by(desc(Agent.extraversion)).limit(limit).all()

    def get_builders(self, limit: int = 50) -> List[Agent]:
        """Get agents with builder personality type and high conscientiousness"""
        return self.session.query(Agent).filter(
            and_(
                Agent.personality_type == "builder",
                Agent.conscientiousness >= 0.7
            )
        ).order_by(desc(Agent.conscientiousness)).limit(limit).all()

    def get_connectors(self, limit: int = 50) -> List[Agent]:
        """Get agents with connector personality type and high agreeableness"""
        return self.session.query(Agent).filter(
            and_(
                Agent.personality_type == "connector",
                Agent.agreeableness >= 0.7,
                Agent.extraversion >= 0.6
            )
        ).order_by(desc(Agent.agreeableness)).limit(limit).all()

    def get_innovators(self, limit: int = 50) -> List[Agent]:
        """Get agents with innovator personality type and high openness"""
        return self.session.query(Agent).filter(
            and_(
                Agent.personality_type == "innovator",
                Agent.openness >= 0.8,
                Agent.conscientiousness >= 0.6
            )
        ).order_by(desc(Agent.openness)).limit(limit).all()

    def get_agents_by_compatibility(self, target_agent: Agent, min_compatibility: float = 0.6) -> List[Dict[str, Any]]:
        """Get agents compatible with target agent based on personality traits"""
        compatible_agents = []
        all_agents = self.session.query(Agent).filter(
            Agent.id != target_agent.id
        ).all()

        for agent in all_agents:
            # Calculate simple compatibility score based on Big Five traits
            compatibility = self._calculate_personality_compatibility(target_agent, agent)

            if compatibility >= min_compatibility:
                compatible_agents.append({
                    'agent': agent,
                    'compatibility_score': compatibility
                })

        # Sort by compatibility score
        compatible_agents.sort(key=lambda x: x['compatibility_score'], reverse=True)
        return compatible_agents

    def get_agents_similar_to(self, target_agent: Agent, limit: int = 10) -> List[Agent]:
        """Get agents similar to target agent"""
        similar_agents = []

        # Find agents with same personality type
        same_type_agents = self.session.query(Agent).filter(
            and_(
                Agent.id != target_agent.id,
                Agent.personality_type == target_agent.personality_type
            )
        ).all()

        # If not enough same type, find similar trait profiles
        if len(same_type_agents) < limit:
            all_agents = self.session.query(Agent).filter(
                Agent.id != target_agent.id
            ).all()

            for agent in all_agents:
                similarity = self._calculate_personality_similarity(target_agent, agent)
                if similarity >= 0.7:  # 70% similarity threshold
                    similar_agents.append((agent, similarity))

            # Sort by similarity and take top results
            similar_agents.sort(key=lambda x: x[1], reverse=True)
            similar_agents = [agent for agent, _ in similar_agents[:limit]]
        else:
            similar_agents = same_type_agents[:limit]

        return similar_agents

    def get_personality_statistics(self) -> Dict[str, Any]:
        """Get personality statistics for all agents"""
        agents = self.get_all()
        if not agents:
            return {}

        total_agents = len(agents)

        # Personality type distribution
        type_counts = {}
        for agent in agents:
            personality_type = agent.personality_type
            type_counts[personality_type] = type_counts.get(personality_type, 0) + 1

        # Average trait scores
        traits = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
        trait_averages = {}
        for trait in traits:
            total = sum(getattr(agent, trait, 0) for agent in agents)
            trait_averages[trait] = round(total / total_agents, 3)

        return {
            'total_agents': total_agents,
            'personality_type_distribution': type_counts,
            'average_trait_scores': trait_averages,
            'most_common_type': max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else None
        }

    def create(self, data: Dict[str, Any]) -> Agent:
        """Override create to add validation"""
        # Validate personality type
        if 'personality_type' in data:
            valid_types = [
                "balanced", "explorer", "builder", "connector", "leader", "innovator"
            ]
            if data['personality_type'] not in valid_types:
                raise ValueError(f"Invalid personality_type: {data['personality_type']}. "
                               f"Must be one of: {valid_types}")

        # Validate trait ranges
        traits = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
        for trait in traits:
            if trait in data:
                value = data[trait]
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"{trait} must be between 0.0 and 1.0")

        return super().create(data)

    def update(self, record_id: int, data: Dict[str, Any]) -> Optional[Agent]:
        """Override update to add validation"""
        # Apply same validation as create for updated fields
        if 'personality_type' in data:
            valid_types = [
                "balanced", "explorer", "builder", "connector", "leader", "innovator"
            ]
            if data['personality_type'] not in valid_types:
                raise ValueError(f"Invalid personality_type: {data['personality_type']}. "
                               f"Must be one of: {valid_types}")

        # Validate trait ranges
        traits = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
        for trait in traits:
            if trait in data:
                value = data[trait]
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"{trait} must be between 0.0 and 1.0")

        return super().update(record_id, data)

    def get_agents_with_friendships_count(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get agents with their friendship counts"""
        from sqlalchemy import func

        # Count initiated friendships
        initiated_count = self.session.query(
            Agent.id,
            Agent.name,
            func.count(Agent.initiated_friendships).label('initiated_count')
        ).outerjoin(Agent.initiated_friendships).group_by(Agent.id).all()

        # Count received friendships
        received_count = self.session.query(
            Agent.id,
            func.count(Agent.received_friendships).label('received_count')
        ).outerjoin(Agent.received_friendships).group_by(Agent.id).all()

        # Combine results
        result = []
        for agent_id, name, init_count in initiated_count:
            received = next((rec_count for ag_id, rec_count in received_count if ag_id == agent_id), 0)
            result.append({
                'agent_id': agent_id,
                'name': name,
                'initiated_friendships': init_count,
                'received_friendships': received,
                'total_friendships': init_count + received
            })

        # Sort by total friendships and limit
        result.sort(key=lambda x: x['total_friendships'], reverse=True)
        return result[:limit]

    def _calculate_personality_compatibility(self, agent1: Agent, agent2: Agent) -> float:
        """Calculate compatibility score between two agents"""
        # Simple compatibility based on complementary traits
        # High agreeableness is generally compatible with most types
        # Extraversion complementarity
        # Openness similarity for shared interests

        compatibility = 0.0

        # Openness similarity (shared interests)
        openness_diff = abs(agent1.openness - agent2.openness)
        openness_score = 1.0 - openness_diff
        compatibility += openness_score * 0.3

        # Extraversion complementarity
        # Moderate difference in extraversion can be complementary
        extraversion_diff = abs(agent1.extraversion - agent2.extraversion)
        if extraversion_diff <= 0.3:  # Similar
            extraversion_score = 0.8
        elif extraversion_diff <= 0.6:  # Complementary
            extraversion_score = 1.0
        else:  # Too different
            extraversion_score = 0.3
        compatibility += extraversion_score * 0.3

        # Agreeableness bonus (high agreeableness is generally good)
        agreeableness_score = (agent1.agreeableness + agent2.agreeableness) / 2.0
        compatibility += agreeableness_score * 0.2

        # Conscientiousness similarity (work ethic compatibility)
        conscientiousness_diff = abs(agent1.conscientiousness - agent2.conscientiousness)
        conscientiousness_score = 1.0 - conscientiousness_diff
        compatibility += conscientiousness_score * 0.2

        return round(compatibility, 3)

    def _calculate_personality_similarity(self, agent1: Agent, agent2: Agent) -> float:
        """Calculate personality similarity between two agents"""
        traits = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
        total_diff = 0.0

        for trait in traits:
            trait1 = getattr(agent1, trait, 0.5)
            trait2 = getattr(agent2, trait, 0.5)
            total_diff += abs(trait1 - trait2)

        # Convert difference to similarity (0-1 scale)
        avg_diff = total_diff / len(traits)
        similarity = 1.0 - avg_diff

        return round(similarity, 3)