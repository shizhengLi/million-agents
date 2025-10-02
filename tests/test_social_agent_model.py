"""
SocialAgent model tests
"""

import pytest
from datetime import datetime
from unittest.mock import patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session


class TestSocialAgentModel:
    """Test SocialAgent ORM model"""

    def setup_method(self):
        """Set up test database session"""
        from sqlalchemy import create_engine as sa_create_engine
        from src.database.models import Base

        # Create in-memory SQLite database for testing
        self.engine = sa_create_engine('sqlite:///:memory:')
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def teardown_method(self):
        """Clean up after test"""
        self.engine.dispose()

    def test_social_agent_model_creation(self):
        """Test SocialAgent model creation"""
        from src.database.models import Agent, SocialAgent

        with self.SessionLocal() as session:
            # Create base agent first
            agent = Agent(
                name="social_test_agent",
                personality_type="connector",
                openness=0.8,
                extraversion=0.9,
                agreeableness=0.7
            )
            session.add(agent)
            session.commit()
            session.refresh(agent)

            # Create social agent
            social_agent = SocialAgent(
                agent_id=agent.id,
                bio="Test bio for social agent",
                avatar_url="https://example.com/avatar.png",
                reputation_score=75.5,
                activity_level=0.8,
                social_preference="extroverted",
                communication_style="friendly"
            )

            session.add(social_agent)
            session.commit()
            session.refresh(social_agent)

            assert social_agent.id is not None
            assert social_agent.agent_id == agent.id
            assert social_agent.bio == "Test bio for social agent"
            assert social_agent.avatar_url == "https://example.com/avatar.png"
            assert social_agent.reputation_score == 75.5
            assert social_agent.activity_level == 0.8
            assert social_agent.social_preference == "extroverted"
            assert social_agent.communication_style == "friendly"
            assert social_agent.created_at is not None
            assert social_agent.updated_at is not None

    def test_social_agent_model_defaults(self):
        """Test SocialAgent model default values"""
        from src.database.models import Agent, SocialAgent

        with self.SessionLocal() as session:
            # Create base agent first
            agent = Agent(name="minimal_agent")
            session.add(agent)
            session.commit()
            session.refresh(agent)

            # Create social agent with minimal data
            social_agent = SocialAgent(agent_id=agent.id)
            session.add(social_agent)
            session.commit()
            session.refresh(social_agent)

            assert social_agent.bio is None
            assert social_agent.avatar_url is None
            assert social_agent.reputation_score == 50.0  # Default
            assert social_agent.activity_level == 0.5  # Default
            assert social_agent.social_preference == "balanced"  # Default
            assert social_agent.communication_style == "neutral"  # Default
            assert social_agent.friends_count == 0
            assert social_agent.interactions_count == 0
            assert social_agent.communities_count == 0

    def test_social_agent_model_validation(self):
        """Test SocialAgent model field validation"""
        from src.database.models import Agent, SocialAgent

        with self.SessionLocal() as session:
            # Create base agent first
            agent = Agent(name="validation_agent")
            session.add(agent)
            session.commit()
            session.refresh(agent)

            # Test invalid social preference
            with pytest.raises(ValueError):
                social_agent = SocialAgent(
                    agent_id=agent.id,
                    social_preference="invalid_preference"
                )

            # Test invalid communication style
            with pytest.raises(ValueError):
                social_agent = SocialAgent(
                    agent_id=agent.id,
                    communication_style="invalid_style"
                )

            # Test invalid reputation score (out of bounds)
            with pytest.raises(ValueError):
                social_agent = SocialAgent(
                    agent_id=agent.id,
                    reputation_score=150.0  # Should be 0-100
                )

            # Test invalid activity level (out of bounds)
            with pytest.raises(ValueError):
                social_agent = SocialAgent(
                    agent_id=agent.id,
                    activity_level=1.5  # Should be 0-1
                )

    def test_social_agent_agent_relationship(self):
        """Test SocialAgent to Agent relationship"""
        from src.database.models import Agent, SocialAgent

        with self.SessionLocal() as session:
            # Create base agent
            agent = Agent(name="relationship_agent")
            session.add(agent)
            session.commit()
            session.refresh(agent)

            # Create social agent
            social_agent = SocialAgent(
                agent_id=agent.id,
                bio="Testing relationship"
            )
            session.add(social_agent)
            session.commit()
            session.refresh(social_agent)

            # Test bidirectional relationship
            assert social_agent.agent.id == agent.id
            assert social_agent.agent.name == "relationship_agent"

    def test_social_agent_unique_agent(self):
        """Test SocialAgent unique agent constraint"""
        from src.database.models import Agent, SocialAgent

        with self.SessionLocal() as session:
            # Create base agent
            agent = Agent(name="unique_agent")
            session.add(agent)
            session.commit()
            session.refresh(agent)

            # Create first social agent
            social_agent1 = SocialAgent(agent_id=agent.id)
            session.add(social_agent1)
            session.commit()

            # Try to create second social agent for same agent
            social_agent2 = SocialAgent(agent_id=agent.id)
            session.add(social_agent2)

            with pytest.raises(Exception):  # Should raise IntegrityError
                session.commit()

    def test_social_agent_update_reputation(self):
        """Test updating social agent reputation"""
        from src.database.models import Agent, SocialAgent

        with self.SessionLocal() as session:
            # Create base agent
            agent = Agent(name="reputation_agent")
            session.add(agent)
            session.commit()
            session.refresh(agent)

            # Create social agent
            social_agent = SocialAgent(
                agent_id=agent.id,
                reputation_score=50.0
            )
            session.add(social_agent)
            session.commit()
            session.refresh(social_agent)

            original_updated = social_agent.updated_at

            # Update reputation
            social_agent.reputation_score = 85.5
            session.commit()
            session.refresh(social_agent)

            assert social_agent.reputation_score == 85.5
            assert social_agent.updated_at > original_updated

    def test_social_agent_counter_updates(self):
        """Test updating social agent counters"""
        from src.database.models import Agent, SocialAgent

        with self.SessionLocal() as session:
            # Create base agent
            agent = Agent(name="counter_agent")
            session.add(agent)
            session.commit()
            session.refresh(agent)

            # Create social agent
            social_agent = SocialAgent(agent_id=agent.id)
            session.add(social_agent)
            session.commit()
            session.refresh(social_agent)

            # Update counters
            social_agent.friends_count = 10
            social_agent.interactions_count = 25
            social_agent.communities_count = 3
            session.commit()
            session.refresh(social_agent)

            assert social_agent.friends_count == 10
            assert social_agent.interactions_count == 25
            assert social_agent.communities_count == 3

    def test_social_agent_string_representation(self):
        """Test SocialAgent model string representation"""
        from src.database.models import Agent, SocialAgent

        with self.SessionLocal() as session:
            # Create base agent
            agent = Agent(name="repr_agent")
            session.add(agent)
            session.commit()
            session.refresh(agent)

            # Create social agent
            social_agent = SocialAgent(
                agent_id=agent.id,
                social_preference="extroverted",
                reputation_score=75.0
            )
            session.add(social_agent)
            session.commit()
            session.refresh(social_agent)

            str_repr = str(social_agent)
            assert "repr_agent" in str_repr
            assert "extroverted" in str_repr
            assert "75.0" in str_repr

    def test_social_agent_query_by_reputation(self):
        """Test querying social agents by reputation score"""
        from src.database.models import Agent, SocialAgent

        with self.SessionLocal() as session:
            # Create multiple agents and social agents
            agents_data = [
                ("high_rep_agent", 90.0),
                ("low_rep_agent", 20.0),
                ("medium_rep_agent", 50.0),
                ("high_rep_agent2", 85.0)
            ]

            for name, reputation in agents_data:
                agent = Agent(name=name)
                session.add(agent)
                session.commit()
                session.refresh(agent)

                social_agent = SocialAgent(
                    agent_id=agent.id,
                    reputation_score=reputation
                )
                session.add(social_agent)

            session.commit()

            # Query high reputation agents (>80)
            high_rep_agents = session.query(SocialAgent).filter(
                SocialAgent.reputation_score >= 80.0
            ).all()

            assert len(high_rep_agents) == 2
            assert all(sa.reputation_score >= 80.0 for sa in high_rep_agents)

    def test_social_agent_query_by_social_preference(self):
        """Test querying social agents by social preference"""
        from src.database.models import Agent, SocialAgent

        with self.SessionLocal() as session:
            # Create agents with different social preferences
            preferences_data = [
                ("extroverted1", "extroverted"),
                ("introverted1", "introverted"),
                ("extroverted2", "extroverted"),
                ("balanced1", "balanced")
            ]

            for name, preference in preferences_data:
                agent = Agent(name=name)
                session.add(agent)
                session.commit()
                session.refresh(agent)

                social_agent = SocialAgent(
                    agent_id=agent.id,
                    social_preference=preference
                )
                session.add(social_agent)

            session.commit()

            # Query extroverted agents
            extroverted_agents = session.query(SocialAgent).filter(
                SocialAgent.social_preference == "extroverted"
            ).all()

            assert len(extroverted_agents) == 2
            assert all(sa.social_preference == "extroverted" for sa in extroverted_agents)

    def test_social_agent_activity_score_calculation(self):
        """Test social agent activity score calculation"""
        from src.database.models import Agent, SocialAgent

        with self.SessionLocal() as session:
            # Create base agent
            agent = Agent(name="activity_agent")
            session.add(agent)
            session.commit()
            session.refresh(agent)

            # Create social agent
            social_agent = SocialAgent(
                agent_id=agent.id,
                activity_level=0.8,
                friends_count=10,
                interactions_count=50,
                communities_count=3
            )
            session.add(social_agent)
            session.commit()
            session.refresh(social_agent)

            # Test activity score calculation (if implemented)
            if hasattr(social_agent, 'get_activity_score'):
                activity_score = social_agent.get_activity_score()
                assert isinstance(activity_score, float)
                assert 0.0 <= activity_score <= 1.0