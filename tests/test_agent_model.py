"""
Agent model tests
"""

import pytest
from datetime import datetime
from unittest.mock import patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session


class TestAgentModel:
    """Test Agent ORM model"""

    def setup_method(self):
        """Set up test database session"""
        from src.database.config import create_engine
        from src.database.models import Base

        # Create in-memory SQLite database for testing
        from sqlalchemy import create_engine as sa_create_engine
        self.engine = sa_create_engine('sqlite:///:memory:')
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def teardown_method(self):
        """Clean up after test"""
        self.engine.dispose()

    def test_agent_model_creation(self):
        """Test Agent model creation"""
        from src.database.models import Agent

        with self.SessionLocal() as session:
            agent = Agent(
                name="test_agent",
                personality_type="explorer",
                openness=0.8,
                conscientiousness=0.7,
                extraversion=0.6,
                agreeableness=0.5,
                neuroticism=0.4
            )

            session.add(agent)
            session.commit()
            session.refresh(agent)

            assert agent.id is not None
            assert agent.name == "test_agent"
            assert agent.personality_type == "explorer"
            assert agent.openness == 0.8
            assert agent.conscientiousness == 0.7
            assert agent.extraversion == 0.6
            assert agent.agreeableness == 0.5
            assert agent.neuroticism == 0.4
            assert agent.created_at is not None
            assert agent.updated_at is not None

    def test_agent_model_required_fields(self):
        """Test Agent model required fields"""
        from src.database.models import Agent

        with self.SessionLocal() as session:
            # Test missing name - should fail at constructor level
            with pytest.raises(TypeError):  # Should raise TypeError for missing name
                agent = Agent()

    def test_agent_model_defaults(self):
        """Test Agent model default values"""
        from src.database.models import Agent

        with self.SessionLocal() as session:
            agent = Agent(name="test_agent")
            session.add(agent)
            session.commit()
            session.refresh(agent)

            assert agent.personality_type == "balanced"
            assert agent.openness == 0.5
            assert agent.conscientiousness == 0.5
            assert agent.extraversion == 0.5
            assert agent.agreeableness == 0.5
            assert agent.neuroticism == 0.5

    def test_agent_model_validation(self):
        """Test Agent model field validation"""
        from src.database.models import Agent

        with self.SessionLocal() as session:
            # Test invalid personality type - should fail at constructor level
            with pytest.raises(ValueError):  # Should raise ValueError for invalid personality
                agent = Agent(
                    name="test_agent",
                    personality_type="invalid_type"
                )

    def test_agent_model_personality_bounds(self):
        """Test Agent model personality trait bounds"""
        from src.database.models import Agent

        with self.SessionLocal() as session:
            # Test out of bounds values - should fail at constructor level
            with pytest.raises(ValueError):  # Should raise ValueError for out of bounds
                agent = Agent(
                    name="test_agent",
                    openness=1.5,  # Should be 0-1
                    conscientiousness=-0.5  # Should be 0-1
                )

    def test_agent_model_timestamps(self):
        """Test Agent model timestamp functionality"""
        from src.database.models import Agent
        from datetime import datetime

        with self.SessionLocal() as session:
            agent = Agent(name="test_agent")
            session.add(agent)
            session.commit()
            session.refresh(agent)

            created_time = agent.created_at
            updated_time = agent.updated_at

            assert created_time is not None
            assert updated_time is not None
            assert isinstance(created_time, datetime)
            assert isinstance(updated_time, datetime)

            # Wait a tiny bit and update to ensure timestamp changes
            import time
            time.sleep(0.01)

            agent.personality_type = "explorer"
            session.commit()
            session.refresh(agent)

            # Updated timestamp should be more recent
            assert agent.updated_at > updated_time

    def test_agent_model_relationships(self):
        """Test Agent model relationships (placeholder for future implementation)"""
        from src.database.models import Agent

        with self.SessionLocal() as session:
            agent = Agent(name="test_agent")
            session.add(agent)
            session.commit()
            session.refresh(agent)

            # Basic agent should be created successfully
            assert agent.id is not None
            # Relationships will be added when related models are implemented

    def test_agent_model_string_representation(self):
        """Test Agent model string representation"""
        from src.database.models import Agent

        with self.SessionLocal() as session:
            agent = Agent(name="test_agent", personality_type="explorer")
            session.add(agent)
            session.commit()
            session.refresh(agent)

            str_repr = str(agent)
            assert "test_agent" in str_repr
            assert "explorer" in str_repr

    def test_agent_model_unique_name(self):
        """Test Agent model name uniqueness constraint"""
        from src.database.models import Agent

        with self.SessionLocal() as session:
            agent1 = Agent(name="duplicate_name")
            agent2 = Agent(name="duplicate_name")

            session.add(agent1)
            session.add(agent2)

            with pytest.raises(Exception):  # Should raise IntegrityError
                session.commit()

    def test_agent_model_query_by_personality(self):
        """Test querying agents by personality type"""
        from src.database.models import Agent

        with self.SessionLocal() as session:
            # Create agents with different personalities
            agents = [
                Agent(name="agent1", personality_type="explorer"),
                Agent(name="agent2", personality_type="builder"),
                Agent(name="agent3", personality_type="explorer")
            ]

            for agent in agents:
                session.add(agent)
            session.commit()

            # Query explorers
            explorers = session.query(Agent).filter(
                Agent.personality_type == "explorer"
            ).all()

            assert len(explorers) == 2
            assert all(a.personality_type == "explorer" for a in explorers)

    def test_agent_model_update_traits(self):
        """Test updating agent personality traits"""
        from src.database.models import Agent

        with self.SessionLocal() as session:
            agent = Agent(
                name="test_agent",
                openness=0.5,
                conscientiousness=0.5,
                extraversion=0.5,
                agreeableness=0.5,
                neuroticism=0.5
            )

            session.add(agent)
            session.commit()
            session.refresh(agent)

            # Update traits
            agent.openness = 0.9
            agent.extraversion = 0.1
            session.commit()
            session.refresh(agent)

            assert agent.openness == 0.9
            assert agent.extraversion == 0.1
            assert agent.updated_at > agent.created_at