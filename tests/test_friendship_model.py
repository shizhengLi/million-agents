"""
Friendship model tests
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session


class TestFriendshipModel:
    """Test Friendship ORM model"""

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

    def create_test_agents(self, session, count=2):
        """Helper method to create test agents"""
        from src.database.models import Agent

        agents = []
        for i in range(count):
            agent = Agent(name=f"test_agent_{i}")
            session.add(agent)
            session.commit()
            session.refresh(agent)
            agents.append(agent)

        return agents

    def test_friendship_model_creation(self):
        """Test Friendship model creation"""
        from src.database.models import Agent, Friendship

        with self.SessionLocal() as session:
            # Create test agents
            agent1, agent2 = self.create_test_agents(session, 2)

            # Create friendship
            friendship = Friendship(
                initiator_id=agent1.id,
                recipient_id=agent2.id,
                friendship_status="pending"
            )

            session.add(friendship)
            session.commit()
            session.refresh(friendship)

            assert friendship.id is not None
            assert friendship.initiator_id == agent1.id
            assert friendship.recipient_id == agent2.id
            assert friendship.friendship_status == "pending"
            assert friendship.strength_level == 0.5  # Default
            assert friendship.created_at is not None
            assert friendship.updated_at is not None

    def test_friendship_model_defaults(self):
        """Test Friendship model default values"""
        from src.database.models import Agent, Friendship

        with self.SessionLocal() as session:
            # Create test agents
            agent1, agent2 = self.create_test_agents(session, 2)

            # Create friendship with minimal data
            friendship = Friendship(
                initiator_id=agent1.id,
                recipient_id=agent2.id
            )

            session.add(friendship)
            session.commit()
            session.refresh(friendship)

            assert friendship.friendship_status == "pending"  # Default
            assert friendship.strength_level == 0.5  # Default
            assert friendship.last_interaction is None
            assert friendship.interaction_count == 0

    def test_friendship_model_validation(self):
        """Test Friendship model field validation"""
        from src.database.models import Agent, Friendship

        with self.SessionLocal() as session:
            # Create test agents
            agent1, agent2 = self.create_test_agents(session, 2)

            # Test invalid friendship status
            with pytest.raises(ValueError):
                friendship = Friendship(
                    initiator_id=agent1.id,
                    recipient_id=agent2.id,
                    friendship_status="invalid_status"
                )

            # Test invalid strength level (out of bounds)
            with pytest.raises(ValueError):
                friendship = Friendship(
                    initiator_id=agent1.id,
                    recipient_id=agent2.id,
                    strength_level=1.5  # Should be 0-1
                )

            # Test invalid strength level (negative)
            with pytest.raises(ValueError):
                friendship = Friendship(
                    initiator_id=agent1.id,
                    recipient_id=agent2.id,
                    strength_level=-0.5  # Should be 0-1
                )

    def test_friendship_self_friendship_prevention(self):
        """Test prevention of self-friendship"""
        from src.database.models import Agent, Friendship

        with self.SessionLocal() as session:
            # Create single test agent
            agent, = self.create_test_agents(session, 1)

            # Try to create self-friendship
            with pytest.raises(ValueError):
                friendship = Friendship(
                    initiator_id=agent.id,
                    recipient_id=agent.id
                )

    def test_friendship_unique_constraint(self):
        """Test friendship uniqueness constraint"""
        from src.database.models import Agent, Friendship

        with self.SessionLocal() as session:
            # Create test agents
            agent1, agent2 = self.create_test_agents(session, 2)

            # Create first friendship
            friendship1 = Friendship(
                initiator_id=agent1.id,
                recipient_id=agent2.id,
                friendship_status="accepted"
            )
            session.add(friendship1)
            session.commit()

            # Try to create duplicate friendship (same direction)
            friendship2 = Friendship(
                initiator_id=agent1.id,
                recipient_id=agent2.id,
                friendship_status="pending"
            )
            session.add(friendship2)

            with pytest.raises(Exception):  # Should raise IntegrityError
                session.commit()

    def test_friendship_bidirectional_relationships(self):
        """Test friendship bidirectional relationships"""
        from src.database.models import Agent, Friendship

        with self.SessionLocal() as session:
            # Create test agents
            agent1, agent2 = self.create_test_agents(session, 2)

            # Create friendship
            friendship = Friendship(
                initiator_id=agent1.id,
                recipient_id=agent2.id,
                friendship_status="accepted"
            )

            session.add(friendship)
            session.commit()
            session.refresh(friendship)

            # Test relationships
            assert friendship.initiator.id == agent1.id
            assert friendship.initiator.name == "test_agent_0"
            assert friendship.recipient.id == agent2.id
            assert friendship.recipient.name == "test_agent_1"

    def test_friendship_status_transitions(self):
        """Test friendship status transitions"""
        from src.database.models import Agent, Friendship

        with self.SessionLocal() as session:
            # Create test agents
            agent1, agent2 = self.create_test_agents(session, 2)

            # Create friendship with pending status
            friendship = Friendship(
                initiator_id=agent1.id,
                recipient_id=agent2.id,
                friendship_status="pending"
            )

            session.add(friendship)
            session.commit()
            session.refresh(friendship)

            assert friendship.friendship_status == "pending"

            # Accept friendship
            friendship.update_status("accepted")
            session.commit()
            session.refresh(friendship)

            assert friendship.friendship_status == "accepted"

            # Block friendship
            friendship.update_status("blocked")
            session.commit()
            session.refresh(friendship)

            assert friendship.friendship_status == "blocked"

    def test_friendship_strength_level_updates(self):
        """Test friendship strength level updates"""
        from src.database.models import Agent, Friendship

        with self.SessionLocal() as session:
            # Create test agents
            agent1, agent2 = self.create_test_agents(session, 2)

            # Create friendship
            friendship = Friendship(
                initiator_id=agent1.id,
                recipient_id=agent2.id,
                strength_level=0.3
            )

            session.add(friendship)
            session.commit()
            session.refresh(friendship)

            assert friendship.strength_level == 0.3

            # Update strength level
            friendship.update_strength_level(0.8)
            session.commit()
            session.refresh(friendship)

            assert friendship.strength_level == 0.8

    def test_friendship_interaction_tracking(self):
        """Test friendship interaction tracking"""
        from src.database.models import Agent, Friendship

        with self.SessionLocal() as session:
            # Create test agents
            agent1, agent2 = self.create_test_agents(session, 2)

            # Create friendship
            friendship = Friendship(
                initiator_id=agent1.id,
                recipient_id=agent2.id
            )

            session.add(friendship)
            session.commit()
            session.refresh(friendship)

            original_count = friendship.interaction_count
            original_interaction = friendship.last_interaction

            # Record interaction
            friendship.record_interaction()
            session.commit()
            session.refresh(friendship)

            assert friendship.interaction_count == original_count + 1
            if original_interaction is not None:
                assert friendship.last_interaction > original_interaction
            else:
                assert friendship.last_interaction is not None

    def test_friendship_is_active(self):
        """Test friendship active status checking"""
        from src.database.models import Agent, Friendship

        with self.SessionLocal() as session:
            # Create test agents
            agent1, agent2 = self.create_test_agents(session, 2)

            # Create accepted friendship
            accepted_friendship = Friendship(
                initiator_id=agent1.id,
                recipient_id=agent2.id,
                friendship_status="accepted"
            )
            session.add(accepted_friendship)

            # Create pending friendship
            agent3 = Agent(name="test_agent_2")
            session.add(agent3)
            session.commit()

            pending_friendship = Friendship(
                initiator_id=agent1.id,
                recipient_id=agent3.id,
                friendship_status="pending"
            )
            session.add(pending_friendship)

            # Create blocked friendship
            agent4 = Agent(name="test_agent_3")
            session.add(agent4)
            session.commit()

            blocked_friendship = Friendship(
                initiator_id=agent1.id,
                recipient_id=agent4.id,
                friendship_status="blocked"
            )
            session.add(blocked_friendship)

            session.commit()

            # Test active status
            assert accepted_friendship.is_active() is True
            assert pending_friendship.is_active() is False
            assert blocked_friendship.is_active() is False

    def test_friendship_is_strong(self):
        """Test friendship strength checking"""
        from src.database.models import Agent, Friendship

        with self.SessionLocal() as session:
            # Create test agents
            agent1, agent2 = self.create_test_agents(session, 2)

            # Create weak friendship
            weak_friendship = Friendship(
                initiator_id=agent1.id,
                recipient_id=agent2.id,
                strength_level=0.3
            )
            session.add(weak_friendship)

            # Create strong friendship
            agent3 = Agent(name="test_agent_2")
            session.add(agent3)
            session.commit()

            strong_friendship = Friendship(
                initiator_id=agent1.id,
                recipient_id=agent3.id,
                strength_level=0.8
            )
            session.add(strong_friendship)

            session.commit()

            # Test strength checking
            assert weak_friendship.is_strong() is False
            assert strong_friendship.is_strong() is True

    def test_friendship_string_representation(self):
        """Test Friendship model string representation"""
        from src.database.models import Agent, Friendship

        with self.SessionLocal() as session:
            # Create test agents
            agent1, agent2 = self.create_test_agents(session, 2)

            # Create friendship
            friendship = Friendship(
                initiator_id=agent1.id,
                recipient_id=agent2.id,
                friendship_status="accepted",
                strength_level=0.7
            )

            session.add(friendship)
            session.commit()
            session.refresh(friendship)

            str_repr = str(friendship)
            assert "test_agent_0" in str_repr
            assert "test_agent_1" in str_repr
            assert "accepted" in str_repr
            assert "0.7" in str_repr

    def test_friendship_query_by_status(self):
        """Test querying friendships by status"""
        from src.database.models import Agent, Friendship

        with self.SessionLocal() as session:
            # Create test agents
            agents = self.create_test_agents(session, 4)

            # Create friendships with different statuses
            friendships_data = [
                (agents[0].id, agents[1].id, "accepted"),
                (agents[0].id, agents[2].id, "pending"),
                (agents[1].id, agents[2].id, "accepted"),
                (agents[0].id, agents[3].id, "blocked")
            ]

            for initiator_id, recipient_id, status in friendships_data:
                friendship = Friendship(
                    initiator_id=initiator_id,
                    recipient_id=recipient_id,
                    friendship_status=status
                )
                session.add(friendship)

            session.commit()

            # Query accepted friendships
            accepted_friendships = session.query(Friendship).filter(
                Friendship.friendship_status == "accepted"
            ).all()

            assert len(accepted_friendships) == 2
            assert all(f.friendship_status == "accepted" for f in accepted_friendships)

    def test_friendship_query_by_strength(self):
        """Test querying friendships by strength level"""
        from src.database.models import Agent, Friendship

        with self.SessionLocal() as session:
            # Create test agents
            agents = self.create_test_agents(session, 4)

            # Create friendships with different strength levels
            friendships_data = [
                (agents[0].id, agents[1].id, 0.2),
                (agents[0].id, agents[2].id, 0.5),
                (agents[1].id, agents[2].id, 0.8),
                (agents[0].id, agents[3].id, 0.9)
            ]

            for initiator_id, recipient_id, strength in friendships_data:
                friendship = Friendship(
                    initiator_id=initiator_id,
                    recipient_id=recipient_id,
                    strength_level=strength,
                    friendship_status="accepted"
                )
                session.add(friendship)

            session.commit()

            # Query strong friendships (strength >= 0.7)
            strong_friendships = session.query(Friendship).filter(
                Friendship.strength_level >= 0.7,
                Friendship.friendship_status == "accepted"
            ).all()

            assert len(strong_friendships) == 2
            assert all(f.strength_level >= 0.7 for f in strong_friendships)

    def test_friendship_get_friendship_summary(self):
        """Test friendship summary generation"""
        from src.database.models import Agent, Friendship

        with self.SessionLocal() as session:
            # Create test agents
            agent1, agent2 = self.create_test_agents(session, 2)

            # Create friendship
            friendship = Friendship(
                initiator_id=agent1.id,
                recipient_id=agent2.id,
                friendship_status="accepted",
                strength_level=0.7,
                interaction_count=15
            )

            session.add(friendship)
            session.commit()
            session.refresh(friendship)

            # Test summary generation (if implemented)
            if hasattr(friendship, 'get_friendship_summary'):
                summary = friendship.get_friendship_summary()
                assert isinstance(summary, dict)
                assert summary['initiator_id'] == agent1.id
                assert summary['recipient_id'] == agent2.id
                assert summary['friendship_status'] == "accepted"
                assert summary['strength_level'] == 0.7
                assert summary['interaction_count'] == 15

    @staticmethod
    def test_friendship_get_valid_statuses():
        """Test getting valid friendship statuses"""
        from src.database.models import Friendship

        # Test static method (if implemented)
        if hasattr(Friendship, 'get_valid_statuses'):
            valid_statuses = Friendship.get_valid_statuses()
            assert isinstance(valid_statuses, list)
            assert "pending" in valid_statuses
            assert "accepted" in valid_statuses
            assert "blocked" in valid_statuses