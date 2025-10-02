"""
Friendship Repository tests
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import IntegrityError


class TestFriendshipRepository:
    """Test Friendship Repository pattern implementation"""

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

    def test_friendship_repository_create_friendship(self):
        """Test creating a friendship through repository"""
        from src.repositories.friendship_repository import FriendshipRepository
        from src.database.models import Agent, Friendship

        with self.SessionLocal() as session:
            repo = FriendshipRepository(session)

            # Create test agents
            agent1, agent2 = self.create_test_agents(session, 2)

            # Create friendship
            friendship_data = {
                "initiator_id": agent1.id,
                "recipient_id": agent2.id,
                "friendship_status": "pending",
                "strength_level": 0.5
            }

            friendship = repo.create(friendship_data)

            assert friendship is not None
            assert friendship.id is not None
            assert friendship.initiator_id == agent1.id
            assert friendship.recipient_id == agent2.id
            assert friendship.friendship_status == "pending"
            assert friendship.strength_level == 0.5

    def test_friendship_repository_get_by_id(self):
        """Test getting friendship by ID through repository"""
        from src.repositories.friendship_repository import FriendshipRepository
        from src.database.models import Agent, Friendship

        with self.SessionLocal() as session:
            repo = FriendshipRepository(session)

            # Create test agents and friendship
            agent1, agent2 = self.create_test_agents(session, 2)
            friendship_data = {
                "initiator_id": agent1.id,
                "recipient_id": agent2.id,
                "friendship_status": "pending"
            }
            created_friendship = repo.create(friendship_data)

            # Get by ID
            retrieved_friendship = repo.get_by_id(created_friendship.id)

            assert retrieved_friendship is not None
            assert retrieved_friendship.id == created_friendship.id
            assert retrieved_friendship.initiator_id == agent1.id
            assert retrieved_friendship.recipient_id == agent2.id

    def test_friendship_repository_get_friendships_for_agent(self):
        """Test getting friendships for a specific agent"""
        from src.repositories.friendship_repository import FriendshipRepository
        from src.database.models import Agent, Friendship

        with self.SessionLocal() as session:
            repo = FriendshipRepository(session)

            # Create test agents
            agents = self.create_test_agents(session, 4)

            # Create friendships for agent 0
            repo.create({
                "initiator_id": agents[0].id,
                "recipient_id": agents[1].id,
                "friendship_status": "accepted"
            })
            repo.create({
                "initiator_id": agents[2].id,
                "recipient_id": agents[0].id,
                "friendship_status": "accepted"
            })
            repo.create({
                "initiator_id": agents[1].id,
                "recipient_id": agents[3].id,
                "friendship_status": "pending"
            })

            # Get friendships for agent 0
            agent_friendships = repo.get_friendships_for_agent(agents[0].id)

            assert len(agent_friendships) == 2
            assert all(
                friendship.initiator_id == agents[0].id or
                friendship.recipient_id == agents[0].id
                for friendship in agent_friendships
            )

    def test_friendship_repository_get_active_friendships(self):
        """Test getting active (accepted) friendships"""
        from src.repositories.friendship_repository import FriendshipRepository
        from src.database.models import Agent, Friendship

        with self.SessionLocal() as session:
            repo = FriendshipRepository(session)

            # Create test agents
            agents = self.create_test_agents(session, 4)

            # Create friendships with different statuses
            repo.create({
                "initiator_id": agents[0].id,
                "recipient_id": agents[1].id,
                "friendship_status": "accepted"
            })
            repo.create({
                "initiator_id": agents[1].id,
                "recipient_id": agents[2].id,
                "friendship_status": "pending"
            })
            repo.create({
                "initiator_id": agents[2].id,
                "recipient_id": agents[3].id,
                "friendship_status": "accepted"
            })

            # Get active friendships
            active_friendships = repo.get_active_friendships()

            assert len(active_friendships) >= 2
            assert all(friendship.friendship_status == "accepted" for friendship in active_friendships)

    def test_friendship_repository_get_friendships_by_status(self):
        """Test getting friendships by status"""
        from src.repositories.friendship_repository import FriendshipRepository
        from src.database.models import Agent, Friendship

        with self.SessionLocal() as session:
            repo = FriendshipRepository(session)

            # Create test agents
            agents = self.create_test_agents(session, 6)

            # Create friendships with different statuses
            repo.create({
                "initiator_id": agents[0].id,
                "recipient_id": agents[1].id,
                "friendship_status": "pending"
            })
            repo.create({
                "initiator_id": agents[2].id,
                "recipient_id": agents[3].id,
                "friendship_status": "accepted"
            })
            repo.create({
                "initiator_id": agents[4].id,
                "recipient_id": agents[5].id,
                "friendship_status": "blocked"
            })

            # Get friendships by status
            pending_friendships = repo.get_by_status("pending")
            accepted_friendships = repo.get_by_status("accepted")
            blocked_friendships = repo.get_by_status("blocked")

            assert len(pending_friendships) >= 1
            assert all(friendship.friendship_status == "pending" for friendship in pending_friendships)
            assert len(accepted_friendships) >= 1
            assert all(friendship.friendship_status == "accepted" for friendship in accepted_friendships)
            assert len(blocked_friendships) >= 1
            assert all(friendship.friendship_status == "blocked" for friendship in blocked_friendships)

    def test_friendship_repository_update_friendship_status(self):
        """Test updating friendship status through repository"""
        from src.repositories.friendship_repository import FriendshipRepository
        from src.database.models import Agent, Friendship

        with self.SessionLocal() as session:
            repo = FriendshipRepository(session)

            # Create test agents and friendship
            agent1, agent2 = self.create_test_agents(session, 2)
            friendship = repo.create({
                "initiator_id": agent1.id,
                "recipient_id": agent2.id,
                "friendship_status": "pending"
            })

            # Update friendship status
            updated_friendship = repo.update_status(friendship.id, "accepted")

            assert updated_friendship is not None
            assert updated_friendship.id == friendship.id
            assert updated_friendship.friendship_status == "accepted"

    def test_friendship_repository_update_strength_level(self):
        """Test updating friendship strength level through repository"""
        from src.repositories.friendship_repository import FriendshipRepository
        from src.database.models import Agent, Friendship

        with self.SessionLocal() as session:
            repo = FriendshipRepository(session)

            # Create test agents and friendship
            agent1, agent2 = self.create_test_agents(session, 2)
            friendship = repo.create({
                "initiator_id": agent1.id,
                "recipient_id": agent2.id,
                "strength_level": 0.3
            })

            # Update strength level
            updated_friendship = repo.update_strength_level(friendship.id, 0.8)

            assert updated_friendship is not None
            assert updated_friendship.id == friendship.id
            assert updated_friendship.strength_level == 0.8

    def test_friendship_repository_delete_friendship(self):
        """Test deleting friendship through repository"""
        from src.repositories.friendship_repository import FriendshipRepository
        from src.database.models import Agent, Friendship

        with self.SessionLocal() as session:
            repo = FriendshipRepository(session)

            # Create test agents and friendship
            agent1, agent2 = self.create_test_agents(session, 2)
            friendship = repo.create({
                "initiator_id": agent1.id,
                "recipient_id": agent2.id,
                "friendship_status": "accepted"
            })

            # Delete friendship
            result = repo.delete(friendship.id)

            assert result is True

            # Verify deletion
            deleted_friendship = repo.get_by_id(friendship.id)
            assert deleted_friendship is None

    def test_friendship_repository_get_friendship_between_agents(self):
        """Test getting friendship between two specific agents"""
        from src.repositories.friendship_repository import FriendshipRepository
        from src.database.models import Agent, Friendship

        with self.SessionLocal() as session:
            repo = FriendshipRepository(session)

            # Create test agents
            agents = self.create_test_agents(session, 3)

            # Create friendship between agent 0 and 1
            repo.create({
                "initiator_id": agents[0].id,
                "recipient_id": agents[1].id,
                "friendship_status": "accepted"
            })

            # Get friendship between agent 0 and 1
            friendship = repo.get_friendship_between_agents(agents[0].id, agents[1].id)

            assert friendship is not None
            assert (friendship.initiator_id == agents[0].id and friendship.recipient_id == agents[1].id) or \
                   (friendship.initiator_id == agents[1].id and friendship.recipient_id == agents[0].id)

            # Try to get non-existent friendship
            no_friendship = repo.get_friendship_between_agents(agents[0].id, agents[2].id)
            assert no_friendship is None

    def test_friendship_repository_get_strong_friendships(self):
        """Test getting strong friendships"""
        from src.repositories.friendship_repository import FriendshipRepository
        from src.database.models import Agent, Friendship

        with self.SessionLocal() as session:
            repo = FriendshipRepository(session)

            # Create test agents
            agents = self.create_test_agents(session, 6)

            # Create friendships with different strength levels
            repo.create({
                "initiator_id": agents[0].id,
                "recipient_id": agents[1].id,
                "friendship_status": "accepted",
                "strength_level": 0.9
            })
            repo.create({
                "initiator_id": agents[2].id,
                "recipient_id": agents[3].id,
                "friendship_status": "accepted",
                "strength_level": 0.2
            })
            repo.create({
                "initiator_id": agents[4].id,
                "recipient_id": agents[5].id,
                "friendship_status": "accepted",
                "strength_level": 0.8
            })

            # Get strong friendships (threshold 0.7)
            strong_friendships = repo.get_strong_friendships(threshold=0.7)

            assert len(strong_friendships) >= 2
            assert all(friendship.strength_level >= 0.7 for friendship in strong_friendships)

    def test_friendship_repository_get_pending_friendships_for_agent(self):
        """Test getting pending friendships for a specific agent"""
        from src.repositories.friendship_repository import FriendshipRepository
        from src.database.models import Agent, Friendship

        with self.SessionLocal() as session:
            repo = FriendshipRepository(session)

            # Create test agents
            agents = self.create_test_agents(session, 4)

            # Create pending friendships where agent 0 is recipient
            repo.create({
                "initiator_id": agents[1].id,
                "recipient_id": agents[0].id,
                "friendship_status": "pending"
            })
            repo.create({
                "initiator_id": agents[2].id,
                "recipient_id": agents[0].id,
                "friendship_status": "pending"
            })
            # Create accepted friendship (should not be in pending)
            repo.create({
                "initiator_id": agents[0].id,
                "recipient_id": agents[3].id,
                "friendship_status": "accepted"
            })

            # Get pending friendships for agent 0
            pending_friendships = repo.get_pending_friendships_for_agent(agents[0].id)

            assert len(pending_friendships) == 2
            assert all(friendship.recipient_id == agents[0].id for friendship in pending_friendships)
            assert all(friendship.friendship_status == "pending" for friendship in pending_friendships)

    def test_friendship_repository_record_interaction(self):
        """Test recording friendship interaction through repository"""
        from src.repositories.friendship_repository import FriendshipRepository
        from src.database.models import Agent, Friendship

        with self.SessionLocal() as session:
            repo = FriendshipRepository(session)

            # Create test agents and friendship
            agent1, agent2 = self.create_test_agents(session, 2)
            friendship = repo.create({
                "initiator_id": agent1.id,
                "recipient_id": agent2.id,
                "friendship_status": "accepted",
                "interaction_count": 5
            })

            # Record interaction
            result = repo.record_interaction(friendship.id)

            assert result is True

            # Verify interaction recorded
            updated_friendship = repo.get_by_id(friendship.id)
            assert updated_friendship.interaction_count == 6
            assert updated_friendship.last_interaction is not None

    def test_friendship_repository_get_friendships_with_pagination(self):
        """Test getting friendships with pagination"""
        from src.repositories.friendship_repository import FriendshipRepository
        from src.database.models import Agent, Friendship

        with self.SessionLocal() as session:
            repo = FriendshipRepository(session)

            # Create test agents
            agents = self.create_test_agents(session, 10)

            # Create many friendships
            for i in range(0, 9, 2):
                repo.create({
                    "initiator_id": agents[i].id,
                    "recipient_id": agents[i+1].id,
                    "friendship_status": "accepted"
                })

            # Get first page
            page1 = repo.get_with_pagination(page=1, per_page=2)
            assert len(page1) == 2

            # Get second page
            page2 = repo.get_with_pagination(page=2, per_page=2)
            assert len(page2) == 2

            # Verify no overlap
            page1_ids = {friendship.id for friendship in page1}
            page2_ids = {friendship.id for friendship in page2}
            assert len(page1_ids.intersection(page2_ids)) == 0

    def test_friendship_repository_count_friendships(self):
        """Test counting friendships"""
        from src.repositories.friendship_repository import FriendshipRepository
        from src.database.models import Agent, Friendship

        with self.SessionLocal() as session:
            repo = FriendshipRepository(session)

            # Initial count
            initial_count = repo.count()

            # Create test agents and friendships
            agents = self.create_test_agents(session, 6)
            for i in range(0, 5, 2):
                repo.create({
                    "initiator_id": agents[i].id,
                    "recipient_id": agents[i+1].id,
                    "friendship_status": "accepted"
                })

            # New count
            new_count = repo.count()

            assert new_count == initial_count + 3

    def test_friendship_repository_create_self_friendship_error(self):
        """Test preventing self-friendship creation"""
        from src.repositories.friendship_repository import FriendshipRepository
        from src.database.models import Agent

        with self.SessionLocal() as session:
            repo = FriendshipRepository(session)

            # Create test agent
            agent, = self.create_test_agents(session, 1)

            # Try to create self-friendship
            with pytest.raises(ValueError):
                repo.create({
                    "initiator_id": agent.id,
                    "recipient_id": agent.id,
                    "friendship_status": "pending"
                })

    def test_friendship_repository_create_duplicate_friendship_error(self):
        """Test preventing duplicate friendship creation"""
        from src.repositories.friendship_repository import FriendshipRepository
        from src.database.models import Agent, Friendship

        with self.SessionLocal() as session:
            repo = FriendshipRepository(session)

            # Create test agents
            agent1, agent2 = self.create_test_agents(session, 2)

            # Create first friendship
            repo.create({
                "initiator_id": agent1.id,
                "recipient_id": agent2.id,
                "friendship_status": "accepted"
            })

            # Try to create duplicate friendship
            with pytest.raises(IntegrityError):
                repo.create({
                    "initiator_id": agent1.id,
                    "recipient_id": agent2.id,
                    "friendship_status": "pending"
                })

    def test_friendship_repository_update_nonexistent_friendship(self):
        """Test updating non-existent friendship"""
        from src.repositories.friendship_repository import FriendshipRepository

        with self.SessionLocal() as session:
            repo = FriendshipRepository(session)

            # Try to update non-existent friendship
            result = repo.update_status(99999, "accepted")
            assert result is None

            result = repo.update_strength_level(99999, 0.8)
            assert result is None

            result = repo.record_interaction(99999)
            assert result is False