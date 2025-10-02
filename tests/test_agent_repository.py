"""
Agent Repository tests
"""

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import IntegrityError


class TestAgentRepository:
    """Test Agent Repository pattern implementation"""

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

    def test_agent_repository_create_agent(self):
        """Test creating an agent through repository"""
        from src.repositories.agent_repository import AgentRepository
        from src.database.models import Agent

        with self.SessionLocal() as session:
            repo = AgentRepository(session)

            # Create agent
            agent_data = {
                "name": "test_agent",
                "personality_type": "explorer",
                "openness": 0.8,
                "conscientiousness": 0.6,
                "extraversion": 0.7,
                "agreeableness": 0.5,
                "neuroticism": 0.3
            }

            agent = repo.create(agent_data)

            assert agent is not None
            assert agent.id is not None
            assert agent.name == "test_agent"
            assert agent.personality_type == "explorer"
            assert agent.openness == 0.8

    def test_agent_repository_get_by_id(self):
        """Test getting agent by ID through repository"""
        from src.repositories.agent_repository import AgentRepository
        from src.database.models import Agent

        with self.SessionLocal() as session:
            repo = AgentRepository(session)

            # Create agent first
            agent_data = {"name": "test_agent_get"}
            created_agent = repo.create(agent_data)

            # Get by ID
            retrieved_agent = repo.get_by_id(created_agent.id)

            assert retrieved_agent is not None
            assert retrieved_agent.id == created_agent.id
            assert retrieved_agent.name == "test_agent_get"

    def test_agent_repository_get_by_name(self):
        """Test getting agent by name through repository"""
        from src.repositories.agent_repository import AgentRepository
        from src.database.models import Agent

        with self.SessionLocal() as session:
            repo = AgentRepository(session)

            # Create agent first
            agent_data = {"name": "unique_agent_name"}
            created_agent = repo.create(agent_data)

            # Get by name
            retrieved_agent = repo.get_by_name("unique_agent_name")

            assert retrieved_agent is not None
            assert retrieved_agent.id == created_agent.id
            assert retrieved_agent.name == "unique_agent_name"

    def test_agent_repository_get_all(self):
        """Test getting all agents through repository"""
        from src.repositories.agent_repository import AgentRepository
        from src.database.models import Agent

        with self.SessionLocal() as session:
            repo = AgentRepository(session)

            # Create multiple agents
            agent_names = ["agent1", "agent2", "agent3"]
            for name in agent_names:
                repo.create({"name": name})

            # Get all agents
            all_agents = repo.get_all()

            assert len(all_agents) >= 3
            created_names = [agent.name for agent in all_agents if agent.name in agent_names]
            assert len(created_names) == 3
            assert all(name in created_names for name in agent_names)

    def test_agent_repository_update_agent(self):
        """Test updating agent through repository"""
        from src.repositories.agent_repository import AgentRepository
        from src.database.models import Agent

        with self.SessionLocal() as session:
            repo = AgentRepository(session)

            # Create agent
            agent = repo.create({"name": "update_agent", "personality_type": "balanced"})

            # Update agent
            update_data = {
                "personality_type": "leader",
                "openness": 0.9,
                "extraversion": 0.8
            }
            updated_agent = repo.update(agent.id, update_data)

            assert updated_agent is not None
            assert updated_agent.id == agent.id
            assert updated_agent.personality_type == "leader"
            assert updated_agent.openness == 0.9
            assert updated_agent.extraversion == 0.8
            assert updated_agent.name == "update_agent"  # Unchanged

    def test_agent_repository_delete_agent(self):
        """Test deleting agent through repository"""
        from src.repositories.agent_repository import AgentRepository
        from src.database.models import Agent

        with self.SessionLocal() as session:
            repo = AgentRepository(session)

            # Create agent
            agent = repo.create({"name": "delete_agent"})

            # Delete agent
            result = repo.delete(agent.id)

            assert result is True

            # Verify deletion
            deleted_agent = repo.get_by_id(agent.id)
            assert deleted_agent is None

    def test_agent_repository_delete_nonexistent_agent(self):
        """Test deleting non-existent agent"""
        from src.repositories.agent_repository import AgentRepository

        with self.SessionLocal() as session:
            repo = AgentRepository(session)

            # Try to delete non-existent agent
            result = repo.delete(99999)

            assert result is False

    def test_agent_repository_get_by_personality_type(self):
        """Test getting agents by personality type"""
        from src.repositories.agent_repository import AgentRepository
        from src.database.models import Agent

        with self.SessionLocal() as session:
            repo = AgentRepository(session)

            # Create agents with different personality types
            personality_types = ["explorer", "leader", "builder"]
            for personality_type in personality_types:
                repo.create({
                    "name": f"agent_{personality_type}",
                    "personality_type": personality_type
                })

            # Get agents by personality type
            explorers = repo.get_by_personality_type("explorer")
            leaders = repo.get_by_personality_type("leader")

            assert len(explorers) >= 1
            assert all(agent.personality_type == "explorer" for agent in explorers)
            assert len(leaders) >= 1
            assert all(agent.personality_type == "leader" for agent in leaders)

    def test_agent_repository_search_by_name(self):
        """Test searching agents by name"""
        from src.repositories.agent_repository import AgentRepository
        from src.database.models import Agent

        with self.SessionLocal() as session:
            repo = AgentRepository(session)

            # Create agents with searchable names
            search_names = ["search_agent_alpha", "search_agent_beta", "other_agent"]
            for name in search_names:
                repo.create({"name": name})

            # Search for agents
            search_results = repo.search_by_name("search_agent")

            assert len(search_results) >= 2
            assert all("search_agent" in agent.name for agent in search_results)

    def test_agent_repository_get_agents_with_pagination(self):
        """Test getting agents with pagination"""
        from src.repositories.agent_repository import AgentRepository
        from src.database.models import Agent

        with self.SessionLocal() as session:
            repo = AgentRepository(session)

            # Create many agents
            for i in range(15):
                repo.create({"name": f"page_agent_{i}"})

            # Get first page
            page1 = repo.get_with_pagination(page=1, per_page=5)
            assert len(page1) == 5

            # Get second page
            page2 = repo.get_with_pagination(page=2, per_page=5)
            assert len(page2) == 5

            # Verify no overlap
            page1_ids = {agent.id for agent in page1}
            page2_ids = {agent.id for agent in page2}
            assert len(page1_ids.intersection(page2_ids)) == 0

    def test_agent_repository_count_agents(self):
        """Test counting agents"""
        from src.repositories.agent_repository import AgentRepository
        from src.database.models import Agent

        with self.SessionLocal() as session:
            repo = AgentRepository(session)

            # Initial count
            initial_count = repo.count()

            # Create some agents
            for i in range(5):
                repo.create({"name": f"count_agent_{i}"})

            # New count
            new_count = repo.count()

            assert new_count == initial_count + 5

    def test_agent_repository_get_agents_created_after(self):
        """Test getting agents created after a certain date"""
        from src.repositories.agent_repository import AgentRepository
        from src.database.models import Agent

        with self.SessionLocal() as session:
            repo = AgentRepository(session)

            # Create agent at specific time
            cutoff_time = datetime.utcnow()

            # Create agents after cutoff
            import time
            time.sleep(0.01)  # Ensure different timestamps

            repo.create({"name": "after_agent_1"})
            time.sleep(0.01)
            repo.create({"name": "after_agent_2"})

            # Get agents created after cutoff
            recent_agents = repo.get_created_after(cutoff_time)

            assert len(recent_agents) >= 2
            assert all(agent.created_at > cutoff_time for agent in recent_agents)

    def test_agent_repository_bulk_create(self):
        """Test bulk creating agents"""
        from src.repositories.agent_repository import AgentRepository
        from src.database.models import Agent

        with self.SessionLocal() as session:
            repo = AgentRepository(session)

            # Prepare bulk data
            agent_data_list = [
                {"name": "bulk_agent_1", "personality_type": "explorer"},
                {"name": "bulk_agent_2", "personality_type": "leader"},
                {"name": "bulk_agent_3", "personality_type": "builder"}
            ]

            # Bulk create
            created_agents = repo.bulk_create(agent_data_list)

            assert len(created_agents) == 3
            assert all(agent.id is not None for agent in created_agents)
            created_names = [agent.name for agent in created_agents]
            assert all(name in ["bulk_agent_1", "bulk_agent_2", "bulk_agent_3"] for name in created_names)

    def test_agent_repository_get_by_trait_range(self):
        """Test getting agents by personality trait range"""
        from src.repositories.agent_repository import AgentRepository
        from src.database.models import Agent

        with self.SessionLocal() as session:
            repo = AgentRepository(session)

            # Create agents with different openness levels
            repo.create({"name": "open_agent", "openness": 0.9})
            repo.create({"name": "moderate_agent", "openness": 0.5})
            repo.create({"name": "closed_agent", "openness": 0.1})

            # Get agents with high openness
            open_agents = repo.get_by_trait_range("openness", 0.7, 1.0)

            assert len(open_agents) >= 1
            assert all(agent.openness >= 0.7 for agent in open_agents)

    def test_agent_repository_create_duplicate_name(self):
        """Test handling duplicate name creation"""
        from src.repositories.agent_repository import AgentRepository
        from src.database.models import Agent

        with self.SessionLocal() as session:
            repo = AgentRepository(session)

            # Create first agent
            repo.create({"name": "duplicate_test"})

            # Try to create duplicate
            with pytest.raises(IntegrityError):
                repo.create({"name": "duplicate_test"})

    def test_agent_repository_update_nonexistent(self):
        """Test updating non-existent agent"""
        from src.repositories.agent_repository import AgentRepository

        with self.SessionLocal() as session:
            repo = AgentRepository(session)

            # Try to update non-existent agent
            result = repo.update(99999, {"name": "updated_name"})

            assert result is None

    def test_agent_repository_transaction_rollback(self):
        """Test transaction rollback on error"""
        from src.repositories.agent_repository import AgentRepository
        from src.database.models import Agent

        with self.SessionLocal() as session:
            repo = AgentRepository(session)

            # Count before
            initial_count = repo.count()

            # Try to create invalid agent (should fail)
            try:
                repo.create({
                    "name": "invalid_agent",
                    "personality_type": "invalid_type"  # This should cause validation error
                })
            except ValueError:
                pass

            # Count after - should be unchanged
            final_count = repo.count()
            assert final_count == initial_count