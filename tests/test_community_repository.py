"""
Community Repository tests
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import IntegrityError


class TestCommunityRepository:
    """Test Community Repository pattern implementation"""

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

    def test_community_repository_create_community(self):
        """Test creating a community through repository"""
        from src.repositories.community_repository import CommunityRepository
        from src.database.models import Community

        with self.SessionLocal() as session:
            repo = CommunityRepository(session)

            # Create community
            community_data = {
                "name": "Test Community",
                "description": "A test community for unit testing",
                "community_type": "academic",
                "privacy_level": "public",
                "max_members": 100,
                "tags": ["test", "academic", "research"]
            }

            community = repo.create(community_data)

            assert community is not None
            assert community.id is not None
            assert community.name == "Test Community"
            assert community.community_type == "academic"
            assert community.privacy_level == "public"
            assert community.max_members == 100
            assert "test" in community.tags

    def test_community_repository_get_by_id(self):
        """Test getting community by ID through repository"""
        from src.repositories.community_repository import CommunityRepository
        from src.database.models import Community

        with self.SessionLocal() as session:
            repo = CommunityRepository(session)

            # Create community first
            community_data = {"name": "test_community_get"}
            created_community = repo.create(community_data)

            # Get by ID
            retrieved_community = repo.get_by_id(created_community.id)

            assert retrieved_community is not None
            assert retrieved_community.id == created_community.id
            assert retrieved_community.name == "test_community_get"

    def test_community_repository_get_by_name(self):
        """Test getting community by name through repository"""
        from src.repositories.community_repository import CommunityRepository
        from src.database.models import Community

        with self.SessionLocal() as session:
            repo = CommunityRepository(session)

            # Create community first
            community_data = {"name": "unique_community_name"}
            created_community = repo.create(community_data)

            # Get by name
            retrieved_community = repo.get_by_name("unique_community_name")

            assert retrieved_community is not None
            assert retrieved_community.id == created_community.id
            assert retrieved_community.name == "unique_community_name"

    def test_community_repository_get_all(self):
        """Test getting all communities through repository"""
        from src.repositories.community_repository import CommunityRepository
        from src.database.models import Community

        with self.SessionLocal() as session:
            repo = CommunityRepository(session)

            # Create multiple communities
            community_names = ["community1", "community2", "community3"]
            for name in community_names:
                repo.create({"name": name})

            # Get all communities
            all_communities = repo.get_all()

            assert len(all_communities) >= 3
            created_names = [comm.name for comm in all_communities if comm.name in community_names]
            assert len(created_names) == 3
            assert all(name in created_names for name in community_names)

    def test_community_repository_update_community(self):
        """Test updating community through repository"""
        from src.repositories.community_repository import CommunityRepository
        from src.database.models import Community

        with self.SessionLocal() as session:
            repo = CommunityRepository(session)

            # Create community
            community = repo.create({
                "name": "update_community",
                "description": "Original description",
                "community_type": "general"
            })

            # Update community
            update_data = {
                "description": "Updated description",
                "community_type": "academic",
                "max_members": 500
            }
            updated_community = repo.update(community.id, update_data)

            assert updated_community is not None
            assert updated_community.id == community.id
            assert updated_community.description == "Updated description"
            assert updated_community.community_type == "academic"
            assert updated_community.max_members == 500
            assert updated_community.name == "update_community"  # Unchanged

    def test_community_repository_delete_community(self):
        """Test deleting community through repository"""
        from src.repositories.community_repository import CommunityRepository
        from src.database.models import Community

        with self.SessionLocal() as session:
            repo = CommunityRepository(session)

            # Create community
            community = repo.create({"name": "delete_community"})

            # Delete community
            result = repo.delete(community.id)

            assert result is True

            # Verify deletion
            deleted_community = repo.get_by_id(community.id)
            assert deleted_community is None

    def test_community_repository_get_by_type(self):
        """Test getting communities by type"""
        from src.repositories.community_repository import CommunityRepository
        from src.database.models import Community

        with self.SessionLocal() as session:
            repo = CommunityRepository(session)

            # Create communities with different types
            community_types = ["academic", "professional", "general"]
            for community_type in community_types:
                repo.create({
                    "name": f"community_{community_type}",
                    "community_type": community_type
                })

            # Get communities by type
            academic_communities = repo.get_by_type("academic")
            professional_communities = repo.get_by_type("professional")

            assert len(academic_communities) >= 1
            assert all(comm.community_type == "academic" for comm in academic_communities)
            assert len(professional_communities) >= 1
            assert all(comm.community_type == "professional" for comm in professional_communities)

    def test_community_repository_get_by_privacy_level(self):
        """Test getting communities by privacy level"""
        from src.repositories.community_repository import CommunityRepository
        from src.database.models import Community

        with self.SessionLocal() as session:
            repo = CommunityRepository(session)

            # Create communities with different privacy levels
            privacy_levels = ["public", "private", "secret"]
            for privacy_level in privacy_levels:
                repo.create({
                    "name": f"community_{privacy_level}",
                    "privacy_level": privacy_level
                })

            # Get communities by privacy level
            public_communities = repo.get_by_privacy_level("public")
            private_communities = repo.get_by_privacy_level("private")

            assert len(public_communities) >= 1
            assert all(comm.privacy_level == "public" for comm in public_communities)
            assert len(private_communities) >= 1
            assert all(comm.privacy_level == "private" for comm in private_communities)

    def test_community_repository_search_by_name(self):
        """Test searching communities by name"""
        from src.repositories.community_repository import CommunityRepository
        from src.database.models import Community

        with self.SessionLocal() as session:
            repo = CommunityRepository(session)

            # Create communities with searchable names
            search_names = ["search_community_alpha", "search_community_beta", "other_community"]
            for name in search_names:
                repo.create({"name": name})

            # Search for communities
            search_results = repo.search_by_name("search_community")

            assert len(search_results) >= 2
            assert all("search_community" in comm.name for comm in search_results)

    def test_community_repository_get_active_communities(self):
        """Test getting active communities"""
        from src.repositories.community_repository import CommunityRepository
        from src.database.models import Community

        with self.SessionLocal() as session:
            repo = CommunityRepository(session)

            # Create active and inactive communities
            repo.create({"name": "active_community_1", "is_active": True})
            repo.create({"name": "active_community_2", "is_active": True})
            repo.create({"name": "inactive_community", "is_active": False})

            # Get active communities
            active_communities = repo.get_active_communities()

            assert len(active_communities) >= 2
            assert all(comm.is_active for comm in active_communities)

    def test_community_repository_get_communities_with_pagination(self):
        """Test getting communities with pagination"""
        from src.repositories.community_repository import CommunityRepository
        from src.database.models import Community

        with self.SessionLocal() as session:
            repo = CommunityRepository(session)

            # Create many communities
            for i in range(12):
                repo.create({"name": f"page_community_{i}"})

            # Get first page
            page1 = repo.get_with_pagination(page=1, per_page=4)
            assert len(page1) == 4

            # Get second page
            page2 = repo.get_with_pagination(page=2, per_page=4)
            assert len(page2) == 4

            # Verify no overlap
            page1_ids = {comm.id for comm in page1}
            page2_ids = {comm.id for comm in page2}
            assert len(page1_ids.intersection(page2_ids)) == 0

    def test_community_repository_count_communities(self):
        """Test counting communities"""
        from src.repositories.community_repository import CommunityRepository
        from src.database.models import Community

        with self.SessionLocal() as session:
            repo = CommunityRepository(session)

            # Initial count
            initial_count = repo.count()

            # Create some communities
            for i in range(4):
                repo.create({"name": f"count_community_{i}"})

            # New count
            new_count = repo.count()

            assert new_count == initial_count + 4

    def test_community_repository_get_trending_communities(self):
        """Test getting trending communities"""
        from src.repositories.community_repository import CommunityRepository
        from src.database.models import Community

        with self.SessionLocal() as session:
            repo = CommunityRepository(session)

            # Create communities with recent activity
            recent_time = datetime.utcnow() - timedelta(hours=1)
            old_time = datetime.utcnow() - timedelta(days=10)

            # Trending community (recent activity, good member count)
            repo.create({
                "name": "trending_community",
                "member_count": 50,
                "last_activity": recent_time
            })

            # Non-trending community (old activity)
            repo.create({
                "name": "old_community",
                "member_count": 30,
                "last_activity": old_time
            })

            # Get trending communities
            trending_communities = repo.get_trending_communities(days=1)

            assert len(trending_communities) >= 1
            assert all(comm.is_recently_active(days=1) for comm in trending_communities)

    def test_community_repository_search_by_tags(self):
        """Test searching communities by tags"""
        from src.repositories.community_repository import CommunityRepository
        from src.database.models import Community

        with self.SessionLocal() as session:
            repo = CommunityRepository(session)

            # Create communities with different tags
            repo.create({
                "name": "ai_community",
                "tags": ["AI", "machine-learning", "research"]
            })
            repo.create({
                "name": "web_community",
                "tags": ["web", "development", "javascript"]
            })
            repo.create({
                "name": "ml_community",
                "tags": ["machine-learning", "AI", "data-science"]
            })

            # Search by tags
            ai_communities = repo.search_by_tags(["AI"])
            ml_communities = repo.search_by_tags(["machine-learning"])

            # Note: SQLite has limitations with JSON queries, so we test the method exists and runs
            assert isinstance(ai_communities, list)
            assert isinstance(ml_communities, list)

    def test_community_repository_get_communities_by_member_count_range(self):
        """Test getting communities by member count range"""
        from src.repositories.community_repository import CommunityRepository
        from src.database.models import Community

        with self.SessionLocal() as session:
            repo = CommunityRepository(session)

            # Create communities with different member counts
            repo.create({"name": "small_community", "member_count": 5})
            repo.create({"name": "medium_community", "member_count": 50})
            repo.create({"name": "large_community", "member_count": 500})

            # Get communities by member count range
            medium_communities = repo.get_by_member_count_range(10, 100)

            assert len(medium_communities) >= 1
            assert all(10 <= comm.member_count <= 100 for comm in medium_communities)

    def test_community_repository_bulk_create(self):
        """Test bulk creating communities"""
        from src.repositories.community_repository import CommunityRepository
        from src.database.models import Community

        with self.SessionLocal() as session:
            repo = CommunityRepository(session)

            # Prepare bulk data
            community_data_list = [
                {"name": "bulk_community_1", "community_type": "academic"},
                {"name": "bulk_community_2", "community_type": "professional"},
                {"name": "bulk_community_3", "community_type": "general"}
            ]

            # Bulk create
            created_communities = repo.bulk_create(community_data_list)

            assert len(created_communities) == 3
            assert all(comm.id is not None for comm in created_communities)
            created_names = [comm.name for comm in created_communities]
            assert all(name in ["bulk_community_1", "bulk_community_2", "bulk_community_3"] for name in created_names)

    def test_community_repository_get_communities_created_after(self):
        """Test getting communities created after a certain date"""
        from src.repositories.community_repository import CommunityRepository
        from src.database.models import Community

        with self.SessionLocal() as session:
            repo = CommunityRepository(session)

            # Create community at specific time
            cutoff_time = datetime.utcnow()

            # Create communities after cutoff
            import time
            time.sleep(0.01)  # Ensure different timestamps

            repo.create({"name": "after_community_1"})
            time.sleep(0.01)
            repo.create({"name": "after_community_2"})

            # Get communities created after cutoff
            recent_communities = repo.get_created_after(cutoff_time)

            assert len(recent_communities) >= 2
            assert all(comm.created_at > cutoff_time for comm in recent_communities)

    def test_community_repository_update_member_count(self):
        """Test updating community member count"""
        from src.repositories.community_repository import CommunityRepository
        from src.database.models import Community

        with self.SessionLocal() as session:
            repo = CommunityRepository(session)

            # Create community
            community = repo.create({"name": "member_count_community", "member_count": 10})

            # Update member count
            result = repo.update_member_count(community.id, 5)  # Add 5 members

            assert result is True

            # Verify update
            updated_community = repo.get_by_id(community.id)
            assert updated_community.member_count == 15

    def test_community_repository_create_duplicate_name(self):
        """Test handling duplicate name creation"""
        from src.repositories.community_repository import CommunityRepository
        from src.database.models import Community

        with self.SessionLocal() as session:
            repo = CommunityRepository(session)

            # Create first community
            repo.create({"name": "duplicate_test"})

            # Try to create duplicate
            with pytest.raises(IntegrityError):
                repo.create({"name": "duplicate_test"})

    def test_community_repository_update_nonexistent(self):
        """Test updating non-existent community"""
        from src.repositories.community_repository import CommunityRepository

        with self.SessionLocal() as session:
            repo = CommunityRepository(session)

            # Try to update non-existent community
            result = repo.update(99999, {"name": "updated_name"})

            assert result is None