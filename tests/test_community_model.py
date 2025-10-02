"""
Community model tests
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session


class TestCommunityModel:
    """Test Community ORM model"""

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

    def test_community_model_creation(self):
        """Test Community model creation"""
        from src.database.models import Community

        with self.SessionLocal() as session:
            community = Community(
                name="AI Research Community",
                description="A community for AI research and development",
                community_type="academic",
                privacy_level="public",
                max_members=1000,
                tags=["AI", "research", "machine-learning"]
            )

            session.add(community)
            session.commit()
            session.refresh(community)

            assert community.id is not None
            assert community.name == "AI Research Community"
            assert community.description == "A community for AI research and development"
            assert community.community_type == "academic"
            assert community.privacy_level == "public"
            assert community.max_members == 1000
            assert "AI" in community.tags
            assert "research" in community.tags
            assert community.created_at is not None
            assert community.updated_at is not None

    def test_community_model_defaults(self):
        """Test Community model default values"""
        from src.database.models import Community

        with self.SessionLocal() as session:
            community = Community(name="Default Community")
            session.add(community)
            session.commit()
            session.refresh(community)

            assert community.description is None
            assert community.community_type == "general"
            assert community.privacy_level == "public"
            assert community.max_members == 10000  # Default
            assert community.member_count == 0
            assert community.is_active is True
            assert community.tags == []

    def test_community_model_validation(self):
        """Test Community model field validation"""
        from src.database.models import Community

        with self.SessionLocal() as session:
            # Test invalid community type
            with pytest.raises(ValueError):
                community = Community(
                    name="Test Community",
                    community_type="invalid_type"
                )

            # Test invalid privacy level
            with pytest.raises(ValueError):
                community = Community(
                    name="Test Community",
                    privacy_level="invalid_privacy"
                )

            # Test invalid max_members (negative)
            with pytest.raises(ValueError):
                community = Community(
                    name="Test Community",
                    max_members=-100
                )

            # Test invalid max_members (too large)
            with pytest.raises(ValueError):
                community = Community(
                    name="Test Community",
                    max_members=10000001  # Over 10 million limit
                )

    def test_community_model_unique_name(self):
        """Test Community model name uniqueness constraint"""
        from src.database.models import Community

        with self.SessionLocal() as session:
            # Create first community
            community1 = Community(name="Unique Name")
            session.add(community1)
            session.commit()

            # Try to create second community with same name
            community2 = Community(name="Unique Name")
            session.add(community2)

            with pytest.raises(Exception):  # Should raise IntegrityError
                session.commit()

    def test_community_model_timestamps(self):
        """Test Community model timestamp functionality"""
        from src.database.models import Community

        with self.SessionLocal() as session:
            community = Community(name="Timestamp Test")
            session.add(community)
            session.commit()
            session.refresh(community)

            created_time = community.created_at
            updated_time = community.updated_at

            assert created_time is not None
            assert updated_time is not None
            assert isinstance(created_time, datetime)
            assert isinstance(updated_time, datetime)

            # Wait a tiny bit and update to ensure timestamp changes
            import time
            time.sleep(0.01)

            community.description = "Updated description"
            session.commit()
            session.refresh(community)

            # Updated timestamp should be more recent
            assert community.updated_at > updated_time

    def test_community_model_update_member_count(self):
        """Test updating community member count"""
        from src.database.models import Community

        with self.SessionLocal() as session:
            community = Community(name="Member Count Test")
            session.add(community)
            session.commit()
            session.refresh(community)

            # Increment member count
            community.update_member_count(5)
            session.commit()
            session.refresh(community)

            assert community.member_count == 5

            # Decrement member count
            community.update_member_count(-2)
            session.commit()
            session.refresh(community)

            assert community.member_count == 3

            # Prevent negative count
            community.update_member_count(-10)
            session.commit()
            session.refresh(community)

            assert community.member_count == 0  # Should not go negative

    def test_community_model_is_full(self):
        """Test community capacity checking"""
        from src.database.models import Community

        with self.SessionLocal() as session:
            # Create community with small capacity
            community = Community(
                name="Small Community",
                max_members=5
            )
            session.add(community)
            session.commit()
            session.refresh(community)

            # Not full initially
            assert community.is_full() is False

            # Add members up to capacity
            community.update_member_count(5)
            session.commit()
            session.refresh(community)

            # Should be full now
            assert community.is_full() is True

            # Over capacity (should be prevented by validation)
            community.update_member_count(1)
            session.commit()
            session.refresh(community)

            # Member count should not exceed max_members
            assert community.member_count <= community.max_members

    def test_community_model_deactivation(self):
        """Test community deactivation"""
        from src.database.models import Community

        with self.SessionLocal() as session:
            community = Community(name="Deactivation Test")
            session.add(community)
            session.commit()
            session.refresh(community)

            # Should be active initially
            assert community.is_active is True

            # Deactivate community
            community.deactivate()
            session.commit()
            session.refresh(community)

            assert community.is_active is False

    def test_community_model_string_representation(self):
        """Test Community model string representation"""
        from src.database.models import Community

        with self.SessionLocal() as session:
            community = Community(
                name="Representation Test",
                community_type="academic",
                member_count=50
            )
            session.add(community)
            session.commit()
            session.refresh(community)

            str_repr = str(community)
            assert "Representation Test" in str_repr
            assert "academic" in str_repr
            assert "50" in str_repr

    def test_community_model_query_by_type(self):
        """Test querying communities by type"""
        from src.database.models import Community

        with self.SessionLocal() as session:
            # Create communities of different types
            communities_data = [
                ("AI Research", "academic"),
                ("Tech Startup", "professional"),
                ("Hobby Group", "general"),
                ("ML Engineers", "professional")
            ]

            for name, community_type in communities_data:
                community = Community(name=name, community_type=community_type)
                session.add(community)

            session.commit()

            # Query professional communities
            professional_communities = session.query(Community).filter(
                Community.community_type == "professional"
            ).all()

            assert len(professional_communities) == 2
            assert all(c.community_type == "professional" for c in professional_communities)

    def test_community_model_query_by_privacy(self):
        """Test querying communities by privacy level"""
        from src.database.models import Community

        with self.SessionLocal() as session:
            # Create communities with different privacy levels
            privacy_data = [
                ("Public Group", "public"),
                ("Private Club", "private"),
                ("Secret Society", "secret"),
                ("Open Forum", "public")
            ]

            for name, privacy_level in privacy_data:
                community = Community(name=name, privacy_level=privacy_level)
                session.add(community)

            session.commit()

            # Query public communities
            public_communities = session.query(Community).filter(
                Community.privacy_level == "public"
            ).all()

            assert len(public_communities) == 2
            assert all(c.privacy_level == "public" for c in public_communities)

    def test_community_model_tags_operations(self):
        """Test community tags operations"""
        from src.database.models import Community

        with self.SessionLocal() as session:
            community = Community(
                name="Tags Test",
                tags=["AI", "machine-learning"]
            )
            session.add(community)
            session.commit()
            session.refresh(community)

            # Test initial tags
            assert "AI" in community.tags
            assert "machine-learning" in community.tags

            # Test tag existence methods
            assert community.has_tag("AI") is True
            assert community.has_tag("nonexistent") is False

            # Test adding tags (in-memory operation)
            community.add_tag("research")
            assert "research" in community.tags

            # Test removing tags (in-memory operation)
            community.remove_tag("machine-learning")
            assert "machine-learning" not in community.tags
            assert "AI" in community.tags  # Should still be there

    def test_community_model_activity_tracking(self):
        """Test community activity tracking"""
        from src.database.models import Community

        with self.SessionLocal() as session:
            community = Community(
                name="Activity Test",
                member_count=10,
                last_activity=datetime.utcnow() - timedelta(days=1)
            )
            session.add(community)
            session.commit()
            session.refresh(community)

            original_activity = community.last_activity

            # Record activity
            community.record_activity()
            session.commit()
            session.refresh(community)

            assert community.last_activity > original_activity

            # Check if community is active
            assert community.is_recently_active() is True

            # Test inactive community
            inactive_community = Community(
                name="Inactive Test",
                last_activity=datetime.utcnow() - timedelta(days=30)
            )
            session.add(inactive_community)
            session.commit()
            session.refresh(inactive_community)

            assert inactive_community.is_recently_active() is False

    def test_community_model_popularity_score(self):
        """Test community popularity score calculation"""
        from src.database.models import Community

        with self.SessionLocal() as session:
            # Create community with various metrics
            community = Community(
                name="Popularity Test",
                member_count=100,
                max_members=500,
                created_at=datetime.utcnow() - timedelta(days=30)
            )
            session.add(community)
            session.commit()
            session.refresh(community)

            # Test popularity score calculation (if implemented)
            if hasattr(community, 'get_popularity_score'):
                score = community.get_popularity_score()
                assert isinstance(score, float)
                assert 0.0 <= score <= 1.0

            # Test popularity level
            if hasattr(community, 'get_popularity_level'):
                level = community.get_popularity_level()
                assert level in ["low", "medium", "high", "very_high"]

    def test_community_model_search_functionality(self):
        """Test community search functionality"""
        from src.database.models import Community

        with self.SessionLocal() as session:
            # Create communities with searchable content
            communities_data = [
                ("Machine Learning Research", "Focus on ML algorithms and research", ["ML", "research"]),
                ("AI Development", "Building AI applications", ["AI", "development"]),
                ("Data Science Community", "Data analysis and visualization", ["data", "science"]),
                ("Python Programming", "Python language discussions", ["python", "programming"])
            ]

            for name, description, tags in communities_data:
                community = Community(
                    name=name,
                    description=description,
                    tags=tags
                )
                session.add(community)

            session.commit()

            # Test search by name (if implemented)
            if hasattr(Community, 'search_by_name'):
                results = Community.search_by_name(session, "machine")
                assert len(results) >= 1
                assert any("machine" in c.name.lower() for c in results)

            # Test search by tags (if implemented)
            if hasattr(Community, 'search_by_tags'):
                results = Community.search_by_tags(session, ["AI"])
                # Just test that the method runs without error
                assert isinstance(results, list)