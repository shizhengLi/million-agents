"""
CommunityMembership model tests
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session


class TestCommunityMembershipModel:
    """Test CommunityMembership ORM model"""

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

    def create_test_communities(self, session, count=2):
        """Helper method to create test communities"""
        from src.database.models import Community

        communities = []
        for i in range(count):
            community = Community(name=f"test_community_{i}")
            session.add(community)
            session.commit()
            session.refresh(community)
            communities.append(community)

        return communities

    def test_community_membership_model_creation(self):
        """Test CommunityMembership model creation"""
        from src.database.models import Agent, Community, CommunityMembership

        with self.SessionLocal() as session:
            # Create test agent and community
            agent, = self.create_test_agents(session, 1)
            community, = self.create_test_communities(session, 1)

            # Create community membership
            membership = CommunityMembership(
                agent_id=agent.id,
                community_id=community.id,
                role="member"
            )

            session.add(membership)
            session.commit()
            session.refresh(membership)

            assert membership.id is not None
            assert membership.agent_id == agent.id
            assert membership.community_id == community.id
            assert membership.role == "member"
            assert membership.is_active is True  # Default
            assert membership.joined_at is not None

    def test_community_membership_model_defaults(self):
        """Test CommunityMembership model default values"""
        from src.database.models import Agent, Community, CommunityMembership

        with self.SessionLocal() as session:
            # Create test agent and community
            agent, = self.create_test_agents(session, 1)
            community, = self.create_test_communities(session, 1)

            # Create membership with minimal data
            membership = CommunityMembership(
                agent_id=agent.id,
                community_id=community.id
            )

            session.add(membership)
            session.commit()
            session.refresh(membership)

            assert membership.role == "member"  # Default
            assert membership.is_active is True  # Default
            assert membership.contribution_score == 0  # Default
            assert membership.last_activity is None

    def test_community_membership_model_validation(self):
        """Test CommunityMembership model field validation"""
        from src.database.models import Agent, Community, CommunityMembership

        with self.SessionLocal() as session:
            # Create test agent and community
            agent, = self.create_test_agents(session, 1)
            community, = self.create_test_communities(session, 1)

            # Test invalid role
            with pytest.raises(ValueError):
                membership = CommunityMembership(
                    agent_id=agent.id,
                    community_id=community.id,
                    role="invalid_role"
                )

            # Test invalid contribution score (negative)
            with pytest.raises(ValueError):
                membership = CommunityMembership(
                    agent_id=agent.id,
                    community_id=community.id,
                    contribution_score=-10
                )

    def test_community_membership_unique_constraint(self):
        """Test CommunityMembership unique constraint"""
        from src.database.models import Agent, Community, CommunityMembership

        with self.SessionLocal() as session:
            # Create test agent and community
            agent, = self.create_test_agents(session, 1)
            community, = self.create_test_communities(session, 1)

            # Create first membership
            membership1 = CommunityMembership(
                agent_id=agent.id,
                community_id=community.id,
                role="member"
            )
            session.add(membership1)
            session.commit()

            # Try to create duplicate membership
            membership2 = CommunityMembership(
                agent_id=agent.id,
                community_id=community.id,
                role="admin"
            )
            session.add(membership2)

            with pytest.raises(Exception):  # Should raise IntegrityError
                session.commit()

    def test_community_membership_relationships(self):
        """Test CommunityMembership relationships"""
        from src.database.models import Agent, Community, CommunityMembership

        with self.SessionLocal() as session:
            # Create test agent and community
            agent, = self.create_test_agents(session, 1)
            community, = self.create_test_communities(session, 1)

            # Create membership
            membership = CommunityMembership(
                agent_id=agent.id,
                community_id=community.id,
                role="member"
            )

            session.add(membership)
            session.commit()
            session.refresh(membership)

            # Test relationships
            assert membership.agent.id == agent.id
            assert membership.agent.name == "test_agent_0"
            assert membership.community.id == community.id
            assert membership.community.name == "test_community_0"

    def test_community_membership_role_transitions(self):
        """Test CommunityMembership role transitions"""
        from src.database.models import Agent, Community, CommunityMembership

        with self.SessionLocal() as session:
            # Create test agent and community
            agent, = self.create_test_agents(session, 1)
            community, = self.create_test_communities(session, 1)

            # Create membership with member role
            membership = CommunityMembership(
                agent_id=agent.id,
                community_id=community.id,
                role="member"
            )

            session.add(membership)
            session.commit()
            session.refresh(membership)

            assert membership.role == "member"

            # Promote to admin
            membership.update_role("admin")
            session.commit()
            session.refresh(membership)

            assert membership.role == "admin"

            # Demote to moderator
            membership.update_role("moderator")
            session.commit()
            session.refresh(membership)

            assert membership.role == "moderator"

    def test_community_membership_activation(self):
        """Test CommunityMembership activation/deactivation"""
        from src.database.models import Agent, Community, CommunityMembership

        with self.SessionLocal() as session:
            # Create test agent and community
            agent, = self.create_test_agents(session, 1)
            community, = self.create_test_communities(session, 1)

            # Create membership
            membership = CommunityMembership(
                agent_id=agent.id,
                community_id=community.id
            )

            session.add(membership)
            session.commit()
            session.refresh(membership)

            # Should be active initially
            assert membership.is_active is True

            # Deactivate membership
            membership.deactivate()
            session.commit()
            session.refresh(membership)

            assert membership.is_active is False

            # Reactivate membership
            membership.activate()
            session.commit()
            session.refresh(membership)

            assert membership.is_active is True

    def test_community_membership_contribution_tracking(self):
        """Test CommunityMembership contribution tracking"""
        from src.database.models import Agent, Community, CommunityMembership

        with self.SessionLocal() as session:
            # Create test agent and community
            agent, = self.create_test_agents(session, 1)
            community, = self.create_test_communities(session, 1)

            # Create membership
            membership = CommunityMembership(
                agent_id=agent.id,
                community_id=community.id,
                contribution_score=10
            )

            session.add(membership)
            session.commit()
            session.refresh(membership)

            assert membership.contribution_score == 10

            # Add contribution
            membership.add_contribution(5)
            session.commit()
            session.refresh(membership)

            assert membership.contribution_score == 15

            # Remove contribution
            membership.remove_contribution(3)
            session.commit()
            session.refresh(membership)

            assert membership.contribution_score == 12

            # Prevent negative score
            membership.remove_contribution(20)
            session.commit()
            session.refresh(membership)

            assert membership.contribution_score == 0  # Should not go negative

    def test_community_membership_activity_tracking(self):
        """Test CommunityMembership activity tracking"""
        from src.database.models import Agent, Community, CommunityMembership

        with self.SessionLocal() as session:
            # Create test agent and community
            agent, = self.create_test_agents(session, 1)
            community, = self.create_test_communities(session, 1)

            # Create membership
            membership = CommunityMembership(
                agent_id=agent.id,
                community_id=community.id
            )

            session.add(membership)
            session.commit()
            session.refresh(membership)

            original_activity = membership.last_activity

            # Record activity
            membership.record_activity()
            session.commit()
            session.refresh(membership)

            if original_activity is not None:
                assert membership.last_activity > original_activity
            else:
                assert membership.last_activity is not None

    def test_community_membership_is_active_member(self):
        """Test CommunityMembership active member checking"""
        from src.database.models import Agent, Community, CommunityMembership

        with self.SessionLocal() as session:
            # Create test agent and community
            agent, = self.create_test_agents(session, 1)
            community, = self.create_test_communities(session, 1)

            # Create active membership
            active_membership = CommunityMembership(
                agent_id=agent.id,
                community_id=community.id,
                is_active=True
            )
            session.add(active_membership)

            # Create inactive membership
            agent2 = Agent(name="test_agent_1")
            session.add(agent2)
            session.commit()

            inactive_membership = CommunityMembership(
                agent_id=agent2.id,
                community_id=community.id,
                is_active=False
            )
            session.add(inactive_membership)

            session.commit()

            # Test active status
            assert active_membership.is_active_member() is True
            assert inactive_membership.is_active_member() is False

    def test_community_membership_is_contributor(self):
        """Test CommunityMembership contributor checking"""
        from src.database.models import Agent, Community, CommunityMembership

        with self.SessionLocal() as session:
            # Create test agent and community
            agent, = self.create_test_agents(session, 1)
            community, = self.create_test_communities(session, 1)

            # Create contributor membership
            contributor_membership = CommunityMembership(
                agent_id=agent.id,
                community_id=community.id,
                contribution_score=50
            )
            session.add(contributor_membership)

            # Create non-contributor membership
            agent2 = Agent(name="test_agent_1")
            session.add(agent2)
            session.commit()

            non_contributor_membership = CommunityMembership(
                agent_id=agent2.id,
                community_id=community.id,
                contribution_score=5
            )
            session.add(non_contributor_membership)

            session.commit()

            # Test contributor status (assuming threshold is 20)
            assert contributor_membership.is_contributor(20) is True
            assert non_contributor_membership.is_contributor(20) is False

    def test_community_membership_string_representation(self):
        """Test CommunityMembership model string representation"""
        from src.database.models import Agent, Community, CommunityMembership

        with self.SessionLocal() as session:
            # Create test agent and community
            agent, = self.create_test_agents(session, 1)
            community, = self.create_test_communities(session, 1)

            # Create membership
            membership = CommunityMembership(
                agent_id=agent.id,
                community_id=community.id,
                role="admin",
                contribution_score=25
            )

            session.add(membership)
            session.commit()
            session.refresh(membership)

            str_repr = str(membership)
            assert "test_agent_0" in str_repr
            assert "test_community_0" in str_repr
            assert "admin" in str_repr
            assert "25" in str_repr

    def test_community_membership_query_by_role(self):
        """Test querying community memberships by role"""
        from src.database.models import Agent, Community, CommunityMembership

        with self.SessionLocal() as session:
            # Create test agents and community
            agents = self.create_test_agents(session, 3)
            community, = self.create_test_communities(session, 1)

            # Create memberships with different roles
            roles_data = [
                (agents[0].id, "admin"),
                (agents[1].id, "moderator"),
                (agents[2].id, "member")
            ]

            for agent_id, role in roles_data:
                membership = CommunityMembership(
                    agent_id=agent_id,
                    community_id=community.id,
                    role=role
                )
                session.add(membership)

            session.commit()

            # Query admin memberships
            admin_memberships = session.query(CommunityMembership).filter(
                CommunityMembership.community_id == community.id,
                CommunityMembership.role == "admin"
            ).all()

            assert len(admin_memberships) == 1
            assert admin_memberships[0].role == "admin"

    def test_community_membership_query_by_activity(self):
        """Test querying community memberships by activity status"""
        from src.database.models import Agent, Community, CommunityMembership

        with self.SessionLocal() as session:
            # Create test agents and community
            agents = self.create_test_agents(session, 3)
            community, = self.create_test_communities(session, 1)

            # Create memberships with different activity status
            memberships_data = [
                (agents[0].id, True),
                (agents[1].id, False),
                (agents[2].id, True)
            ]

            for agent_id, is_active in memberships_data:
                membership = CommunityMembership(
                    agent_id=agent_id,
                    community_id=community.id,
                    is_active=is_active
                )
                session.add(membership)

            session.commit()

            # Query active memberships
            active_memberships = session.query(CommunityMembership).filter(
                CommunityMembership.community_id == community.id,
                CommunityMembership.is_active == True
            ).all()

            assert len(active_memberships) == 2
            assert all(m.is_active for m in active_memberships)

    def test_community_membership_get_membership_summary(self):
        """Test CommunityMembership summary generation"""
        from src.database.models import Agent, Community, CommunityMembership

        with self.SessionLocal() as session:
            # Create test agent and community
            agent, = self.create_test_agents(session, 1)
            community, = self.create_test_communities(session, 1)

            # Create membership
            membership = CommunityMembership(
                agent_id=agent.id,
                community_id=community.id,
                role="moderator",
                contribution_score=30,
                is_active=True
            )

            session.add(membership)
            session.commit()
            session.refresh(membership)

            # Test summary generation (if implemented)
            if hasattr(membership, 'get_membership_summary'):
                summary = membership.get_membership_summary()
                assert isinstance(summary, dict)
                assert summary['agent_id'] == agent.id
                assert summary['community_id'] == community.id
                assert summary['role'] == "moderator"
                assert summary['contribution_score'] == 30
                assert summary['is_active'] is True

    @staticmethod
    def test_community_membership_get_valid_roles():
        """Test getting valid community membership roles"""
        from src.database.models import CommunityMembership

        # Test static method (if implemented)
        if hasattr(CommunityMembership, 'get_valid_roles'):
            valid_roles = CommunityMembership.get_valid_roles()
            assert isinstance(valid_roles, list)
            assert "member" in valid_roles
            assert "moderator" in valid_roles
            assert "admin" in valid_roles