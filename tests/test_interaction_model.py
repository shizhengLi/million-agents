"""
Interaction model tests
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session


class TestInteractionModel:
    """Test Interaction ORM model"""

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
        """Helper method to create test agents with social profiles"""
        from src.database.models import Agent, SocialAgent

        agents = []
        for i in range(count):
            agent = Agent(name=f"test_agent_{i}")
            session.add(agent)
            session.commit()
            session.refresh(agent)

            social_agent = SocialAgent(
                agent_id=agent.id,
                social_preference="balanced"
            )
            session.add(social_agent)
            session.commit()

            agents.append(agent)

        return agents

    def test_interaction_model_creation(self):
        """Test Interaction model creation"""
        from src.database.models import Agent, SocialAgent, Interaction

        with self.SessionLocal() as session:
            # Create test agents
            initiator, recipient = self.create_test_agents(session, 2)

            # Create interaction
            interaction = Interaction(
                initiator_id=initiator.id,
                recipient_id=recipient.id,
                interaction_type="conversation",
                content="Hello, this is a test message!",
                sentiment_score=0.7,
                engagement_score=0.8,
                interaction_metadata={"topic": "general", "duration": 120}
            )

            session.add(interaction)
            session.commit()
            session.refresh(interaction)

            assert interaction.id is not None
            assert interaction.initiator_id == initiator.id
            assert interaction.recipient_id == recipient.id
            assert interaction.interaction_type == "conversation"
            assert interaction.content == "Hello, this is a test message!"
            assert interaction.sentiment_score == 0.7
            assert interaction.engagement_score == 0.8
            assert interaction.interaction_metadata["topic"] == "general"
            assert interaction.interaction_time is not None
            assert interaction.created_at is not None

    def test_interaction_model_defaults(self):
        """Test Interaction model default values"""
        from src.database.models import Agent, SocialAgent, Interaction

        with self.SessionLocal() as session:
            # Create test agents
            initiator, recipient = self.create_test_agents(session, 2)

            # Create interaction with minimal data
            interaction = Interaction(
                initiator_id=initiator.id,
                recipient_id=recipient.id,
                interaction_type="message"
            )

            session.add(interaction)
            session.commit()
            session.refresh(interaction)

            assert interaction.content is None
            assert interaction.sentiment_score is None
            assert interaction.engagement_score is None
            assert interaction.interaction_metadata is None
            assert interaction.interaction_type == "message"
            assert interaction.interaction_time is not None

    def test_interaction_model_validation(self):
        """Test Interaction model field validation"""
        from src.database.models import Agent, SocialAgent, Interaction

        with self.SessionLocal() as session:
            # Create test agents
            initiator, recipient = self.create_test_agents(session, 2)

            # Test invalid interaction type
            with pytest.raises(ValueError):
                interaction = Interaction(
                    initiator_id=initiator.id,
                    recipient_id=recipient.id,
                    interaction_type="invalid_type"
                )

            # Test invalid sentiment score (out of bounds)
            with pytest.raises(ValueError):
                interaction = Interaction(
                    initiator_id=initiator.id,
                    recipient_id=recipient.id,
                    interaction_type="conversation",
                    sentiment_score=1.5  # Should be -1 to 1
                )

            # Test invalid engagement score (out of bounds)
            with pytest.raises(ValueError):
                interaction = Interaction(
                    initiator_id=initiator.id,
                    recipient_id=recipient.id,
                    interaction_type="conversation",
                    engagement_score=-0.5  # Should be 0 to 1
                )

    def test_interaction_agent_relationships(self):
        """Test Interaction relationships with agents"""
        from src.database.models import Agent, SocialAgent, Interaction

        with self.SessionLocal() as session:
            # Create test agents
            initiator, recipient = self.create_test_agents(session, 2)

            # Create interaction
            interaction = Interaction(
                initiator_id=initiator.id,
                recipient_id=recipient.id,
                interaction_type="conversation",
                content="Test relationship"
            )

            session.add(interaction)
            session.commit()
            session.refresh(interaction)

            # Test relationships
            assert interaction.initiator.id == initiator.id
            assert interaction.initiator.name == "test_agent_0"
            assert interaction.recipient.id == recipient.id
            assert interaction.recipient.name == "test_agent_1"

    def test_interaction_self_interaction_prevention(self):
        """Test prevention of self-interactions"""
        from src.database.models import Agent, SocialAgent, Interaction

        with self.SessionLocal() as session:
            # Create single test agent
            agent, = self.create_test_agents(session, 1)

            # Try to create self-interaction
            with pytest.raises(ValueError):
                interaction = Interaction(
                    initiator_id=agent.id,
                    recipient_id=agent.id,
                    interaction_type="conversation"
                )

    def test_interaction_timestamps(self):
        """Test Interaction timestamp functionality"""
        from src.database.models import Agent, SocialAgent, Interaction

        with self.SessionLocal() as session:
            # Create test agents
            initiator, recipient = self.create_test_agents(session, 2)

            # Create interaction with specific interaction time
            specific_time = datetime.utcnow() - timedelta(hours=1)
            interaction = Interaction(
                initiator_id=initiator.id,
                recipient_id=recipient.id,
                interaction_type="conversation",
                interaction_time=specific_time
            )

            session.add(interaction)
            session.commit()
            session.refresh(interaction)

            assert interaction.interaction_time == specific_time
            assert interaction.created_at > specific_time  # Created after interaction time

    def test_interaction_update_scores(self):
        """Test updating interaction scores"""
        from src.database.models import Agent, SocialAgent, Interaction

        with self.SessionLocal() as session:
            # Create test agents
            initiator, recipient = self.create_test_agents(session, 2)

            # Create interaction
            interaction = Interaction(
                initiator_id=initiator.id,
                recipient_id=recipient.id,
                interaction_type="conversation"
            )

            session.add(interaction)
            session.commit()
            session.refresh(interaction)

            # Update scores
            interaction.sentiment_score = 0.9
            interaction.engagement_score = 0.7
            session.commit()
            session.refresh(interaction)

            assert interaction.sentiment_score == 0.9
            assert interaction.engagement_score == 0.7

    def test_interaction_query_by_type(self):
        """Test querying interactions by type"""
        from src.database.models import Agent, SocialAgent, Interaction

        with self.SessionLocal() as session:
            # Create test agents
            initiator, recipient = self.create_test_agents(session, 2)

            # Create different types of interactions
            interactions_data = [
                ("conversation", "Let's chat"),
                ("message", "Quick message"),
                ("collaboration", "Working together"),
                ("conversation", "Another chat")
            ]

            for interaction_type, content in interactions_data:
                interaction = Interaction(
                    initiator_id=initiator.id,
                    recipient_id=recipient.id,
                    interaction_type=interaction_type,
                    content=content
                )
                session.add(interaction)

            session.commit()

            # Query conversations
            conversations = session.query(Interaction).filter(
                Interaction.interaction_type == "conversation"
            ).all()

            assert len(conversations) == 2
            assert all(i.interaction_type == "conversation" for i in conversations)

    def test_interaction_query_by_time_range(self):
        """Test querying interactions by time range"""
        from src.database.models import Agent, SocialAgent, Interaction

        with self.SessionLocal() as session:
            # Create test agents
            initiator, recipient = self.create_test_agents(session, 2)

            base_time = datetime.utcnow()

            # Create interactions at different times
            time_offsets = [-2, -1, 0, 1]  # hours ago
            for offset in time_offsets:
                interaction = Interaction(
                    initiator_id=initiator.id,
                    recipient_id=recipient.id,
                    interaction_type="message",
                    interaction_time=base_time + timedelta(hours=offset)
                )
                session.add(interaction)

            session.commit()

            # Query interactions in last hour
            recent_time = base_time + timedelta(minutes=-30)
            recent_interactions = session.query(Interaction).filter(
                Interaction.interaction_time >= recent_time
            ).all()

            assert len(recent_interactions) == 2

    def test_interaction_string_representation(self):
        """Test Interaction model string representation"""
        from src.database.models import Agent, SocialAgent, Interaction

        with self.SessionLocal() as session:
            # Create test agents
            initiator, recipient = self.create_test_agents(session, 2)

            # Create interaction
            interaction = Interaction(
                initiator_id=initiator.id,
                recipient_id=recipient.id,
                interaction_type="conversation",
                content="Test representation"
            )

            session.add(interaction)
            session.commit()
            session.refresh(interaction)

            str_repr = str(interaction)
            assert "test_agent_0" in str_repr
            assert "test_agent_1" in str_repr
            assert "conversation" in str_repr

    def test_interaction_sentiment_analysis_methods(self):
        """Test interaction sentiment analysis helper methods"""
        from src.database.models import Agent, SocialAgent, Interaction

        with self.SessionLocal() as session:
            # Create test agents
            initiator, recipient = self.create_test_agents(session, 2)

            # Create interaction with sentiment and engagement
            interaction = Interaction(
                initiator_id=initiator.id,
                recipient_id=recipient.id,
                interaction_type="conversation",
                sentiment_score=0.8,
                engagement_score=0.8,
                content="This is great!"
            )

            session.add(interaction)
            session.commit()
            session.refresh(interaction)

            # Test sentiment classification (if implemented)
            if hasattr(interaction, 'get_sentiment_label'):
                label = interaction.get_sentiment_label()
                assert label in ["positive", "neutral", "negative"]
                assert label == "positive"  # 0.8 should be positive

            # Test engagement classification (if implemented)
            if hasattr(interaction, 'get_engagement_level'):
                level = interaction.get_engagement_level()
                assert level in ["low", "medium", "high"]
                assert level == "high"  # 0.8 should be high

    def test_interaction_metadata_operations(self):
        """Test interaction metadata operations"""
        from src.database.models import Agent, SocialAgent, Interaction

        with self.SessionLocal() as session:
            # Create test agents
            initiator, recipient = self.create_test_agents(session, 2)

            # Create interaction with complex metadata
            interaction_metadata = {
                "topic": "technology",
                "duration": 300,
                "keywords": ["AI", "machine learning"],
                "context": {
                    "platform": "chat",
                    "channel": "general"
                }
            }

            interaction = Interaction(
                initiator_id=initiator.id,
                recipient_id=recipient.id,
                interaction_type="collaboration",
                interaction_metadata=interaction_metadata
            )

            session.add(interaction)
            session.commit()
            session.refresh(interaction)

            # Test metadata retrieval
            assert interaction.interaction_metadata["topic"] == "technology"
            assert interaction.interaction_metadata["keywords"] == ["AI", "machine learning"]
            assert interaction.interaction_metadata["context"]["platform"] == "chat"