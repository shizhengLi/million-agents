"""
Social agent class for million agents social application
"""

from typing import List, Dict, Any, Set, Optional
import time
import openai
from src.config import Settings
from .base import BaseAgent


class SocialAgent(BaseAgent):
    """Social agent with personality, interests, and social interactions"""

    # Valid personality types
    VALID_PERSONALITIES = {
        "friendly": "outgoing and sociable",
        "analytical": "logical and detail-oriented",
        "creative": "innovative and artistic",
        "formal": "professional and respectful",
        "casual": "relaxed and informal",
        "curious": "inquisitive and learning-focused",
        "helpful": "supportive and cooperative"
    }

    def __init__(
        self,
        agent_id: Optional[str] = None,
        name: Optional[str] = None,
        personality: str = "friendly",
        interests: Optional[List[str]] = None,
        bio: Optional[str] = None
    ):
        """Initialize a social agent

        Args:
            agent_id: Unique identifier for the agent
            name: Display name for the agent
            personality: Personality type (must be in VALID_PERSONALITIES)
            interests: List of interests the agent has
            bio: Short biography or description
        """
        super().__init__(
            agent_id=agent_id,
            name=name,
            role="social_agent",
            description=bio or f"Social agent with {personality} personality"
        )

        # Validate personality
        if personality not in self.VALID_PERSONALITIES:
            raise ValueError(
                f"Invalid personality '{personality}'. "
                f"Valid options: {', '.join(self.VALID_PERSONALITIES.keys())}"
            )

        self.personality = personality
        self.interests = interests or []
        self.bio = bio or self.description

        # Social network features
        self.friends: Set[SocialAgent] = set()
        self.communities: Set[str] = set()
        self.interaction_history: List[Dict[str, Any]] = []

        # OpenAI client for message generation
        self.settings = Settings()
        self.openai_client = openai.OpenAI(
            api_key=self.settings.openai_api_key,
            base_url=self.settings.openai_base_url
        )

    def add_friend(self, other_agent: 'SocialAgent') -> bool:
        """Add another agent as a friend

        Args:
            other_agent: The agent to add as friend

        Returns:
            True if friend was added, False if already a friend
        """
        if other_agent not in self.friends and other_agent != self:
            self.friends.add(other_agent)
            other_agent.friends.add(self)  # Mutual friendship
            self.update_activity()
            return True
        return False

    def remove_friend(self, other_agent: 'SocialAgent') -> bool:
        """Remove an agent from friends

        Args:
            other_agent: The agent to remove

        Returns:
            True if friend was removed, False if not a friend
        """
        if other_agent in self.friends:
            self.friends.remove(other_agent)
            other_agent.friends.discard(self)  # Remove mutual friendship
            self.update_activity()
            return True
        return False

    def join_community(self, community_id: str) -> bool:
        """Join a community

        Args:
            community_id: ID of the community to join

        Returns:
            True if joined successfully, False if already a member
        """
        if community_id not in self.communities:
            self.communities.add(community_id)
            self.update_activity()
            return True
        return False

    def leave_community(self, community_id: str) -> bool:
        """Leave a community

        Args:
            community_id: ID of the community to leave

        Returns:
            True if left successfully, False if not a member
        """
        if community_id in self.communities:
            self.communities.remove(community_id)
            self.update_activity()
            return True
        return False

    def record_interaction(
        self,
        with_agent_id: str,
        message: str,
        interaction_type: str = "message",
        context: Optional[Dict[str, Any]] = None
    ):
        """Record an interaction with another agent

        Args:
            with_agent_id: ID of the agent interacted with
            message: The message/content of interaction
            interaction_type: Type of interaction (message, greeting, question, etc.)
            context: Additional context about the interaction
        """
        interaction = {
            "with": with_agent_id,
            "message": message,
            "type": interaction_type,
            "timestamp": time.time(),
            "context": context or {}
        }
        self.interaction_history.append(interaction)
        self.update_activity()

    def check_compatibility(self, other_agent: 'SocialAgent') -> float:
        """Calculate compatibility score with another agent

        Args:
            other_agent: The agent to check compatibility with

        Returns:
            Compatibility score between 0.0 and 1.0
        """
        score = 0.0

        # Personality compatibility (30% weight)
        if self.personality == other_agent.personality:
            score += 0.3
        elif self.personality in ["friendly", "helpful"] or other_agent.personality in ["friendly", "helpful"]:
            score += 0.2  # Friendly agents are generally compatible

        # Interest overlap (50% weight)
        common_interests = set(self.interests) & set(other_agent.interests)
        if len(self.interests) > 0 and len(other_agent.interests) > 0:
            interest_overlap = len(common_interests) / min(len(self.interests), len(other_agent.interests))
            score += interest_overlap * 0.5
        elif len(self.interests) == 0 and len(other_agent.interests) == 0:
            score += 0.25  # Both have no interests - moderate compatibility

        # Community overlap (20% weight)
        common_communities = self.communities & other_agent.communities
        if len(self.communities) > 0 and len(other_agent.communities) > 0:
            community_overlap = len(common_communities) / min(len(self.communities), len(other_agent.communities))
            score += community_overlap * 0.2

        return min(score, 1.0)

    def generate_message(
        self,
        context: str,
        prompt: Optional[str] = None,
        max_length: int = 100
    ) -> str:
        """Generate a message based on context using OpenAI

        Args:
            context: The context for message generation
            prompt: Optional custom prompt
            max_length: Maximum message length

        Returns:
            Generated message
        """
        if prompt is None:
            prompt = self._create_default_prompt(context)

        try:
            response = self.openai_client.chat.completions.create(
                model=self.settings.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": f"You are {self.name}, a {self.personality} social agent. "
                        f"Your interests are: {', '.join(self.interests)}. "
                        f"Personality: {self.VALID_PERSONALITIES[self.personality]}. "
                        f"Generate a short, natural response (max {max_length} characters)."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_length * 2,  # Rough estimate
                temperature=0.7
            )

            message = response.choices[0].message.content.strip()
            return message[:max_length]  # Ensure max length

        except Exception as e:
            # Fallback to simple responses if OpenAI fails
            return self._generate_fallback_message(context)

    def _create_default_prompt(self, context: str) -> str:
        """Create a default prompt based on context"""
        prompt = f"Context: {context}\n\n"
        prompt += f"As {self.name} (a {self.personality} agent), respond naturally. "
        if self.interests:
            prompt += f"Consider your interests: {', '.join(self.interests)}. "
        prompt += "Keep your response brief and conversational."
        return prompt

    def _generate_fallback_message(self, context: str) -> str:
        """Generate a fallback message when OpenAI is unavailable"""
        responses = {
            "friendly": [
                "That sounds interesting!",
                "I'd love to chat more about that.",
                "Thanks for sharing!",
                "That's great to hear!"
            ],
            "analytical": [
                "Let me think about that analytically.",
                "That's an interesting perspective.",
                "I'd like to analyze this further.",
                "The data points suggest..."
            ],
            "creative": [
                "That sparks my imagination!",
                "What a creative way to think about it!",
                "I see some interesting possibilities here.",
                "Let's explore this creatively!"
            ],
            "formal": [
                "Thank you for sharing this information.",
                "I appreciate your perspective on this matter.",
                "That's a valid point worth considering.",
                "I understand your position on this."
            ]
        }

        import random
        default_responses = ["Interesting!", "I see.", "Thanks for sharing.", "Great point!"]
        personality_responses = responses.get(self.personality, default_responses)
        return random.choice(personality_responses)

    def get_stats(self) -> Dict[str, Any]:
        """Get social statistics for this agent

        Returns:
            Dictionary containing social statistics
        """
        interaction_partners = set()
        interaction_types = {}

        for interaction in self.interaction_history:
            interaction_partners.add(interaction["with"])
            interaction_type = interaction["type"]
            interaction_types[interaction_type] = interaction_types.get(interaction_type, 0) + 1

        return {
            "total_friends": len(self.friends),
            "total_communities": len(self.communities),
            "total_interactions": len(self.interaction_history),
            "interaction_partners": list(interaction_partners),
            "interaction_types": interaction_types,
            "most_common_interaction": max(interaction_types.items(), key=lambda x: x[1])[0] if interaction_types else None,
            "personality": self.personality,
            "interests_count": len(self.interests),
            "age": self.get_age(),
            "idle_time": self.get_idle_time()
        }

    def __str__(self) -> str:
        """String representation of the social agent"""
        return (
            f"SocialAgent(id={self.agent_id}, "
            f"name={self.name}, "
            f"personality={self.personality}, "
            f"friends={len(self.friends)}, "
            f"communities={len(self.communities)})"
        )