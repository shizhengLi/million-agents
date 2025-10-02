"""
Async Social Agent with enhanced async capabilities
Supports async message generation and interaction processing
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
import openai
from openai import AsyncOpenAI

from .social_agent import SocialAgent
try:
    from ..config.settings import Settings
except ImportError:
    from config.settings import Settings


@dataclass
class AsyncInteraction:
    """Data class for async interaction results"""
    agent_id: str
    message: str
    context: str
    timestamp: float
    processing_time: float
    tokens_used: Optional[int] = None
    error: Optional[str] = None


class AsyncSocialAgent(SocialAgent):
    """Enhanced social agent with async capabilities"""

    def __init__(
        self,
        name: str,
        personality: str = "friendly",
        interests: Optional[List[str]] = None,
        bio: Optional[str] = None,
        openai_client: Optional[AsyncOpenAI] = None
    ):
        """Initialize async social agent

        Args:
            name: Agent name
            personality: Agent personality type
            interests: List of agent interests
            bio: Agent biography
            openai_client: Async OpenAI client instance
        """
        super().__init__(name=name, personality=personality, interests=interests, bio=bio)

        # Async OpenAI client
        if openai_client is None:
            settings = Settings()
            self.openai_client = AsyncOpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url
            )
        else:
            self.openai_client = openai_client

        self.logger = logging.getLogger(f"{__name__}.{self.agent_id}")
        self._generation_cache = {}
        self._rate_limiter = asyncio.Semaphore(5)  # Limit concurrent API calls

    async def generate_message_async(
        self,
        context: str,
        max_tokens: int = 150,
        temperature: float = 0.7,
        use_cache: bool = True
    ) -> AsyncInteraction:
        """Generate a message asynchronously using OpenAI API

        Args:
            context: Context for message generation
            max_tokens: Maximum tokens in response
            temperature: Response randomness
            use_cache: Whether to use cached responses

        Returns:
            AsyncInteraction object with generated message
        """
        start_time = time.time()
        cache_key = f"{self.agent_id}:{context}:{max_tokens}:{temperature}"

        # Check cache first
        if use_cache and cache_key in self._generation_cache:
            cached_result = self._generation_cache[cache_key]
            return AsyncInteraction(
                agent_id=self.agent_id,
                message=cached_result['message'],
                context=context,
                timestamp=time.time(),
                processing_time=0.001,  # Cached response is very fast
                tokens_used=cached_result['tokens_used']
            )

        async with self._rate_limiter:
            try:
                # Create prompt based on agent personality and context
                prompt = self._create_async_prompt(context)

                # Generate response using async OpenAI client
                response = await self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": context}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature
                )

                message = response.choices[0].message.content
                tokens_used = response.usage.total_tokens if response.usage else None
                processing_time = time.time() - start_time

                # Cache the result
                if use_cache:
                    self._generation_cache[cache_key] = {
                        'message': message,
                        'tokens_used': tokens_used
                    }

                # Add to interaction history
                self.record_interaction(context, message)

                return AsyncInteraction(
                    agent_id=self.agent_id,
                    message=message,
                    context=context,
                    timestamp=time.time(),
                    processing_time=processing_time,
                    tokens_used=tokens_used
                )

            except Exception as e:
                self.logger.error(f"Failed to generate async message for {self.agent_id}: {e}")
                processing_time = time.time() - start_time

                # Fallback to sync method if async fails
                try:
                    fallback_message = self.generate_message(context)
                    return AsyncInteraction(
                        agent_id=self.agent_id,
                        message=fallback_message,
                        context=context,
                        timestamp=time.time(),
                        processing_time=processing_time,
                        error=f"Async failed, used fallback: {str(e)}"
                    )
                except Exception as fallback_error:
                    return AsyncInteraction(
                        agent_id=self.agent_id,
                        message=f"I'm having trouble responding right now.",
                        context=context,
                        timestamp=time.time(),
                        processing_time=processing_time,
                        error=f"Both async and sync failed: {str(fallback_error)}"
                    )

    def _create_async_prompt(self, context: str) -> str:
        """Create a detailed prompt for async message generation"""
        interests_str = ", ".join(self.interests) if self.interests else "various topics"

        prompt = f"""You are {self.name}, a {self.personality} social agent.

Bio: {self.bio}
Interests: {interests_str}

Generate a natural, engaging response that reflects your personality and interests.
Keep your response concise and relevant to the context.
Be authentic to your persona while being helpful and engaging.

Context: {context}

Response:"""

        return prompt

    async def interact_with_async(
        self,
        other_agent: 'AsyncSocialAgent',
        context: str,
        max_tokens: int = 100
    ) -> Dict[str, AsyncInteraction]:
        """Have an async interaction with another agent

        Args:
            other_agent: The other agent to interact with
            context: Context for the interaction
            max_tokens: Maximum tokens for responses

        Returns:
            Dictionary with both agents' responses
        """
        # Generate both agents' responses concurrently
        tasks = [
            self.generate_message_async(context, max_tokens),
            other_agent.generate_message_async(context, max_tokens)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        interactions = {}
        for i, agent in enumerate([self, other_agent]):
            if isinstance(results[i], Exception):
                interactions[agent.agent_id] = AsyncInteraction(
                    agent_id=agent.agent_id,
                    message="Unable to generate response",
                    context=context,
                    timestamp=time.time(),
                    processing_time=0.0,
                    error=str(results[i])
                )
            else:
                interactions[agent.agent_id] = results[i]

        return interactions

    async def batch_interact_async(
        self,
        agents: List['AsyncSocialAgent'],
        context: str,
        max_concurrent: int = 3
    ) -> List[Dict[str, AsyncInteraction]]:
        """Interact with multiple agents asynchronously

        Args:
            agents: List of agents to interact with
            context: Context for interactions
            max_concurrent: Maximum concurrent interactions

        Returns:
            List of interaction results
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def interact_with_single(agent: 'AsyncSocialAgent') -> Dict[str, AsyncInteraction]:
            async with semaphore:
                return await self.interact_with_async(agent, context)

        # Execute interactions concurrently
        tasks = [interact_with_single(agent) for agent in agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter exceptions
        valid_interactions = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Batch interaction failed: {result}")
            else:
                valid_interactions.append(result)

        return valid_interactions

    async def update_profile_async(
        self,
        name: Optional[str] = None,
        personality: Optional[str] = None,
        interests: Optional[List[str]] = None,
        bio: Optional[str] = None
    ) -> None:
        """Update agent profile asynchronously

        Args:
            name: New name (optional)
            personality: New personality (optional)
            interests: New interests (optional)
            bio: New biography (optional)
        """
        # Simulate async profile update (e.g., database update)
        await asyncio.sleep(0.001)

        # Update profile
        if name is not None:
            self.name = name
        if personality is not None:
            self.personality = personality
        if interests is not None:
            self.interests = interests
        if bio is not None:
            self.bio = bio

        # Clear cache since profile changed
        self._generation_cache.clear()

    async def get_compatibility_score_async(
        self,
        other_agent: 'AsyncSocialAgent'
    ) -> float:
        """Calculate compatibility with another agent asynchronously

        Args:
            other_agent: Agent to calculate compatibility with

        Returns:
            Compatibility score (0.0 to 1.0)
        """
        # Simulate async compatibility calculation
        await asyncio.sleep(0.001)

        # Use existing compatibility calculation
        return self.check_compatibility(other_agent)

    async def analyze_sentiment_async(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text asynchronously

        Args:
            text: Text to analyze

        Returns:
            Dictionary with sentiment scores
        """
        # Simulate async sentiment analysis
        await asyncio.sleep(0.001)

        # Simple sentiment analysis based on keywords
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst', 'hate']

        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)

        total_words = len(words)
        if total_words == 0:
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}

        positive_score = positive_count / total_words
        negative_score = negative_count / total_words
        neutral_score = 1.0 - positive_score - negative_score

        return {
            'positive': max(0.0, positive_score),
            'negative': max(0.0, negative_score),
            'neutral': max(0.0, neutral_score)
        }

    def get_async_stats(self) -> Dict[str, Any]:
        """Get async-specific statistics

        Returns:
            Dictionary with async performance stats
        """
        return {
            'agent_id': self.agent_id,
            'name': self.name,
            'personality': self.personality,
            'cache_size': len(self._generation_cache),
            'total_interactions': len(self.interaction_history),
            'friends_count': len(self.friends),
            'communities_count': len(self.communities) if self.communities else 0,
            'interests': self.interests,
            'openai_client_configured': self.openai_client is not None
        }

    def clear_cache(self):
        """Clear the generation cache"""
        self._generation_cache.clear()

    async def cleanup_async(self):
        """Cleanup resources asynchronously"""
        self.clear_cache()
        # Any other async cleanup can be added here
        await asyncio.sleep(0.001)