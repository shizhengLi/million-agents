"""
Unit tests for configuration module
"""

import os
import pytest
from pathlib import Path


class TestConfig:
    """Test configuration loading and validation"""

    def test_config_loads_successfully(self, mock_env_vars):
        """Test that configuration loads successfully with all required env vars"""
        from src.config.settings import Settings

        settings = Settings()

        assert settings.openai_api_key == 'test-api-key-12345'
        assert settings.openai_base_url == 'https://api.openai.com/v1'
        assert settings.openai_model == 'gpt-4o-mini'
        assert settings.max_agents == 1000
        assert settings.agent_batch_size == 10
        assert settings.interaction_interval == 1
        assert settings.community_size_limit == 100
        assert settings.database_url == 'sqlite:///:memory:'
        assert settings.log_level == 'DEBUG'

    def test_config_missing_required_vars(self, missing_env_vars):
        """Test that configuration raises error for missing required variables"""
        # Reset singleton instance
        from src.config.settings import Settings
        Settings._instance = None

        with pytest.raises(ValueError, match="Missing required environment variable"):
            Settings()

    def test_config_default_values(self, mock_env_vars):
        """Test that configuration uses default values correctly"""
        from src.config.settings import Settings

        # Reset singleton
        Settings._instance = None

        # Test with minimal required vars only
        # Clear all optional vars
        optional_vars = ['MAX_AGENTS', 'AGENT_BATCH_SIZE', 'INTERACTION_INTERVAL',
                        'COMMUNITY_SIZE_LIMIT', 'DATABASE_URL', 'LOG_LEVEL']
        for var in optional_vars:
            os.environ.pop(var, None)

        settings = Settings()

        # Test default values
        assert settings.max_agents == 1000000  # Default value
        assert settings.agent_batch_size == 100
        assert settings.interaction_interval == 5
        assert settings.community_size_limit == 1000
        assert 'sqlite:///' in settings.database_url
        assert settings.log_level == 'INFO'

    def test_config_invalid_numeric_values(self, mock_env_vars):
        """Test that configuration validates numeric values"""
        from src.config.settings import Settings

        # Test invalid max_agents
        Settings._instance = None
        os.environ['MAX_AGENTS'] = 'invalid'
        with pytest.raises(ValueError, match="Invalid numeric value in environment variables"):
            Settings()

        # Test negative value
        Settings._instance = None
        os.environ['MAX_AGENTS'] = '-1'
        with pytest.raises(ValueError, match="MAX_AGENTS must be a positive integer"):
            Settings()

    def test_config_validation_ranges(self, mock_env_vars):
        """Test that configuration validates value ranges"""
        from src.config.settings import Settings

        # Test batch size too large
        Settings._instance = None
        env_copy = mock_env_vars.copy()
        env_copy['AGENT_BATCH_SIZE'] = '10000'
        os.environ.update(env_copy)
        with pytest.raises(ValueError, match="AGENT_BATCH_SIZE cannot exceed 1000"):
            Settings()

        # Test interaction interval too small
        Settings._instance = None
        env_copy = mock_env_vars.copy()
        env_copy['INTERACTION_INTERVAL'] = '0'
        os.environ.update(env_copy)
        with pytest.raises(ValueError, match="INTERACTION_INTERVAL must be at least 1"):
            Settings()

        # Test batch size larger than max_agents
        Settings._instance = None
        env_copy = mock_env_vars.copy()
        env_copy['MAX_AGENTS'] = '50'
        env_copy['AGENT_BATCH_SIZE'] = '100'
        env_copy['INTERACTION_INTERVAL'] = '1'  # Reset to valid value
        os.environ.update(env_copy)
        with pytest.raises(ValueError, match="AGENT_BATCH_SIZE cannot exceed MAX_AGENTS"):
            Settings()

        # Test negative agent_batch_size
        Settings._instance = None
        env_copy = mock_env_vars.copy()
        env_copy['AGENT_BATCH_SIZE'] = '-1'
        env_copy['INTERACTION_INTERVAL'] = '1'  # Reset to valid value
        os.environ.update(env_copy)
        with pytest.raises(ValueError, match="AGENT_BATCH_SIZE must be a positive integer"):
            Settings()

        # Test negative community_size_limit
        Settings._instance = None
        env_copy = mock_env_vars.copy()
        env_copy['COMMUNITY_SIZE_LIMIT'] = '-1'
        env_copy['AGENT_BATCH_SIZE'] = '10'  # Reset to valid value
        env_copy['INTERACTION_INTERVAL'] = '1'  # Reset to valid value
        os.environ.update(env_copy)
        with pytest.raises(ValueError, match="COMMUNITY_SIZE_LIMIT must be a positive integer"):
            Settings()

        # Test invalid log level
        Settings._instance = None
        env_copy = mock_env_vars.copy()
        env_copy['LOG_LEVEL'] = 'INVALID'
        env_copy['AGENT_BATCH_SIZE'] = '10'  # Reset to valid value
        env_copy['COMMUNITY_SIZE_LIMIT'] = '100'  # Reset to valid value
        env_copy['INTERACTION_INTERVAL'] = '1'  # Reset to valid value
        os.environ.update(env_copy)
        with pytest.raises(ValueError, match="LOG_LEVEL must be one of"):
            Settings()

    def test_config_singleton_behavior(self, mock_env_vars):
        """Test that Settings behaves as singleton"""
        from src.config.settings import Settings

        # Reset singleton to test fresh
        Settings._instance = None

        settings1 = Settings()
        settings2 = Settings()

        assert settings1 is settings2

    def test_config_str_representation(self, mock_env_vars):
        """Test string representation hides sensitive data"""
        from src.config.settings import Settings

        # Reset singleton
        Settings._instance = None

        settings = Settings()
        settings_str = str(settings)

        # Should hide API key
        assert 'test-api-key-12345' not in settings_str
        assert '***' in settings_str

    def test_config_short_api_key(self, mock_env_vars):
        """Test string representation with short API key"""
        from src.config.settings import Settings

        # Reset singleton and set short API key
        Settings._instance = None
        os.environ['OPENAI_API_KEY'] = 'short'

        settings = Settings()
        settings_str = str(settings)

        # Should handle short API key gracefully
        assert 'short' not in settings_str
        assert '***' in settings_str

    def test_config_empty_api_key(self, mock_env_vars):
        """Test configuration with empty API key"""
        from src.config.settings import Settings

        # Reset singleton and set empty API key
        Settings._instance = None
        os.environ['OPENAI_API_KEY'] = '   '  # Whitespace only

        with pytest.raises(ValueError, match="Missing required environment variable"):
            Settings()

    def test_config_whitespace_api_key(self, mock_env_vars):
        """Test configuration with whitespace-only API key"""
        from src.config.settings import Settings

        # Reset singleton and set whitespace API key
        Settings._instance = None
        os.environ['OPENAI_API_KEY'] = '\t\n  '  # Whitespace only

        with pytest.raises(ValueError, match="Missing required environment variable"):
            Settings()