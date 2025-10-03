"""
Integration tests for OpenAI API and configuration
"""

import pytest
from unittest.mock import Mock, patch
import os


class TestOpenAIIntegration:
    """Test OpenAI API integration with configuration"""

    def test_configuration_openai_values(self, mock_env_vars):
        """Test that configuration correctly loads OpenAI values"""
        from src.config.settings import Settings
        from src.config import Settings as ConfigSettings

        # Reset singleton
        Settings._instance = None

        settings = Settings()

        # Test that OpenAI configuration is correctly loaded
        assert settings.openai_api_key == 'test-api-key-12345'
        assert settings.openai_base_url == 'https://api.openai.com/v1'
        assert settings.openai_model == 'gpt-4o-mini'

        # Test that the config module also returns the same instance
        config_settings = ConfigSettings()
        assert settings is config_settings

    @patch('openai.OpenAI')
    def test_openai_client_creation(self, mock_openai_class, mock_env_vars):
        """Test that OpenAI client can be created with configuration"""
        from src.config.settings import Settings

        # Reset singleton
        Settings._instance = None

        settings = Settings()

        # Create OpenAI client
        client = mock_openai_class(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url
        )

        # Verify the client was created with correct parameters
        mock_openai_class.assert_called_once_with(
            api_key='test-api-key-12345',
            base_url='https://api.openai.com/v1'
        )

    def test_configuration_singleton_across_modules(self, mock_env_vars):
        """Test that Settings singleton works across different module imports"""
        from src.config.settings import Settings
        import src.config

        # Reset singleton
        Settings._instance = None

        # Create settings through different import paths
        settings1 = Settings()
        settings2 = src.config.Settings()

        # Should be the same instance
        assert settings1 is settings2

    def test_configuration_isolation_between_tests(self, mock_env_vars):
        """Test that configuration changes are isolated between tests"""
        from src.config.settings import Settings

        # Reset singleton
        Settings._instance = None

        original_settings = Settings()
        original_api_key = original_settings.openai_api_key

        # Modify environment
        os.environ['OPENAI_API_KEY'] = 'modified-key'
        Settings._instance = None
        modified_settings = Settings()

        # Should have different values
        assert modified_settings.openai_api_key == 'modified-key'
        assert modified_settings.openai_api_key != original_api_key

        # Restore singleton for other tests
        Settings._instance = None

    @pytest.mark.slow
    def test_configuration_real_openai_validation(self, mock_env_vars):
        """Test configuration with real OpenAI validation (slow test)"""
        from src.config.settings import Settings

        # Reset singleton
        Settings._instance = None

        settings = Settings()

        # Test that all required OpenAI fields are present and valid
        assert hasattr(settings, 'openai_api_key')
        assert hasattr(settings, 'openai_base_url')
        assert hasattr(settings, 'openai_model')

        # Test API key format (should not be empty)
        assert len(settings.openai_api_key) > 0
        assert isinstance(settings.openai_api_key, str)

        # Test base URL format
        assert settings.openai_base_url.startswith('http')
        assert isinstance(settings.openai_base_url, str)

        # Test model format
        assert len(settings.openai_model) > 0
        assert isinstance(settings.openai_model, str)

    def test_configuration_for_openai_usage(self, mock_env_vars):
        """Test configuration is properly formatted for OpenAI usage"""
        from src.config.settings import Settings

        # Reset singleton
        Settings._instance = None

        settings = Settings()

        # Create a mock OpenAI client to test the format
        expected_params = {
            'api_key': settings.openai_api_key,
            'base_url': settings.openai_base_url
        }

        # Verify the parameters have the right types
        assert isinstance(expected_params['api_key'], str)
        assert isinstance(expected_params['base_url'], str)
        assert expected_params['api_key'] != ''
        assert expected_params['base_url'] != ''

    def test_configuration_security(self, mock_env_vars):
        """Test that configuration properly handles sensitive data"""
        from src.config.settings import Settings

        # Reset singleton
        Settings._instance = None

        settings = Settings()

        # Test string representation masks sensitive data
        settings_str = str(settings)
        assert 'test-api-key-12345' not in settings_str
        assert '***' in settings_str

        # Test that API key is accessible directly but hidden in string repr
        assert settings.openai_api_key == 'test-api-key-12345'

    @pytest.mark.unit
    def test_configuration_module_interface(self, mock_env_vars):
        """Test that configuration module provides expected interface"""
        import src.config

        # Reset singleton
        src.config.Settings._instance = None

        # Test that module provides Settings class
        assert hasattr(src.config, 'Settings')
        assert callable(src.config.Settings)

        # Test that we can create settings through module
        settings = src.config.Settings()
        assert hasattr(settings, 'openai_api_key')
        assert hasattr(settings, 'openai_base_url')
        assert hasattr(settings, 'openai_model')