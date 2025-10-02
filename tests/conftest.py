"""
pytest configuration and fixtures
"""

import pytest
import os
import tempfile
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing"""
    original_env = os.environ.copy()

    test_env = {
        'OPENAI_API_KEY': 'test-api-key-12345',
        'OPENAI_BASE_URL': 'https://api.openai.com/v1',
        'OPENAI_MODEL': 'gpt-4o-mini',
        'MAX_AGENTS': '1000',
        'AGENT_BATCH_SIZE': '10',
        'INTERACTION_INTERVAL': '1',
        'COMMUNITY_SIZE_LIMIT': '100',
        'DATABASE_URL': 'sqlite:///:memory:',
        'LOG_LEVEL': 'DEBUG'
    }

    os.environ.update(test_env)
    yield test_env

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def missing_env_vars():
    """Test environment with missing required variables"""
    import dotenv

    original_env = os.environ.copy()

    # Clear all environment variables
    os.environ.clear()

    # Ensure .env file is not loaded during this test
    # Temporarily rename .env file if it exists
    env_file_path = Path(__file__).parent.parent / '.env'
    temp_env_path = None
    if env_file_path.exists():
        temp_env_path = env_file_path.with_suffix('.env.tmp')
        env_file_path.rename(temp_env_path)

    yield

    # Restore .env file if it was renamed
    if temp_env_path and temp_env_path.exists():
        temp_env_path.rename(env_file_path)

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)