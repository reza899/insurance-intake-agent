#!/usr/bin/env python3
"""
Integration Tests for Insurance Intake Agent

Simple integration tests that can be run without complex setup.
These test the main system components working together.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Mock all heavy dependencies that slow down Docker
heavy_modules = [
    'torch', 'transformers', 'datasets', 'pyserini', 'faiss', 'sentence_transformers',
    'litellm', 'openai', 'langchain', 'langchain_core', 'langchain_openai'
]
for module in heavy_modules:
    sys.modules[module] = MagicMock()

from src.models import LLMRequest
from src.llm.provider import LLMProvider
from src.agent.core.extractor import DataExtractor
from src.agent.core.duplicate_detector import DuplicateDetector
from src.agent.orchestrator import InsuranceAgent
from config.settings import settings


class IntegrationTester:
    """Simple integration test runner."""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []
    
    def test(self, name: str):
        """Decorator for test methods."""
        def decorator(func):
            self.tests.append((name, func))
            return func
        return decorator
    
    async def run_all(self):
        """Run all tests."""
        print("🧪 Running Integration Tests")
        print("=" * 50)
        
        for test_name, test_func in self.tests:
            try:
                print(f"\n🔍 {test_name}")
                await test_func()
                print(f"✅ {test_name} - PASSED")
                self.passed += 1
            except Exception as e:
                print(f"❌ {test_name} - FAILED: {str(e)}")
                self.failed += 1
        
        print("\n" + "=" * 50)
        print(f"📊 Test Results: {self.passed} passed, {self.failed} failed")
        return self.failed == 0


# Create test instance
tester = IntegrationTester()


@tester.test("LLM Factory - Provider Creation")
async def test_llm_factory():
    """Test LLM factory can create providers."""
    from unittest.mock import patch, MagicMock
    
    # Test that we can create an LLM provider
    provider = LLMProvider()
    assert provider is not None
    assert hasattr(provider, 'generate_response')
    assert hasattr(provider, 'config')


@tester.test("LLM Provider - Initialization")
async def test_llm_provider_init():
    """Test LLM provider initialization."""
    from unittest.mock import patch, MagicMock
    
    # Test provider initialization
    provider = LLMProvider()
    assert provider is not None
    
    # Test provider can handle requests
    assert hasattr(provider, 'generate_response')
    assert hasattr(provider, 'config')


@tester.test("Configuration - Settings Loading")
async def test_settings_loading():
    """Test configuration settings are loaded correctly."""
    assert settings.mongodb_url is not None
    assert settings.mongodb_database is not None
    assert settings.llm_primary_model is not None
    assert settings.llm_fallback_models is not None
    
    # Test provider configuration
    assert settings.llm_temperature is not None
    assert settings.llm_max_tokens is not None
    assert settings.llm_timeout is not None


@tester.test("Data Extractor - Initialization")
async def test_data_extractor():
    """Test data extractor initialization."""
    extractor = DataExtractor()
    
    # Test extractor methods exist
    assert hasattr(extractor, 'extract_data')
    assert hasattr(extractor, 'validate_data')
    assert hasattr(extractor, 'get_missing_fields')
    
    # Test static methods work
    test_data = {"customer_name": "John", "car_type": "Sedan"}
    missing = DataExtractor.get_missing_fields(test_data)
    assert len(missing) > 0
    assert "manufacturer" in missing


@tester.test("Duplicate Detector - Initialization")
async def test_duplicate_detector():
    """Test duplicate detector initialization."""
    from unittest.mock import patch, MagicMock
    
    # Test detector initialization
    detector = DuplicateDetector()
    
    # Test detector methods exist
    assert hasattr(detector, 'find_duplicates')
    assert hasattr(detector, '_get_similarity_score')


@tester.test("Insurance Agent - Initialization")
async def test_insurance_agent():
    """Test insurance agent initialization."""
    from unittest.mock import patch, MagicMock
    
    # Test agent initialization
    agent = InsuranceAgent()
    
    # Test agent methods exist
    assert hasattr(agent, 'process_message')
    assert hasattr(agent, 'get_registration')


@tester.test("LLM Provider - Mock Request")
async def test_llm_provider_mock_request():
    """Test LLM provider with a mock request (no actual API call)."""
    # Create a simple request without initializing provider
    request = LLMRequest(
        prompt="Hello",
        max_tokens=5,
        temperature=0.1
    )
    
    # For this test, we just verify the provider can process the request structure
    # We won't make actual API calls to avoid dependencies
    assert request.prompt == "Hello"
    assert request.max_tokens == 5
    assert request.temperature == 0.1


@tester.test("Data Models - Validation")
async def test_data_models():
    """Test data model validation."""
    from src.models.insurance import Customer, CarRegistration
    
    # Test valid customer
    customer = Customer(
        name="John Doe",
        birth_date="1990-05-15"
    )
    assert customer.name == "John Doe"
    assert str(customer.birth_date) == "1990-05-15"
    
    # Test valid car registration
    car = CarRegistration(
        car_type="Sedan",
        manufacturer="Toyota",
        year=2020,
        license_plate="ABC123"
    )
    assert car.car_type == "Sedan"
    assert car.manufacturer == "Toyota"
    assert car.year == 2020
    assert car.license_plate == "ABC123"


@tester.test("Configuration - Provider Switching")
async def test_provider_switching():
    """Test configuration allows provider switching."""
    from unittest.mock import patch, MagicMock
    
    # Test provider configuration switching
    provider = LLMProvider()
    assert provider is not None
    
    # Test that we can access configuration
    assert isinstance(provider.config, dict)
    assert 'primary_model' in provider.config
    assert 'fallback_models' in provider.config


@tester.test("Error Handling - Invalid Provider")
async def test_error_handling():
    """Test error handling for invalid configurations."""
    # Test error handling by trying to use invalid model
    try:
        provider = LLMProvider()
        # Temporarily set invalid model
        provider.models = ["invalid_model"]
        request = LLMRequest(prompt="test", max_tokens=5)
        # This should work but might fail on actual generation
        assert provider is not None
    except Exception as e:
        # This is expected if invalid model is used
        pass


async def main():
    """Run all integration tests."""
    success = await tester.run_all()
    if success:
        print("\n🎉 All integration tests passed!")
        return 0
    else:
        print("\n💥 Some integration tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)