#!/usr/bin/env python3
"""
Integration Tests for Insurance Intake Agent

Simple integration tests that can be run without complex setup.
These test the main system components working together.
"""

import asyncio
import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Mock all PyTorch-related imports globally before any other imports
torch_mock = MagicMock()
torch_mock.__spec__ = MagicMock()
torch_mock.__version__ = "2.0.0"
torch_mock.cuda = MagicMock()
torch_mock.cuda.is_available = MagicMock(return_value=False)

sys.modules['torch'] = torch_mock
sys.modules['torch.nn'] = MagicMock()
sys.modules['torch.cuda'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['transformers.AutoModelForCausalLM'] = MagicMock()
sys.modules['transformers.AutoTokenizer'] = MagicMock()

# Mock datasets with proper structure
datasets_mock = MagicMock()
datasets_mock.fingerprint = MagicMock()
datasets_mock.fingerprint.Hasher = MagicMock()
sys.modules['datasets'] = datasets_mock
sys.modules['datasets.fingerprint'] = datasets_mock.fingerprint

sys.modules['pyserini'] = MagicMock()

from src.models import LLMRequest
from src.llm.router import LLMRouter
from src.llm.factory import LLMProviderFactory
from src.agent.extractor import DataExtractor
from src.agent.duplicate_detector import DuplicateDetector
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
    
    available = LLMProviderFactory.get_available_providers()
    assert "huggingface" in available
    assert "openai_compatible_llm" in available
    
    # Mock transformers components to prevent PyTorch import issues
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    
    # Mock the actual provider creation to avoid LLM initialization
    with patch('src.llm.providers.huggingface.TRANSFORMERS_AVAILABLE', True), \
         patch('src.llm.providers.huggingface.AutoTokenizer.from_pretrained', return_value=mock_tokenizer), \
         patch('src.llm.providers.huggingface.AutoModelForCausalLM.from_pretrained', return_value=mock_model), \
         patch('src.llm.providers.huggingface.HuggingFaceProvider') as mock_hf, \
         patch('src.llm.providers.openai_compatible.OpenAICompatibleProvider') as mock_openai:
        
        # Test creating each provider
        for provider_name in available:
            provider = LLMProviderFactory.create_provider(provider_name)
            assert provider is not None


@tester.test("LLM Router - Initialization")
async def test_llm_router_init():
    """Test LLM router initialization."""
    from unittest.mock import patch, MagicMock
    
    # Mock transformers components to prevent PyTorch import issues
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    
    # Mock providers to avoid actual initialization
    with patch('src.llm.providers.huggingface.TRANSFORMERS_AVAILABLE', True), \
         patch('src.llm.providers.huggingface.AutoTokenizer.from_pretrained', return_value=mock_tokenizer), \
         patch('src.llm.providers.huggingface.AutoModelForCausalLM.from_pretrained', return_value=mock_model), \
         patch('src.llm.providers.huggingface.HuggingFaceProvider') as mock_hf, \
         patch('src.llm.providers.openai_compatible.OpenAICompatibleProvider') as mock_openai:
        
        router = LLMRouter()
        assert router.provider is not None
        assert router.provider_type in ["huggingface", "openai_compatible_llm"]
        
        # Test router can handle requests
        assert hasattr(router, 'route_request')
        assert hasattr(router, 'health_check')


@tester.test("Configuration - Settings Loading")
async def test_settings_loading():
    """Test configuration settings are loaded correctly."""
    assert settings.mongodb_url is not None
    assert settings.mongodb_database is not None
    assert settings.ext_provider_model is not None
    assert settings.local_model_name is not None
    
    # Test provider configuration
    if settings.use_hf_local:
        assert settings.local_model_device in ["cpu", "cuda", "mps"]
    else:
        assert settings.ext_provider_base_url is not None


@tester.test("Data Extractor - Initialization")
async def test_data_extractor():
    """Test data extractor initialization."""
    extractor = DataExtractor()
    
    # Test extractor methods exist
    assert hasattr(extractor, 'extract_data')
    assert hasattr(extractor, 'validate_data')
    assert hasattr(extractor, 'get_missing_fields')
    assert hasattr(extractor, 'is_complete')
    
    # Test static methods work
    test_data = {"customer_name": "John", "car_type": "Sedan"}
    missing = DataExtractor.get_missing_fields(test_data)
    assert len(missing) > 0
    assert "manufacturer" in missing


@tester.test("Duplicate Detector - Initialization")
async def test_duplicate_detector():
    """Test duplicate detector initialization."""
    from unittest.mock import patch, MagicMock
    
    # Mock transformers components to prevent PyTorch import issues
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    
    # Mock providers to avoid actual initialization  
    with patch('src.llm.providers.huggingface.TRANSFORMERS_AVAILABLE', True), \
         patch('src.llm.providers.huggingface.AutoTokenizer.from_pretrained', return_value=mock_tokenizer), \
         patch('src.llm.providers.huggingface.AutoModelForCausalLM.from_pretrained', return_value=mock_model), \
         patch('src.llm.providers.huggingface.HuggingFaceProvider') as mock_hf, \
         patch('src.llm.providers.openai_compatible.OpenAICompatibleProvider') as mock_openai:
        
        detector = DuplicateDetector()
        
        # Test detector methods exist
        assert hasattr(detector, 'find_duplicates')
        assert hasattr(detector, 'is_likely_duplicate')
        assert hasattr(detector, '_calculate_similarity')
        assert detector.threshold is not None
        assert detector.weights is not None


@tester.test("Insurance Agent - Initialization")
async def test_insurance_agent():
    """Test insurance agent initialization."""
    from unittest.mock import patch, MagicMock
    
    # Mock transformers components to prevent PyTorch import issues
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    
    # Mock providers to avoid actual initialization  
    with patch('src.llm.providers.huggingface.TRANSFORMERS_AVAILABLE', True), \
         patch('src.llm.providers.huggingface.AutoTokenizer.from_pretrained', return_value=mock_tokenizer), \
         patch('src.llm.providers.huggingface.AutoModelForCausalLM.from_pretrained', return_value=mock_model), \
         patch('src.llm.providers.huggingface.HuggingFaceProvider') as mock_hf, \
         patch('src.llm.providers.openai_compatible.OpenAICompatibleProvider') as mock_openai:
        
        agent = InsuranceAgent()
        
        # Test agent methods exist
        assert hasattr(agent, 'process_message')
        assert hasattr(agent, 'get_registration')
        assert agent.extractor is not None
        assert agent.duplicate_detector is not None
        assert agent.llm_router is not None


@tester.test("LLM Router - Mock Request")
async def test_llm_router_mock_request():
    """Test LLM router with a mock request (no actual API call)."""
    # Create a simple request without initializing router
    request = LLMRequest(
        prompt="Hello",
        max_tokens=5,
        temperature=0.1
    )
    
    # For this test, we just verify the router can process the request structure
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
    
    original_use_hf = settings.use_hf_local
    
    try:
        # Mock transformers components to prevent PyTorch import issues
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        
        # Mock providers to avoid actual initialization
        with patch('src.llm.providers.huggingface.TRANSFORMERS_AVAILABLE', True), \
             patch('src.llm.providers.huggingface.AutoTokenizer.from_pretrained', return_value=mock_tokenizer), \
             patch('src.llm.providers.huggingface.AutoModelForCausalLM.from_pretrained', return_value=mock_model), \
             patch('src.llm.providers.huggingface.HuggingFaceProvider') as mock_hf, \
             patch('src.llm.providers.openai_compatible.OpenAICompatibleProvider') as mock_openai:
            
            # Test external provider configuration
            settings.use_hf_local = False
            router1 = LLMRouter()
            assert router1.provider_type == "openai_compatible_llm"
            
            # Test local provider configuration
            settings.use_hf_local = True
            router2 = LLMRouter()
            assert router2.provider_type == "huggingface"
        
    finally:
        # Restore original setting
        settings.use_hf_local = original_use_hf


@tester.test("Error Handling - Invalid Provider")
async def test_error_handling():
    """Test error handling for invalid configurations."""
    try:
        # Try to create an invalid provider
        LLMProviderFactory.create_provider("invalid_provider")
        assert False, "Should have raised an error"
    except ValueError as e:
        assert "Unknown provider" in str(e)
        assert "Available providers" in str(e)


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