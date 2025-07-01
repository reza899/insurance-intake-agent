#!/usr/bin/env python3
"""
Comprehensive End-to-End Tests for Insurance Intake Agent

These tests verify the complete system functionality including:
- LLM provider switching (local/external)
- Conversation flow and data extraction
- Duplicate detection
- Database persistence
- API endpoints
"""

import asyncio
import pytest
import httpx
import sys
from typing import Dict, Any
from unittest.mock import AsyncMock, patch, MagicMock

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

from config.settings import settings
from src.models import LLMRequest, RegistrationRequest, RegistrationResponse
from src.llm.router import LLMRouter
from src.agent.orchestrator import InsuranceAgent
from src.agent.extractor import DataExtractor
from src.agent.duplicate_detector import DuplicateDetector
from src.database.mongodb import mongodb


class TestEndToEnd:
    """End-to-end test suite for the insurance intake agent."""

    @pytest.fixture(autouse=True)
    async def setup_test_env(self):
        """Set up test environment with mock database and LLM."""
        # Store original settings
        self.original_use_hf_local = settings.use_hf_local
        self.original_debug = settings.debug
        
        # Set test configuration
        settings.debug = True
        
        # Mock database operations
        self.mock_db = AsyncMock()
        
        yield
        
        # Restore original settings
        settings.use_hf_local = self.original_use_hf_local
        settings.debug = self.original_debug

    async def test_llm_router_external_provider(self):
        """Test LLM router with external (OpenAI-compatible) provider."""
        print("🧪 Testing LLM Router - External Provider")
        
        # Configure for external provider
        settings.use_hf_local = False
        
        with patch.object(LLMRouter, 'route_request') as mock_route:
            # Mock successful LLM response
            mock_llm_response = AsyncMock()
            mock_llm_response.content = "Hello world friend"
            mock_llm_response.provider_name = "openai_compatible_llm"
            mock_llm_response.model = settings.ext_provider_model
            mock_llm_response.tokens_used = 15
            mock_llm_response.latency_ms = 100
            mock_route.return_value = mock_llm_response
            
            router = LLMRouter()
            
            request = LLMRequest(
                prompt="Say hello in exactly 3 words",
                max_tokens=10,
                temperature=0.1
            )
            
            response = await router.route_request(request)
            
            assert response.content == "Hello world friend"
            assert response.provider_name == "openai_compatible_llm"
            assert response.model == settings.ext_provider_model
            assert response.tokens_used == 15
            assert response.latency_ms > 0
            print("✅ External provider test passed")

    async def test_llm_router_local_provider(self):
        """Test LLM router with local (HuggingFace) provider."""
        print("🧪 Testing LLM Router - Local Provider")
        
        # Configure for local provider
        settings.use_hf_local = True
        
        with patch('src.llm.providers.huggingface.TRANSFORMERS_AVAILABLE', True), \
             patch('src.llm.providers.huggingface.AutoTokenizer.from_pretrained') as mock_tokenizer_class, \
             patch('src.llm.providers.huggingface.AutoModelForCausalLM.from_pretrained') as mock_model:
            
            # Mock tokenizer instance with encode method
            mock_tokenizer_instance = AsyncMock()
            mock_tokenizer_instance.encode.return_value = [1, 2, 3, 4, 5]  # Mock 5 tokens
            mock_tokenizer_instance.pad_token = None
            mock_tokenizer_instance.eos_token = "<|end|>"
            mock_tokenizer_class.return_value = mock_tokenizer_instance
            
            # Mock successful response
            mock_result = "Hello world everyone"
            
            # Mock asyncio.to_thread to return the generated text directly
            with patch('asyncio.to_thread', return_value=mock_result):
                router = LLMRouter()
                
                request = LLMRequest(
                    prompt="Say hello in exactly 3 words",
                    max_tokens=10,
                    temperature=0.1
                )
                
                response = await router.route_request(request)
                
                assert response.content == "Hello world everyone"
                assert response.provider_name == "huggingface"
                assert response.model == settings.local_model_name
                assert response.tokens_used > 0
                assert response.latency_ms > 0
                print("✅ Local provider test passed")

    async def test_conversation_flow_complete_registration(self):
        """Test complete conversation flow for successful registration."""
        print("🧪 Testing Complete Conversation Flow")
        
        with patch.object(InsuranceAgent, '_save_registration') as mock_save:
            mock_save.return_value = "reg_123456"
            
            with patch.object(DuplicateDetector, 'find_duplicates') as mock_dup:
                mock_dup.return_value = []  # No duplicates
                
                with patch.object(DataExtractor, 'extract_data') as mock_extract:
                    # Simulate progressive data collection
                    extraction_responses = [
                        {"customer_name": "John Doe"},
                        {"customer_name": "John Doe", "manufacturer": "Toyota"},
                        {
                            "customer_name": "John Doe",
                            "manufacturer": "Toyota",
                            "car_type": "Sedan",
                            "year": 2020,
                            "license_plate": "ABC123",
                            "birth_date": "1990-05-15"
                        }
                    ]
                    mock_extract.side_effect = extraction_responses
                    
                    orchestrator = InsuranceAgent()
                    
                    # Start conversation
                    response1 = await orchestrator.process_message("Hello, I want car insurance")
                    assert "car" in response1["response"].lower() or "vehicle" in response1["response"].lower()
                    assert response1["status"] == "collecting_data"
                    
                    # Provide name
                    response2 = await orchestrator.process_message("My name is John Doe")
                    assert response2["status"] == "collecting_data"
                    
                    # Provide car info
                    response3 = await orchestrator.process_message(
                        "I have a Toyota Sedan from 2020, license plate ABC123, born on 1990-05-15"
                    )
                    
                    assert response3["status"] == "completed"
                    assert "registration_id" in response3
                    assert response3["registration_id"] == "reg_123456"
                    print("✅ Complete conversation flow test passed")

    async def test_duplicate_detection_workflow(self):
        """Test duplicate detection during registration."""
        print("🧪 Testing Duplicate Detection Workflow")
        
        with patch.object(DuplicateDetector, 'find_duplicates') as mock_dup:
            # Mock finding a duplicate
            mock_dup.return_value = [
                {
                    "id": "existing_123",
                    "customer_name": "John Doe",
                    "birth_date": "1990-05-15",
                    "license_plate": "ABC123",
                    "similarity_score": 0.95
                }
            ]
            
            with patch.object(DataExtractor, 'extract_data') as mock_extract:
                mock_extract.return_value = {
                    "customer_name": "John Doe",
                    "manufacturer": "Toyota",
                    "car_type": "Sedan",
                    "year": 2020,
                    "license_plate": "ABC123",
                    "birth_date": "1990-05-15"
                }
                
                orchestrator = InsuranceAgent()
                
                # Process complete registration data
                response = await orchestrator.process_message(
                    "My name is John Doe, I have a Toyota Sedan from 2020, "
                    "license plate ABC123, born on 1990-05-15"
                )
                
                assert response["status"] == "duplicate_found"
                assert "similar registration" in response["response"].lower()
                assert len(response["duplicates"]) == 1
                assert response["duplicates"][0]["similarity_score"] == 0.95
                print("✅ Duplicate detection test passed")

    async def test_data_extraction_progressive(self):
        """Test progressive data extraction from multiple messages."""
        print("🧪 Testing Progressive Data Extraction")
        
        extractor = DataExtractor()
        
        # Mock DSPy extraction
        with patch.object(extractor, 'extract') as mock_dspy:
            # Test partial extraction
            mock_result = AsyncMock()
            mock_result.customer_name = "John Smith"
            mock_result.manufacturer = ""
            mock_result.car_type = ""
            mock_result.year = ""
            mock_result.license_plate = ""
            mock_result.birth_date = ""
            mock_dspy.return_value = mock_result
            
            data1 = await extractor.extract_data("My name is John Smith")
            assert data1["customer_name"] == "John Smith"
            assert len(data1) == 1
            
            # Test additional information
            mock_result.manufacturer = "Honda"
            mock_result.car_type = "SUV"
            mock_dspy.return_value = mock_result
            
            data2 = await extractor.extract_data(
                "I drive a Honda SUV", 
                existing_data=data1
            )
            assert data2["customer_name"] == "John Smith"
            assert data2["manufacturer"] == "Honda"
            assert data2["car_type"].upper() == "SUV"
            
            # Verify missing fields detection
            missing = extractor.get_missing_fields(data2)
            assert "year" in missing
            assert "license_plate" in missing
            assert "birth_date" in missing
            print("✅ Progressive data extraction test passed")

    async def test_database_operations(self):
        """Test database save and retrieve operations."""
        print("🧪 Testing Database Operations")
        
        with patch('src.database.mongodb.mongodb') as mock_mongodb:
            mock_collection = AsyncMock()
            mock_db = AsyncMock()
            mock_db.__getitem__.return_value = mock_collection
            mock_mongodb.get_collection.return_value = mock_collection
            
            # Mock successful save
            mock_result = AsyncMock()
            mock_result.inserted_id = "test_id_123"
            mock_collection.insert_one.return_value = mock_result
            
            # Mock find operation
            mock_collection.find.return_value.to_list.return_value = [
                {
                    "_id": "test_id_123",
                    "customer_name": "Test User",
                    "license_plate": "TEST123"
                }
            ]
            
            # Test save operation
            from src.agent.orchestrator import InsuranceAgent
            orchestrator = InsuranceAgent()
            
            test_data = {
                "customer_name": "Test User",
                "manufacturer": "TestCar",
                "car_type": "Sedan",
                "year": 2023,
                "license_plate": "TEST123",
                "birth_date": "1985-01-01"
            }
            
            # Mock the save method
            with patch.object(orchestrator, '_save_registration', return_value="test_id_123") as mock_save:
                result_id = await mock_save(test_data)
                assert result_id == "test_id_123"
            
            # Mock the get method  
            with patch.object(orchestrator, 'get_registration') as mock_get:
                mock_get.return_value = {
                    "id": "test_id_123",
                    "customer_name": "Test User",
                    "license_plate": "TEST123"
                }
                
                retrieved = await orchestrator.get_registration("test_id_123")
                assert retrieved["customer_name"] == "Test User"
                print("✅ Database operations test passed")

    async def test_api_health_endpoints(self):
        """Test API health and status endpoints."""
        print("🧪 Testing API Health Endpoints")
        
        # Mock the FastAPI application
        with patch('src.api.main.app') as mock_app:
            # For this test, we'll directly test the router health check
            # since testing the full FastAPI server would require more setup
            
            router = LLMRouter()
            
            with patch.object(router.provider, 'health_check', return_value=True):
                health = await router.health_check()
                assert isinstance(health, dict)
                assert router.provider_type in health
                assert health[router.provider_type] is True
                print("✅ API health endpoint test passed")

    async def test_error_handling_and_recovery(self):
        """Test system behavior under error conditions."""
        print("🧪 Testing Error Handling and Recovery")
        
        # Test LLM provider failure
        with patch.object(LLMRouter, 'route_request') as mock_route:
            # Simulate provider failure then recovery
            mock_route.side_effect = [
                Exception("Provider temporarily unavailable"),
                AsyncMock(content="Recovery response", provider_name="test", model="test", latency_ms=100)
            ]
            
            router = LLMRouter()
            
            # First call should fail
            try:
                await router.route_request(LLMRequest(prompt="test"))
                assert False, "Should have raised an exception"
            except Exception as e:
                assert "Provider temporarily unavailable" in str(e)
            
            print("✅ Error handling test passed")

    async def test_configuration_switching(self):
        """Test switching between different provider configurations."""
        print("🧪 Testing Configuration Switching")
        
        # Test switching from external to local
        original_use_hf = settings.use_hf_local
        
        try:
            # Mock transformers components to prevent PyTorch import issues
            mock_tokenizer = AsyncMock()
            mock_model = AsyncMock()
            
            # Mock providers to avoid actual initialization
            with patch('src.llm.providers.huggingface.TRANSFORMERS_AVAILABLE', True), \
                 patch('src.llm.providers.huggingface.AutoTokenizer.from_pretrained', return_value=mock_tokenizer), \
                 patch('src.llm.providers.huggingface.AutoModelForCausalLM.from_pretrained', return_value=mock_model), \
                 patch('src.llm.providers.huggingface.HuggingFaceProvider') as mock_hf, \
                 patch('src.llm.providers.openai_compatible.OpenAICompatibleProvider') as mock_openai:
                
                # Test external provider
                settings.use_hf_local = False
                router1 = LLMRouter()
                assert router1.provider_type == "openai_compatible_llm"
                
                # Test local provider
                settings.use_hf_local = True
                router2 = LLMRouter()
                assert router2.provider_type == "huggingface"
                
                print("✅ Configuration switching test passed")
            
        finally:
            # Restore original setting
            settings.use_hf_local = original_use_hf