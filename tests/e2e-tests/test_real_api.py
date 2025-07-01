#!/usr/bin/env python3
"""
Real API Integration Tests

These tests make actual API calls to verify the system works end-to-end.
Run only when you want to test with real providers and have proper API keys.

Usage:
    python tests/e2e-tests/test_real_api.py
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models import LLMRequest
from src.llm.router import LLMRouter
from src.agent.orchestrator import InsuranceAgent
from config.settings import settings


class RealAPITester:
    """Real API test runner."""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
    
    async def test_external_provider_real_call(self):
        """Test external provider with real API call."""
        print("🌐 Testing External Provider (Real API Call)")
        
        if not settings.ext_provider_api_key:
            print("⏭️  Skipped - No API key configured")
            self.skipped += 1
            return
        
        # Configure for external provider
        original_use_hf = settings.use_hf_local
        settings.use_hf_local = False
        
        try:
            router = LLMRouter()
            
            request = LLMRequest(
                prompt="Say 'API test successful' and nothing else",
                max_tokens=10,
                temperature=0.1
            )
            
            response = await router.route_request(request)
            
            print(f"✅ Provider: {response.provider_name}")
            print(f"✅ Model: {response.model}")
            print(f"✅ Response: {response.content}")
            print(f"✅ Latency: {response.latency_ms:.2f}ms")
            print(f"✅ Tokens: {response.tokens_used}")
            
            assert response.content is not None
            assert len(response.content.strip()) > 0
            assert response.latency_ms > 0
            
            self.passed += 1
            
        except Exception as e:
            print(f"❌ External provider test failed: {e}")
            self.failed += 1
        finally:
            settings.use_hf_local = original_use_hf
    
    async def test_local_provider_real_call(self):
        """Test local provider with real model loading."""
        print("🏠 Testing Local Provider (Real Model)")
        
        # Check if we have local model capabilities
        try:
            import transformers
        except ImportError:
            print("⏭️  Skipped - Transformers not available")
            self.skipped += 1
            return
        
        # Configure for local provider
        original_use_hf = settings.use_hf_local
        settings.use_hf_local = True
        
        try:
            router = LLMRouter()
            
            request = LLMRequest(
                prompt="Hello",
                max_tokens=5,
                temperature=0.1
            )
            
            print("📥 Loading local model (this may take time)...")
            response = await router.route_request(request)
            
            print(f"✅ Provider: {response.provider_name}")
            print(f"✅ Model: {response.model}")
            print(f"✅ Response: {response.content}")
            print(f"✅ Latency: {response.latency_ms:.2f}ms")
            print(f"✅ Tokens: {response.tokens_used}")
            
            assert response.content is not None
            assert response.provider_name == "huggingface"
            assert response.latency_ms > 0
            
            self.passed += 1
            
        except Exception as e:
            print(f"❌ Local provider test failed: {e}")
            print("   This might be due to insufficient memory or missing model files")
            self.failed += 1
        finally:
            settings.use_hf_local = original_use_hf
    
    async def test_conversation_flow_real(self):
        """Test real conversation flow."""
        print("💬 Testing Real Conversation Flow")
        
        if not settings.ext_provider_api_key:
            print("⏭️  Skipped - No API key configured for conversation test")
            self.skipped += 1
            return
        
        # Use external provider for more reliable conversation
        original_use_hf = settings.use_hf_local
        settings.use_hf_local = False
        
        try:
            orchestrator = InsuranceAgent()
            
            # Start conversation
            response1 = await orchestrator.process_message(
                "Hello, I need car insurance"
            )
            print(f"✅ Initial response: {response1.message[:100]}...")
            assert response1.status in ["collecting_data", "clarification_needed"]
            
            # Provide some information
            response2 = await orchestrator.process_message(
                "My name is Test User"
            )
            print(f"✅ After name: {response2.message[:100]}...")
            assert response2.status == "collecting_data"
            
            # Check that we're collecting data
            assert orchestrator.session_data is not None
            if "customer_name" in orchestrator.session_data:
                print(f"✅ Extracted name: {orchestrator.session_data['customer_name']}")
            
            self.passed += 1
            
        except Exception as e:
            print(f"❌ Conversation flow test failed: {e}")
            self.failed += 1
        finally:
            settings.use_hf_local = original_use_hf
    
    async def test_provider_fallback_real(self):
        """Test provider fallback mechanism."""
        print("🔄 Testing Provider Fallback")
        
        if not settings.ext_provider_api_key:
            print("⏭️  Skipped - No API key configured for fallback test")
            self.skipped += 1
            return
        
        # Configure for external provider with fallback
        original_use_hf = settings.use_hf_local
        settings.use_hf_local = False
        
        try:
            router = LLMRouter()
            
            # Test that the provider has fallback configured
            provider = router.provider
            assert hasattr(provider, 'chain')
            
            # Make a request that should work
            request = LLMRequest(
                prompt="Test fallback",
                max_tokens=5,
                temperature=0.1
            )
            
            response = await router.route_request(request)
            print(f"✅ Fallback test response: {response.content}")
            
            assert response.content is not None
            self.passed += 1
            
        except Exception as e:
            print(f"❌ Provider fallback test failed: {e}")
            self.failed += 1
        finally:
            settings.use_hf_local = original_use_hf
    
    async def test_health_checks_real(self):
        """Test health checks with real providers."""
        print("🏥 Testing Health Checks")
        
        try:
            router = LLMRouter()
            health = await router.health_check()
            
            print(f"✅ Health check result: {health}")
            assert isinstance(health, dict)
            assert router.provider_type in health
            
            # Health should be boolean
            health_status = health[router.provider_type]
            assert isinstance(health_status, bool)
            
            if health_status:
                print("✅ Provider is healthy")
            else:
                print("⚠️  Provider reported unhealthy (this might be expected)")
            
            self.passed += 1
            
        except Exception as e:
            print(f"❌ Health check test failed: {e}")
            self.failed += 1
    
    async def run_all(self):
        """Run all real API tests."""
        print("🌐 Running Real API Integration Tests")
        print("=" * 60)
        print("⚠️  These tests make real API calls and may take time")
        print("=" * 60)
        
        # Run tests
        await self.test_external_provider_real_call()
        await self.test_local_provider_real_call()
        await self.test_conversation_flow_real()
        await self.test_provider_fallback_real()
        await self.test_health_checks_real()
        
        # Report results
        print("\n" + "=" * 60)
        print(f"📊 Real API Test Results:")
        print(f"   ✅ Passed: {self.passed}")
        print(f"   ❌ Failed: {self.failed}")
        print(f"   ⏭️  Skipped: {self.skipped}")
        
        total_run = self.passed + self.failed
        if total_run > 0:
            success_rate = (self.passed / total_run) * 100
            print(f"   📈 Success Rate: {success_rate:.1f}%")
        
        return self.failed == 0


async def main():
    """Run real API tests."""
    print("🚀 Starting Real API Tests")
    print("Make sure you have:")
    print("1. Valid API keys configured in .env")
    print("2. Internet connection for external APIs")
    print("3. Sufficient memory for local models")
    print()
    
    tester = RealAPITester()
    success = await tester.run_all()
    
    if success:
        print("\n🎉 All real API tests completed successfully!")
    else:
        print("\n💥 Some real API tests failed - check configuration")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)