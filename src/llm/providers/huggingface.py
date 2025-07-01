import asyncio
import time
from typing import Dict, Union

from config.settings import settings
from src.models import LLMError, LLMRequest, LLMResponse

from .base import BaseLLMProvider

# Defer transformers import to avoid PyTorch dependency issues in testing
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    # Set to None so we can check and handle gracefully
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    TRANSFORMERS_AVAILABLE = False


class HuggingFaceProvider(BaseLLMProvider):
    """Provider for local HuggingFace models."""

    def __init__(self) -> None:
        """Initialize HuggingFace provider from environment settings."""
        config = {"name": "huggingface"}
        super().__init__(config)

        self.provider_name = "huggingface"
        self.model_name = settings.local_model_name
        self.device = settings.local_model_device

        # Initialize model and tokenizer
        try:
            # Check if transformers is available
            if not TRANSFORMERS_AVAILABLE:
                raise LLMError(
                    provider=self.provider_name,
                    message="Transformers library not available. Install with: pip install transformers torch",
                    retryable=False,
                )

            # Load tokenizer and model separately for better control
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Set device mapping
            device_map: Union[str, Dict[str, str], None] = None
            if self.device == "cuda":
                device_map = "auto"  # Automatically distribute across GPUs
            elif self.device == "mps":
                device_map = {"": "mps"}  # Apple Silicon GPU
            else:
                device_map = {"": "cpu"}  # CPU only

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=device_map,
                torch_dtype="auto",  # Use optimal precision
                trust_remote_code=True,  # Allow custom model code
            )

            # Ensure tokenizer has pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        except Exception as e:
            raise LLMError(
                provider=self.provider_name,
                message=f"Failed to load model {self.model_name}: {str(e)}",
                retryable=False,
            )

    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        """Generate response using local HuggingFace model."""
        start_time = time.time()

        try:
            # Build the prompt with context if provided
            full_prompt = self._format_prompt(request.prompt, request.context)

            # Set generation parameters
            max_tokens = request.max_tokens or 512
            temperature = request.temperature or 0.1

            # Generate response using direct model inference
            generated_text = await asyncio.to_thread(self._generate_with_model, full_prompt, max_tokens, temperature)

            latency_ms = (time.time() - start_time) * 1000

            # Count tokens properly using the model's tokenizer
            tokens_used = self._count_tokens(full_prompt + generated_text)

            return LLMResponse(
                content=generated_text,
                provider_name=self.provider_name,
                model=self.model_name,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
            )

        except Exception as e:
            raise LLMError(
                provider=self.provider_name, message=f"Failed to generate response: {str(e)}", retryable=True
            )

    async def health_check(self) -> bool:
        """Check if the provider is healthy."""
        try:
            # Simple test request
            test_request = LLMRequest(prompt="Hello", context=None, max_tokens=10, temperature=0.1)
            await self.generate_response(test_request)
            return True
        except Exception:
            return False

    def _generate_with_model(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate text using the model directly."""
        import torch

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")

        # Move to model device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate with model
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )

        # Decode only the new tokens (exclude input)
        generated_ids = outputs[0][inputs["input_ids"].shape[-1] :]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return generated_text.strip()

    def _count_tokens(self, text: str) -> int:
        """Count tokens using the model's tokenizer."""
        try:
            if self.tokenizer:
                tokens = self.tokenizer.encode(text)
                return len(tokens)
            else:
                # Fallback to word count if tokenizer not available
                return len(text.split())
        except Exception:
            # Fallback to word count on any error
            return len(text.split())

    def _format_prompt(self, prompt: str, context: str = None) -> str:
        """Format prompt based on model type (currently supports Phi-3, easily extensible)."""
        # Check if model is Phi-3 based (default format)
        if "phi" in self.model_name.lower():
            if context:
                return f"<|system|>\n{context}<|end|>\n<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
            else:
                return f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"

        # Add support for other model formats as needed
        # elif "llama" in self.model_name.lower():
        #     return f"[INST] {context + ' ' if context else ''}{prompt} [/INST]"

        # Default format for unknown models
        if context:
            return f"System: {context}\nUser: {prompt}\nAssistant:"
        else:
            return f"User: {prompt}\nAssistant:"
