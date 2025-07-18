"""
Local LLM implementation supporting multiple backends.
Supports Ollama, Hugging Face Transformers, and OpenAI-compatible APIs.
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import httpx
import os

from ..utils.logger import get_logger
from ..utils.config_loader import ConfigLoader

logger = get_logger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM backends."""
    backend: str  # 'ollama', 'huggingface', 'openai_compatible'
    model_name: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    max_tokens: int = 2048
    temperature: float = 0.7
    timeout: int = 60


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> str:
        """Generate text from prompt."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the backend is healthy."""
        pass


class OllamaBackend(LLMBackend):
    """Ollama backend for local LLM inference."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.base_url = config.base_url or "http://localhost:11434"
        self.model_name = config.model_name
        self.client = httpx.AsyncClient(timeout=config.timeout)
    
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> str:
        """Generate text using Ollama API."""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    **kwargs
                }
            }
            
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "")
            
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check Ollama health."""
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except:
            return False
    
    async def close(self):
        """Close the client."""
        await self.client.aclose()


class HuggingFaceBackend(LLMBackend):
    """Hugging Face Transformers backend."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.model_name = config.model_name
        self._model = None
        self._tokenizer = None
        self._device = None
    
    async def _initialize_model(self):
        """Initialize model and tokenizer."""
        if self._model is not None:
            return
        
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            # Determine device
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self._device}")
            
            # Load tokenizer and model
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
                device_map="auto" if self._device == "cuda" else None
            )
            
            # Add padding token if needed
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            
            logger.info(f"Loaded model: {self.model_name}")
            
        except ImportError:
            raise ImportError("transformers and torch are required for HuggingFace backend")
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace model: {e}")
            raise
    
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> str:
        """Generate text using HuggingFace transformers."""
        await self._initialize_model()
        
        try:
            import torch
            
            # Tokenize input
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096
            ).to(self._device)
            
            # Generate
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self._tokenizer.pad_token_id,
                    **kwargs
                )
            
            # Decode response
            generated_text = self._tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"HuggingFace generation error: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check if model is loaded."""
        try:
            await self._initialize_model()
            return self._model is not None
        except:
            return False


class OpenAICompatibleBackend(LLMBackend):
    """OpenAI-compatible API backend."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.base_url = config.base_url or "https://api.openai.com/v1"
        self.api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = config.model_name
        self.client = httpx.AsyncClient(timeout=config.timeout)
    
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> str:
        """Generate text using OpenAI-compatible API."""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                **kwargs
            }
            
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            logger.error(f"OpenAI-compatible API error: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check API health."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}"
            }
            response = await self.client.get(
                f"{self.base_url}/models",
                headers=headers
            )
            return response.status_code == 200
        except:
            return False
    
    async def close(self):
        """Close the client."""
        await self.client.aclose()


class LocalLLM:
    """Main LLM interface supporting multiple backends."""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        if config is None:
            config_loader = ConfigLoader()
            llm_config = config_loader.get("llm", {})
            config = LLMConfig(**llm_config)
        
        self.config = config
        self.backend = self._create_backend(config)
    
    def _create_backend(self, config: LLMConfig) -> LLMBackend:
        """Create appropriate backend based on configuration."""
        if config.backend == "ollama":
            return OllamaBackend(config)
        elif config.backend == "huggingface":
            return HuggingFaceBackend(config)
        elif config.backend == "openai_compatible":
            return OpenAICompatibleBackend(config)
        else:
            raise ValueError(f"Unknown backend: {config.backend}")
    
    async def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate text using the configured backend."""
        temperature = temperature or self.config.temperature
        max_tokens = max_tokens or self.config.max_tokens
        
        return await self.backend.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    
    async def health_check(self) -> bool:
        """Check if the LLM backend is healthy."""
        return await self.backend.health_check()
    
    async def close(self):
        """Close the backend connection."""
        if hasattr(self.backend, 'close'):
            await self.backend.close()
    
    def get_config(self) -> LLMConfig:
        """Get current configuration."""
        return self.config
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about the current backend."""
        return {
            "backend": self.config.backend,
            "model": self.config.model_name,
            "base_url": self.config.base_url,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        }


# Factory function for easy creation
def create_llm(
    backend: str = "ollama",
    model_name: str = "llama2",
    **kwargs
) -> LocalLLM:
    """Create a LocalLLM instance with specified configuration."""
    config = LLMConfig(
        backend=backend,
        model_name=model_name,
        **kwargs
    )
    return LocalLLM(config)