#!/usr/bin/env python3
"""
AFlowasync_llm
AFlow/scripts/async_llm.py
Formattercall_with_format()
"""
import os
import asyncio
import yaml
import hashlib
import threading
from collections import OrderedDict
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from dataclasses import dataclass, field
from openai import AsyncOpenAI
import httpx

if TYPE_CHECKING:
    from scripts.formatter import BaseFormatter


# ============================================
# Global LLM Response Cache (optional)
# ============================================

_GLOBAL_LLM_CACHE: "OrderedDict[str, str]" = OrderedDict()
_GLOBAL_LLM_CACHE_LOCK = threading.RLock()
_GLOBAL_LLM_CACHE_STATS: Optional[Dict[str, int]] = None


@dataclass
class LLMConfig:
    """LLM"""
    api_type: str = "openai"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 4096
    timeout: float = 300.0

    def __post_init__(self):
        if self.api_key == "${OPENAI_API_KEY}" or not self.api_key:
            self.api_key = os.environ.get('OPENAI_API_KEY', 'sk-dummy')


@dataclass
class LLMsConfig:
    """LLM"""
    models: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'LLMsConfig':
        """YAML"""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        models = {}
        for name, config in data.get('models', {}).items():
            api_key = config.get('api_key', '')
            if api_key == "${OPENAI_API_KEY}":
                api_key = os.environ.get('OPENAI_API_KEY', '')
            config['api_key'] = api_key


            if config.get('base_urls'):
                models[name] = config
            else:
                models[name] = LLMConfig(
                    api_type=config.get('api_type', 'openai'),
                    base_url=config.get('base_url', 'https://api.openai.com/v1'),
                    api_key=api_key,
                    model_name=config.get('model_name', name),
                    temperature=config.get('temperature', 0.0),
                    top_p=config.get('top_p', 1.0),
                    max_tokens=config.get('max_tokens', 4096)
                )

        return cls(models=models)


class AsyncLLM:
    """LLM - AFlow"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 4096,
        timeout: float = 300.0,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

        proxy_url = os.environ.get('ALL_PROXY') or os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
        is_external_api = 'api.openai.com' in base_url or 'openai' in base_url.lower()

        if proxy_url and is_external_api:
            http_client = httpx.AsyncClient(
                proxy=proxy_url,
                timeout=float(timeout or 300.0),
            )
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=float(timeout or 300.0),
                http_client=http_client,
            )
        else:
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=float(timeout or 300.0),
            )

        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0

        # Global LRU cache (optional; speeds up repeated AFlow operator calls)
        self._cache_enabled = os.environ.get("AFLOW_LLM_CACHE", "0").strip() not in ("", "0", "false", "False")
        self._cache_max_items = int(os.environ.get("AFLOW_LLM_CACHE_MAX_ITEMS", "2048"))

        global _GLOBAL_LLM_CACHE_STATS
        if self._cache_enabled and _GLOBAL_LLM_CACHE_STATS is None:
            _GLOBAL_LLM_CACHE_STATS = {"hits": 0, "misses": 0, "sets": 0}

    def _make_cache_key(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> str:
        h = hashlib.sha256()
        h.update(str(self.base_url).encode("utf-8", "ignore"))
        h.update(b"\0")
        h.update(str(self.model).encode("utf-8", "ignore"))
        h.update(b"\0")
        h.update(str(system_prompt or "").encode("utf-8", "ignore"))
        h.update(b"\0")
        h.update(str(temperature).encode("utf-8", "ignore"))
        h.update(b"\0")
        h.update(str(self.top_p).encode("utf-8", "ignore"))
        h.update(b"\0")
        h.update(str(max_tokens).encode("utf-8", "ignore"))
        h.update(b"\0")
        h.update(str(prompt).encode("utf-8", "ignore"))
        return h.hexdigest()

    async def __call__(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """LLM"""
        temp = temperature if temperature is not None else self.temperature
        max_toks = max_tokens or self.max_tokens

        if self._cache_enabled:
            key = self._make_cache_key(prompt, system_prompt, temp, max_toks)
            with _GLOBAL_LLM_CACHE_LOCK:
                cached = _GLOBAL_LLM_CACHE.get(key)
                if cached is not None:
                    _GLOBAL_LLM_CACHE.move_to_end(key)
                    if _GLOBAL_LLM_CACHE_STATS is not None:
                        _GLOBAL_LLM_CACHE_STATS["hits"] += 1
                    return cached
                if _GLOBAL_LLM_CACHE_STATS is not None:
                    _GLOBAL_LLM_CACHE_STATS["misses"] += 1

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temp,
                top_p=self.top_p,
                max_tokens=max_toks
            )

            if response.usage:
                self.total_input_tokens += response.usage.prompt_tokens
                self.total_output_tokens += response.usage.completion_tokens
            self.total_calls += 1

            content = response.choices[0].message.content or ""

            if self._cache_enabled:
                with _GLOBAL_LLM_CACHE_LOCK:
                    _GLOBAL_LLM_CACHE[key] = content
                    _GLOBAL_LLM_CACHE.move_to_end(key)
                    if _GLOBAL_LLM_CACHE_STATS is not None:
                        _GLOBAL_LLM_CACHE_STATS["sets"] += 1
                    while len(_GLOBAL_LLM_CACHE) > max(1, self._cache_max_items):
                        _GLOBAL_LLM_CACHE.popitem(last=False)

            return content
        except Exception as e:
            print(f"LLM: {e}")
            raise

    async def batch_call(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        max_concurrent: int = 10
    ) -> List[tuple]:
        """"""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def call_with_semaphore(prompt: str):
            async with semaphore:
                try:
                    result = await self(prompt, system_prompt)
                    return (True, result)
                except Exception as e:
                    return (False, str(e))

        tasks = [call_with_semaphore(p) for p in prompts]
        return await asyncio.gather(*tasks)

    def get_usage_summary(self) -> Dict[str, Any]:
        """"""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_calls": self.total_calls,
            "total_cost": 0.0
        }

    async def call_with_format(
        self,
        prompt: str,
        formatter: "BaseFormatter",
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        FormatterLLM - AFlow

        Args:
            prompt: 
            formatter: FormatterXmlFormatter/CodeFormatter/TextFormatter
            system_prompt: 
            temperature: 
            max_tokens: token

        Returns:

        Raises:
            FormatError: 
        """
        from scripts.formatter import FormatError

        formatted_prompt = formatter.prepare_prompt(prompt)

        response = await self(
            formatted_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )

        is_valid, parsed = formatter.validate_response(response)

        if not is_valid:
            raise FormatError(formatter.format_error_message())

        return parsed


def create_llm_instance(
    config: Dict[str, Any],
    **kwargs
) -> AsyncLLM:
    """
    LLM - AFlow

    Args:
        config: LLMLLMConfig

    Returns:
        AsyncLLM
    """
    if isinstance(config, LLMConfig):
        return AsyncLLM(
            api_key=config.api_key,
            base_url=config.base_url,
            model=config.model_name,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            timeout=getattr(config, "timeout", 300.0),
        )

    if isinstance(config, str):
        api_key = os.environ.get('OPENAI_API_KEY', 'sk-dummy')
        model_mapping = {
            'gpt-oss-120b': 'gpt-4o-mini',
            'gpt-4o': 'gpt-4o',
            'gpt-4o-mini': 'gpt-4o-mini',
            'gpt-3.5-turbo': 'gpt-3.5-turbo',
        }
        model_name = model_mapping.get(config, 'gpt-4o-mini')
        return AsyncLLM(
            api_key=api_key,
            base_url='https://api.openai.com/v1',
            model=model_name,
            temperature=0.0,
            top_p=1.0,
            max_tokens=4096,
            timeout=float(os.environ.get("AFLOW_LLM_TIMEOUT", "300") or 300),
        )

    api_key = config.get('api_key', '')
    if api_key == "${OPENAI_API_KEY}" or not api_key:
        api_key = os.environ.get('OPENAI_API_KEY', 'sk-dummy')


    base_urls = config.get('base_urls', None)
    if base_urls:
        print(f"  🔄 : {len(base_urls)}")
        from src.optimized_async_llm import create_optimized_llm_instance
        return create_optimized_llm_instance(config)

    timeout = config.get("timeout", None)
    if timeout is None:
        timeout = os.environ.get("AFLOW_LLM_TIMEOUT", None)

    return AsyncLLM(
        api_key=api_key,
        base_url=config.get('base_url', 'https://api.openai.com/v1'),
        model=config.get('model_name', config.get('model', 'gpt-4o-mini')),
        temperature=config.get('temperature', 0.0),
        top_p=config.get('top_p', 1.0),
        max_tokens=config.get('max_tokens', 4096),
        timeout=float(timeout or 300.0),
    )


if __name__ == "__main__":
    async def test():
        config = {
            "api_key": os.environ.get('OPENAI_API_KEY'),
            "base_url": "https://api.openai.com/v1",
            "model_name": "gpt-4o-mini"
        }
        llm = create_llm_instance(config)
        result = await llm("Say hello!")
        print(f"Result: {result}")

    asyncio.run(test())
