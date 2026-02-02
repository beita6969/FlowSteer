import os
import gc
import torch
import asyncio
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
import time
import logging

logger = logging.getLogger("InteractiveBatchInference")


# ============================================
# ============================================

@dataclass
class BatchInferenceConfig:
    """"""
    max_batch_size: int = 4
    max_concurrent_api: int = 48

    clear_cache_every_n_batches: int = 1
    force_gc: bool = True

    generation_timeout: float = 120.0
    execution_timeout: float = 300.0

    max_retries: int = 2
    retry_delay: float = 1.0


# ============================================
# ============================================

class BatchGenerationManager:
    """

     Prompt-R1 :
    1. Super-batch: 
    2. : CUDA
    3. : Semaphore
    """

    def __init__(
        self,
        generate_fn: Callable[[str], str],
        tokenizer: Any = None,
        config: Optional[BatchInferenceConfig] = None,
        use_vllm_api: bool = False,
    ):
        """
        Args:
            generate_fn:  (prompt -> response)
            tokenizer: Tokenizer (token)
            config: 
            use_vllm_api: vLLM API
        """
        self.generate_fn = generate_fn
        self.tokenizer = tokenizer
        self.config = config or BatchInferenceConfig()
        self.use_vllm_api = use_vllm_api

        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_api)
        self._generation_lock = asyncio.Lock()

        self.total_generations = 0
        self.total_tokens = 0
        self.batch_count = 0

    async def generate_single(self, prompt: str) -> str:
        """
         ()

        Args:
            prompt: 

        Returns:
        """
        if self.use_vllm_api:
            async with self._semaphore:
                return await self._generate_with_retry(prompt)
        else:
            async with self._generation_lock:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, self.generate_fn, prompt)

    async def _generate_with_retry(self, prompt: str) -> str:
        """"""
        for attempt in range(self.config.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(self.generate_fn):
                    return await self.generate_fn(prompt)
                else:
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, self.generate_fn, prompt)
            except Exception as e:
                if attempt < self.config.max_retries:
                    logger.warning(f"Generation failed (attempt {attempt+1}): {e}, retrying...")
                    await asyncio.sleep(self.config.retry_delay)
                else:
                    logger.error(f"Generation failed after {self.config.max_retries+1} attempts: {e}")
                    raise

    async def generate_batch(self, prompts: List[str]) -> List[str]:
        """
         (Super-batch )

        Args:
            prompts: 

        Returns:
        """
        if not prompts:
            return []

        results = []
        batch_size = self.config.max_batch_size

        for batch_start in range(0, len(prompts), batch_size):
            batch_end = min(batch_start + batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]

            if self.batch_count > 0 and self.batch_count % self.config.clear_cache_every_n_batches == 0:
                self._clear_memory()

            tasks = [self.generate_single(p) for p in batch_prompts]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Batch generation failed for item {batch_start + i}: {result}")
                    results.append("")
                else:
                    results.append(result)

            self.batch_count += 1
            self.total_generations += len(batch_prompts)

        return results

    def _clear_memory(self):
        """ (Prompt-R1)"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if self.config.force_gc:
            gc.collect()

        logger.debug("Memory cleared")

    def get_stats(self) -> Dict[str, Any]:
        """"""
        return {
            "total_generations": self.total_generations,
            "total_tokens": self.total_tokens,
            "batch_count": self.batch_count,
        }


# ============================================
# ============================================

class BatchInteractiveLoopManager:
    """

    :
    1. : 
    2. : 
    3. : 
    """

    def __init__(
        self,
        batch_generator: BatchGenerationManager,
        env_factory: Callable,
        max_rounds: int = 100,
        verbose: bool = False,
    ):
        """
        Args:
            batch_generator: 
            env_factory:  (problem -> env)
            max_rounds: 
            verbose: 
        """
        self.batch_generator = batch_generator
        self.env_factory = env_factory
        self.max_rounds = max_rounds
        self.verbose = verbose

    async def run_batch_loops(
        self,
        problems: List[Dict],
    ) -> List[Dict]:
        """

        Args:
            problems:  [{problem, problem_type, ground_truth}, ...]

        Returns:
             [{trajectory, final_dsl, total_rounds, ...}, ...]
        """
        n = len(problems)

        envs = []
        for prob in problems:
            env = self.env_factory(
                problem=prob['problem'],
                max_rounds=self.max_rounds,
            )
            envs.append(env)

        active = [True] * n
        prompts_history = [[] for _ in range(n)]
        responses_history = [[] for _ in range(n)]

        for round_idx in range(self.max_rounds):
            current_prompts = []
            prompt_indices = []

            for i in range(n):
                if active[i]:
                    prompt = envs[i].get_current_prompt()
                    current_prompts.append(prompt)
                    prompt_indices.append(i)

            if not current_prompts:
                break

            if self.verbose:
                logger.info(f"Round {round_idx + 1}: {len(current_prompts)} active problems")

            responses = await self.batch_generator.generate_batch(current_prompts)

            for j, (idx, response) in enumerate(zip(prompt_indices, responses)):
                prompts_history[idx].append(current_prompts[j])
                responses_history[idx].append(response)

                feedback, success, still_active = envs[idx].step(response)

                active[idx] = still_active

        results = []
        for i in range(n):
            result = {
                'problem': problems[i]['problem'],
                'problem_type': problems[i].get('problem_type', 'math'),
                'final_dsl': envs[i].get_dsl(),
                'total_rounds': envs[i].round_count,
                'prompts_history': prompts_history[i],
                'responses_history': responses_history[i],
                'env': envs[i],
            }
            results.append(result)

        return results


# ============================================
# ============================================

class OptimizedAsyncLLMClient:
    """
    LLM

     Prompt-R1  optimized_async_llm.py:
    1. HTTP
    2. Keep-alive
    3. HTTP/2
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        max_connections: int = 50,
        timeout: float = 300.0,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.max_connections = max_connections
        self.timeout = timeout
        self._client = None

    async def _get_client(self):
        """HTTP"""
        if self._client is None:
            try:
                import httpx

                limits = httpx.Limits(
                    max_connections=self.max_connections,
                    max_keepalive_connections=self.max_connections,
                    keepalive_expiry=30.0,
                )

                timeout = httpx.Timeout(
                    connect=10.0,
                    read=self.timeout,
                    write=30.0,
                    pool=10.0,
                )

                self._client = httpx.AsyncClient(
                    limits=limits,
                    timeout=timeout,
                    http2=True,  # HTTP/2 for better concurrency
                )
            except ImportError:
                logger.warning("httpx not installed, using aiohttp fallback")
                self._client = "fallback"

        return self._client

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        """

        Args:
            prompt: 
            temperature: 
            max_tokens: token

        Returns:
        """
        client = await self._get_client()

        if client == "fallback":
            from openai import AsyncOpenAI
            aclient = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key or "dummy")
            response = await aclient.chat.completions.create(
                model="default",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        data = {
            "model": "default",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        response = await client.post(
            f"{self.base_url}/chat/completions",
            json=data,
            headers=headers,
        )
        response.raise_for_status()

        result = response.json()
        return result["choices"][0]["message"]["content"]

    async def batch_generate(
        self,
        prompts: List[str],
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> List[str]:
        """

        Args:
            prompts: 
            temperature: 
            max_tokens: token

        Returns:
        """
        tasks = [
            self.generate(p, temperature, max_tokens)
            for p in prompts
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed = []
        for r in results:
            if isinstance(r, Exception):
                logger.error(f"Batch generation error: {r}")
                processed.append("")
            else:
                processed.append(r)

        return processed

    async def close(self):
        """"""
        if self._client and self._client != "fallback":
            await self._client.aclose()
            self._client = None


# ============================================
# ============================================

def create_batch_generator(
    generate_fn: Callable,
    tokenizer: Any = None,
    max_batch_size: int = 4,
    use_vllm_api: bool = False,
) -> BatchGenerationManager:
    """

    Args:
        generate_fn: 
        tokenizer: Tokenizer
        max_batch_size: 
        use_vllm_api: vLLM API

    Returns:
        BatchGenerationManager
    """
    config = BatchInferenceConfig(
        max_batch_size=max_batch_size,
    )

    return BatchGenerationManager(
        generate_fn=generate_fn,
        tokenizer=tokenizer,
        config=config,
        use_vllm_api=use_vllm_api,
    )


def create_optimized_client(
    base_url: str,
    api_key: str = "",
    max_connections: int = 50,
) -> OptimizedAsyncLLMClient:
    """
    LLM

    Args:
        base_url: APIURL
        api_key: API
        max_connections: 

    Returns:
        OptimizedAsyncLLMClient
    """
    return OptimizedAsyncLLMClient(
        base_url=base_url,
        api_key=api_key,
        max_connections=max_connections,
    )
