#!/usr/bin/env python3
"""
LLM - Plan A + Plan B 

Plan A: httpx
- httpx
- max_connections50
- 

Plan B: API
- prompt
- asyncio.gather
"""
import asyncio
from typing import List, Optional, Dict, Any, Tuple
from openai import AsyncOpenAI



class OptimizedAsyncLLM:
    """
    LLM

    :
    - Plan A: httpxHTTP
    - Plan B: APIprompt
    - MultiURL: 
    - AsyncLLM
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 4096,
        max_connections: int = 50,
        max_concurrent: int = 20,
        system_msg: Optional[str] = None,
        base_urls: List[str] = None,
        timeout: float = 300.0,
    ):
        """
        LLM

        Args:
            api_key: API
            base_url: APIURL
            model: 
            temperature: 
            top_p: Top-p
            max_connections: HTTP (Plan A)
            max_concurrent:  (Plan B)
            system_msg: 
            base_urls: MultiURL URL
        """
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = int(max_tokens or 4096)
        self.system_msg = system_msg


        self.base_urls = base_urls or [base_url]
        self._url_index = 0
        self.clients = []
        for url in self.base_urls:
            self.clients.append(AsyncOpenAI(
                api_key=api_key,
                base_url=url,
                timeout=float(timeout or 300.0),
                max_retries=2,
            ))
        self.client = self.clients[0]

        self._max_concurrent = max_concurrent
        self._semaphore = None

        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0

    @property
    def semaphore(self) -> asyncio.Semaphore:
        """MultiURL: """
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._max_concurrent)
        return self._semaphore

    def _get_client(self):
        """MultiURL: """
        client = self.clients[self._url_index % len(self.clients)]
        self._url_index += 1
        return client

    async def __call__(self, prompt: str) -> str:
        """
        prompt

        Args:
            prompt: 

        Returns:
            LLM
        """
        async with self.semaphore:
            messages = []
            if self.system_msg:
                messages.append({"role": "system", "content": self.system_msg})
            messages.append({"role": "user", "content": prompt})

            client = self._get_client()
            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
            )

            if response.usage:
                self.total_input_tokens += response.usage.prompt_tokens
                self.total_output_tokens += response.usage.completion_tokens
            self.total_calls += 1

            return response.choices[0].message.content or ""

    async def aask(self, msg: str, system_msgs: list = None) -> str:
        """MetaGPTaask"""
        original_sys_msg = self.system_msg
        if system_msgs:
            self.system_msg = system_msgs[0] if isinstance(system_msgs, list) else system_msgs

        try:
            return await self.__call__(msg)
        finally:
            self.system_msg = original_sys_msg

    async def call_with_format(
        self,
        prompt: str,
        formatter,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        """
        from scripts.formatter import FormatError

        formatted_prompt = formatter.prepare_prompt(prompt)

        async with self.semaphore:
            messages = []
            if system_prompt or self.system_msg:
                messages.append({"role": "system", "content": system_prompt or self.system_msg})
            messages.append({"role": "user", "content": formatted_prompt})

            client = self._get_client()
            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                top_p=self.top_p,
                max_tokens=int(max_tokens or self.max_tokens),
            )

            if response.usage:
                self.total_input_tokens += response.usage.prompt_tokens
                self.total_output_tokens += response.usage.completion_tokens
            self.total_calls += 1

            response_text = response.choices[0].message.content or ""

        is_valid, parsed = formatter.validate_response(response_text)

        if not is_valid:
            raise FormatError(formatter.format_error_message())

        return parsed

    async def batch_call(
        self,
        prompts: List[str],
        return_exceptions: bool = True
    ) -> List[Tuple[bool, Any]]:
        """
        Plan B: prompt

        asyncio.gather

        Args:
            prompts: prompt
            return_exceptions: 

        Returns:
            List of (success: bool, result_or_error: Any)
        """
        async def safe_call(prompt: str) -> Tuple[bool, Any]:
            try:
                result = await self.__call__(prompt)
                return (True, result)
            except Exception as e:
                if return_exceptions:
                    return (False, str(e))
                raise

        results = await asyncio.gather(
            *[safe_call(p) for p in prompts],
            return_exceptions=return_exceptions
        )

        processed = []
        for r in results:
            if isinstance(r, Exception):
                processed.append((False, str(r)))
            else:
                processed.append(r)

        return processed

    async def batch_call_with_messages(
        self,
        messages_list: List[List[Dict[str, str]]],
        return_exceptions: bool = True
    ) -> List[Tuple[bool, Any]]:
        """
        Plan B: 

        Args:
            messages_list: 
            return_exceptions: 

        Returns:
            List of (success, result_or_error)
        """
        async def call_with_messages(messages: List[Dict[str, str]]) -> Tuple[bool, Any]:
            try:
                async with self.semaphore:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        max_tokens=self.max_tokens,
                    )

                    if response.usage:
                        self.total_input_tokens += response.usage.prompt_tokens
                        self.total_output_tokens += response.usage.completion_tokens
                    self.total_calls += 1

                    return (True, response.choices[0].message.content or "")
            except Exception as e:
                if return_exceptions:
                    return (False, str(e))
                raise

        results = await asyncio.gather(
            *[call_with_messages(m) for m in messages_list],
            return_exceptions=return_exceptions
        )

        processed = []
        for r in results:
            if isinstance(r, Exception):
                processed.append((False, str(r)))
            else:
                processed.append(r)

        return processed

    def get_usage_summary(self) -> Dict[str, Any]:
        """"""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_calls": self.total_calls,
            "total_cost": 0.0
        }


def create_optimized_llm_instance(
    config: Dict[str, Any],
    max_connections: int = 50,
    max_concurrent: int = 20
) -> OptimizedAsyncLLM:
    """
    LLM

    Args:
        config: LLM:
            - api_key: API
            - base_url: APIURL
            - base_urls: MultiURL URL
            - model_name: 
            - temperature:  ()
            - top_p: Top-p ()
        max_connections: Plan A - HTTP
        max_concurrent: Plan B - 

    Returns:
        OptimizedAsyncLLM
    """
    base_urls = config.get("base_urls", None)
    if base_urls:
        print(f"  🔄 MultiURL: : {len(base_urls)}")

    return OptimizedAsyncLLM(
        api_key=config.get("api_key", "dummy"),
        base_url=config.get("base_url", "http://localhost:8002/v1"),
        model=config.get("model_name", config.get("model", "gpt-oss-120b")),
        temperature=config.get("temperature", 0.7),
        top_p=config.get("top_p", 1.0),
        max_tokens=int(config.get("max_tokens", 4096) or 4096),
        max_connections=max_connections,
        base_urls=base_urls,
        max_concurrent=max_concurrent,
        timeout=float(config.get("timeout", 300.0) or 300.0),
    )


async def cleanup_global_resources():
    """- MultiURL"""
    pass


# ============================================================
# ============================================================

async def test_optimized_llm():
    """LLM"""
    print("\n" + "=" * 60)
    print("🧪 LLM")
    print("=" * 60)

    config = {
        "api_key": "dummy",
        "base_url": "http://localhost:8002/v1",
        "model_name": "/path/to/executor-model",
        "temperature": 0.7,
    }

    llm = create_optimized_llm_instance(
        config,
        max_connections=50,
        max_concurrent=20
    )

    print("\n📝 ...")
    try:
        result = await llm("What is 2 + 2?")
        print(f"  : {result[:100]}...")
    except Exception as e:
        print(f"  ❌ : {e}")

    print("\n📝  (3prompt)...")
    prompts = [
        "What is 1 + 1?",
        "What is the capital of France?",
        "Write a haiku about coding."
    ]

    import time
    start = time.time()
    try:
        results = await llm.batch_call(prompts)
        elapsed = time.time() - start

        print(f"  : {elapsed:.2f}")
        for i, (success, result) in enumerate(results):
            status = "✅" if success else "❌"
            preview = str(result)[:50] if result else "N/A"
            print(f"  {status} Prompt {i+1}: {preview}...")
    except Exception as e:
        print(f"  ❌ : {e}")

    print(f"\n📊 :")
    usage = llm.get_usage_summary()
    print(f"  : {usage['total_calls']}")
    print(f"  Token: {usage['total_tokens']}")

    await cleanup_global_resources()


if __name__ == "__main__":
    asyncio.run(test_optimized_llm())
