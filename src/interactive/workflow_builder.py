from typing import Dict, Any, Optional, List, Tuple, Callable, Union
from dataclasses import dataclass, field
import time
import copy
import asyncio

from .workflow_graph import WorkflowGraph, VALID_OPERATORS
from .workflow_env import InteractiveWorkflowEnv, StepResult, create_env
from .action_parser import ActionParser, ParsedAction, ActionType


@dataclass
class TurnRecord:
    round_idx: int
    model_response: str
    model_token_ids: List[int] = field(default_factory=list)
    feedback: str = ""
    feedback_token_ids: List[int] = field(default_factory=list)
    action: Optional[Dict[str, Any]] = None
    success: bool = False
    dsl_snapshot: str = ""
    execution_result: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class Trajectory:
    """

    - 
    - Tokenaction_mask
    - 
    """
    problem: str
    problem_type: str = "math"

    turns: List[TurnRecord] = field(default_factory=list)

    all_token_ids: List[int] = field(default_factory=list)
    action_mask: List[int] = field(default_factory=list)

    final_dsl: str = ""
    final_answer: Optional[str] = None
    is_correct: bool = False

    reward: float = 0.0
    reward_breakdown: Dict[str, float] = field(default_factory=dict)

    total_rounds: int = 0
    total_model_tokens: int = 0
    total_feedback_tokens: int = 0
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0

    def add_turn(self, turn: TurnRecord):
        """"""
        self.turns.append(turn)
        self.total_rounds = len(self.turns)

    def finalize(self, final_dsl: str, final_answer: Optional[str] = None):
        """"""
        self.final_dsl = final_dsl
        self.final_answer = final_answer
        self.end_time = time.time()

        self.total_model_tokens = sum(len(t.model_token_ids) for t in self.turns)
        self.total_feedback_tokens = sum(len(t.feedback_token_ids) for t in self.turns)

    def get_duration(self) -> float:
        """()"""
        if self.end_time > 0:
            return self.end_time - self.start_time
        return time.time() - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        """"""
        return {
            "problem": self.problem,
            "problem_type": self.problem_type,
            "turns": [
                {
                    "round": t.round_idx,
                    "model_response": t.model_response,
                    "feedback": t.feedback,
                    "action": t.action,
                    "success": t.success,
                    "dsl": t.dsl_snapshot,
                }
                for t in self.turns
            ],
            "final_dsl": self.final_dsl,
            "final_answer": self.final_answer,
            "is_correct": self.is_correct,
            "reward": self.reward,
            "reward_breakdown": self.reward_breakdown,
            "total_rounds": self.total_rounds,
            "total_model_tokens": self.total_model_tokens,
            "total_feedback_tokens": self.total_feedback_tokens,
            "duration": self.get_duration(),
        }


# ============================================
# ============================================

def create_action_mask(
    model_token_ids: List[int],
    feedback_token_ids: List[int]
) -> Tuple[List[int], List[int]]:
    """ action_mask

    token: mask = 1
    token: mask = 0

    Args:
        model_token_ids: token ID
        feedback_token_ids: token ID

    Returns:
        (combined_ids, action_mask)
    """
    combined_ids = model_token_ids + feedback_token_ids
    action_mask = [1] * len(model_token_ids) + [0] * len(feedback_token_ids)
    return combined_ids, action_mask


def merge_trajectory_masks(turns: List[TurnRecord]) -> Tuple[List[int], List[int]]:
    """tokenmask

    Args:
        turns: 

    Returns:
        (all_token_ids, all_action_masks)
    """
    all_ids = []
    all_masks = []

    for turn in turns:
        all_ids.extend(turn.model_token_ids)
        all_masks.extend([1] * len(turn.model_token_ids))

        all_ids.extend(turn.feedback_token_ids)
        all_masks.extend([0] * len(turn.feedback_token_ids))

    return all_ids, all_masks


# ============================================
# ============================================

class InteractiveWorkflowBuilder:
    """

     Prompt-R1  ToolGenerationManager
    1. 
    2. Tokenaction_mask
    3. 

    :
    ```python
    builder = InteractiveWorkflowBuilder(
        generate_fn=my_model.generate,
        tokenizer=my_tokenizer,
        executor=my_executor,
    )
    trajectory = builder.run_loop(
        problem="What is 2+2?",
        max_rounds=10,
    )
    ```
    """

    def __init__(
        self,
        generate_fn: Optional[Callable[[str], str]] = None,
        tokenizer: Optional[Any] = None,
        executor: Optional[Callable[[str, str], Tuple[str, bool]]] = None,
        max_rounds: int = 20,
        execute_each_step: bool = True,
        verbose: bool = False,
    ):
        """

        Args:
            generate_fn:  prompt -> response
            tokenizer: tokenizer (encode/decode)
            executor: workflow (dsl, problem) -> (result, success)
            max_rounds: 
            execute_each_step: workflow
            verbose: 
        """
        self.generate_fn = generate_fn
        self.tokenizer = tokenizer
        self.executor = executor
        self.max_rounds = max_rounds
        self.execute_each_step = execute_each_step
        self.verbose = verbose

        self.parser = ActionParser(strict=False)

    def run_loop(
        self,
        problem: str,
        problem_type: str = "math",
        system_prompt: Optional[str] = None,
        max_rounds: Optional[int] = None,
        **kwargs
    ) -> Trajectory:
        """

        Args:
            problem: 
            problem_type: 
            system_prompt:  ()
            max_rounds:  ()
            **kwargs: executor

        Returns:
            Trajectory 
        """
        env = InteractiveWorkflowEnv(
            problem=problem,
            problem_type=problem_type,
            executor=self.executor,
            executor_kwargs=kwargs or None,
            max_rounds=max_rounds or self.max_rounds,
            execute_each_step=self.execute_each_step,
            use_two_step_interaction=kwargs.get("use_two_step_interaction", True) if kwargs else True,
            dataset=kwargs.get("dataset", "") if kwargs else "",
            simple_feedback=kwargs.get("simple_feedback", True) if kwargs else True,
        )

        trajectory = Trajectory(
            problem=problem,
            problem_type=problem_type,
        )

        current_prompt = system_prompt or env.system_prompt

        round_idx = 0
        active = True

        while active and round_idx < (max_rounds or self.max_rounds):
            round_idx += 1

            if self.verbose:
                print(f"\n{'='*50}")
                print(f"Round {round_idx}")
                print(f"{'='*50}")

            if self.generate_fn is None:
                raise ValueError("generate_fn is required for run_loop")

            model_response = self.generate_fn(current_prompt)

            if self.verbose:
                print(f"Model: {model_response[:200]}...")

            model_token_ids = []
            if self.tokenizer:
                model_token_ids = self.tokenizer.encode(
                    model_response,
                    add_special_tokens=False
                )

            feedback, success_list, active = env.step(model_response)

            if self.verbose:
                print(f"Feedback: {feedback[:200]}...")
                print(f"Active: {active}")

            # 4. Tokenize feedback
            feedback_token_ids = []
            if self.tokenizer:
                feedback_token_ids = self.tokenizer.encode(
                    feedback,
                    add_special_tokens=False
                )

            action = self.parser.parse(model_response)

            turn = TurnRecord(
                round_idx=round_idx,
                model_response=model_response,
                model_token_ids=model_token_ids,
                feedback=feedback,
                feedback_token_ids=feedback_token_ids,
                action=action.to_dict() if action else None,
                success=success_list[0] if success_list else False,
                dsl_snapshot=env.get_dsl(),
                execution_result=env.last_execution_result,
            )
            trajectory.add_turn(turn)

            current_prompt = current_prompt + "\n" + model_response + "\n" + feedback

            if not active:
                if self.verbose:
                    print(f"\n[Loop ended] Reason: active=False")
                break

        trajectory.all_token_ids, trajectory.action_mask = merge_trajectory_masks(
            trajectory.turns
        )

        trajectory.finalize(
            final_dsl=env.get_dsl(),
            final_answer=env.last_execution_result,
        )

        if self.verbose:
            print(f"\n{'='*50}")
            print(f"Trajectory completed:")
            print(f"  Rounds: {trajectory.total_rounds}")
            print(f"  Final DSL: {trajectory.final_dsl}")
            print(f"  Model tokens: {trajectory.total_model_tokens}")
            print(f"  Feedback tokens: {trajectory.total_feedback_tokens}")
            print(f"  Duration: {trajectory.get_duration():.2f}s")
            print(f"{'='*50}")

        return trajectory

    async def run_loop_async(
        self,
        problem: str,
        problem_type: str = "math",
        system_prompt: Optional[str] = None,
        max_rounds: Optional[int] = None,
        **kwargs
    ) -> Trajectory:
        """run_loop (async executor)"""
        env = InteractiveWorkflowEnv(
            problem=problem,
            problem_type=problem_type,
            executor=None,
            max_rounds=max_rounds or self.max_rounds,
            execute_each_step=False,
            use_two_step_interaction=kwargs.get("use_two_step_interaction", True) if kwargs else True,
            dataset=kwargs.get("dataset", "") if kwargs else "",
            simple_feedback=kwargs.get("simple_feedback", True) if kwargs else True,
        )

        trajectory = Trajectory(
            problem=problem,
            problem_type=problem_type,
        )

        current_prompt = system_prompt or env.system_prompt
        round_idx = 0
        active = True

        while active and round_idx < (max_rounds or self.max_rounds):
            round_idx += 1

            if self.generate_fn is None:
                raise ValueError("generate_fn is required")
            model_response = self.generate_fn(current_prompt)

            # 2. Tokenize
            model_token_ids = []
            if self.tokenizer:
                model_token_ids = self.tokenizer.encode(
                    model_response,
                    add_special_tokens=False
                )

            feedback, success_list, active = env.step(model_response)

            exec_result = None
            if self.execute_each_step and self.executor and not env.graph.is_empty():
                dsl = env.get_dsl()
                try:
                    if asyncio.iscoroutinefunction(self.executor):
                        exec_result, _ = await self.executor(dsl, problem)
                    else:
                        exec_result, _ = self.executor(dsl, problem)
                    env.last_execution_result = exec_result
                except Exception as e:
                    exec_result = f"Execution error: {str(e)}"
                    env.last_execution_result = exec_result

            # 5. Tokenize feedback
            feedback_token_ids = []
            if self.tokenizer:
                feedback_token_ids = self.tokenizer.encode(
                    feedback,
                    add_special_tokens=False
                )

            action = self.parser.parse(model_response)

            turn = TurnRecord(
                round_idx=round_idx,
                model_response=model_response,
                model_token_ids=model_token_ids,
                feedback=feedback,
                feedback_token_ids=feedback_token_ids,
                action=action.to_dict() if action else None,
                success=success_list[0] if success_list else False,
                dsl_snapshot=env.get_dsl(),
                execution_result=exec_result,
            )
            trajectory.add_turn(turn)

            current_prompt = current_prompt + "\n" + model_response + "\n" + feedback

            if not active:
                break

        trajectory.all_token_ids, trajectory.action_mask = merge_trajectory_masks(
            trajectory.turns
        )

        trajectory.finalize(
            final_dsl=env.get_dsl(),
            final_answer=env.last_execution_result,
        )

        return trajectory


# ============================================
# ============================================

class BatchWorkflowBuilder:
    """

    rollout
    """

    def __init__(
        self,
        generate_fn: Optional[Callable[[List[str]], List[str]]] = None,
        tokenizer: Optional[Any] = None,
        executor: Optional[Callable[[str, str], Tuple[str, bool]]] = None,
        max_rounds: int = 20,
        execute_each_step: bool = True,
    ):
        """

        Args:
            generate_fn:  List[prompt] -> List[response]
            tokenizer: tokenizer
            executor: workflow
            max_rounds: 
            execute_each_step: 
        """
        self.generate_fn = generate_fn
        self.tokenizer = tokenizer
        self.executor = executor
        self.max_rounds = max_rounds
        self.execute_each_step = execute_each_step
        self.parser = ActionParser(strict=False)

    def run_batch_loop(
        self,
        problems: List[str],
        problem_types: Optional[List[str]] = None,
        system_prompts: Optional[List[str]] = None,
        use_two_step_interaction: bool = True,
        datasets: Optional[List[str]] = None,
        simple_feedback: bool = True,
    ) -> List[Trajectory]:
        """

        Args:
            problems: 
            problem_types: 
            system_prompts: 
            use_two_step_interaction: False
            datasets: 
            simple_feedback: feedbackTrue

        Returns:
        """
        batch_size = len(problems)
        problem_types = problem_types or ["math"] * batch_size
        datasets = datasets or [""] * batch_size

        envs = [
            InteractiveWorkflowEnv(
                problem=problems[i],
                problem_type=problem_types[i],
                executor=self.executor,
                max_rounds=self.max_rounds,
                execute_each_step=self.execute_each_step,
                use_two_step_interaction=use_two_step_interaction,
                dataset=datasets[i] if datasets else "",
                simple_feedback=simple_feedback,
            )
            for i in range(batch_size)
        ]

        trajectories = [
            Trajectory(problem=problems[i], problem_type=problem_types[i])
            for i in range(batch_size)
        ]

        if system_prompts:
            current_prompts = list(system_prompts)
        else:
            current_prompts = [env.system_prompt for env in envs]

        active_mask = [True] * batch_size

        for round_idx in range(1, self.max_rounds + 1):
            if not any(active_mask):
                break

            active_indices = [i for i, a in enumerate(active_mask) if a]
            active_prompts = [current_prompts[i] for i in active_indices]

            if self.generate_fn is None:
                raise ValueError("generate_fn is required")
            active_responses = self.generate_fn(active_prompts)

            for local_idx, global_idx in enumerate(active_indices):
                model_response = active_responses[local_idx]
                env = envs[global_idx]
                trajectory = trajectories[global_idx]

                # Tokenize
                model_token_ids = []
                if self.tokenizer:
                    model_token_ids = self.tokenizer.encode(
                        model_response,
                        add_special_tokens=False
                    )

                feedback, success_list, active = env.step(model_response)

                # Tokenize feedback
                feedback_token_ids = []
                if self.tokenizer:
                    feedback_token_ids = self.tokenizer.encode(
                        feedback,
                        add_special_tokens=False
                    )

                action = self.parser.parse(model_response)

                turn = TurnRecord(
                    round_idx=round_idx,
                    model_response=model_response,
                    model_token_ids=model_token_ids,
                    feedback=feedback,
                    feedback_token_ids=feedback_token_ids,
                    action=action.to_dict() if action else None,
                    success=success_list[0] if success_list else False,
                    dsl_snapshot=env.get_dsl(),
                    execution_result=env.last_execution_result,
                )
                trajectory.add_turn(turn)

                current_prompts[global_idx] = (
                    current_prompts[global_idx] + "\n" +
                    model_response + "\n" + feedback
                )

                active_mask[global_idx] = active

        for i, trajectory in enumerate(trajectories):
            trajectory.all_token_ids, trajectory.action_mask = merge_trajectory_masks(
                trajectory.turns
            )
            trajectory.finalize(
                final_dsl=envs[i].get_dsl(),
                final_answer=envs[i].last_execution_result,
            )

        return trajectories


# ============================================
# ============================================

def create_aflow_executor_wrapper(
    aflow_executor,
    problem_type: str = "math"
) -> Callable[[str, str], Tuple[str, bool]]:
    """ AFlowExecutor 

     AFlowExecutor.execute_workflow()  (dsl, problem, prompts) -> (result, success) 

    🔧  custom_prompts 

    Args:
        aflow_executor: AFlowExecutor 
        problem_type: 

    Returns:
        executor
    """
    try:
        from ..vllm_workflow_generator import WorkflowCodeGenerator
    except ImportError:
        from src.vllm_workflow_generator import WorkflowCodeGenerator

    code_generator = WorkflowCodeGenerator(problem_type=problem_type)
    # Node-level incremental cache: {node_id: (input_hash, raw_result)}
    # Lives for the lifetime of this executor wrapper (one env/problem).
    node_cache: Dict[str, Any] = {}

    def executor_fn(dsl: str, problem: str, prompts: dict = None, **exec_kwargs) -> Tuple[str, bool]:
        """DSL

        Args:
            dsl: DSL 
            problem: 
            prompts:  custom_prompts {node_id: {"operator": str, "prompt": str}}
            **exec_kwargs:  code  entry_point/test
        """
        try:
            # Prune caches for nodes removed from the current graph.
            # (Downstream invalidation is handled by input_hash mismatches and per-node overwrites.)
            if prompts:
                keep_ids = set(prompts.keys())
                for cached_id in list(node_cache.keys()):
                    if str(cached_id).startswith("stage_"):
                        continue
                    if cached_id not in keep_ids:
                        node_cache.pop(cached_id, None)

            if prompts:
                workflow_code = code_generator._generate_workflow_code_with_custom_prompts(dsl, prompts)
                is_valid = True
                gen_error = None
                print(f"  🔧  custom_prompts  ({len(prompts)} prompts)")
            else:
                workflow_code, is_valid, gen_error = code_generator.generate(dsl)

            if not is_valid:
                return f"Code generation error: {gen_error}", False

            aflow_kwargs = {}
            if exec_kwargs:
                if exec_kwargs.get("entry_point") is not None:
                    aflow_kwargs["entry_point"] = exec_kwargs.get("entry_point")
                if exec_kwargs.get("test") is not None:
                    aflow_kwargs["test"] = exec_kwargs.get("test")
            # Incremental execution cache (per node_id)
            aflow_kwargs["node_cache"] = node_cache

            try:
                loop = asyncio.get_running_loop()
                loop_running = True
            except RuntimeError:
                loop_running = False

            if loop_running:
                import concurrent.futures
                exec_timeout = int(getattr(aflow_executor, "timeout", 300) or 300)
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(
                        asyncio.run,
                        aflow_executor.execute_workflow(
                            workflow_code, problem, problem_type, **aflow_kwargs
                        )
                    )
                    # Align thread waiting timeout with executor timeout (plus small overhead margin).
                    result, cost, metadata = future.result(timeout=exec_timeout + 15)
            else:
                result, cost, metadata = asyncio.run(
                    aflow_executor.execute_workflow(
                        workflow_code, problem, problem_type, **aflow_kwargs
                    )
                )

            if metadata and metadata.get('programmer_code'):
                if isinstance(result, dict):
                    result['code'] = metadata['programmer_code']
                else:
                    result = {'output': str(result), 'code': metadata['programmer_code']}
                print(f"  📦 CodeCache: metadataprogrammer_code")
                return result, True
            elif isinstance(result, dict):
                return result, True
            elif isinstance(result, str):
                return result, True
            elif 'is_correct_DISABLED' in str(result):
                if 'is_correct' in result:
                    if 'answer' in result and result['answer']:
                        answer = str(result['answer'])
                        print(f"  ✅ OperatorOutput:  Verify  answer: {answer[:50]}...")
                    else:
                        answer = f"[VERIFY_OUTPUT] is_correct={result.get('is_correct')}"
                        print(f"  ⚠️ OperatorOutput: Verify  answer ")
                elif 'review_result' in result:
                    import re
                    feedback = result.get('feedback', '')
                    answer_patterns = [
                        r'(?:the\s+)?(?:correct\s+)?answer\s+(?:is|should\s+be|=)\s*[:\s]*([^\.,\n]+)',
                        r'expected\s+(?:output|result|answer)\s*[:\s]*([^\.,\n]+)',
                    ]
                    extracted = None
                    for pattern in answer_patterns:
                        match = re.search(pattern, str(feedback), re.IGNORECASE)
                        if match:
                            extracted = match.group(1).strip()
                            break
                    if extracted:
                        answer = extracted
                        print(f"  ✅ OperatorOutput:  Review feedback : {answer[:50]}...")
                    else:
                        answer = f"[REVIEW_OUTPUT] result={result.get('review_result')}"
                        print(f"  ⚠️ OperatorOutput: Review ")
                elif 'result' in result and 'solution' in result:
                    if result.get('result') == True:
                        answer = f"[TEST_PASSED] solution available"
                    else:
                        answer = f"[TEST_FAILED] {str(result)[:100]}"
                    print(f"  ⚠️ OperatorOutput: Test operator : {answer[:50]}...")
                elif 'output' in result and result['output']:
                    answer = str(result['output'])
                elif 'response' in result and result['response']:
                    answer = str(result['response'])
                elif 'answer' in result:
                    answer = str(result['answer'])
                elif 'solution' in result:
                    answer = str(result['solution'])
                else:
                    answer = str(result)
            else:
                answer = str(result) if result else ""

            return answer, True

        except Exception as e:
            return f"Execution error: {str(e)}", False

    return executor_fn


# ============================================
# ============================================

def create_builder(
    generate_fn: Optional[Callable] = None,
    tokenizer: Optional[Any] = None,
    executor: Optional[Callable] = None,
    aflow_executor: Optional[Any] = None,
    problem_type: str = "math",
    max_rounds: int = 20,
    execute_each_step: bool = True,
    verbose: bool = False,
) -> InteractiveWorkflowBuilder:
    """

    Args:
        generate_fn: 
        tokenizer: tokenizer
        executor:  ()
        aflow_executor: AFlowExecutor ()
        problem_type: 
        max_rounds: 
        execute_each_step: 
        verbose: 

    Returns:
        InteractiveWorkflowBuilder 
    """
    if aflow_executor is not None and executor is None:
        executor = create_aflow_executor_wrapper(aflow_executor, problem_type)

    return InteractiveWorkflowBuilder(
        generate_fn=generate_fn,
        tokenizer=tokenizer,
        executor=executor,
        max_rounds=max_rounds,
        execute_each_step=execute_each_step,
        verbose=verbose,
    )


# ============================================
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("InteractiveWorkflowBuilder ")
    print("=" * 60)

    responses = [
        "<action>add</action><operator>Plan</operator>",
        "<action>add</action><operator>Programmer</operator>",
        "<action>add</action><structure>parallel</structure><operators>Custom, Custom</operators>",
        "<action>add</action><structure>conditional</structure><condition>Review</condition><true>Revise</true><false>done</false>",
        "<action>finish</action>",
    ]
    response_idx = [0]

    def mock_generate(prompt: str) -> str:
        idx = response_idx[0]
        response_idx[0] += 1
        if idx < len(responses):
            return responses[idx]
        return "<action>finish</action>"

    class MockTokenizer:
        def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
            return list(range(len(text)))

        def decode(self, ids: List[int]) -> str:
            return f"[{len(ids)} tokens]"

    builder = InteractiveWorkflowBuilder(
        generate_fn=mock_generate,
        tokenizer=MockTokenizer(),
        executor=None,
        max_rounds=10,
        execute_each_step=False,
        verbose=True,
    )

    trajectory = builder.run_loop(
        problem="What is 15 * 23?",
        problem_type="math",
    )

    print("\n" + "=" * 60)
    print(":")
    print("=" * 60)
    print(f": {trajectory.problem}")
    print(f"DSL: {trajectory.final_dsl}")
    print(f": {trajectory.total_rounds}")
    print(f"tokens: {trajectory.total_model_tokens}")
    print(f"tokens: {trajectory.total_feedback_tokens}")
    print(f"Action mask: {len(trajectory.action_mask)}")
    print(f"  tokens (mask=1): {sum(trajectory.action_mask)}")
    print(f"  tokens (mask=0): {len(trajectory.action_mask) - sum(trajectory.action_mask)}")

    print("\n:")
    for turn in trajectory.turns:
        print(f"  Round {turn.round_idx}:")
        print(f"    : {turn.action}")
        print(f"    : {turn.success}")
        print(f"    DSL: {turn.dsl_snapshot}")

    print("\n" + "=" * 60)
    print("!")
