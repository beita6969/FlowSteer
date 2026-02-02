# -*- coding: utf-8 -*-
"""
Interactive Workflow Building - Workflow Environment
=====================================================
Workflow interactive environment, similar to Prompt-R1's BaseToolEnv.

Core functions:
- step(raw_response): Execute one action step, return feedback
- stop(raw_response): Determine if should stop
- reset(): Reset environment
- get_system_prompt(): Get system prompt

Two-step interaction (v2):
- After adding operator, force enter AWAITING_PROMPT state
- Model must use set_prompt to set custom prompt for that operator
- This reduces information load for small models, improving effectiveness

Author: Claude Code
Date: 2024-12-09
"""

from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import time

from .workflow_graph import WorkflowGraph, VALID_OPERATORS
from .action_parser import (
    ActionParser, ParsedAction, ActionType, StructureType
)
from .prompt_templates import get_problem_type_hint
from .operator_descriptions import (
    OPERATOR_TEMPLATES, get_operator_template, format_prompt_guidance
)
from .workflow_memory import WorkflowMemory


class EnvState(Enum):
    """Environment state enum (two-step interaction)"""
    BUILDING = "building"          # Normal building state, can add/delete/modify/finish
    AWAITING_PROMPT = "awaiting_prompt"  # Waiting for prompt state, can only set_prompt


@dataclass
class StepResult:
    """Single step execution result"""
    feedback: str                    # Feedback text
    success_list: List[bool]         # Success status for each action
    active: bool                     # Whether to continue (False = terminate)
    action: Optional[ParsedAction] = None  # Parsed action
    execution_result: Optional[str] = None # Workflow execution result
    dsl: str = ""                    # Current DSL
    statistics: Dict[str, Any] = field(default_factory=dict)  # Statistics
    verify_result: Optional[Dict[str, Any]] = None  # Verify operator results (is_correct, confidence, etc.)


class InteractiveWorkflowEnv:
    """Interactive Workflow Building Environment

    Implements interface similar to Prompt-R1 BaseToolEnv:
    - step(raw_response) -> (feedback, success_list, active)
    - stop(raw_response) -> bool

    Usage:
    ```python
    env = InteractiveWorkflowEnv(problem="What is 2+2?")
    while True:
        response = model.generate(prompt + env.get_current_state())
        feedback, success, active = env.step(response)
        if not active:
            break
    final_dsl = env.get_dsl()
    ```
    """

    def __init__(
        self,
        problem: str,
        problem_type: str = "math",
        executor: Optional[Callable[[str, str], Tuple[str, bool]]] = None,
        executor_kwargs: Optional[Dict[str, Any]] = None,
        max_rounds: int = 100,  # Relaxed limit, let model decide
        execute_each_step: bool = True,
        parser: Optional[ActionParser] = None,
        # Finish constraints: to avoid "finish immediately after one step" degenerate strategy
        finish_min_total_operators: int = 1,
        finish_require_checker: bool = False,
        finish_require_structure: bool = False,
        # Ablation: whether to pass small-model custom prompts into executor
        use_custom_prompts_in_execution: bool = True,
    ):
        """Initialize environment

        Args:
            problem: Problem to solve
            executor: Workflow executor function (dsl, problem) -> (result, success)
            executor_kwargs: Extra parameters for executor (e.g., entry_point/test for code)
            max_rounds: Maximum interaction rounds
            execute_each_step: Whether to execute workflow at each step
            parser: Action parser
        """
        self.problem = problem
        self.problem_type = str(problem_type or "math")
        self.executor = executor
        self.executor_kwargs: Dict[str, Any] = dict(executor_kwargs or {})
        self.max_rounds = max_rounds
        self.execute_each_step = execute_each_step
        self.parser = parser or ActionParser(strict=False)
        self.use_custom_prompts_in_execution = bool(use_custom_prompts_in_execution)

        self.disabled_operators: set = set()
        if self._is_multiple_choice_problem(problem) or self.problem_type in ("mathqa_mc", "mc", "multiple_choice"):
            self.disabled_operators.add("Programmer")
            print(f"[DEBUG] MC problem detected, disabled operators: {self.disabled_operators}", flush=True)

        # Finish constraints (optional)
        self.finish_min_total_operators = max(0, int(finish_min_total_operators))
        self.finish_require_checker = bool(finish_require_checker)
        self.finish_require_structure = bool(finish_require_structure)
        self.finish_checker_ops = {"Verify", "Test", "Review"}
        # Code task: if official test provided, force Test operator before finish (avoid only using Review/Verify without actually running tests)
        self.finish_require_test = bool(str(self.executor_kwargs.get("test", "") or "").strip())
        print(f"[DEBUG] finish_require_test={self.finish_require_test}, executor_kwargs keys={list(self.executor_kwargs.keys())}", flush=True)
        # "Solving/producing answer" operators (Plan/Decompose only produce intermediate structure, not counted as solver)
        self.finish_solver_ops = {"Custom", "Programmer", "AnswerGenerate", "Aggregate", "Revise"}

        # State
        self.graph = WorkflowGraph()
        self.round_count = 0
        self.history: List[Dict[str, Any]] = []  # Interaction history
        self.is_finished = False
        self.last_execution_result: Optional[str] = None
        self.last_solver_result: Optional[Any] = None
        self.memory = WorkflowMemory()

        # Two-step interaction state
        self.env_state: EnvState = EnvState.BUILDING  # Current environment state
        self.pending_node_id: Optional[str] = None    # Node ID waiting for prompt
        self.pending_operator: Optional[str] = None   # Operator name waiting for prompt
        self.pending_action: Optional[ParsedAction] = None  # Pending add action (delayed execution)
        # Structure ADD multi-prompt collection (parallel/conditional/loop)
        self.pending_prompt_targets: List[Tuple[str, str]] = []  # [(node_id, operator_name), ...]
        self.pending_prompt_cursor: int = 0
        self.pending_prompt_total: int = 0
        self.pending_prompt_structure: Optional[str] = None  # "parallel" / "conditional" / "loop"
        self.pending_prompt_parent_id: Optional[str] = None  # Top-level structure node id, e.g., node_3
        self.pending_prompt_context: Optional[str] = None  # Context hint for prompt_request
        # Debug/observability: last accepted prompt content (sanitized)
        self.last_prompt_set: Optional[Dict[str, Any]] = None
        self.force_finish_flag: bool = False

    def _check_finish_constraints(self) -> Tuple[bool, str, str]:
        """Check if finish is allowed.

        Returns:
            (ok, message, hint)
        """
        stats = self.graph.get_statistics()
        total_ops = int(stats.get("total_operators", 0) or 0)

        if total_ops <= 0:
            return False, "Finish rejected: workflow is empty.", "Add at least one operator before finishing."

        if total_ops < self.finish_min_total_operators:
            return (
                False,
                f"Finish rejected: workflow is too short ({total_ops} operators).",
                "Add more operators before finishing.",
            )

        if self.finish_require_structure:
            has_any_structure = bool(
                stats.get("has_parallel") or stats.get("has_conditional") or stats.get("has_loop")
            )
            if not has_any_structure:
                return (
                    False,
                    "Finish rejected: no structure used (parallel/conditional/loop).",
                    "Add at least one structure (parallel/conditional/loop) before finishing.",
                )

        if self.finish_require_checker:
            ops = set(stats.get("operator_list", []) or [])
            if not (ops & self.finish_solver_ops):
                return (
                    False,
                    "Finish rejected: no solver operator used (Custom/Programmer/AnswerGenerate/...).",
                    "Add a solver operator (e.g., Programmer/Custom/AnswerGenerate) before finishing.",
                )
            if not (ops & self.finish_checker_ops):
                return (
                    False,
                    "Finish rejected: no checker operator used (Verify/Test/Review).",
                    "Add Verify/Test/Review before finishing.",
                )
            if self.finish_require_test and ("Test" not in ops):
                return (
                    False,
                    "Finish rejected: code task requires Test operator before finishing.",
                    "Add Test to run the provided unit tests before finishing.",
                )

        if self.problem_type == "code":
            ops = set(stats.get("operator_list", []) or [])
            if "Programmer" not in ops and "Test" not in ops:
                return (
                    False,
                    "Finish rejected: code task requires Programmer to generate code.",
                    "Add Programmer operator to generate executable Python code. Plan/Decompose only produce text descriptions, not code.",
                )

        # if stats.get("has_parallel"):
        #     ops = set(stats.get("operator_list", []) or [])
        #     aggregation_ops = {'ScEnsemble', 'Aggregate', 'MdEnsemble'}
        #     if not (ops & aggregation_ops):
        #         return (
        #             False,
        #             "Finish rejected: parallel structure without aggregation.",
        #             "Add ScEnsemble or Aggregate after parallel to merge multiple results into single answer."
        #         )

        if self.last_execution_result:
            if isinstance(self.last_execution_result, dict):
                exec_str = str(self.last_execution_result.get('output', self.last_execution_result))
            else:
                exec_str = str(self.last_execution_result)

            is_long = (
                len(exec_str) > 60 or
                '\n' in exec_str or
                any(p in exec_str for p in ['Step ', 'To solve', 'Because', 'First,', 'The solution', '1. ', '2. '])
            )
            has_format = 'Format' in (stats.get('operator_list', []) or [])

            if is_long and not has_format:
                return (
                    False,
                    "Finish rejected: answer not concise.",
                    "Add Format to extract a concise answer.",
                )

        return True, "", ""

    def reset(self, problem: Optional[str] = None) -> str:
        """Reset environment

        Args:
            problem: New problem (optional)

        Returns:
            Initial state description
        """
        if problem:
            self.problem = problem

        self.graph = WorkflowGraph()
        self.round_count = 0
        self.history.clear()
        self.is_finished = False
        self.memory.reset()
        self.last_execution_result = None
        self.last_solver_result = None
        self.last_prompt_set = None

        # Reset two-step interaction state
        self.env_state = EnvState.BUILDING
        self.pending_node_id = None
        self.pending_operator = None
        self.pending_prompt_targets = []
        self.pending_prompt_cursor = 0
        self.pending_prompt_total = 0
        self.pending_prompt_structure = None
        self.pending_prompt_parent_id = None
        self.pending_prompt_context = None

        return self.get_initial_state()

    def _set_current_prompt_target(self) -> None:
        """Update current waiting prompt based on pending_prompt_cursor (node_id, operator, context)"""
        if not self.pending_prompt_targets or self.pending_prompt_cursor >= len(self.pending_prompt_targets):
            self.pending_node_id = None
            self.pending_operator = None
            self.pending_prompt_context = None
            return

        node_id, operator = self.pending_prompt_targets[self.pending_prompt_cursor]
        self.pending_node_id = node_id
        self.pending_operator = operator

        if self.pending_prompt_structure and self.pending_prompt_total:
            # parallel / conditional / loop: let model know which prompt this is (encourage heterogeneous parallel/differentiation)
            base_context = (
                f"{self.pending_prompt_structure} prompt {self.pending_prompt_cursor + 1}/{self.pending_prompt_total} "
                f"(node_id={node_id}, parent={self.pending_prompt_parent_id})"
            )
        else:
            base_context = f"node_id={node_id}"

        print(f"[DEBUG] _set_current_prompt_target: structure={self.pending_prompt_structure}, operator={operator}, has_result={self.last_execution_result is not None}", flush=True)
        if self.pending_prompt_structure == "conditional" and operator == "Programmer" and self.last_execution_result:
            exec_str = str(self.last_execution_result.get('output', self.last_execution_result)) if isinstance(self.last_execution_result, dict) else str(self.last_execution_result)
            print(f"[DEBUG] conditional+Programmer: exec_str[:100]={exec_str[:100]}", flush=True)
            if any(err in exec_str.upper() for err in ['ERROR', 'FAILED', 'EXCEPTION']):
                error_preview = exec_str[:300].replace('\n', ' ')
                base_context = f"{base_context}\n[LAST_ERROR]: {error_preview}\nIMPORTANT: Include this error in your prompt so Programmer knows what to fix!"
                print(f"[DEBUG] Injected error into base_context, len={len(base_context)}", flush=True)
                print(f"[DEBUG] base_context has LAST_ERROR: {'LAST_ERROR' in base_context}", flush=True)

        # 🔧 Code problem: inject official test/entry_point (truncated) into prompt_request context
        # so controller can align return type when writing custom prompte.g., "Yes"/"No" vs True/False.
        if self.problem_type == "code":
            # Be robust to different field names used by different pipelines.
            test = ""
            for k in ("public_tests", "test", "test_cases", "tests"):
                v = self.executor_kwargs.get(k)
                s = str(v or "").strip()
                if s:
                    test = s
                    break

            entry_point = str(self.executor_kwargs.get("entry_point", "") or "").strip()
            if not entry_point and test:
                import re
                # HumanEval: candidate = func
                m = re.search(r"candidate\\s*=\\s*([a-zA-Z_][a-zA-Z0-9_]*)", test)
                if m:
                    entry_point = m.group(1)
                else:
                    # BigCodeBench: result = func(...)
                    m = re.search(r"result\\s*=\\s*([a-zA-Z_][a-zA-Z0-9_]*)\\s*\\(", test)
                    if m:
                        entry_point = m.group(1)
                    else:
                        # Common: assert func(...)
                        m = re.search(r"assert\\s*\\(?\\s*([a-zA-Z_][a-zA-Z0-9_]*)\\s*\\(", test)
                        if m:
                            entry_point = m.group(1)

            # Always inject for code tasks: MBPP-style tasks often hide constants/signature in tests only.
            if test or entry_point:
                extra_parts: List[str] = []
                if entry_point:
                    extra_parts.append(f"entry_point={entry_point}")

                if test:
                    # Prefer assert-heavy snippets and include both head+tail so constants at the end aren't missed.
                    non_empty = [ln.rstrip() for ln in test.splitlines() if ln.strip()]
                    assert_like = [
                        ln.strip() for ln in non_empty
                        if ln.lstrip().startswith("assert") or "assert " in ln or "self.assert" in ln
                    ]
                    candidates = assert_like if len(assert_like) >= 3 else non_empty

                    head_n, tail_n = 12, 8
                    preview_lines: List[str] = []
                    if len(candidates) <= head_n + tail_n:
                        preview_lines = candidates[: head_n + tail_n]
                    else:
                        preview_lines = candidates[:head_n] + ["..."] + candidates[-tail_n:]

                    test_preview = "\n".join(preview_lines)
                    if len(test_preview) > 1200:
                        test_preview = test_preview[:1200] + "... [truncated]"

                    extra_parts.append("public_tests (truncated):")
                    extra_parts.append(test_preview)

                base_context = base_context + "\n" + "\n".join(extra_parts)

        self.pending_prompt_context = base_context
        print(f"[DEBUG] Final pending_prompt_context has LAST_ERROR: {'LAST_ERROR' in (self.pending_prompt_context or '')}", flush=True)

    def _format_pending_structure_prompt_request(self, operator_name: str, prompt_guidance: str) -> str:
        """Structure ADD prompt requestcollect prompt for each sub-operator one by one"""
        context = self.pending_prompt_context or ""
        lines = ["<feedback>"]
        lines.append("[Status]: Pending - Awaiting Prompt")
        if context:
            lines.append(f"[Message]: Write a custom prompt for {operator_name}. ({context})")
            print(f"[DEBUG] context len={len(context)}, has LAST_ERROR={'LAST_ERROR' in context}", flush=True)
            if "LAST_ERROR" in context:
                idx = context.find("LAST_ERROR")
                print(f"[DEBUG] LAST_ERROR at index {idx}, context[idx:idx+100]={context[idx:idx+100]}", flush=True)
            # Extra hint: encourage parallel branch differentiation
            if self.pending_prompt_structure == "parallel":
                lines.append("[Hint]: Make this branch DIFFERENT from other parallel branches (different method, checks, or perspective).")
        else:
            lines.append(f"[Message]: Write a custom prompt for {operator_name}.")

        lines.append(f"[Current DSL]: {self.graph.to_dsl() or '(empty)'}")
        lines.append("")
        lines.append("=" * 50)
        lines.append(f"WRITE YOUR PROMPT FOR {operator_name}")
        lines.append("=" * 50)
        lines.append("")
        lines.append(prompt_guidance)
        lines.append("")
        lines.append("IMPORTANT: Just write your prompt directly. No action tags needed.")
        lines.append("Example: 'Solve this math problem step by step. Show all calculations.'")
        lines.append("")
        lines.append(f"[Round]: {self.round_count}/{self.max_rounds}")
        lines.append("</feedback>")
        return "\n".join(lines)

    def step(self, raw_response: str) -> Tuple[str, List[bool], bool]:
        """Execute one action step

        Args:
            raw_response: Model raw output

        Returns:
            (feedback, success_list, active)
            - feedback: Feedback text, fed to model for continued generation
            - success_list: Success status for each action
            - active: Whether to continue (False = terminate)
        """
        result = self._step_internal(raw_response)

        # Record history
        self.history.append({
            "round": self.round_count,
            "raw_response": raw_response,
            "action": result.action.to_dict() if result.action else None,
            "feedback": result.feedback,
            "success": result.success_list,
            "active": result.active,
            "dsl": result.dsl,
            "execution_result": result.execution_result,
        })

        return result.feedback, result.success_list, result.active

    def _step_internal(self, raw_response: str) -> StepResult:
        """Internal step execution logic"""
        self.round_count += 1

        # Check if exceeded max rounds
        if self.round_count > self.max_rounds:
            if self.last_solver_result is not None:
                self.last_execution_result = self.last_solver_result
            return StepResult(
                feedback=self._format_feedback(
                    success=False,
                    message=f"Maximum rounds ({self.max_rounds}) reached. Workflow auto-finished.",
                    force_finish=True
                ),
                success_list=[False],
                active=False,
                dsl=self.graph.to_dsl(),
                statistics=self.graph.get_statistics()
            )

        # ====== Two-step interaction state machine ======
        # If in AWAITING_PROMPT state, treat any input as prompt content
        if self.env_state == EnvState.AWAITING_PROMPT:
            return self._handle_prompt_input(raw_response)

        # Parse action (only in BUILDING state)
        action = self.parser.parse(raw_response)

        # ====== Normal BUILDING state ======
        # Handle invalid action
        if not action.is_valid() and action.action_type == ActionType.INVALID:
            return StepResult(
                feedback=self._format_feedback(
                    success=False,
                    message=f"Invalid action: {action.parse_error}",
                    hint="Please use the correct XML format."
                ),
                success_list=[False],
                active=True,  # Continue, give model chance to retry
                action=action,
                dsl=self.graph.to_dsl(),
                statistics=self.graph.get_statistics()
            )

        # Handle FINISH action
        if action.action_type == ActionType.FINISH:
            ok, msg, hint = self._check_finish_constraints()
            if not ok:
                return StepResult(
                    feedback=self._format_feedback(
                        success=False,
                        message=msg,
                        hint=hint,
                    ),
                    success_list=[False],
                    active=True,
                    action=action,
                    dsl=self.graph.to_dsl(),
                    statistics=self.graph.get_statistics(),
                )

            self.is_finished = True

            self.memory.add_action(
                step=self.round_count,
                action_type="finish",
                result="success"
            )

            # Final execution
            if self.execute_each_step and self.last_execution_result is not None:
                exec_result = self.last_execution_result
                exec_success = True
                print(f"  🔧 ReuseResult: ")
            else:
                exec_result, exec_success = self._execute_workflow()

            if action.final_answer:
                final_answer = action.final_answer
                print(f"  ✅ FinalAnswer: Using model-provided final_answer: {action.final_answer[:50]}...")
            elif self.last_solver_result is not None:
                final_answer = self.last_solver_result
                print(f"  ✅ AvoidLeakage: Using last_solver_result to avoid Plan leakage")
            else:
                final_answer = exec_result
                print(f"  ⚠️ FinalAnswer: Model did not provide final_answer, using execution result")

            return StepResult(
                feedback=self._format_feedback(
                    success=True,
                    message="Workflow construction finished.",
                    final=True,
                    execution_result=final_answer
                ),
                success_list=[True],
                active=False,  # Terminate
                action=action,
                execution_result=final_answer,
                dsl=self.graph.to_dsl(),
                statistics=self.graph.get_statistics()
            )

        # Handle ADD action
        if action.action_type == ActionType.ADD:
            return self._handle_add(action)

        # Handle DELETE action
        if action.action_type == ActionType.DELETE:
            return self._handle_delete(action)

        # Handle MODIFY action
        if action.action_type == ActionType.MODIFY:
            return self._handle_modify(action)

        if action.action_type == ActionType.SET_PROMPT:
            if not action.target:
                return StepResult(
                    feedback=self._format_feedback(
                        success=False,
                        message="set_prompt rejected: missing <target> node id.",
                        hint="Use: <action>set_prompt</action><target>node_3</target><prompt>...</prompt>"
                    ),
                    success_list=[False],
                    active=True,
                    action=action,
                    dsl=self.graph.to_dsl(),
                    statistics=self.graph.get_statistics()
                )

            prompt = (action.custom_prompt or "").strip()
            if len(prompt) < 5:
                return StepResult(
                    feedback=self._format_feedback(
                        success=False,
                        message="set_prompt rejected: prompt is too short or empty.",
                        hint="Provide a more specific prompt (at least a few words)."
                    ),
                    success_list=[False],
                    active=True,
                    action=action,
                    dsl=self.graph.to_dsl(),
                    statistics=self.graph.get_statistics()
                )

            set_result = self.graph.set_node_prompt(action.target, prompt)
            if not set_result or not set_result.get("success"):
                error_msg = set_result.get("message", "Failed to set prompt") if set_result else "Unknown error"
                return StepResult(
                    feedback=self._format_feedback(
                        success=False,
                        message=error_msg,
                        hint="Check the node id in [Workflow Nodes] and try again."
                    ),
                    success_list=[False],
                    active=True,
                    action=action,
                    dsl=self.graph.to_dsl(),
                    statistics=self.graph.get_statistics()
                )

            exec_result = None
            if self.execute_each_step:
                exec_result, _ = self._execute_workflow()

            node_info = self.graph.get_node_info(action.target) or {}
            operator_name = set_result.get("operator") or node_info.get("operator")
            feedback_str, verify_result = self._format_prompt_set_success(
                operator=operator_name or "Unknown",
                node_id=action.target,
                prompt_preview=prompt[:100] + "..." if len(prompt) > 100 else prompt,
                execution_result=exec_result
            )
            return StepResult(
                feedback=feedback_str,
                success_list=[True],
                active=True,
                action=action,
                execution_result=exec_result,
                dsl=self.graph.to_dsl(),
                statistics=self.graph.get_statistics()
            )

        # Other cases
        return StepResult(
            feedback=self._format_feedback(
                success=False,
                message=f"Unknown action type: {action.action_type.value}"
            ),
            success_list=[False],
            active=True,
            action=action,
            dsl=self.graph.to_dsl(),
            statistics=self.graph.get_statistics()
        )

    def _handle_add(self, action: ParsedAction) -> StepResult:
        """Handle ADD action - Delayed execution mode: collect prompt first, then execute add"""
        # OPERATOR: Keep original two-step (request prompt first, then actually add)
        if action.structure_type == StructureType.OPERATOR:
            operator_name = action.operator

            if operator_name in self.disabled_operators:
                return StepResult(
                    feedback=self._format_feedback(
                        success=False,
                        message=f"Operator '{operator_name}' is disabled for this problem type.",
                        hint=f"For multiple-choice problems, use Custom or AnswerGenerate instead of Programmer. Available: {', '.join(sorted(self._get_available_operators()))}"
                    ),
                    success_list=[False],
                    active=True,
                    action=action,
                    dsl=self.graph.to_dsl(),
                    statistics=self.graph.get_statistics()
                )

            self.env_state = EnvState.AWAITING_PROMPT
            self.pending_action = action
            self.pending_operator = operator_name
            self.pending_node_id = None
            self.pending_prompt_targets = []
            self.pending_prompt_cursor = 0
            self.pending_prompt_total = 0
            self.pending_prompt_structure = None
            self.pending_prompt_parent_id = None
            self.pending_prompt_context = None

            prompt_guidance = format_prompt_guidance(operator_name, self.problem, self.problem_type)
            return StepResult(
                feedback=self._format_pending_add_prompt_request(
                    operator_name=operator_name,
                    prompt_guidance=prompt_guidance
                ),
                success_list=[True],
                active=True,
                action=action,
                execution_result=None,
                dsl=self.graph.to_dsl(),
                statistics=self.graph.get_statistics()
            )

        # STRUCTURE: Execute add first (generate child node ids), then collect child node prompts one by one
        ops_to_check = []
        if action.structure_type == StructureType.PARALLEL:
            ops_to_check = action.operators or []
        elif action.structure_type == StructureType.CONDITIONAL:
            ops_to_check = [action.condition]
            if action.true_branch and action.true_branch.lower() != 'done':
                ops_to_check.append(action.true_branch)
            if action.false_branch and action.false_branch.lower() != 'done':
                ops_to_check.append(action.false_branch)
        elif action.structure_type == StructureType.LOOP:
            ops_to_check = action.operators or []

        disabled_in_structure = [op for op in ops_to_check if op in self.disabled_operators]
        if disabled_in_structure:
            return StepResult(
                feedback=self._format_feedback(
                    success=False,
                    message=f"Structure contains disabled operator(s): {', '.join(disabled_in_structure)}",
                    hint=f"For multiple-choice problems, use Custom or AnswerGenerate instead of Programmer. Available: {', '.join(sorted(self._get_available_operators()))}"
                ),
                success_list=[False],
                active=True,
                action=action,
                dsl=self.graph.to_dsl(),
                statistics=self.graph.get_statistics()
            )

        result = None
        if action.structure_type == StructureType.PARALLEL:
            result = self.graph.add_parallel(action.operators, position=action.position)
            structure_name = "parallel"
        elif action.structure_type == StructureType.CONDITIONAL:
            result = self.graph.add_conditional(
                action.condition, action.true_branch, action.false_branch, position=action.position
            )
            structure_name = "conditional"
        elif action.structure_type == StructureType.LOOP:
            result = self.graph.add_loop(action.operators, count=action.loop_count, position=action.position)
            structure_name = "loop"
        else:
            return StepResult(
                feedback=self._format_feedback(
                    success=False,
                    message=f"Unknown structure type: {action.structure_type}"
                ),
                success_list=[False],
                active=True,
                action=action,
                dsl=self.graph.to_dsl(),
                statistics=self.graph.get_statistics()
            )

        if not result or not result.get("success"):
            error_msg = result.get("message", "Failed to add structure") if result else "Unknown error"
            return StepResult(
                feedback=self._format_feedback(
                    success=False,
                    message=error_msg,
                    hint="Please try again."
                ),
                success_list=[False],
                active=True,
                action=action,
                dsl=self.graph.to_dsl(),
                statistics=self.graph.get_statistics()
            )

        parent_id = result.get("node_id")

        # Generate list of child nodes needing prompt collection (in build order)
        prompt_targets: List[Tuple[str, str]] = []
        if action.structure_type == StructureType.PARALLEL:
            for i, op in enumerate(action.operators):
                prompt_targets.append((f"{parent_id}_p{i}", op))
        elif action.structure_type == StructureType.CONDITIONAL:
            prompt_targets.append((f"{parent_id}_cond", action.condition))
            prompt_targets.append((f"{parent_id}_true", action.true_branch))
            if action.false_branch:
                prompt_targets.append((f"{parent_id}_false", action.false_branch))
        elif action.structure_type == StructureType.LOOP:
            for i, op in enumerate(action.operators):
                prompt_targets.append((f"{parent_id}_l{i}", op))

        if not prompt_targets:
            # Should not happen in theory
            return StepResult(
                feedback=self._format_feedback(
                    success=True,
                    message=f"Added {structure_name} structure ({parent_id}). No prompt targets found.",
                    node_id=parent_id
                ),
                success_list=[True],
                active=True,
                action=action,
                dsl=self.graph.to_dsl(),
                statistics=self.graph.get_statistics()
            )

        # Enter multi-prompt collection mode
        self.env_state = EnvState.AWAITING_PROMPT
        self.pending_action = None  # Structure already added
        self.pending_prompt_targets = prompt_targets
        self.pending_prompt_cursor = 0
        self.pending_prompt_total = len(prompt_targets)
        self.pending_prompt_structure = structure_name
        self.pending_prompt_parent_id = parent_id
        self._set_current_prompt_target()

        prompt_guidance = format_prompt_guidance(self.pending_operator, self.problem, self.problem_type)
        return StepResult(
            feedback=self._format_pending_structure_prompt_request(
                operator_name=self.pending_operator,
                prompt_guidance=prompt_guidance
            ),
            success_list=[True],
            active=True,
            action=action,
            execution_result=None,
            dsl=self.graph.to_dsl(),
            statistics=self.graph.get_statistics()
        )

    def _handle_prompt_input(self, raw_response: str) -> StepResult:
        """Handle prompt input (delayed execution mode)

        In AWAITING_PROMPT state, treat any model output as prompt content
        Then execute previously delayed add action and set prompt
        """
        # Extract prompt content (remove possible action tags, keep only core content)
        prompt = self._extract_prompt_content(raw_response)

        if not prompt or len(prompt.strip()) < 5:
            # 🔧 Key fix: detect Think-Only case, give more targeted feedback
            import re
            is_think_only = bool(
                re.search(r'<think>', raw_response, re.IGNORECASE) or
                re.search(r'<think_only>', raw_response, re.IGNORECASE)
            )

            if is_think_only:
                # Think-Only: model only output thinking, no actual prompt
                error_msg = "Think-Only detected: Your thinking content was filtered out."
                hint_msg = (
                    f"DO NOT use <think> tags here! "
                    f"Just write your prompt DIRECTLY for {self.pending_operator}. "
                    f"Example: 'Solve this step by step and show all calculations.'"
                )
            else:
                # Normal empty/short prompt
                error_msg = "Prompt is too short or empty."
                hint_msg = "Please provide a detailed prompt for the operator."

            return StepResult(
                feedback=self._format_feedback(
                    success=False,
                    message=error_msg,
                    hint=hint_msg
                ),
                success_list=[False],
                active=True,
                dsl=self.graph.to_dsl(),
                statistics=self.graph.get_statistics()
            )

        # Case A: delayed action (single operator ADD / MODIFY)
        if self.pending_action:
            action = self.pending_action
            result = None
            node_id = None

            # QA dynamic prompt shortening: make operator output more scorable short answers, reduce false negatives from extraction/format
            prompt = self._apply_dynamic_prompt_rules(prompt, self.pending_operator)

            if action.action_type == ActionType.MODIFY:
                node_id = self.pending_node_id
                result = self.graph.modify_node(action.target, action.operator)
                if result.get("success"):
                    self.graph.set_node_prompt(node_id, prompt)
            else:
                # ADD operator
                result = self.graph.add_operator(action.operator, position=action.position)
                if result and result.get("success"):
                    node_id = result.get("node_id")
                    self.graph.set_node_prompt(node_id, prompt)

            if not result or not result.get("success"):
                error_msg = result.get("message", "Failed to execute action") if result else "Unknown error"
                return StepResult(
                    feedback=self._format_feedback(
                        success=False,
                        message=error_msg,
                        hint="Please try again."
                    ),
                    success_list=[False],
                    active=True,
                    dsl=self.graph.to_dsl(),
                    statistics=self.graph.get_statistics()
                )

            operator_name = self.pending_operator
            # Observability: expose sanitized prompt that was accepted
            self.last_prompt_set = {
                "round": self.round_count,
                "node_id": node_id,
                "operator": operator_name,
                "problem_type": self.problem_type,
                "context": self.pending_prompt_context,
                "prompt": prompt,
            }
            self.env_state = EnvState.BUILDING
            self.pending_action = None
            self.pending_operator = None
            self.pending_node_id = None
            self.pending_prompt_context = None

            self.memory.add_action(
                step=self.round_count,
                action_type="add",
                operator=operator_name,
                prompt_summary=prompt[:50] if prompt else None,
                result="success"
            )

            exec_result = None
            exec_success = True
            if self.execute_each_step:
                exec_result, exec_success = self._execute_workflow()
                preview = str(exec_result)[:100] if exec_result else 'None'
                print(f"[DEBUG] _handle_prompt_input: exec_result={preview}...", flush=True)
                if not exec_success and exec_result:
                    self.memory.add_error(
                        step=self.round_count,
                        error_type="EXECUTION_ERROR",
                        error_msg=str(exec_result)[:200],
                        operator=operator_name
                    )
                elif exec_success and self.memory.current_error:
                    self.memory.mark_error_resolved()
            else:
                print(f"[DEBUG] _handle_prompt_input: execute_each_step is False", flush=True)

            feedback_str, verify_result = self._format_prompt_set_success(
                operator=operator_name,
                node_id=node_id,
                prompt_preview=prompt[:100] + "..." if len(prompt) > 100 else prompt,
                execution_result=exec_result
            )
            return StepResult(
                feedback=feedback_str,
                success_list=[True],
                active=True,
                execution_result=exec_result,
                dsl=self.graph.to_dsl(),
                statistics=self.graph.get_statistics()
            )

        # Case B: structure ADD multi-prompt collection (parallel/conditional/loop)
        if not self.pending_prompt_targets or self.pending_node_id is None or self.pending_operator is None:
            return StepResult(
                feedback=self._format_feedback(
                    success=False,
                    message="Internal error: No pending prompt target.",
                    hint="This shouldn't happen. Please report this issue."
                ),
                success_list=[False],
                active=True,
                dsl=self.graph.to_dsl(),
                statistics=self.graph.get_statistics()
            )

        current_node_id = self.pending_node_id
        current_operator = self.pending_operator

        # QA dynamic prompt shortening: Each sub-operator of structure ADD also gets enhancement
        prompt = self._apply_dynamic_prompt_rules(prompt, current_operator)

        if (self.pending_prompt_structure == "conditional" and
            current_operator == "Programmer" and
            self.last_execution_result):
            exec_str = str(self.last_execution_result.get('output', self.last_execution_result)) if isinstance(self.last_execution_result, dict) else str(self.last_execution_result)
            if any(err in exec_str.upper() for err in ['ERROR', 'FAILED', 'EXCEPTION']):
                error_preview = exec_str[:300].replace('\n', ' ')
                prompt = f"[PREVIOUS_ERROR]: {error_preview}\n\nFix the code based on this error.\n\n{prompt}"
                print(f"[DEBUG] Injected error into prompt, len={len(prompt)}", flush=True)

        set_result = self.graph.set_node_prompt(current_node_id, prompt)
        if not set_result or not set_result.get("success"):
            error_msg = set_result.get("message", "Failed to set prompt") if set_result else "Unknown error"
            return StepResult(
                feedback=self._format_feedback(
                    success=False,
                    message=error_msg,
                    hint="Please try again."
                ),
                success_list=[False],
                active=True,
                dsl=self.graph.to_dsl(),
                statistics=self.graph.get_statistics()
            )

        # Observability: expose sanitized prompt that was accepted
        self.last_prompt_set = {
            "round": self.round_count,
            "node_id": current_node_id,
            "operator": current_operator,
            "problem_type": self.problem_type,
            "context": self.pending_prompt_context,
            "prompt": prompt,
        }

        # Advance to next prompt target
        self.pending_prompt_cursor += 1

        # Still have remaining prompts
        if self.pending_prompt_cursor < len(self.pending_prompt_targets):
            self._set_current_prompt_target()
            prompt_guidance = format_prompt_guidance(self.pending_operator, self.problem, self.problem_type)
            return StepResult(
                feedback=self._format_pending_structure_prompt_request(
                    operator_name=self.pending_operator,
                    prompt_guidance=prompt_guidance
                ),
                success_list=[True],
                active=True,
                dsl=self.graph.to_dsl(),
                statistics=self.graph.get_statistics()
            )

        # All prompts collected: clean up state and return to BUILDING
        self.env_state = EnvState.BUILDING
        self.pending_prompt_targets = []
        self.pending_prompt_cursor = 0
        self.pending_prompt_total = 0
        self.pending_prompt_structure = None
        self.pending_prompt_parent_id = None
        self.pending_prompt_context = None
        self.pending_node_id = None
        self.pending_operator = None

        exec_result = None
        if self.execute_each_step:
            exec_result, _ = self._execute_workflow()
            preview = str(exec_result)[:100] if exec_result else 'None'
            print(f"[DEBUG] _handle_prompt_input: exec_result={preview}...", flush=True)

        feedback_str, verify_result = self._format_prompt_set_success(
            operator=current_operator,
            node_id=current_node_id,
            prompt_preview=prompt[:100] + "..." if len(prompt) > 100 else prompt,
            execution_result=exec_result
        )
        return StepResult(
            feedback=feedback_str,
            success_list=[True],
            active=True,
            execution_result=exec_result,
            dsl=self.graph.to_dsl(),
            statistics=self.graph.get_statistics()
        )

    def _apply_dynamic_prompt_rules(self, prompt: str, operator_name: str) -> str:
        """Light dynamic enhancement on model-provided custom prompt (no forced workflow structure).

        Goal: reduce cases where model is correct but marked wrong due to long answer/inconsistent format, especially QA.
        """
        prompt = str(prompt or "").strip()
        op = str(operator_name or "").strip()
        pt = str(self.problem_type or "").strip().lower()

        if not prompt or not op:
            return prompt

        # === QA: short answer constraint (reduce false negatives from extraction/format) ===
        if pt == "qa":
            target_ops = {"Custom", "Verify", "AnswerGenerate"}
            if op not in target_ops:
                return prompt

            marker = "[QA_SHORT_ANSWER]"
            if marker in prompt:
                return prompt

            if op == "Verify":
                guidance = (
                    f"{marker}\n"
                    "- In <answer>, output ONLY the final answer (usually 1-5 words).\n"
                    "- If the proposed answer is correct but verbose, shorten it to the minimal answer.\n"
                    "- If the answer is a number/year/date, output ONLY the number/year/date (no extra words/units unless explicitly asked).\n"
                )
            elif op == "AnswerGenerate":
                guidance = (
                    f"{marker}\n"
                    "- In <answer>, output ONLY the final answer (usually 1-5 words).\n"
                    "- No explanations, no full sentences.\n"
                    "- If the answer is a number/year/date, output ONLY the number/year/date.\n"
                )
            else:  # Custom
                guidance = (
                    f"{marker}\n"
                    "- Return ONLY the final answer, as short as possible (usually 1-5 words).\n"
                    "- No explanation, no full sentence.\n"
                    "- If the answer is a number/year/date, output ONLY the number/year/date.\n"
                )

            return f"{guidance}\n{prompt}".strip()

        # === CODE: enforce code-only instruction for Programmer ===
        # Motivation: controller sometimes writes "Step 1/2/3" style prompts that cause code leakage (non-code text).
        if pt == "code" and op == "Programmer":
            marker = "[CODE_PYTHON_ONLY]"
            if marker in prompt:
                return prompt
            guidance = (
                f"{marker}\n"
                "- Your output MUST be ONLY Python code (no explanations, no markdown fences, no numbered steps).\n"
                "- Follow the REQUIRED function/class signature EXACTLY; do NOT rename.\n"
                "- Do NOT add input()/print() unless explicitly required by the problem/tests.\n"
                "- Include necessary imports; keep the implementation minimal and deterministic.\n"
            )
            return f"{guidance}\n{prompt}".strip()

        if pt in ("mathqa_mc", "mc", "multiple_choice"):
            if op == "Custom":
                marker = "[MC_PRECISE_CALC]"
                if marker in prompt:
                    return prompt
                guidance = (
                    f"{marker}\n"
                    "- This is a MULTIPLE CHOICE problem. Your answer MUST be a single letter (a, b, c, d, or e).\n"
                    "- Calculate step by step with EXACT numbers (no rounding, no approximation).\n"
                    "- After calculation, VERIFY by substituting your answer back into the original equation.\n"
                    "- Compare your result with EACH option to find the matching one.\n"
                    "- Output format: 'The answer is [letter]' where [letter] is a, b, c, d, or e.\n"
                )
                return f"{guidance}\n{prompt}".strip()
            elif op == "Programmer":
                marker = "[MC_CODE_SOLVE]"
                if marker in prompt:
                    return prompt
                guidance = (
                    f"{marker}\n"
                    "- Write Python code to solve this multiple choice problem.\n"
                    "- Calculate the exact numerical answer using Python.\n"
                    "- Compare the result with each option (a, b, c, d, e) to find the match.\n"
                    "- Print ONLY the matching option letter (a, b, c, d, or e).\n"
                    "- Use fractions or Decimal for precise calculation if needed.\n"
                )
                return f"{guidance}\n{prompt}".strip()
            elif op in ("Verify", "Review"):
                marker = "[MC_VERIFY]"
                if marker in prompt:
                    return prompt
                guidance = (
                    f"{marker}\n"
                    "- Verify the proposed answer by recalculating from scratch.\n"
                    "- Check if the answer matches one of the options (a, b, c, d, e).\n"
                    "- If incorrect, identify the correct option.\n"
                    "- Output: 'Correct option: [letter]' where [letter] is a, b, c, d, or e.\n"
                )
                return f"{guidance}\n{prompt}".strip()
            elif op == "Format":
                marker = "[MC_FORMAT]"
                if marker in prompt:
                    return prompt
                guidance = (
                    f"{marker}\n"
                    "- Extract ONLY the final answer letter (a, b, c, d, or e).\n"
                    "- Output a SINGLE lowercase letter, nothing else.\n"
                )
                return f"{guidance}\n{prompt}".strip()

        return prompt

    def _extract_prompt_content(self, raw_response: str) -> str:
        """Extract prompt content from model output

        Supports multiple formats:
        1. Plain text
        2. <prompt>content</prompt>
        3. Response with action tags (extract text content)

        🔧 Key fix: filter out <think> and <think_only> content, keep only actual prompt after
        """
        import re

        # Try to extract content from <prompt> tag (highest priority)
        prompt_match = re.search(r'<prompt[^>]*>(.*?)</prompt>', raw_response, re.DOTALL | re.IGNORECASE)
        if prompt_match:
            return prompt_match.group(1).strip()

        # 🔧 Key fix: remove <think>...</think> blocks, keep only content after
        # This way Qwen3 thinking mode output can be handled correctly
        cleaned = raw_response

        # Remove complete <think>...</think> blocks
        cleaned = re.sub(r'<think>.*?</think>\s*', '', cleaned, flags=re.DOTALL | re.IGNORECASE)

        # Remove <think_only>...</think_only> blocks (Think-Only content wrapped by generation function)
        cleaned = re.sub(r'<think_only>.*?</think_only>\s*', '', cleaned, flags=re.DOTALL | re.IGNORECASE)

        # Remove incomplete <think> tags (truncated cases)
        cleaned = re.sub(r'<think>.*', '', cleaned, flags=re.DOTALL | re.IGNORECASE)

        # Remove common action tags, keep other content
        # Remove <action>...</action>
        cleaned = re.sub(r'<action[^>]*>.*?</action>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        # Remove <operator>...</operator>
        cleaned = re.sub(r'<operator[^>]*>.*?</operator>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        # Remove other common tags
        cleaned = re.sub(r'</?reasoning[^>]*>', '', cleaned, flags=re.IGNORECASE)

        # Return cleaned text
        return cleaned.strip()

    def _format_pending_add_prompt_request(self, operator_name: str, prompt_guidance: str) -> str:
        """Format prompt request for operator to be added (delayed execution mode)"""
        lines = ["<feedback>"]
        lines.append("[Status]: Pending - Awaiting Prompt")
        lines.append(f"[Message]: You want to add {operator_name}. Now write a custom prompt for it.")
        lines.append(f"[Current DSL]: {self.graph.to_dsl() or '(empty)'}")
        lines.append("")
        lines.append("=" * 50)
        lines.append(f"WRITE YOUR PROMPT FOR {operator_name}")
        lines.append("=" * 50)
        lines.append("")
        if operator_name == "Programmer":
            lines.append("CRITICAL: Your prompt will be sent to a code generation model.")
            lines.append("DO NOT write step descriptions like 'Step 1: Validate input...'")
            lines.append("Write instructions like 'Implement function xxx, handle empty list edge case'")
            lines.append("")
        lines.append(prompt_guidance)
        lines.append("")
        lines.append("IMPORTANT: Just write your prompt directly. No action tags needed.")
        lines.append("Example: 'Solve this math problem step by step. Show all calculations.'")
        lines.append("")
        lines.append(f"[Round]: {self.round_count}/{self.max_rounds}")
        lines.append("</feedback>")

        return "\n".join(lines)

    def _format_pending_modify_prompt_request(self, target_node: str, operator_name: str, prompt_guidance: str) -> str:
        """Format prompt request for modify operation

        🔧 Key: modify also requires new prompt, otherwise large model generates same code
        """
        lines = ["<feedback>"]
        lines.append("[Status]: Pending - Awaiting New Prompt for Modify")
        lines.append(f"[Message]: You want to modify {target_node} to {operator_name}. Now write a NEW custom prompt.")
        lines.append(f"[Current DSL]: {self.graph.to_dsl() or '(empty)'}")
        lines.append("")
        lines.append("=" * 50)
        lines.append(f"WRITE A NEW PROMPT FOR {operator_name}")
        lines.append("=" * 50)
        lines.append("")
        lines.append(prompt_guidance)
        lines.append("")
        lines.append("IMPORTANT: Write a DIFFERENT prompt than before to get different results!")
        lines.append("The previous prompt didn't work. Try a new approach.")
        lines.append("")
        lines.append(f"[Round]: {self.round_count}/{self.max_rounds}")
        lines.append("</feedback>")

        return "\n".join(lines)

    def _handle_delete(self, action: ParsedAction) -> StepResult:
        """Handle DELETE action"""
        result = self.graph.remove_node(action.target)
        success = result["success"]

        if success:
            self.memory.add_action(
                step=self.round_count,
                action_type="delete",
                operator=action.target,
                result="success"
            )

        exec_result = None
        if success and self.execute_each_step and not self.graph.is_empty():
            exec_result, _ = self._execute_workflow()

        return StepResult(
            feedback=self._format_feedback(
                success=success,
                message=result["message"],
                execution_result=exec_result
            ),
            success_list=[success],
            active=True,
            action=action,
            execution_result=exec_result,
            dsl=self.graph.to_dsl(),
            statistics=self.graph.get_statistics()
        )

    def _handle_modify(self, action: ParsedAction) -> StepResult:
        """Handle MODIFY action

        🔧 Key fix: modify also needs to enter AWAITING_PROMPT state, let small model write new prompt
        Otherwise each modify uses old prompt, large model generates same code, causing infinite loop
        """
        operator_name = action.operator
        target_node = action.target

        # Verify target node exists (using _node_map or get_node_info)
        node_exists = self.graph.get_node_info(target_node) is not None
        if not node_exists:
            return StepResult(
                feedback=self._format_feedback(
                    success=False,
                    message=f"Target node '{target_node}' not found",
                    hint="Use a valid node ID from the workflow."
                ),
                success_list=[False],
                active=True,
                action=action,
                dsl=self.graph.to_dsl(),
                statistics=self.graph.get_statistics()
            )

        # Verify operator is valid (VALID_OPERATORS imported from workflow_graph at top of file)
        if operator_name not in VALID_OPERATORS:
            return StepResult(
                feedback=self._format_feedback(
                    success=False,
                    message=f"Invalid operator: {operator_name}",
                    hint=f"Valid operators: {', '.join(VALID_OPERATORS)}"
                ),
                success_list=[False],
                active=True,
                action=action,
                dsl=self.graph.to_dsl(),
                statistics=self.graph.get_statistics()
            )

        if operator_name in self.disabled_operators:
            return StepResult(
                feedback=self._format_feedback(
                    success=False,
                    message=f"Operator '{operator_name}' is disabled for this problem type.",
                    hint=f"For multiple-choice problems, use reasoning operators (Custom, AnswerGenerate) instead of code generation. Available operators: {', '.join(sorted(self._get_available_operators()))}"
                ),
                success_list=[False],
                active=True,
                action=action,
                dsl=self.graph.to_dsl(),
                statistics=self.graph.get_statistics()
            )

        # ====== 🔧 Key fix: enter AWAITING_PROMPT state ======
        self.env_state = EnvState.AWAITING_PROMPT
        self.pending_action = action
        self.pending_operator = operator_name
        self.pending_node_id = target_node  # modify already has node_id

        # Generate prompt template guidance
        from .operator_descriptions import format_prompt_guidance
        prompt_guidance = format_prompt_guidance(operator_name, self.problem, self.problem_type)

        return StepResult(
            feedback=self._format_pending_modify_prompt_request(
                target_node=target_node,
                operator_name=operator_name,
                prompt_guidance=prompt_guidance
            ),
            success_list=[True],
            active=True,
            action=action,
            execution_result=None,
            dsl=self.graph.to_dsl(),
            statistics=self.graph.get_statistics()
        )

    def _handle_set_prompt(self, action: ParsedAction) -> StepResult:
        """Handle SET_PROMPT action (Second step of two-step interaction)

        Set custom_prompt to pending node, then execute workflow
        """
        if not self.pending_node_id:
            return StepResult(
                feedback=self._format_feedback(
                    success=False,
                    message="No pending node to set prompt for.",
                    hint="This shouldn't happen. Please report this issue."
                ),
                success_list=[False],
                active=True,
                action=action,
                dsl=self.graph.to_dsl(),
                statistics=self.graph.get_statistics()
            )

        # Set prompt to node
        result = self.graph.set_node_prompt(self.pending_node_id, action.custom_prompt)
        success = result["success"]

        if success:
            # ====== Return to BUILDING state ======
            self.env_state = EnvState.BUILDING
            node_id = self.pending_node_id
            operator = self.pending_operator
            self.pending_node_id = None
            self.pending_operator = None

            self.memory.add_action(
                step=self.round_count,
                action_type="set_prompt",
                operator=operator,
                prompt_summary=action.custom_prompt[:50] if action.custom_prompt else None,
                result="success"
            )

            # Now execute workflow
            exec_result = None
            exec_success = True
            if self.execute_each_step:
                exec_result, exec_success = self._execute_workflow()
                if not exec_success and exec_result:
                    self.memory.add_error(
                        step=self.round_count,
                        error_type="EXECUTION_ERROR",
                        error_msg=str(exec_result)[:200],
                        operator=operator
                    )
                elif exec_success and self.memory.current_error:
                    # Execution succeeded, mark error as resolved
                    self.memory.mark_error_resolved()

            feedback_str, verify_result = self._format_prompt_set_success(
                operator=operator,
                node_id=node_id,
                prompt_preview=action.custom_prompt[:100] + "..." if len(action.custom_prompt) > 100 else action.custom_prompt,
                execution_result=exec_result
            )

            return StepResult(
                feedback=feedback_str,
                success_list=[success],
                active=True,  # Continue building
                action=action,
                execution_result=exec_result,
                dsl=self.graph.to_dsl(),
                statistics=self.graph.get_statistics(),
                verify_result=verify_result
            )
        else:
            # Set failed, keep AWAITING_PROMPT state
            return StepResult(
                feedback=self._format_feedback(
                    success=False,
                    message=result["message"],
                    hint="Please try again with a valid prompt."
                ),
                success_list=[success],
                active=True,
                action=action,
                dsl=self.graph.to_dsl(),
                statistics=self.graph.get_statistics()
            )

    def _format_add_success_with_prompt_guidance(
        self,
        result: Dict[str, Any],
        operator_name: str,
        prompt_guidance: str
    ) -> str:
        """Format feedback after successful add (includes prompt template guidance)"""
        lines = ["<feedback>"]
        lines.append("[Status]: Success - Operator Added")
        lines.append(f"[Message]: {result['message']}")
        lines.append(f"[Node ID]: {result.get('node_id', 'N/A')}")
        lines.append(f"[Current DSL]: {self.graph.to_dsl() or '(empty)'}")

        # Two-step interaction key: require model to set prompt
        lines.append("")
        lines.append("=" * 50)
        lines.append(f"[REQUIRED ACTION]: Set prompt for {operator_name}")
        lines.append("=" * 50)
        lines.append("")
        lines.append(prompt_guidance)
        lines.append("")
        lines.append("Use this format:")
        lines.append("<action>set_prompt</action>")
        lines.append("<prompt>Your customized prompt here</prompt>")
        lines.append("")
        lines.append(f"[Round]: {self.round_count}/{self.max_rounds}")
        lines.append("</feedback>")

        return "\n".join(lines)

    def _format_awaiting_prompt_error(self, action: ParsedAction) -> str:
        """Format error when receiving non-set_prompt action in AWAITING_PROMPT state"""
        lines = ["<feedback>"]
        lines.append("[Status]: Failed - Wrong Action")
        lines.append(f"[Message]: You must set prompt for {self.pending_operator} ({self.pending_node_id}) first.")
        lines.append(f"[Received Action]: {action.action_type.value}")
        lines.append("")
        lines.append("You are currently in AWAITING_PROMPT state.")
        lines.append("Before continuing to build the workflow, you must set the prompt for the operator you just added.")
        lines.append("")

        # Show prompt template again
        prompt_guidance = format_prompt_guidance(self.pending_operator, self.problem, self.problem_type)
        lines.append(prompt_guidance)
        lines.append("")
        lines.append("Use this format:")
        lines.append("<action>set_prompt</action>")
        lines.append("<prompt>Your customized prompt here</prompt>")
        lines.append("")
        lines.append(f"[Round]: {self.round_count}/{self.max_rounds}")
        lines.append("</feedback>")

        return "\n".join(lines)

    def _format_prompt_set_success(
        self,
        operator: str,
        node_id: str,
        prompt_preview: str,
        execution_result: Optional[str] = None
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Format feedback after successful prompt set

        Key change: put execution result and Action Hint at front to avoid truncation!

        Returns:
            Tuple[feedback_str, verify_result_dict]
        """
        lines = ["<feedback>"]
        verify_result = None

        # Statistics
        stats = self.graph.get_statistics()
        total_ops = stats['total_operators']
        all_operators = self.graph.get_all_operators()

        block_finish = 0
        next_action = None
        signal_reason = None

        # ====== Key change: put execution result and judgment at front!======
        if execution_result:
            if isinstance(execution_result, dict):
                # Programmer returned dict, extract output for display
                exec_str = str(execution_result.get('output', execution_result))
            else:
                exec_str = str(execution_result)

            # Check if error result
            # Treat common non-answer executor outputs as errors to avoid routing into Verify/Format prematurely.
            is_error = any(
                err in exec_str.upper()
                for err in [
                    'ERROR',
                    'EXCEPTION',
                    'FAILED',
                    'TIMEOUT',
                    'WORKFLOW_ERROR',
                    'NO CODE GENERATED',
                    "FUNCTION 'SOLVE' NOT FOUND",
                    'PROHIBITED IMPORT',
                    'FORMAT ERROR',
                ]
            )

            # Truncate execution result
            exec_preview = exec_str[:300] if len(exec_str) > 300 else exec_str

            # Truncate problem display
            problem_preview = self.problem[:200] if len(self.problem) > 200 else self.problem

            if is_error:
                lines.append(f"[EXECUTION ERROR]: {exec_preview}")
                # v19: Smart error analysis - provide specific fix suggestions to model
                error_suggestion = self._analyze_error_and_suggest(execution_result)
                if error_suggestion:
                    lines.append(f"[FIX SUGGESTION]: {error_suggestion}")
                lines.append(f"[DECISION]: The execution failed. Consider fixing the workflow.")
            else:
                # Successful execution - show problem and answer, remind model to make decision
                lines.append(f"[PROBLEM]: {problem_preview}")
                lines.append(f"[EXECUTION RESULT]: {exec_preview}")

                if operator in ('Programmer', 'Custom', 'AnswerGenerate'):
                    prev_output = self.memory.get_previous_step_output()
                    if prev_output:
                        lines.append(f"[PREVIOUS STEP OUTPUT]: {prev_output}")
                        lines.append("Use this result as context for the current step.")

                print(f"[DEBUG] operator='{operator}', exec_result type={type(execution_result).__name__}, is_dict={isinstance(execution_result, dict)}")
                if isinstance(execution_result, dict):
                    print(f"[DEBUG] dict keys={list(execution_result.keys())}")
                if operator == 'Plan' and isinstance(execution_result, dict):
                    steps_list = execution_result.get('steps_list', [])
                    print(f"[DEBUG] operator={operator}, steps_list={steps_list[:2] if steps_list else 'empty'}")
                    if steps_list:
                        lines.append(f"[PLAN STEPS]: {len(steps_list)} steps identified")
                        print(f"[DEBUG] ✓ Added [PLAN STEPS] to lines")
                        for i, step in enumerate(steps_list[:5], 1):
                            step_preview = step[:100] + '...' if len(step) > 100 else step
                            lines.append(f"  Step {i}: {step_preview}")
                        if len(steps_list) > 5:
                            lines.append(f"  ... and {len(steps_list) - 5} more steps")
                        lines.append(f"[HINT]: Plan created {len(steps_list)} steps. Add an operator (Programmer/Custom/AnswerGenerate) to execute these steps.")
                elif operator == 'Decompose' and isinstance(execution_result, dict):
                    sub_problems = execution_result.get('sub_problems_list', [])
                    if sub_problems:
                        n_subs = len(sub_problems)
                        lines.append(f"[SUB-PROBLEMS]: {n_subs} independent sub-problems identified")
                        for i, sub in enumerate(sub_problems[:5], 1):
                            sub_preview = sub[:100] + '...' if len(sub) > 100 else sub
                            lines.append(f"  Sub-problem {i}: {sub_preview}")
                        if n_subs > 5:
                            lines.append(f"  ... and {n_subs - 5} more sub-problems")
                        lines.append(f"[HINT]: Add {n_subs} Programmer operators (one per sub-problem) to solve them in parallel, then add Aggregate to combine results.")

                if operator == "Format":
                    lines.append(f"[HINT]: Format extracted concise answer. Consider finishing if satisfied.")
                else:
                    finish_hint, verify_result = self._detect_verification_success(execution_result, exec_str)
                    if finish_hint:
                        lines.append(f"[HINT]: {finish_hint}")
                    elif operator in ('Verify', 'Review', 'Test'):
                        is_long_answer = (
                            len(exec_str) > 40 or
                            '\n' in exec_str or
                            any(p in exec_str for p in ['Step ', 'To solve', 'Because', 'First,', 'The solution'])
                        )
                        if is_long_answer:
                            lines.append(f"[HINT]: Add Format next to extract a concise final answer before finishing.")
                lines.append("[DECISION]: Decide whether to continue improving the workflow or finish if the result answers the problem.")

            solver_ops = {'Programmer', 'AnswerGenerate', 'Custom'}
            has_solver = bool(solver_ops & set(all_operators))
            has_format = 'Format' in all_operators
            problem_type = getattr(self, 'problem_type', 'qa')
            print(f"[DEBUG] operator='{operator}' has_solver={has_solver} all_operators={all_operators}")

            is_concise = (
                len(exec_str) <= 50 and
                '\n' not in exec_str and
                not any(p in exec_str for p in ['Step ', 'To solve', 'Because', 'First,', 'The solution'])
            )

            is_mc_problem = problem_type in ("mathqa_mc", "mc", "multiple_choice")
            is_valid_mc_answer = False
            if is_mc_problem:
                exec_clean = exec_str.strip().lower()
                is_valid_mc_answer = exec_clean in ['a', 'b', 'c', 'd', 'e']
                if not is_valid_mc_answer:
                    print(f"[DEBUG] MC answer not valid letter: '{exec_str[:50]}', will force Format", flush=True)

            revise_count = sum(1 for op in all_operators if op == 'Revise')
            if is_mc_problem and revise_count >= 3 and not is_valid_mc_answer:
                block_finish = 1
                next_action = "Format"
                signal_reason = f"MC problem: {revise_count} Revise attempts failed, force Format to extract option letter"
                print(f"[DEBUG] MC Revise loop detected: {revise_count} Revise, forcing Format", flush=True)
            elif is_error:
                # When execution fails, prioritize re-running a solver that can actually fix the failure.
                # For math/code tasks, failures almost always originate from the Programmer sandbox run.
                has_programmer = "Programmer" in set(all_operators)
                has_test = "Test" in set(all_operators)
                block_finish = 1
                if has_programmer or problem_type in ("math", "code"):
                    error_detail = self._extract_test_error(exec_str)
                    if operator in ('Test', 'Programmer') and has_test and has_programmer:
                        next_action = "conditional(Test,Programmer,done)"
                        if error_detail:
                            signal_reason = f"Test failed: {error_detail[:150]}. IMPORTANT: Write this error in Programmer's prompt so it knows what to fix."
                        else:
                            signal_reason = "Test failed; use conditional structure: Test ? Programmer : done"
                    else:
                        next_action = "Programmer"
                        if error_detail:
                            signal_reason = f"Test failed: {error_detail[:150]}. IMPORTANT: Write this error in Programmer's prompt so it knows what to fix."
                        else:
                            signal_reason = "Execution failed; rerun Programmer to fix runtime/error before verification"
                elif has_solver:
                    next_action = "Revise"
                    signal_reason = "Execution failed; Revise to fix before proceeding"
                else:
                    next_action = "Custom"
                    signal_reason = "Execution failed; add solver to fix"

            elif operator == 'Plan':
                block_finish = 1
                if problem_type in ('math', 'code'):
                    next_action = "Programmer"
                    signal_reason = "Plan is not answer; add Programmer to write executable code"
                else:
                    next_action = "Custom"
                    signal_reason = "Plan is not answer; add Custom to execute steps"

            elif operator == 'Decompose':
                block_finish = 1
                if problem_type in ('math', 'code'):
                    next_action = "Programmer"
                    solver_name = "Programmer"
                else:
                    next_action = "Custom"
                    solver_name = "Custom"
                n_subs = 0
                if isinstance(execution_result, dict):
                    n_subs = len(execution_result.get('sub_problems_list', []))
                if n_subs > 1:
                    signal_reason = f"Decompose created {n_subs} independent sub-problems; add {n_subs} {solver_name} in parallel, then Aggregate"
                else:
                    signal_reason = f"Decompose is not answer; add {solver_name} to solve"

            elif has_solver:
                if operator == 'Format':
                    if is_mc_problem and not is_valid_mc_answer:
                        block_finish = 1
                        next_action = "Format"
                        signal_reason = f"MC answer must be a/b/c/d/e, got '{exec_str[:20]}'; rerun Format to extract option letter"
                    elif is_concise:
                        block_finish = 0
                        next_action = "FINISH"
                        signal_reason = "Answer formatted and concise"
                    else:
                        block_finish = 1
                        next_action = "Revise"
                        signal_reason = "Format output not concise; Revise to simplify"

                elif operator in ('Verify', 'Review', 'Test'):
                    if operator == 'Verify':
                        # verify_result is extracted from structured Verify output (dict/XML). Be conservative on missing data.
                        if not verify_result:
                            block_finish = 1
                            next_action = "Revise"
                            signal_reason = "Verify produced no structured result; Revise or rerun Verify"
                        else:
                            is_correct = verify_result.get('is_correct')
                            confidence = str(verify_result.get('confidence', 'unknown')).lower()
                            verified_answer = str(verify_result.get('answer', '') or '').strip()

                            if is_correct is True:
                                if not has_format:
                                    block_finish = 1
                                    next_action = "Format"
                                    signal_reason = "Verify passed; add Format for concise answer"
                                else:
                                    block_finish = 0
                                    next_action = "FINISH"
                                    signal_reason = "Verify passed and Format exists"
                            elif is_correct is False:
                                # If verifier provides a high-confidence corrected answer, format it directly.
                                if verified_answer and confidence == 'high':
                                    if not has_format:
                                        block_finish = 1
                                        next_action = "Format"
                                        signal_reason = "Verify failed but provided high-confidence corrected answer; Format it"
                                    else:
                                        block_finish = 0
                                        next_action = "FINISH"
                                        signal_reason = "High-confidence corrected answer available and Format exists"
                                else:
                                    block_finish = 1
                                    next_action = "Revise"
                                    signal_reason = "Verify failed; Revise to fix"
                            else:
                                block_finish = 1
                                next_action = "Revise"
                                signal_reason = "Verify result unknown; Revise to fix"
                    else:
                        # Review/Test fallback: without a structured pass/fail signal, assume output needs formatting.
                        if not has_format:
                            block_finish = 1
                            next_action = "Format"
                            signal_reason = f"{operator} executed; add Format for concise answer"
                        else:
                            block_finish = 0
                            next_action = "FINISH"
                            signal_reason = f"{operator} executed and Format exists"

                elif operator in ('Programmer', 'Custom', 'AnswerGenerate'):
                    has_verify = 'Verify' in all_operators
                    has_test = 'Test' in all_operators

                    if is_mc_problem and not is_valid_mc_answer:
                        block_finish = 1
                        next_action = "Format"
                        signal_reason = f"MC answer must be a/b/c/d/e, got '{exec_str[:20]}'; add Format to extract option letter"
                    elif problem_type == 'math' and not has_verify:
                        block_finish = 1
                        next_action = "Verify"
                        signal_reason = "MATH: Must verify answer before finishing"
                    elif problem_type == 'code' and self.finish_require_test and not has_test:
                        block_finish = 1
                        next_action = "Test"
                        signal_reason = "CODE: Must run Test with provided unit tests before finishing"
                    elif is_concise and (has_verify or has_test):
                        block_finish = 0
                        next_action = "FINISH"
                        signal_reason = "Solver output verified and concise"
                    elif is_concise:
                        if problem_type == 'math':
                            block_finish = 1
                            next_action = "Verify"
                            signal_reason = "Add Verify to double-check math answer"
                        elif problem_type == 'code':
                            block_finish = 1
                            next_action = "Test"
                            signal_reason = "Add Test to run unit tests on the code"
                        else:
                            block_finish = 1
                            next_action = "Review"
                            signal_reason = "Add Review to check answer quality"
                    else:
                        if has_verify or has_test:
                            block_finish = 1
                            next_action = "Format"
                            signal_reason = "Solver output needs Format"
                        else:
                            block_finish = 1
                            if problem_type == 'math':
                                next_action = "Verify"
                                signal_reason = "Verify math answer first, then Format"
                            elif problem_type == 'code':
                                next_action = "Test"
                                signal_reason = "Test code first, then Format"
                            else:
                                next_action = "Review"
                                signal_reason = "Review answer quality first, then Format"

                else:
                    if is_mc_problem and not is_valid_mc_answer:
                        block_finish = 1
                        next_action = "Format"
                        signal_reason = f"MC answer must be a/b/c/d/e, got '{exec_str[:20]}'; add Format to extract option letter"
                    elif is_concise:
                        block_finish = 0
                        next_action = "FINISH"
                        signal_reason = "Answer ready"
                    else:
                        block_finish = 1
                        next_action = "Format"
                        signal_reason = "Need Format for concise answer"

            else:
                if total_ops >= 1:
                    block_finish = 1
                    if problem_type == 'math':
                        next_action = "Programmer"
                    else:
                        next_action = "Custom"
                    signal_reason = "Workflow has no solver operator"

            has_parallel = stats.get('has_parallel', False)
            aggregation_ops = {'ScEnsemble', 'Aggregate', 'MdEnsemble'}
            has_aggregation = bool(aggregation_ops & set(all_operators))

            if has_parallel and not has_aggregation and not is_error:
                if next_action != "FINISH":
                    if next_action not in aggregation_ops:
                        block_finish = 1
                        next_action = "ScEnsemble" if problem_type == 'math' else "Aggregate"
                        signal_reason = "Parallel structure needs aggregation before verification/formatting"
                    lines.append(f"[HINT]: Parallel structure detected without aggregation. Add ScEnsemble/Aggregate to merge parallel results into a single answer.")
                    print(f"[DEBUG] Added parallel aggregation hint: has_parallel={has_parallel}, has_aggregation={has_aggregation}")
        else:
            # No execution result (should not happen in theory)
            lines.append(f"[EXECUTION]: No result returned.")

        if block_finish or next_action:
            signal_lines = []
            signal_lines.append(f"[BLOCK_FINISH]={block_finish}")
            if next_action:
                if next_action == "FINISH":
                    signal_lines.append(f"[NEXT]=FINISH (Answer looks correct and concise. Consider finishing to avoid drift.)")
                    lines = [line for line in lines if not line.startswith("[DECISION]")]
                    lines = [line for line in lines if not ("[HINT]" in line and "Add" in line and "next" in line)]
                else:
                    signal_lines.append(f"[NEXT]=ADD:{next_action}")
                    lines = [line for line in lines if not line.startswith("[DECISION]")]
            if signal_reason:
                signal_lines.append(f"[REASON]={signal_reason}")
            print(f"[DEBUG] operator={operator}, block_finish={block_finish}, next_action={next_action}, reason={signal_reason}", flush=True)
            for i, line in enumerate(signal_lines):
                lines.insert(1 + i, line)

        # Then state info
        lines.append(f"[Status]: Success - Prompt Set")
        lines.append(f"[Message]: Custom prompt set for {operator} ({node_id})")

        # DSL - simplified display, avoid too long
        dsl = self.graph.to_dsl() or '(empty)'
        if len(dsl) > 150:
            # Show only head and tail
            dsl_parts = dsl.split(' -> ')
            if len(dsl_parts) > 6:
                dsl = ' -> '.join(dsl_parts[:3]) + ' -> ... -> ' + ' -> '.join(dsl_parts[-2:])
        lines.append(f"[Current DSL]: {dsl}")

        # Statistics
        lines.append(f"[Statistics]: {total_ops} operators, {stats['unique_types']} unique types")
        lines.append(f"[Round]: {self.round_count}/{self.max_rounds}")
        lines.append("</feedback>")

        return "\n".join(lines), verify_result

    def _execute_workflow(self) -> Tuple[str, bool]:
        """Execute current workflow

        Returns:
            (result_text, success)
        """
        if self.graph.is_empty():
            print(f"[DEBUG] _execute_workflow: graph is empty", flush=True)
            return "(Empty workflow - no execution)", False

        dsl = self.graph.to_dsl()
        # 🔧 Key fix: get custom_prompts written by small model
        prompts = self.graph.get_all_prompts()
        print(f"[DEBUG] _execute_workflow: dsl={dsl}, executor={self.executor is not None}, prompts={len(prompts)}", flush=True)

        if self.executor:
            try:
                # 🔧 Pass prompts to executor (if executor supports)
                import inspect
                sig = inspect.signature(self.executor)
                call_kwargs: Dict[str, Any] = {}

                if self.use_custom_prompts_in_execution and 'prompts' in sig.parameters:
                    call_kwargs['prompts'] = prompts

                has_var_kwargs = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
                )
                if self.executor_kwargs:
                    for k, v in self.executor_kwargs.items():
                        if has_var_kwargs or k in sig.parameters:
                            call_kwargs[k] = v

                result, success = self.executor(dsl, self.problem, **call_kwargs)
                self.last_execution_result = result
                SOLVER_OPS = {"Custom", "Programmer", "AnswerGenerate", "Aggregate",
                              "Format", "Revise", "ScEnsemble", "MdEnsemble"}
                stats = self.graph.get_statistics()
                operator_list = stats.get('operator_list', [])
                last_op = operator_list[-1] if operator_list else "unknown"
                if success and last_op in SOLVER_OPS:
                    self.last_solver_result = result
                self.memory.add_step_result(
                    step=self.round_count,
                    operator=last_op,
                    input_ctx=self.problem[:200] if self.problem else "",
                    output=result,
                    success=success
                )
                print(f"[DEBUG] _execute_workflow: result={str(result)[:100] if result else 'None'}..., success={success}", flush=True)
                return result, success
            except Exception as e:
                error_msg = f"Execution error: {str(e)}"
                self.last_execution_result = error_msg
                print(f"[DEBUG] _execute_workflow: exception={str(e)}", flush=True)
                return error_msg, False
        else:
            # Return placeholder when no executor
            placeholder = f"(Execution skipped - DSL: {dsl})"
            self.last_execution_result = placeholder
            print(f"[DEBUG] _execute_workflow: no executor, returning placeholder", flush=True)
            return placeholder, True

    def _extract_test_error(self, exec_str: str) -> str:
        """TestError: Test"""
        import re

        if 'AssertionError' in exec_str:
            match = re.search(r'AssertionError[:\s]*([^\n]+)', exec_str)
            if match:
                return f"AssertionError: {match.group(1)[:150]}"
            return "AssertionError: Output doesn't match expected result"

        error_patterns = [
            (r'NameError[:\s]*([^\n]+)', 'NameError'),
            (r'TypeError[:\s]*([^\n]+)', 'TypeError'),
            (r'IndexError[:\s]*([^\n]+)', 'IndexError'),
            (r'SyntaxError[:\s]*([^\n]+)', 'SyntaxError'),
            (r'ValueError[:\s]*([^\n]+)', 'ValueError'),
            (r'KeyError[:\s]*([^\n]+)', 'KeyError'),
        ]
        for pattern, error_type in error_patterns:
            match = re.search(pattern, exec_str)
            if match:
                return f"{error_type}: {match.group(1)[:150]}"

        if 'TEST_FAILED' in exec_str.upper():
            lines = exec_str.split('\n')
            for line in lines:
                if 'Error' in line or 'error' in line:
                    return line[:200]
            return exec_str[:200]
        return ""

    def _analyze_error_and_suggest(self, error_result) -> Optional[str]:
        """v19: Analyze error and provide specific fix suggestions

        Let model know how to modify prompt to help small model avoid common errors
        """
        if isinstance(error_result, dict):
            error_result = str(error_result.get('output', error_result.get('error', str(error_result))))
        error_result = str(error_result) if error_result else ""
        error_lower = error_result.lower()

        if 'test_failed' in error_lower:
            # Error info already in result, return fix suggestion directly
            if 'assertionerror' in error_lower:
                return "Test assertion failed. The code output doesn't match expected result. Use Revise to fix the logic."
            elif 'executionerror' in error_lower or 'error' in error_lower:
                return "Test execution error. Check the code for bugs and use Revise to fix."
            return "Test failed. Review the error message above and use Revise to fix the code."

        # 1. solve() function naming conflict
        if 'solve() takes 0 positional arguments' in error_result:
            return "The code defines 'def solve():' but also tries to call 'solve(x, y)'. Tell the Programmer to use 'sympy.solve()' or 'sp.solve()' for equation solving, NOT 'solve()'."

        # 2. General parameter count error
        if 'takes' in error_lower and 'positional arguments' in error_lower:
            return "Function argument mismatch. Tell the Programmer to check function signatures and use correct number of arguments."

        # 3. NameError - variable undefined
        if 'nameerror' in error_lower or 'is not defined' in error_lower:
            return "Variable or function not defined. Tell the Programmer to define all variables before using them, or import required modules."

        # 4. SyntaxError - syntax error
        if 'syntaxerror' in error_lower or 'syntax error' in error_lower:
            return "Python syntax error. Tell the Programmer to write valid Python code with correct indentation and syntax."

        # 5. ImportError / ModuleNotFoundError
        if 'importerror' in error_lower or 'modulenotfounderror' in error_lower:
            return (
                "Module import failed. If this is a Code task, prefer running tests in the official BigCodeBench/HumanEval "
                "Docker environment (COLAB_GRPO_CODE_EVAL_BACKEND=docker) so common libs are available; otherwise avoid "
                "uncommon third-party imports and stick to stdlib / widely-available packages (numpy/pandas/matplotlib/sympy)."
            )

        # 6. Timeout - timeout
        if 'timeout' in error_lower:
            return "Code execution timed out. Tell the Programmer to use more efficient algorithms or avoid infinite loops."

        # 7. IndexError / KeyError
        if 'indexerror' in error_lower or 'keyerror' in error_lower:
            return "Index or key access error. Tell the Programmer to check array bounds and dictionary keys before accessing."

        # 8. TypeError - type error
        if 'typeerror' in error_lower:
            return "Type mismatch error. Tell the Programmer to ensure correct data types for operations."

        # 9. RecursionError - recursion too deep
        if 'recursionerror' in error_lower or 'maximum recursion' in error_lower:
            return "Recursion too deep. Tell the Programmer to use iteration instead of recursion, or add proper base cases."

        # 10. ZeroDivisionError - division by zero
        if 'zerodivisionerror' in error_lower or 'division by zero' in error_lower:
            return "Division by zero. Tell the Programmer to add checks for zero before division."

        return None

    def _detect_verification_success(self, execution_result: Any, exec_str: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """VerifyDetect + VerifyResult: Detect large model verification success signal and extract structured verify results

        When gptoss120b Verify/Test operator returns success signal, give small model a hint (not definite conclusion).
        Small model should judge whether to finish itself, this only provides reference info

        Returns:
            Tuple[hint_info, verify_result_dict]
            - hint_info: Hint string, return None if no success signal detected
            - verify_result_dict: Structured verify data with keys:
                * is_correct: bool
                * confidence: str (high/medium/low)
                * answer: str (the correct answer from verifier)
                * error_analysis: Optional[str] (if verification failed)
        """
        exec_lower = exec_str.lower()
        verify_result = None

        # 1. Code task: TEST_PASSED signal - suggest Format instead of finish
        if 'test_passed' in exec_lower:
            return "Tests appear to pass. Consider adding Format to extract concise answer.", None

        # 2. Extract structured Verify operator result from dict
        if isinstance(execution_result, dict):
            # Try to extract structured verify information
            is_correct = execution_result.get('is_correct')
            if is_correct is not None:
                verify_result = {
                    'is_correct': is_correct,
                    'confidence': execution_result.get('confidence', 'unknown'),
                    'answer': execution_result.get('answer', ''),
                }
                if not is_correct:
                    verify_result['error_analysis'] = execution_result.get('error_analysis', 'No error details provided')

                hint = "Verify suggests the answer may be correct. Consider adding Format to extract concise answer." if is_correct else None
                return hint, verify_result

            # Check verification signal in output field (fallback)
            output = str(execution_result.get('output', ''))
            if 'is_correct' in output.lower() and 'true' in output.lower():
                return "Verification indicates positive result. Consider adding Format to extract concise answer.", None

        # 3. Try to parse structured verification from exec_str (XML format)
        import re
        is_correct_match = re.search(r'<is_correct>\s*(true|false)\s*</is_correct>', exec_str, re.IGNORECASE)
        confidence_match = re.search(r'<confidence>\s*(high|medium|low)\s*</confidence>', exec_str, re.IGNORECASE)
        answer_match = re.search(r'<answer>\s*(.+?)\s*</answer>', exec_str, re.DOTALL | re.IGNORECASE)

        if is_correct_match:
            is_correct = is_correct_match.group(1).lower() == 'true'
            verify_result = {
                'is_correct': is_correct,
                'confidence': confidence_match.group(1).lower() if confidence_match else 'unknown',
                'answer': answer_match.group(1).strip() if answer_match else '',
            }
            if not is_correct:
                # Try to extract error analysis
                error_match = re.search(r'\[Step 4\].*?Error Analysis.*?:\s*(.+?)(?=</verification_steps>|$)', exec_str, re.DOTALL | re.IGNORECASE)
                if error_match:
                    verify_result['error_analysis'] = error_match.group(1).strip()[:200]  # Limit to 200 chars

            hint = "Verify suggests the answer may be correct. Consider adding Format to extract concise answer." if is_correct else None
            return hint, verify_result

        # 4. Verification success signal in string - suggest Format instead of finish
        success_patterns = [
            ('is_correct>true', "Verify indicates positive. Consider adding Format to extract concise answer."),
            ('is_correct: true', "Verify indicates positive. Consider adding Format to extract concise answer."),
            ('verification: pass', "Verification suggests pass. Consider adding Format to extract concise answer."),
            ('verified: true', "Verification suggests correct. Consider adding Format to extract concise answer."),
            ('correct answer', "Result may be correct. Consider adding Format to extract concise answer."),
            ('answer is correct', "Result appears correct. Consider adding Format to extract concise answer."),
        ]
        for pattern, hint in success_patterns:
            if pattern in exec_lower:
                return hint, None

        # 5. Check verification result with high confidence - suggest Format instead of finish
        confidence_val_match = re.search(r'confidence[:\s]*([0-9.]+)', exec_lower)
        if confidence_val_match:
            try:
                confidence = float(confidence_val_match.group(1))
                if confidence >= 0.9:
                    return f"High confidence ({confidence:.0%}) from verifier. Consider adding Format to extract concise answer.", None
            except ValueError:
                pass

        return None, verify_result

    def _format_feedback(
        self,
        success: bool,
        message: str,
        node_id: Optional[str] = None,
        hint: Optional[str] = None,
        execution_result: Optional[str] = None,
        force_finish: bool = False,
        final: bool = False,
        verify_result: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Format feedback text

        Format:
        <feedback>
        [Status]: Success/Failed
        [Message]: ...
        [Current DSL]: ...
        [Workflow Nodes]: ... (Node list, so model knows which can be deleted/modified)
        [Statistics]: ...
        [Execution Result]: ... (If available)
        [Verify Result]: ... (VerifyResult: If Verify operator was executed)
        </feedback>
        """
        lines = ["<feedback>"]

        # State
        status = "Success" if success else "Failed"
        lines.append(f"[Status]: {status}")

        # Message
        lines.append(f"[Message]: {message}")

        # Node ID (newly added node)
        if node_id:
            lines.append(f"[Node ID]: {node_id}")

        # Current DSL
        dsl = self.graph.to_dsl() or "(empty)"
        lines.append(f"[Current DSL]: {dsl}")

        # ====== New: node list - let model know which nodes can be deleted/modified ======
        node_list = self.graph.get_node_list()
        if node_list:
            node_descriptions = []
            for node in node_list:
                # Format: node_1: Plan (operator), node_2: [A, B] (parallel)
                node_descriptions.append(f"{node['id']}: {node['dsl']} ({node['type']})")
            lines.append(f"[Workflow Nodes]: {'; '.join(node_descriptions)}")
        else:
            lines.append("[Workflow Nodes]: (none)")

        # Statistics
        stats = self.graph.get_statistics()
        lines.append(f"[Statistics]: {stats['total_operators']} operators, {stats['unique_types']} unique types")

        # Structure info
        structures = []
        if stats['has_parallel']:
            structures.append("parallel")
        if stats['has_conditional']:
            structures.append("conditional")
        if stats['has_loop']:
            structures.append("loop")
        if structures:
            lines.append(f"[Structures]: {', '.join(structures)}")

        memory_summary = self.memory.get_full_summary()
        if memory_summary:
            lines.append(memory_summary)
            print(f"[Memory Memory] Actions: {len(self.memory.actions)}, Errors: {len(self.memory.errors)}", flush=True)

        # Execution result
        if execution_result:
            if isinstance(execution_result, dict):
                exec_str = str(execution_result.get('output', execution_result))
            else:
                exec_str = str(execution_result)

            # Truncate too long execution result
            if len(exec_str) > 500:
                exec_str = exec_str[:500] + "..."

            # ====== Key change: show model original problem+answer, let it judge itself ======
            # Truncate problem display (keep concise)
            problem_preview = self.problem
            if len(problem_preview) > 300:
                problem_preview = problem_preview[:300] + "..."
            lines.append(f"[Original Problem]: {problem_preview}")
            lines.append(f"[Execution Result]: {exec_str}")

            # Judge if should finish (based on execution success and operators count)
            stats = self.graph.get_statistics()
            total_ops = stats['total_operators']

            # Check if error result
            is_error = any(err in exec_str.upper() for err in
                          ['ERROR', 'EXCEPTION', 'FAILED', 'TIMEOUT', 'WORKFLOW_ERROR'])

            if is_error:
                # Execution failed, let model continue trying
                lines.append("[Next Step]: Execution failed. Consider adding/modifying operators or prompts to fix the workflow.")
            elif total_ops >= 1:
                dsl = self.graph.to_dsl() or ''
                last_op_is_format = dsl.rstrip().endswith('Format')

                if last_op_is_format:
                    lines.append(f"[HINT]: Format extracted concise answer. Consider finishing if satisfied.")
                else:
                    finish_hint, extracted_verify = self._detect_verification_success(execution_result, exec_str)
                    # Merge extracted verify_result with passed one (passed one takes precedence)
                    if extracted_verify and not verify_result:
                        verify_result = extracted_verify
                    if finish_hint:
                        lines.append(f"[HINT]: {finish_hint}")
                # Always let model judge itself
                lines.append("[Next Step]: Compare the problem and result above. If it answers the problem, you may finish; otherwise continue improving.")

        # Hint
        if hint:
            lines.append(f"[Hint]: {hint}")

        if verify_result:
            lines.append("")
            lines.append("------------------------------------------")
            lines.append("[Verify Result]: Verification Analysis")
            lines.append("------------------------------------------")
            lines.append(f"  Correctness: {verify_result['is_correct']}")
            lines.append(f"  Confidence: {verify_result.get('confidence', 'unknown')}")

            if not verify_result['is_correct']:
                lines.append("")
                lines.append("  [WARNING] VERIFICATION FAILED - Answer appears incorrect")
                if 'error_analysis' in verify_result:
                    lines.append(f"  [ERROR] {verify_result['error_analysis'][:150]}")
                lines.append("")
                lines.append("  [FIX NOW] Add Revise to fix the current error:")
                lines.append("  <action>add</action><operator>Revise</operator>")
                lines.append("")
                lines.append("  [LEARN] For NEXT problem, use self-correcting structure from the start:")
                lines.append("  Plan -> Programmer -> (Verify ? Revise : done) -> Format")
                lines.append("  XML: <action>add</action><structure>conditional</structure><condition>Verify</condition><true>Revise</true><false>done</false>")
                lines.append("")
            else:
                lines.append("  Verification passed - answer appears correct")
                if 'answer' in verify_result and verify_result['answer']:
                    answer_preview = str(verify_result['answer'])[:100]
                    lines.append(f"  Verified Answer: {answer_preview}")
            lines.append("------------------------------------------")

        # Round info
        lines.append(f"[Round]: {self.round_count}/{self.max_rounds}")

        # Termination marker
        if final:
            lines.append("[Final]: Workflow construction complete")
        elif force_finish:
            lines.append("[Final]: Auto-finished due to max rounds")

        lines.append("</feedback>")

        return "\n".join(lines)

    def stop(self, raw_response: str) -> bool:
        """Determine if should stop

        Args:
            raw_response: Model output

        Returns:
            True means should stop
        """
        if self.is_finished:
            return True

        if self.round_count >= self.max_rounds:
            return True

        # Check if contains finish action
        action = self.parser.parse(raw_response)
        if action.action_type == ActionType.FINISH:
            return True

        return False

    def get_dsl(self) -> str:
        """Get current DSL"""
        return self.graph.to_dsl()

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics"""
        return self.graph.get_statistics()

    def get_initial_state(self) -> str:
        """Get initial state description"""
        available_ops = self._get_available_operators()
        return f"""You are building a workflow to solve this problem:

Problem: {self.problem}

The workflow is currently empty. Start by adding operators.
Available operators: {', '.join(sorted(available_ops))}

Use XML commands to build the workflow step by step.
When you think the workflow is complete, use <action>finish</action>.
"""

    def get_current_state(self) -> str:
        """Get current state description"""
        if self.graph.is_empty():
            return self.get_initial_state()

        stats = self.graph.get_statistics()
        dsl = self.graph.to_dsl()

        state = f"""Current workflow state:
DSL: {dsl}
Operators: {stats['total_operators']} total, {stats['unique_types']} unique types
"""

        if self.last_execution_result:
            if isinstance(self.last_execution_result, dict):
                result_str = str(self.last_execution_result.get('output', self.last_execution_result))
            else:
                result_str = str(self.last_execution_result)

            result_preview = result_str[:200]
            if len(result_str) > 200:
                result_preview += "..."
            state += f"Last execution result: {result_preview}\n"

        state += f"\nRound: {self.round_count}/{self.max_rounds}"

        return state

    def get_history(self) -> List[Dict[str, Any]]:
        """Get interaction history"""
        return self.history.copy()

    @property
    def system_prompt(self) -> str:
        """Get system prompt - emphasize complex workflow version"""
        available_ops = self._get_available_operators()
        ops_count = len(available_ops)

        # Build operator descriptions (only for available operators)
        op_descriptions = []
        op_desc_map = {
            "Programmer": "Execute Python code (MATH: use solve(); CODE: match function signature)",
            "Plan": "Create solution strategy (often useful early)",
            "Custom": "Natural language reasoning (QA, analysis, explanations)",
            "Decompose": "Break into sub-problems (for multi-part problems)",
            "Test": "Run test cases (after code generation)",
            "Review": "Evaluate quality (when uncertain)",
            "Verify": "Double-check logic (when no code to test)",
            "Revise": "Fix issues (after Review/Verify finds problems)",
            "ScEnsemble": "Vote among solutions (for consensus)",
            "Aggregate": "Merge parallel outputs",
            "AnswerGenerate": "Format final answer",
        }
        for op in sorted(available_ops):
            if op in op_desc_map:
                op_descriptions.append(f"- **{op}**: {op_desc_map[op]}")
        ops_section = "\n".join(op_descriptions)

        mc_warning = ""
        if "Programmer" in self.disabled_operators:
            mc_warning = """\n**IMPORTANT - MULTIPLE CHOICE MODE**:
- This is a multiple-choice problem. Do NOT use Programmer operator.
- Use **Custom** for step-by-step reasoning and analysis.
- Use **AnswerGenerate** to select the correct option (a, b, c, d, or e).
- Focus on logical reasoning, not code execution.\n"""

        return f'''You are building a robust, multi-stage workflow to solve problems reliably.

## Problem
{self.problem}

In each turn, output EXACTLY ONE XML action (add/delete/modify/set_prompt/finish or a structure add).

**GOAL**: Build a COMPLEX, RELIABLE workflow with multiple verification stages.

**COMPLEXITY REQUIREMENTS (CRITICAL)**:
- Your workflow MUST have at least 5 operators minimum
- Your workflow MUST include at least ONE structure: parallel, conditional, or loop
- You can combine multiple structures (e.g., parallel + conditional, loop + parallel)
- Simple 2-3 operator workflows are NOT acceptable - they lack robustness

**FINISH POLICY**:
- Do not finish with fewer than 5 operators
- Before finishing, include at least one CHECKER: Verify, Test, or Review
- After Plan/Decompose, add solvers AND checkers before finishing

**STRUCTURE GUIDANCE**:
- Use **parallel** for multiple solving approaches → then Aggregate/ScEnsemble
- Use **conditional** for verification-then-fix patterns → if fail, Revise
- Use **loop** for iterative refinement → repeat until quality improves
- You can combine structures, e.g., parallel → conditional → loop

**CRITICAL**: If you use <think>...</think>, you MUST output an <action> tag AFTER it.
{mc_warning}
## Available Operators ({ops_count} total) - USE ONLY THESE!

{ops_section}

## TASK-SPECIFIC GUIDANCE (CRITICAL!)
{get_problem_type_hint(self.problem_type)}

## Actions

- **add**: <action>add</action><operator>NAME</operator>
- **finish**: <action>finish</action> (optional: <answer>YOUR_ANSWER</answer>)
- **parallel**: <action>add</action><structure>parallel</structure><operators>A,B,C</operators>
- **conditional**: <action>add</action><structure>conditional</structure><condition>Review</condition><true>Revise</true><false>done</false>
- **loop**: <action>add</action><structure>loop</structure><operators>A,B</operators><count>n</count>
- **delete**: <action>delete</action><target>node_ID</target>
- **set_prompt**: <action>set_prompt</action><target>node_ID</target><prompt>YOUR PROMPT</prompt>

STOP IMMEDIATELY after the closing tag!

## Example Workflows (COMPLEX patterns - follow these!)

Example 1 - Parallel + Conditional (6 ops):
<action>add</action><operator>Plan</operator>
<action>add</action><structure>parallel</structure><operators>Programmer,Custom,ScEnsemble</operators>
<action>add</action><operator>Aggregate</operator>
<action>add</action><structure>conditional</structure><condition>Verify</condition><true>Revise</true><false>done</false>
<action>add</action><operator>Review</operator>
<action>add</action><operator>AnswerGenerate</operator>
<action>finish</action><answer>result</answer>

Example 2 - Double conditional verification (8 ops):
<action>add</action><operator>Decompose</operator>
<action>add</action><operator>Custom</operator>
<action>add</action><operator>Programmer</operator>
<action>add</action><structure>conditional</structure><condition>Test</condition><true>Revise</true><false>done</false>
<action>add</action><structure>conditional</structure><condition>Review</condition><true>Custom</true><false>done</false>
<action>add</action><operator>Verify</operator>
<action>add</action><operator>AnswerGenerate</operator>
<action>finish</action><answer>result</answer>

Example 3 - Loop with parallel (6 ops):
<action>add</action><operator>Plan</operator>
<action>add</action><structure>parallel</structure><operators>Programmer,Custom</operators>
<action>add</action><operator>ScEnsemble</operator>
<action>add</action><structure>loop</structure><operators>Review,Revise</operators><count>2</count>
<action>add</action><operator>Verify</operator>
<action>finish</action><answer>result</answer>

Example 4 - Code task with test loop (5 ops):
<action>add</action><operator>Plan</operator>
<action>add</action><operator>Programmer</operator>
<action>add</action><structure>loop</structure><operators>Test,Revise</operators><count>3</count>
<action>add</action><operator>Review</operator>
<action>finish</action><answer>def func(): ...</answer>

## Rules (MUST follow)
- Build workflows with 5+ operators and at least one structure
- Output ONLY XML tags, nothing else after the action
- ONE action per turn
- NO markdown code blocks
- Prefer combining multiple structures for robustness
'''

    @staticmethod
    def _is_multiple_choice_problem(problem: str) -> bool:
        """Detect if problem is multiple choice (MultiChoice)"""
        import re
        mc_patterns = [
            r'\ba\s*\)\s*[\d\w]',  # a ) 5
            r'\(\s*a\s*\)',         # ( a )
            r'options\s*:',         # options:
            r'choose.*(?:a|b|c|d|e)',  # choose from a, b, c, d, e
            r'\ba\s*\.',            # a.
        ]
        problem_lower = problem.lower()
        return any(re.search(p, problem_lower) for p in mc_patterns)

    def _get_available_operators(self) -> set:
        """Get available operators (excluding disabled ones) (MultiChoice)"""
        return VALID_OPERATORS - self.disabled_operators

    def batch_step(self, raw_responses: List[str]) -> Tuple[List[str], List[List[bool]], List[bool]]:
        """Batch execute step

        Note: This creates multiple independent environment copies for processing

        Args:
            raw_responses: Multiple model outputs

        Returns:
            (feedbacks, success_lists, actives)
        """
        feedbacks = []
        success_lists = []
        actives = []

        for response in raw_responses:
            feedback, success_list, active = self.step(response)
            feedbacks.append(feedback)
            success_lists.append(success_list)
            actives.append(active)

        return feedbacks, success_lists, actives


# ============================================
# Factory function
# ============================================

def create_env(
    problem: str,
    problem_type: str = "math",
    executor: Optional[Callable[[str, str], Tuple[str, bool]]] = None,
    executor_kwargs: Optional[Dict[str, Any]] = None,
    max_rounds: int = 20,
    execute_each_step: bool = True,
    finish_min_total_operators: int = 1,
    finish_require_checker: bool = False,
    finish_require_structure: bool = False,
    use_custom_prompts_in_execution: bool = True,
) -> InteractiveWorkflowEnv:
    """Create workflow environment

    Args:
        problem: Problem text
        executor: Executor function
        max_rounds: Max rounds
        execute_each_step: Whether to execute each step

    Returns:
        InteractiveWorkflowEnv Instance
    """
    return InteractiveWorkflowEnv(
        problem=problem,
        problem_type=problem_type,
        executor=executor,
        executor_kwargs=executor_kwargs,
        max_rounds=max_rounds,
        execute_each_step=execute_each_step,
        finish_min_total_operators=finish_min_total_operators,
        finish_require_checker=finish_require_checker,
        finish_require_structure=finish_require_structure,
        use_custom_prompts_in_execution=use_custom_prompts_in_execution,
    )


if __name__ == "__main__":
    # Test code
    print("=" * 60)
    print("InteractiveWorkflowEnv test")
    print("=" * 60)

    # Create environment
    env = InteractiveWorkflowEnv(
        problem="What is 15 * 23?",
        executor=None,  # Not executing yet
        max_rounds=10,
        execute_each_step=False,
    )

    # Print system prompt
    print("\n1. System prompt (first 500 chars):")
    print(env.system_prompt[:500] + "...")

    # Simulate interaction
    print("\n2. Simulate interaction:")

    # Add Plan
    response1 = "<action>add</action><operator>Plan</operator>"
    feedback1, success1, active1 = env.step(response1)
    print(f"\nInput: {response1}")
    print(f"Feedback:\n{feedback1}")
    print(f"Success: {success1}, Continue: {active1}")

    # Add Programmer
    response2 = "<action>add</action><operator>Programmer</operator>"
    feedback2, success2, active2 = env.step(response2)
    print(f"\nInput: {response2}")
    print(f"Feedback:\n{feedback2}")
    print(f"Success: {success2}, Continue: {active2}")

    # Add parallel structure
    response3 = "<action>add</action><structure>parallel</structure><operators>Custom, Custom</operators>"
    feedback3, success3, active3 = env.step(response3)
    print(f"\nInput: {response3}")
    print(f"Feedback:\n{feedback3}")

    # Add conditional structure
    response4 = "<action>add</action><structure>conditional</structure><condition>Review</condition><true>Revise</true><false>done</false>"
    feedback4, success4, active4 = env.step(response4)
    print(f"\nInput: {response4}")
    print(f"Feedback:\n{feedback4}")

    # End
    response5 = "<action>finish</action>"
    feedback5, success5, active5 = env.step(response5)
    print(f"\nInput: {response5}")
    print(f"Feedback:\n{feedback5}")
    print(f"Success: {success5}, Continue: {active5}")

    # Get final DSL
    print(f"\n3. Final DSL: {env.get_dsl()}")

    # Get statistics
    print(f"\n4. Statistics: {env.get_statistics()}")

    # Get history
    print(f"\n5. Interaction rounds: {len(env.get_history())}")

    print("\n" + "=" * 60)
    print("Test complete!")
