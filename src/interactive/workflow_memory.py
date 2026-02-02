"""
WorkflowMemory - workflow

1. 
2. 
3. 
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class ActionRecord:
    """"""
    step: int
    action_type: str  # add, delete, modify, set_prompt, finish
    operator: Optional[str] = None
    prompt_summary: Optional[str] = None
    result: Optional[str] = None  # success/failed
    error_msg: Optional[str] = None


@dataclass
class ErrorRecord:
    """"""
    step: int
    error_type: str  # TEST_FAILED, EXECUTION_ERROR, etc.
    error_msg: str
    operator: Optional[str] = None
    revision_count: int = 0


@dataclass
class StepExecutionResult:
    """step-by-step"""
    step: int
    operator: str
    input_context: str
    output: Any
    success: bool


class WorkflowMemory:
    """Workflow"""

    def __init__(self):
        self.actions: List[ActionRecord] = []
        self.errors: List[ErrorRecord] = []
        self.step_results: List[StepExecutionResult] = []
        self.current_error: Optional[ErrorRecord] = None

    def reset(self):
        """"""
        self.actions.clear()
        self.errors.clear()
        self.step_results.clear()
        self.current_error = None

    def add_action(
        self,
        step: int,
        action_type: str,
        operator: Optional[str] = None,
        prompt_summary: Optional[str] = None,
        result: str = "success",
        error_msg: Optional[str] = None
    ):
        """"""
        if prompt_summary and len(prompt_summary) > 50:
            prompt_summary = prompt_summary[:47] + "..."

        record = ActionRecord(
            step=step,
            action_type=action_type,
            operator=operator,
            prompt_summary=prompt_summary,
            result=result,
            error_msg=error_msg
        )
        self.actions.append(record)

        if error_msg and result == "failed":
            self.add_error(step, "EXECUTION_ERROR", error_msg, operator)

    def add_error(
        self,
        step: int,
        error_type: str,
        error_msg: str,
        operator: Optional[str] = None
    ):
        """"""
        if self.current_error and self._is_similar_error(error_msg, self.current_error.error_msg):
            self.current_error.revision_count += 1
        else:
            error = ErrorRecord(
                step=step,
                error_type=error_type,
                error_msg=error_msg[:200] if len(error_msg) > 200 else error_msg,
                operator=operator,
                revision_count=0
            )
            self.errors.append(error)
            self.current_error = error

    def mark_error_resolved(self):
        """"""
        self.current_error = None

    def add_step_result(
        self,
        step: int,
        operator: str,
        input_ctx: str,
        output: Any,
        success: bool
    ):
        """"""
        if input_ctx and len(input_ctx) > 200:
            input_ctx = input_ctx[:197] + "..."

        self.step_results.append(StepExecutionResult(
            step=step,
            operator=operator,
            input_context=input_ctx,
            output=output,
            success=success
        ))

    def get_previous_step_output(self, max_length: int = 500) -> Optional[str]:
        """"""
        if not self.step_results:
            return None

        last_result = self.step_results[-1]
        if not last_result.success:
            return None

        output = last_result.output
        if output is None:
            return None

        if isinstance(output, dict):
            output_str = str(output.get('output', output))
        else:
            output_str = str(output)

        if len(output_str) > max_length:
            output_str = output_str[:max_length - 3] + "..."

        return output_str

    def get_step_results_summary(self, max_steps: int = 5) -> str:
        """"""
        if not self.step_results:
            return ""

        recent = self.step_results[-max_steps:]
        lines = []
        for sr in recent:
            status = "OK" if sr.success else "FAIL"
            output_preview = str(sr.output)[:50] if sr.output else "(none)"
            lines.append(f"Step {sr.step} [{sr.operator}]: {status} - {output_preview}")

        return "\n".join(lines)

    def _is_similar_error(self, msg1: str, msg2: str) -> bool:
        """"""
        msg1_lower = msg1.lower()
        msg2_lower = msg2.lower()

        error_types = ['assertionerror', 'nameerror', 'typeerror', 'syntaxerror',
                       'indexerror', 'keyerror', 'valueerror', 'test_failed']

        for et in error_types:
            if et in msg1_lower and et in msg2_lower:
                return True
        return False

    def get_action_summary(self, max_actions: int = 10) -> str:
        """"""
        if not self.actions:
            return "(No previous actions)"

        recent = self.actions[-max_actions:]
        lines = []

        for a in recent:
            if a.action_type == "add":
                line = f"Step {a.step}: Added {a.operator}"
            elif a.action_type == "set_prompt":
                line = f"Step {a.step}: Set prompt for {a.operator}"
                if a.prompt_summary:
                    line += f" - \"{a.prompt_summary}\""
            elif a.action_type == "delete":
                line = f"Step {a.step}: Deleted {a.operator}"
            elif a.action_type == "modify":
                line = f"Step {a.step}: Modified {a.operator}"
            elif a.action_type == "finish":
                line = f"Step {a.step}: Finished"
            else:
                line = f"Step {a.step}: {a.action_type}"

            if a.result == "failed":
                line += f" [FAILED: {a.error_msg[:30] if a.error_msg else 'unknown'}]"

            lines.append(line)

        return "\n".join(lines)

    def get_error_context(self) -> Optional[str]:
        """"""
        if not self.current_error:
            return None

        err = self.current_error
        lines = [f"[Current Error]: {err.error_type}"]
        lines.append(f"[Error Message]: {err.error_msg}")

        if err.revision_count > 0:
            lines.append(f"[Revision Attempts]: {err.revision_count} (previous fixes didn't work)")
            if err.revision_count >= 2:
                lines.append("[Suggestion]: Try a different approach instead of similar fixes")

        return "\n".join(lines)

    def get_full_summary(self) -> str:
        """"""
        parts = []

        action_summary = self.get_action_summary()
        if action_summary != "(No previous actions)":
            parts.append(f"[Previous Actions]:\n{action_summary}")

        error_context = self.get_error_context()
        if error_context:
            parts.append(error_context)

        if len(self.errors) > 1:
            parts.append(f"[Total Errors]: {len(self.errors)} errors encountered")

        return "\n".join(parts) if parts else ""

    def to_dict(self) -> Dict[str, Any]:
        """"""
        return {
            "actions": [
                {
                    "step": a.step,
                    "action_type": a.action_type,
                    "operator": a.operator,
                    "prompt_summary": a.prompt_summary,
                    "result": a.result,
                    "error_msg": a.error_msg
                }
                for a in self.actions
            ],
            "errors": [
                {
                    "step": e.step,
                    "error_type": e.error_type,
                    "error_msg": e.error_msg,
                    "operator": e.operator,
                    "revision_count": e.revision_count
                }
                for e in self.errors
            ],
            "step_results": [
                {
                    "step": sr.step,
                    "operator": sr.operator,
                    "input_context": sr.input_context,
                    "output": str(sr.output)[:200] if sr.output else None,
                    "success": sr.success
                }
                for sr in self.step_results
            ],
            "current_error": self.current_error.error_msg if self.current_error else None
        }
