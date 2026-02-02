#!/usr/bin/env python3
"""
AFlow - RL
"""
import sys
import os
import tempfile
import importlib.util
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import asyncio
import time

try:
    from .workflow_validator import WorkflowValidator
    from .response_standardizer import ResponseStandardizer
    from .sympy_code_fixer import SymPyCodeFixer
    from .code_execution import extract_python_code
except ImportError:
    from workflow_validator import WorkflowValidator
    from response_standardizer import ResponseStandardizer
    from sympy_code_fixer import SymPyCodeFixer
    from code_execution import extract_python_code

import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.async_llm import create_llm_instance, LLMsConfig
from scripts import operators as operator_module

class AFlowExecutor:
    """RLAFlow"""

    def __init__(
        self,
        llm_config_path: str = "config/aflow_llm.yaml",
        llm_model_name: str = "gpt-oss-120b",
        timeout: int = 300,
        operator_enhancer: Optional[Any] = None,
        enable_fallback: bool = True
    ):
        """
        Args:
            llm_config_path: AFlow LLM
            llm_model_name: LLM
            timeout: 
            operator_enhancer: Layer 2 operator
            enable_fallback: Fallback
        """
        self.llm_config_path = Path(llm_config_path)
        self.llm_model_name = llm_model_name
        self.timeout = timeout
        self.operator_enhancer = operator_enhancer
        self.enable_fallback = enable_fallback
        self.validator = WorkflowValidator()
        self.standardizer = ResponseStandardizer()
        self.sympy_fixer = SymPyCodeFixer()

        self.checkpoints: List[Dict[str, Any]] = []
        self.enable_checkpoints = True

        self._load_llm_config()

        print(f"✅ AFlow")
        print(f"  LLM: {llm_model_name}")
        print(f"  : {timeout}")
        print(f"  XML: ")
        if operator_enhancer is not None:
            print(f"  Layer 2: ")

    def _load_llm_config(self):
        """LLM"""
        try:
            abs_config_path = self.llm_config_path.absolute()

            from scripts.async_llm import LLMsConfig
            self.llm_configs = LLMsConfig.from_yaml(str(abs_config_path))

            import os
            for model_name, config in self.llm_configs.models.items():
                base_url = getattr(config, 'base_url', '')
                if 'localhost' in str(base_url) or '127.0.0.1' in str(base_url):
                    os.environ['NO_PROXY'] = 'localhost,127.0.0.1'
                    os.environ['no_proxy'] = 'localhost,127.0.0.1'
                    print("  📌  NO_PROXY=localhost,127.0.0.1 (vLLM)")
                    break

            print(f"✅ LLM: {abs_config_path}")

        except Exception as e:
            print(f"⚠️  LLM: {e}")
            import traceback
            traceback.print_exc()
            from scripts.async_llm import LLMsConfig, LLMConfig
            import os
            api_key = os.environ.get('OPENAI_API_KEY', '')
            if api_key:
                print(f"   OPENAI_API_KEY ")
                default_config = LLMConfig(
                    api_type='openai',
                    base_url='https://api.openai.com/v1',
                    api_key=api_key,
                    model_name='gpt-4o-mini',
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=4096
                )
                self.llm_configs = LLMsConfig(models={'gpt-4o-mini': default_config})
                print(f"✅ LLM")
            else:
                print(f"  ⚠️  OPENAI_API_KEY ")
                self.llm_configs = None

    def validate_operator_output(self, output: Any, operator_name: str) -> Dict:
        """
        ResponseStandardizer

        Args:
            output: 
            operator_name: 

        Returns:
        """
        standardized = self.standardizer.standardize(output, operator_name)

        if isinstance(output, dict):
            result = output.copy()
            result.update({
                '__standardized__': standardized,
                'response': standardized['content'],
                'success': standardized['success'],
                'error': standardized.get('error')
            })
            return result
        else:
            return standardized

    def _preprocess_workflow_code(self, workflow_input: str) -> Tuple[str, str]:
        """
         - XMLPython

        Args:
            workflow_input: XMLPython

        Returns:
            (graph_code, prompt_code) - PythonTASK_PROMPT
        """
        import re

        if '<graph>' in workflow_input and '</graph>' in workflow_input:
            print(f"  📝 XML...")
            return self._extract_from_xml(workflow_input)

        if '<workflow>' in workflow_input and '</workflow>' in workflow_input:
            print(f"  📝 <workflow>XML...")
            return self._extract_from_xml(workflow_input)

        return workflow_input, ""

    def _extract_from_xml(self, xml_input: str) -> Tuple[str, str]:
        """
        XMLgraphprompt

        Args:
            xml_input: XML

        Returns:
            (graph_code, prompt_code)
        """
        import re

        graph_code = ""
        prompt_code = ""

        graph_match = re.search(r'<graph>\s*([\s\S]*?)\s*</graph>', xml_input)
        if graph_match:
            graph_code = graph_match.group(1).strip()
            print(f"  ✅ <graph>: {len(graph_code)}")

        prompt_match = re.search(r'<prompt>\s*([\s\S]*?)\s*</prompt>', xml_input)
        if prompt_match:
            prompt_code = prompt_match.group(1).strip()
            print(f"  ✅ <prompt>TASK_PROMPT: {len(prompt_code)}")

        if not graph_code:
            print(f"  ⚠️ <graph>Python")
            cleaned = re.sub(r'</?workflow>', '', xml_input)
            cleaned = re.sub(r'</?graph>', '', cleaned)
            cleaned = re.sub(r'<prompt>.*?</prompt>', '', cleaned, flags=re.DOTALL)
            if 'class Workflow' in cleaned:
                graph_code = cleaned.strip()

        if prompt_code and graph_code and 'TASK_PROMPT' not in graph_code:
            class_match = re.search(r'^class Workflow', graph_code, re.MULTILINE)
            if class_match:
                graph_code = prompt_code + "\n\n" + graph_code
                print(f"  📝 TASK_PROMPTgraph")

        return graph_code, prompt_code

    async def execute_workflow(
        self,
        workflow_code: str,
        problem: str,
        problem_type: str = "math",
        **kwargs
    ) -> Tuple[Any, float, Dict]:
        """

        Args:
            workflow_code: RLWorkflow
            problem: 
            problem_type: 
            **kwargs: entry_point for code

        Returns:
            (answer, cost, metadata)
        """

        start_time = time.time()
        plan_steps_list_captured = None
        sub_problems_list_captured = None

        self.clear_checkpoints()
        self._add_checkpoint("start", {"problem_type": problem_type, "problem_len": len(problem)})

        workflow_code, extracted_prompt = self._preprocess_workflow_code(workflow_code)
        if not workflow_code:
            print(f"⚠️  ")
            if self.enable_fallback:
                return await self._execute_fallback_workflow(
                    problem, problem_type,
                    error_info="Failed to extract workflow code from input",
                    **kwargs
                )
            else:
                raise ValueError("")

        sample_info = {
            "problem": problem,
            "problem_type": problem_type,
            "source": kwargs.get("source", ""),
            "context": kwargs.get("context", []),
        }
        formatted_problem = self._format_problem_by_source(problem, sample_info)
        if formatted_problem != problem:
            print(f"  📝  (source={sample_info['source']})")

        is_valid, msg, validation_details = self.validator.validate_workflow_code(workflow_code, problem_type)

        if not is_valid:
            print(f"⚠️  : {msg}")

            fixed_code = self.validator.fix_common_issues(workflow_code)
            is_valid, msg, _ = self.validator.validate_workflow_code(fixed_code, problem_type)

            if is_valid:
                print(f"✅ ")
                workflow_code = fixed_code
            elif self.enable_fallback:
                print(f"  ")
                return await self._execute_fallback_workflow(problem, problem_type, error_info=f"Validation failed: {msg}", **kwargs)
            else:
                raise ValueError(f"Fallback: {msg}")

        if problem_type == "code" or 'sympy' in workflow_code.lower():
            fixed_code, was_modified, fixes = self.sympy_fixer.fix_code(workflow_code)
            if was_modified:
                print(f"🔧 SymPy: {', '.join(fixes)}")
                workflow_code = fixed_code

        try:
            workflow_class = self._create_workflow_class(workflow_code, problem_type)
            self._add_checkpoint("workflow_created", {"class_name": workflow_class.__name__})

            llm_config = self._get_llm_config()

            if llm_config is None:
                print(f"⚠️  llm_config  None: {self.llm_model_name}")
                llm_config = self.llm_model_name

            try:
                workflow = workflow_class(
                    name="rl_generated_workflow",
                    llm_config=llm_config,
                    dataset=problem_type
                )
            except Exception as e:
                print(f"⚠️  : {e}")
                self._add_checkpoint("instantiation_failed", {"error": str(e)[:200]}, is_error=True)
                import traceback
                traceback.print_exc()
                return await self._execute_fallback_workflow(
                    problem, problem_type,
                    error_info=f"Workflow instantiation failed: {type(e).__name__}: {str(e)[:200]}",
                    **kwargs
                )

            # Optional node-level cache for incremental execution (Interactive GRPO execute_each_step).
            node_cache = kwargs.get("node_cache")
            call_kwargs: Dict[str, Any] = {}
            if node_cache is not None:
                try:
                    import inspect

                    sig = inspect.signature(workflow.__call__)
                    has_var_kwargs = any(
                        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
                    )
                    if has_var_kwargs or "node_cache" in sig.parameters:
                        call_kwargs["node_cache"] = node_cache
                except Exception:
                    # If we can't introspect, don't pass extra kwargs to avoid breaking older workflows.
                    pass

            if kwargs.get("test_type") is not None:
                call_kwargs["test_type"] = kwargs["test_type"]
            if kwargs.get("test_inputs") is not None:
                call_kwargs["test_inputs"] = kwargs["test_inputs"]
            if kwargs.get("test_outputs") is not None:
                call_kwargs["test_outputs"] = kwargs["test_outputs"]
            if kwargs.get("dataset") is not None:
                call_kwargs["dataset"] = kwargs["dataset"]

            # For code problems, try passing entry_point and test (HumanEval format)
            try:
                if problem_type == "code":
                    # Try full HumanEval format first (entry_point + test)
                    if "entry_point" in kwargs and "test" in kwargs:
                        try:
                            result = await asyncio.wait_for(
                                workflow(formatted_problem, kwargs["entry_point"], kwargs["test"], **call_kwargs),
                                timeout=self.timeout
                            )
                        except TypeError as e:
                            # Fallback to just entry_point
                            if "positional argument" in str(e) or "takes" in str(e):
                                print(f"  ⚠️  Workflowtestentry_point")
                                try:
                                    result = await asyncio.wait_for(
                                        workflow(formatted_problem, kwargs["entry_point"], **call_kwargs),
                                        timeout=self.timeout
                                    )
                                except TypeError:
                                    print(f"  ⚠️  Workflowentry_pointproblem")
                                    result = await asyncio.wait_for(
                                        workflow(formatted_problem, **call_kwargs),
                                        timeout=self.timeout
                                    )
                            else:
                                raise
                    elif "entry_point" in kwargs:
                        # Only entry_point available
                        try:
                            result = await asyncio.wait_for(
                                workflow(formatted_problem, kwargs["entry_point"], **call_kwargs),
                                timeout=self.timeout
                            )
                        except TypeError as e:
                            if "positional argument" in str(e):
                                print(f"  ⚠️  Workflowentry_pointproblem")
                                result = await asyncio.wait_for(
                                    workflow(formatted_problem, **call_kwargs),
                                    timeout=self.timeout
                                )
                            else:
                                raise
                    else:
                        # No extra parameters
                        result = await asyncio.wait_for(
                            workflow(formatted_problem, **call_kwargs),
                            timeout=self.timeout
                        )
                else:
                    result = await asyncio.wait_for(
                        workflow(formatted_problem, **call_kwargs),
                        timeout=self.timeout
                    )
            except Exception as e:
                print(f"  ❌ Workflow: {type(e).__name__}")
                print(f"     : {str(e)}")
                self._add_checkpoint("execution_failed", {"error_type": type(e).__name__, "error": str(e)[:200]}, is_error=True)
                import traceback
                print(f"  :")
                traceback.print_exc()

                if self.enable_fallback:
                    print(f"  🔄 ")
                    return await self._execute_fallback_workflow(
                        problem, problem_type,
                        error_info=f"Execution failed: {type(e).__name__}: {str(e)[:200]}",
                        **kwargs
                    )
                else:
                    print(f"  ⚠️  Fallback")
                    raise

            if isinstance(result, tuple):
                if len(result) >= 2:
                    answer, cost = result[0], result[1]

                    if not isinstance(cost, (int, float)):
                        print(f"  : cost ({type(cost).__name__})...")
                        if isinstance(answer, (int, float)) and isinstance(cost, str):
                            print(f"  answercost...")
                            answer, cost = cost, answer
                        else:
                            print(f"  cost0.0")
                            if len(str(cost)) <= 100:
                                print(f"     cost: {cost}")
                            else:
                                print(f"     cost: {str(cost)[:100]}...")
                            cost = 0.0

                elif len(result) == 1:
                    answer, cost = result[0], 0.0
                else:
                    answer, cost = None, 0.0
            else:
                answer, cost = result, 0.0


            if isinstance(answer, tuple):
                print(f"  🔧  tuple : {answer}")
                items = list(answer)
                if len(items) == 1:
                    answer = str(items[0])
                else:
                    import re
                    from collections import Counter

                    def _to_str(x):
                        try:
                            return str(x).strip()
                        except Exception:
                            return repr(x)

                    def _is_number(s: str) -> bool:
                        return bool(re.fullmatch(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)", s)) or bool(re.fullmatch(r"[+-]?\d+/\d+", s))

                    strs = [_to_str(x) for x in items if x is not None]
                    nums = [s for s in strs if _is_number(s)]

                    def _majority_choose(cands: list[str]) -> str:
                        cnt = Counter(cands)
                        most = cnt.most_common()
                        top = most[0][1]
                        tops = [v for v, c in most if c == top]
                        if len(tops) == 1:
                            return tops[0]
                        for s in reversed(cands):
                            if s in tops:
                                return s
                        return cands[-1]

                    if nums:
                        chosen = _majority_choose(nums)
                    else:
                        nonempty = [s for s in strs if s]
                        chosen = _majority_choose(nonempty) if nonempty else ""
                    answer = chosen
                print(f"  ✅  tuple : {answer}")

            elif isinstance(answer, list):
                print(f"  🔧  list : {str(answer)[:100]}")
                items = list(answer)
                flat = []
                for x in items:
                    if isinstance(x, (list, tuple)) and len(x) == 1:
                        flat.append(x[0])
                    else:
                        flat.append(x)

                if len(flat) == 1:
                    answer = str(flat[0])
                elif len(flat) > 1:
                    import re
                    from collections import Counter

                    def _to_str(x):
                        try:
                            return str(x).strip()
                        except Exception:
                            return repr(x)

                    def _is_number(s: str) -> bool:
                        return bool(re.fullmatch(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)", s)) or bool(re.fullmatch(r"[+-]?\d+/\d+", s))

                    strs = [_to_str(x) for x in flat if x is not None]
                    nums = [s for s in strs if _is_number(s)]

                    def _majority_choose(cands: list[str]) -> str:
                        cnt = Counter(cands)
                        most = cnt.most_common()
                        top = most[0][1]
                        tops = [v for v, c in most if c == top]
                        if len(tops) == 1:
                            return tops[0]
                        for s in reversed(cands):
                            if s in tops:
                                return s
                        return cands[cands.__len__()-1]

                    if nums:
                        chosen = _majority_choose(nums)
                    else:
                        nonempty = [s for s in strs if s]
                        chosen = _majority_choose(nonempty) if nonempty else ""
                    answer = chosen
                else:
                    answer = ""
                print(f"  ✅  list : {answer}")

            elif isinstance(answer, dict):
                print(f"  🔧 ")
                print(f"     : {list(answer.keys())}")

                preserve_structured = (('steps_list' in answer and isinstance(answer.get('steps_list'), (list, tuple))) or ('sub_problems_list' in answer and isinstance(answer.get('sub_problems_list'), (list, tuple))))
                if preserve_structured:
                    print(f"  ✨ StructOutput: Plan/Decomposedict")

                if 'code' in answer and 'output' in answer and 'is_correct' not in answer:
                    code_val = answer.get('code', '')
                    if code_val and len(code_val) > 0:
                        _programmer_code_cache = code_val
                        print(f"  📦 CodeCache: Programmer ({len(_programmer_code_cache)} chars)")
                    else:
                        print(f"  ⚠️ EmptyCache: ")


                if 'is_correct' in answer and 'answer' in answer:
                    print(f"  ✅ VerifyOutput: Verify is_correct={answer.get('is_correct')} answer={str(answer.get('answer',''))[:50]}")

                elif 'review_result' in answer and 'feedback' in answer:
                    review_result = answer.get('review_result', False)
                    feedback = answer.get('feedback', '')
                    print(f"  ✅ ReviewOutput: Review review_result={review_result} feedback={str(feedback)[:50]}")

                elif 'output' in answer and answer['output']:
                    output_val = answer['output']

                    if isinstance(output_val, list) and len(output_val) > 0:
                        print(f"  🔧 ListExtract: output: {str(output_val[0])[:50]}...")
                        output_val = output_val[0]

                    if isinstance(output_val, str) and not any(ind in output_val for ind in ['def solve(', 'def main(', 'import ']):
                        print(f"  ✅  output : {str(output_val)[:50]}...")
                        if 'code' in answer:
                            print(f"  📦 CodeCache: dictcode")
                        else:
                            answer = str(output_val)
                    else:
                        print(f"  ⚠️ output ")
                        executed = self._execute_leaked_code(str(output_val))
                        if executed:
                            print(f"  ✅ : {executed}")
                            answer = executed
                        else:
                            answer = str(output_val)

                elif 'response' in answer and answer['response'] and not preserve_structured:
                    print(f"  ✅  response : {str(answer['response'])[:50]}...")
                    answer = str(answer['response'])

                elif 'solution' in answer:
                    solution = answer['solution']
                    print(f"  ⚠️  solution  (Test operator)")
                    if isinstance(solution, str) and '```' in solution:
                        solution = extract_python_code(solution)
                        print(f"  🔧 MarkdownClean: markdown: {solution[:50]}...")
                    if isinstance(solution, str) and any(ind in solution for ind in ['def ', 'import ', 'return ']):
                        print(f"  🔧 solution ")
                        executed = self._execute_leaked_code(solution)
                        if executed:
                            print(f"  ✅ : {executed}")
                            answer = executed
                        else:
                            if answer.get('result') == True:
                                print(f"  ⚠️  result=True")
                                import re
                                print_match = re.search(r'print\s*\(\s*["\']?([^)"\']*)["\'\s]*\)', solution)
                                if print_match:
                                    answer = print_match.group(1).strip()
                                else:
                                    answer = "Code execution failed"
                            else:
                                error_info = []
                                if answer.get('error_type'):
                                    error_info.append(f"Error: {answer['error_type']}")
                                if answer.get('error'):
                                    error_info.append(str(answer['error'])[:200])
                                if error_info:
                                    answer = f"TEST_FAILED: {' - '.join(error_info)}"
                                else:
                                    answer = "Code execution failed"
                    else:
                        if answer.get('result') == False and (answer.get('error') or answer.get('error_type')):
                            error_info = []
                            if answer.get('error_type'):
                                error_info.append(f"Error: {answer['error_type']}")
                            if answer.get('error'):
                                error_info.append(str(answer['error'])[:200])
                            answer = f"TEST_FAILED: {' - '.join(error_info)}"
                        else:
                            answer = str(solution)

                elif 'answer' in answer:
                    print(f"  ✅  answer : {str(answer['answer'])[:50]}...")
                    answer = str(answer['answer'])
                elif 'thought' in answer:
                    print(f"  ✅  thought : {str(answer['thought'])[:50]}...")
                    answer = str(answer['thought'])

                elif 'code' in answer:
                    code_content = answer['code']
                    if isinstance(code_content, str) and '```' in code_content:
                        code_content = extract_python_code(code_content)
                        print(f"  🔧 MarkdownClean: codemarkdown")
                    print(f"  ⚠️  code ")
                    executed = self._execute_leaked_code(code_content)
                    if executed:
                        print(f"  ✅ : {executed}")
                        answer = executed
                    else:
                        print(f"  ⚠️  code ")
                        answer = str(code_content)

                elif 'review_result' in answer:
                    print(f"  ⚠️ OperatorOutput:  Review operator ")
                    feedback = answer.get('feedback', '')
                    import re
                    answer_patterns = [
                        r'(?:the\s+)?(?:correct\s+)?answer\s+(?:is|should\s+be|=)\s*[:\s]*([^\.,\n]+)',
                        r'expected\s+(?:output|result|answer)\s*[:\s]*([^\.,\n]+)',
                        r'result\s*(?:is|=)\s*[:\s]*([^\.,\n]+)',
                    ]
                    extracted = None
                    for pattern in answer_patterns:
                        match = re.search(pattern, str(feedback), re.IGNORECASE)
                        if match:
                            extracted = match.group(1).strip()
                            break
                    if extracted:
                        print(f"  ✅ OperatorOutput:  Review feedback : {extracted[:50]}...")
                        answer = extracted
                    else:
                        answer = f"[REVIEW_OUTPUT] result={answer.get('review_result')}"
                        print(f"  ⚠️ OperatorOutput: Review ")

                elif 'is_correct' in answer:
                    print(f"  ⚠️ OperatorOutput:  Verify operator ")
                    if 'answer' in answer and answer['answer']:
                        print(f"  ✅ OperatorOutput:  Verify  answer ")
                        answer = str(answer['answer'])
                    else:
                        answer = f"[VERIFY_OUTPUT] is_correct={answer.get('is_correct')}"
                        print(f"  ⚠️ OperatorOutput: Verify  answer ")

                elif 'plan' in answer:
                    steps_list = answer.get('steps_list', [])
                    if steps_list and isinstance(steps_list, (list, tuple)) and len(steps_list) > 0:
                        print(f"  ✨ StructOutput: Plan operator  {len(steps_list)} dict")
                        plan_steps_list_captured = steps_list
                        pass
                    else:
                        print(f"  ✅ StructOutput: Plan operator steps_listplan")
                        answer = str(answer.get('plan', ''))

                elif 'sub_problems' in answer or 'sub_problems_list' in answer:
                    sub_problems = answer.get('sub_problems_list', answer.get('sub_problems', []))
                    if sub_problems and isinstance(sub_problems, (list, tuple)) and len(sub_problems) > 0:
                        print(f"  ✨ StructOutput: Decompose operator  {len(sub_problems)} dict")
                        pass
                    else:
                        print(f"  ✅ StructOutput: Decompose sub_problems_list")
                        answer = str(answer.get('sub_problems', ''))

                elif not preserve_structured:
                    print(f"  ⚠️ ")
                    answer = str(answer)

            if not isinstance(cost, (int, float)):
                print(f"  cost0.0")
                cost = 0.0

            execution_time = time.time() - start_time

            if answer is None or (isinstance(answer, str) and not answer.strip()):
                print(f"  ⚠️  (None)fallback")
                if self.enable_fallback:
                    return await self._execute_fallback_workflow(
                        problem, problem_type,
                        error_info="Empty answer returned",
                        **kwargs
                    )
                answer = ""

            if isinstance(answer, str):
                invalid_patterns = ['Based on the feedback', 'Revised Solution:', '```python\n```']
                for pattern in invalid_patterns:
                    if pattern in answer:
                        print(f"  ⚠️  : {pattern[:30]}")
                        answer = answer.replace(pattern, '').strip()

            if isinstance(answer, str) and problem_type in ['math', 'qa']:
                import re
                code_patterns = [
                    r'^\s*def\s+\w+\s*\(',
                    r'^\s*import\s+\w+',
                    r'^\s*from\s+\w+\s+import',
                    r'\ndef\s+\w+\s*\([^)]*\)\s*:',
                    r'if\s+__name__\s*==',            # if __name__ ==
                ]
                is_code_leak = any(re.search(p, answer) for p in code_patterns)
                if is_code_leak:
                    print(f"  🔴 ! answer")
                    print(f"     answer: {answer[:100]}...")

                    executed_answer = self._execute_leaked_code(answer)
                    if executed_answer:
                        print(f"  ✅ ! : {executed_answer}")
                        answer = executed_answer
                    else:
                        print(f"  ⚠️  fallback")
                        if self.enable_fallback:
                            return await self._execute_fallback_workflow(
                                problem, problem_type,
                                error_info="Code leakage detected: Programmer returned code instead of output",
                                **kwargs
                            )

            if isinstance(answer, str):
                import re
                if re.search(r'\\boxed\{\s*\}', answer):
                    print(f"  🔴 boxed")
                    answer = ""

            # if isinstance(answer, str) and self._needs_format(answer, problem_type):
            #     try:
            #         formatted_answer = await self._auto_format(formatted_problem, answer)
            #         if formatted_answer and len(formatted_answer.strip()) > 0:
            #             answer = formatted_answer
            #         else:
            #     except Exception as e:

            metadata = {
                "success": True,
                "execution_time": execution_time,
                "cost": cost,
                "problem_type": problem_type
            }

            if '_programmer_code_cache' in dir() and _programmer_code_cache:
                metadata['programmer_code'] = _programmer_code_cache
                print(f"  📦 CodeCache: Programmermetadata ({len(_programmer_code_cache)} chars)")
            elif isinstance(answer, dict) and 'code' in answer and answer.get('code'):
                metadata['programmer_code'] = answer.get('code')
                print(f"  📦 CodeCache: Programmermetadata ({len(answer.get('code', ''))} chars)")

            self._add_checkpoint("execution_success", {
                "answer_type": type(answer).__name__,
                "answer_len": len(str(answer)) if answer else 0,
                "cost": cost
            })

            print(f"\n{'─'*30}  {'─'*30}")
            print(f"  ⏱️  : {execution_time:.2f}s")
            print(f"  💰 API: ${cost:.4f}")
            answer_preview = str(answer)[:200] if answer else "None"
            print(f"  📤 : {answer_preview}{'...' if answer and len(str(answer)) > 200 else ''}")
            print(f"{'─'*70}", flush=True)

            metadata["checkpoints"] = self.get_checkpoint_summary()

            return answer, cost, metadata

        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            print(f"⏱️   ({self.timeout})")
            self._add_checkpoint("timeout", {"timeout_seconds": self.timeout}, is_error=True)

            metadata = {
                "success": False,
                "error": "timeout",
                "execution_time": execution_time,
                "cost": 0.0,
                "problem_type": problem_type,
                "checkpoints": self.get_checkpoint_summary()
            }

            return None, 0.0, metadata

        except Exception as e:
            execution_time = time.time() - start_time
            print(f"❌ : {str(e)}")
            self._add_checkpoint("exception", {"error": str(e)[:200]}, is_error=True)

            import traceback
            traceback.print_exc()

            metadata = {
                "success": False,
                "error": str(e),
                "execution_time": execution_time,
                "cost": 0.0,
                "problem_type": problem_type,
                "checkpoints": self.get_checkpoint_summary()
            }

            return None, 0.0, metadata

    def _execute_leaked_code(self, code_string: str) -> Optional[str]:
        """
        🔧 P0: 

         workflow  result['code']  result['output'] 

        Args:
            code_string:  Python  def solve(): ...

        Returns:
             None
        """
        import re

        try:
            code = code_string

            unicode_replacements = {
                '\u201c': '"',  # LEFT DOUBLE QUOTATION MARK
                '\u201d': '"',  # RIGHT DOUBLE QUOTATION MARK
                '\u2018': "'",  # LEFT SINGLE QUOTATION MARK
                '\u2019': "'",  # RIGHT SINGLE QUOTATION MARK
                '\u202f': ' ',  # NARROW NO-BREAK SPACE
                '\u00a0': ' ',  # NO-BREAK SPACE
                '\u2009': ' ',  # THIN SPACE
                '\u200b': '',   # ZERO WIDTH SPACE
                '\u2013': '-',  # EN DASH
                '\u2014': '-',  # EM DASH
            }
            for unicode_char, replacement in unicode_replacements.items():
                code = code.replace(unicode_char, replacement)

            boxed_match = re.search(r'\\boxed\{([^}]+(?:\{[^}]*\}[^}]*)*)\}', code)
            if boxed_match:
                code = boxed_match.group(1)

            code_block_match = re.search(r'```python\s*([\s\S]*?)```', code)
            if code_block_match:
                code = code_block_match.group(1)

            if 'def solve' not in code and 'def main' not in code:
                if 'return ' in code:
                    code = f"def solve():\n    " + code.replace('\n', '\n    ')

            import multiprocessing

            def _run_code_with_timeout(code_str, result_queue):
                """"""
                import io
                import sys
                # Ensure "__name__" is "__main__" so leaked code guarded by
                # `if __name__ == "__main__":` can run and print the answer.
                global_namespace = {
                    '__builtins__': __builtins__,
                    '__name__': '__main__',
                }

                try:
                    import math
                    global_namespace['math'] = math
                except:
                    pass

                try:
                    global_namespace['sys'] = sys
                except Exception:
                    pass

                old_stdout = sys.stdout
                sys.stdout = captured_output = io.StringIO()

                try:
                    exec(code_str, global_namespace)

                    stdout_content = captured_output.getvalue().strip()
                    if stdout_content:
                        lines = [l.strip() for l in stdout_content.split('\n') if l.strip()]
                        if lines:
                            result_queue.put(lines[-1])
                            return

                    for func_name in ['solve', 'main', 'answer']:
                        if func_name in global_namespace and callable(global_namespace[func_name]):
                            result = global_namespace[func_name]()
                            if result is not None:
                                result_queue.put(str(result))
                                return
                            break

                    stdout_content = captured_output.getvalue().strip()
                    if stdout_content:
                        lines = [l.strip() for l in stdout_content.split('\n') if l.strip()]
                        if lines:
                            result_queue.put(lines[-1])
                            return

                    result_queue.put(None)
                except Exception as e:
                    result_queue.put(None)
                finally:
                    sys.stdout = old_stdout

            result_queue = multiprocessing.Queue()
            process = multiprocessing.Process(
                target=_run_code_with_timeout,
                args=(code, result_queue)
            )

            try:
                process.start()
                process.join(timeout=30)

                if process.is_alive():
                    print(f"     ⏱️ Timeout: (30s)")
                    process.terminate()
                    process.join(timeout=0.5)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=0.5)
                    return None

                if not result_queue.empty():
                    return result_queue.get_nowait()
                return None

            except Exception as e:
                print(f"     : {e}")
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=0.5)
                return None
            finally:
                try:
                    result_queue.close()
                except:
                    pass

        except Exception as e:
            print(f"     _execute_leaked_code : {e}")
            return None

    def _create_workflow_class(self, workflow_code: str, problem_type: str):
        """Workflow"""
        import re

        print(f"  🔍  _create_workflow_class: {len(workflow_code)}", flush=True)

        operator_pattern = r'self\.(\w+)\s*=\s*operator\.(\w+)\('
        operators_found = re.findall(operator_pattern, workflow_code)
        if operators_found:
            op_list = [f"{name}({op_type})" for name, op_type in operators_found]
            print(f"  📦 Operators: {', '.join(op_list)}", flush=True)
        else:
            print(f"  📦 Operators:  (fallback)", flush=True)

        task_prompt_value = None
        task_prompt_match = re.search(
            r'TASK_PROMPT\s*=\s*(?:"""([^"]*(?:"(?!"")|[^"])*)"""|"([^"]*)"|\'([^\']*)\')',
            workflow_code,
            re.DOTALL
        )
        if task_prompt_match:
            task_prompt_value = task_prompt_match.group(1) or task_prompt_match.group(2) or task_prompt_match.group(3)
            if task_prompt_value:
                print(f"  📝 TASK_PROMPT", flush=True)

        namespace = {
            "operator": operator_module,
            "create_llm_instance": create_llm_instance,
            "DatasetType": str,
            "__TASK_PROMPT__": task_prompt_value
        }

        modified_code = workflow_code.replace(
            f"import workspace.{problem_type}.workflows.template.operator as operator",
            "# operator already imported"
        )

        import ast

        allowed_imports = {
            'operator', 'workspace', 'scripts', 'asyncio', 'typing',
            'json', 're', 'math', 'collections', 'itertools', 'functools',
            'abc', 'copy', 'dataclasses', 'enum', 'inspect', 'os', 'sys',
            'time', 'traceback', 'types', 'warnings', 'random',
            'hashlib',
        }

        try:
            tree = ast.parse(modified_code)
            forbidden_imports = set()

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = alias.name.split('.')[0]
                        if module_name not in allowed_imports:
                            forbidden_imports.add(module_name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module_name = node.module.split('.')[0]
                        if module_name not in allowed_imports:
                            forbidden_imports.add(module_name)

            if forbidden_imports:
                print(f"  🚫 AST: {forbidden_imports}", flush=True)
                for mod in forbidden_imports:
                    import re as re_module
                    modified_code = re_module.sub(
                        rf'^(\s*)(import\s+{mod}[^\n]*)',
                        r'\1# [FILTERED] \2',
                        modified_code,
                        flags=re_module.MULTILINE
                    )
                    modified_code = re_module.sub(
                        rf'^(\s*)(from\s+{mod}[^\n]*)',
                        r'\1# [FILTERED] \2',
                        modified_code,
                        flags=re_module.MULTILINE
                    )
                print(f"  📝  {len(forbidden_imports)} ", flush=True)
        except SyntaxError as e:
            print(f"  ⚠️ AST: {e}", flush=True)
            lines = modified_code.split('\n')
            filtered_lines = []
            filtered_count = 0
            for line in lines:
                stripped = line.strip()
                if stripped.startswith('import ') or stripped.startswith('from '):
                    if stripped.startswith('import '):
                        module = stripped.split()[1].split('.')[0]
                    else:
                        module = stripped.split()[1].split('.')[0]
                    if module not in allowed_imports:
                        print(f"  🚫 : {stripped}", flush=True)
                        filtered_lines.append(f"# [FILTERED] {line}")
                        filtered_count += 1
                        continue
                filtered_lines.append(line)
            modified_code = '\n'.join(filtered_lines)
            if filtered_count > 0:
                print(f"  📝  {filtered_count} ", flush=True)

        modified_code = modified_code.replace("async_lll", "async_llm")
        modified_code = modified_code.replace("create_lll_instance", "create_llm_instance")

        import re
        modified_code = re.sub(r'\bself\.l{3,}m\b', 'self.llm', modified_code)
        modified_code = re.sub(r'\basync_l{3,}m\b', 'async_llm', modified_code)
        modified_code = re.sub(r'\bcreate_l{3,}m_instance\b', 'create_llm_instance', modified_code)

        import re
        lines = modified_code.split('\n')
        fixed_lines = []
        in_async_func = False
        indent_stack = []

        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('async def '):
                in_async_func = True
                indent = len(line) - len(line.lstrip())
                indent_stack.append(indent)
            elif indent_stack and stripped and not stripped.startswith('#'):
                current_indent = len(line) - len(line.lstrip())
                while indent_stack and current_indent <= indent_stack[-1]:
                    indent_stack.pop()
                if not indent_stack:
                    in_async_func = False

            if stripped.startswith('await ') and not in_async_func:
                print(f"  🔧 await: {stripped[:50]}...")
                indent = len(line) - len(line.lstrip())
                wrapper = f"{' ' * indent}# [AUTO-FIXED] Wrapped top-level await\n"
                wrapper += f"{' ' * indent}async def _auto_wrap_await():\n"
                wrapper += f"{' ' * (indent + 4)}return {stripped}\n"
                wrapper += f"{' ' * indent}_result = asyncio.get_event_loop().run_until_complete(_auto_wrap_await())"
                fixed_lines.append(wrapper)
                continue

            fixed_lines.append(line)

        modified_code = '\n'.join(fixed_lines)

        import re as regex_module
        invalid_type_patterns = [
            r'(Tuple|List|Dict|Set|Optional|Union)\.(\w+)',  # Tuple.QA -> Any
            r':\s*(QA|Math|Code)\b',  # : QA -> : Any
            r'->\s*(QA|Math|Code)\b',  # -> QA -> -> Any
        ]
        for pattern in invalid_type_patterns:
            if regex_module.search(pattern, modified_code):
                print(f"  🔧 P2:  {pattern[:30]}...")
                modified_code = regex_module.sub(pattern, r'Any', modified_code)

        if 'Any' in modified_code and 'from typing import' in modified_code:
            if ', Any' not in modified_code and 'Any,' not in modified_code and 'import Any' not in modified_code:
                modified_code = modified_code.replace('from typing import', 'from typing import Any, ')

        call_init_vars = '''
        result = None
        solution = None
        code = None
        answer = None
        prog_result = None
        review_result = None
        test_result = None
        revised = None
        cost = 0.0
        '''
        call_match = regex_module.search(r'(async def __call__\([^)]*\)[^:]*:)\s*\n', modified_code)
        if call_match:
            end_pos = call_match.end()
            next_line_match = regex_module.search(r'^([ \t]+)', modified_code[end_pos:], regex_module.MULTILINE)
            if next_line_match:
                base_indent = next_line_match.group(1)
                formatted_init = '\n'.join(base_indent + line.strip() for line in call_init_vars.strip().split('\n') if line.strip())
                modified_code = modified_code[:end_pos] + formatted_init + '\n' + modified_code[end_pos:]
                print(f"  🔧 P2: __call__")

        try:
            exec(modified_code, namespace)

            if "Workflow" not in namespace:
                raise ValueError("No Workflow class found in generated code")

            WorkflowClass = namespace["Workflow"]

            if task_prompt_value:
                class EnhancedWorkflow:
                    """TASK_PROMPT"""
                    _task_prompt = task_prompt_value
                    _original_class = WorkflowClass

                    def __init__(self, name: str, llm_config, dataset):
                        object.__setattr__(self, '_instance', self._original_class(name, llm_config, dataset))

                    async def __call__(self, problem: str, *args, **kwargs):
                        enhanced_problem = f"{self._task_prompt}\n\nProblem:\n{problem}"
                        result = await self._instance(enhanced_problem, *args, **kwargs)
                        return result

                    def __getattr__(self, name):
                        if name == '_instance':
                            raise AttributeError(f"'{type(self).__name__}' object has no attribute '_instance'")
                        return getattr(object.__getattribute__(self, '_instance'), name)

                print(f"  ✨ EnhancedWorkflowTASK_PROMPT")
                return EnhancedWorkflow

            return WorkflowClass

        except Exception as e:
            print(f"⚠️  : {e}")
            raise ValueError(f"Workflow code compilation failed: {type(e).__name__}: {str(e)[:200]}")

    def _get_llm_config(self):
        """LLM"""
        from scripts.async_llm import LLMsConfig, LLMConfig

        try:
            if self.llm_configs:
                result = self.llm_configs.models.get(self.llm_model_name)
            else:
                print(f"⚠️  llm_configs ")
                return self.llm_model_name

            if isinstance(result, LLMConfig):
                # Align underlying request timeout with workflow timeout.
                try:
                    result.timeout = float(self.timeout or 300)
                except Exception:
                    pass
                return result
            elif isinstance(result, dict):
                if 'base_urls' in result:
                    print(f"  🔄 BaseUrls: base_urls ({len(result['base_urls'])})")
                    cfg = dict(result)
                    cfg["timeout"] = float(self.timeout or 300)
                    return cfg
                print(f"⚠️  get()  dict LLMConfig")
                return LLMConfig(
                    api_type=result.get('api_type', 'openai'),
                    base_url=result.get('base_url', 'https://api.openai.com/v1'),
                    api_key=result.get('api_key', ''),
                    model_name=result.get('model_name', 'gpt-4o-mini'),
                    temperature=result.get('temperature', 0.0),
                    top_p=result.get('top_p', 1.0),
                    max_tokens=result.get('max_tokens', 4096),
                    timeout=float(self.timeout or 300),
                )
            elif isinstance(result, str):
                return result
            else:
                print(f"⚠️  : {type(result)}")
                return self.llm_model_name

        except Exception as e:
            print(f"⚠️  LLM: {e}")
            import traceback
            traceback.print_exc()
            print(f"  : {self.llm_model_name}")
            return self.llm_model_name

    def _format_problem_by_source(self, problem: str, sample: dict) -> str:
        """
        Option A: 

        
        - HotpotQA/SQuAD: contextproblem
        - HumanEval: docstring
        - GSM8K/MATH: problem

        Args:
            problem: 
            sample: sourcecontext

        Returns:
            
        """
        source = sample.get("source", "").lower()
        problem_type = sample.get("problem_type", "math")

        if source == "hotpotqa" or "hotpot" in source:
            context = sample.get("context", [])
            if context:
                context_str = ""
                if isinstance(context, list):
                    for item in context:
                        if isinstance(item, list) and len(item) >= 2:
                            title = item[0] if isinstance(item[0], str) else ""
                            paragraphs = item[1] if isinstance(item[1], list) else []
                            if paragraphs:
                                context_str += f"\n{title}:\n" + " ".join(paragraphs)
                        elif isinstance(item, str):
                            context_str += "\n" + item
                if context_str:
                    return f"Context:{context_str}\n\nQuestion: {problem}\n\nAnswer:"
            return f"Question: {problem}\n\nAnswer:"

        elif source == "squad" or "squad" in source:
            context = sample.get("context", "")
            if context and isinstance(context, str):
                return f"Context: {context}\n\nQuestion: {problem}\n\nAnswer:"
            return f"Question: {problem}\n\nAnswer:"

        elif source == "humaneval" or problem_type == "code":
            return problem

        elif source in ["gsm8k", "math"] or problem_type == "math":
            return problem

        elif problem_type == "qa":
            context = sample.get("context", "")
            if context:
                if isinstance(context, list):
                    context_str = ""
                    for item in context:
                        if isinstance(item, list) and len(item) >= 2:
                            title = item[0] if isinstance(item[0], str) else ""
                            paragraphs = item[1] if isinstance(item[1], list) else []
                            if paragraphs:
                                context_str += f"\n{title}:\n" + " ".join(paragraphs)
                        elif isinstance(item, str):
                            context_str += "\n" + item
                    if context_str:
                        return f"Context:{context_str}\n\nQuestion: {problem}\n\nAnswer:"
                elif isinstance(context, str) and context.strip():
                    return f"Context: {context}\n\nQuestion: {problem}\n\nAnswer:"
            return f"Question: {problem}\n\nPlease answer the question based on your knowledge. Answer:"

        return problem

    def _needs_format(self, answer: str, problem_type: str) -> bool:
        """

        :
        1.  [REVIEW_OUTPUT], [VERIFY_OUTPUT]
        2.  (>200)
        3. dict/json
        4. 
        """
        if not answer or len(answer.strip()) == 0:
            return False

        if '[REVIEW_OUTPUT]' in answer or '[VERIFY_OUTPUT]' in answer:
            return True

        if answer.strip().startswith('{') and answer.strip().endswith('}'):
            return True
        if "'review_result':" in answer or "'is_correct':" in answer:
            return True

        explanation_indicators = [
            'therefore', 'because', 'since', 'thus', 'hence',
            'step by step', 'first,', 'second,', 'finally,',
            'the answer is', 'we can see', 'this means',
            '', '', '', '', '', ''
        ]
        if len(answer) > 200:
            answer_lower = answer.lower()
            if any(ind in answer_lower for ind in explanation_indicators):
                return True

        lines = [l for l in answer.split('\n') if l.strip()]
        if len(lines) > 3:
            return True

        if problem_type == 'code':
            return False

        return False

    async def _auto_format(self, problem: str, solution: str) -> Optional[str]:
        """
        """
        try:
            llm_config = self._get_llm_config()
            if llm_config is None:
                return None

            llm = create_llm_instance(llm_config)

            format_op = operator_module.Format(llm, name="AutoFormat")

            result = await format_op(problem=problem, solution=solution)

            if isinstance(result, dict):
                return result.get('response', result.get('answer', str(result)))
            return str(result) if result else None

        except Exception as e:
            print(f"  ⚠️ AutoFormat: _auto_format: {e}")
            return None

    def _add_checkpoint(self, stage: str, data: Dict[str, Any], is_error: bool = False):
        """

        workflow:
        1. DSL
        2. 
        3. operator
        """
        if not self.enable_checkpoints:
            return

        checkpoint = {
            "stage": stage,
            "timestamp": time.time(),
            "is_error": is_error,
            "data": data
        }
        self.checkpoints.append(checkpoint)

        status = "❌" if is_error else "✓"
        data_preview = str(data)[:100] + "..." if len(str(data)) > 100 else str(data)
        print(f"  📍 Checkpoint Checkpoint [{stage}] {status}: {data_preview}")

    def get_checkpoints(self) -> List[Dict[str, Any]]:
        """"""
        return self.checkpoints.copy()

    def clear_checkpoints(self):
        """"""
        self.checkpoints.clear()

    def get_checkpoint_summary(self) -> Dict[str, Any]:
        """

        workflow
        """
        if not self.checkpoints:
            return {"total": 0, "errors": 0, "stages": []}

        errors = [cp for cp in self.checkpoints if cp.get("is_error")]
        stages = [cp["stage"] for cp in self.checkpoints]

        return {
            "total": len(self.checkpoints),
            "errors": len(errors),
            "error_stages": [e["stage"] for e in errors],
            "stages": stages,
            "first_error": errors[0] if errors else None
        }

    async def _execute_fallback_workflow(
        self,
        problem: str,
        problem_type: str,
        error_info: str = "",
        **kwargs
    ) -> Tuple[Any, float, Dict]:
        """
        Fallback - Qwen

        LLM
        Qwen
        """
        print(f"🔄 Fallback: ")
        start_time = time.time()
        execution_time = time.time() - start_time

        error_description = f"WORKFLOW_ERROR: {error_info}" if error_info else "WORKFLOW_ERROR: Execution failed"

        metadata = {
            "success": False,
            "fallback_used": True,
            "error": error_info or "workflow_execution_failed",
            "execution_time": execution_time,
            "cost": 0.0,
            "problem_type": problem_type,
            "is_error_feedback": True
        }

        print(f"  ⚠️ : {error_description[:100]}...")

        return error_description, 0.0, metadata

    def _get_fallback_workflow_class(self, problem_type: str):
        """

        fallback
        1. LLM
        2. None
        3. Test operator
        """

        class FallbackWorkflow:
            def __init__(self, name: str, llm_config, dataset):
                self.name = name
                self.dataset = dataset
                try:
                    self.llm = create_llm_instance(llm_config)
                except Exception as e:
                    print(f"⚠️  LLM: {e}")
                    self.llm = None

            async def __call__(self, problem: str, *args, **kwargs):
                """fallbackTest operator"""

                if self.llm is not None:
                    try:
                        print(f"  📝 Fallback: LLM")

                        if self.dataset == "code":
                            prompt = f"""Given the following coding problem, provide a Python solution.

Problem:
{problem}

Provide ONLY the Python function code, no explanations."""
                        else:
                            prompt = f"""Solve the following problem step by step and provide the final answer.

Problem:
{problem}

Provide the final answer clearly."""

                        answer = await self.llm(prompt)

                        usage = self.llm.get_usage_summary()
                        if isinstance(usage, dict) and "total_cost" in usage:
                            cost = usage["total_cost"]
                        else:
                            cost = 0.0

                        return answer, cost

                    except Exception as e:
                        print(f"  ⚠️  FallbackLLM: {e}")

                try:
                    print(f"  📝 Fallback: Custom operator")
                    custom = operator_module.Custom(self.llm)
                    result = await custom(
                        input=problem,
                        instruction="Generate a solution without requiring test validation."
                    )

                    if result and 'response' in result:
                        usage = self.llm.get_usage_summary()
                        if isinstance(usage, dict) and "total_cost" in usage:
                            cost = usage["total_cost"]
                        else:
                            cost = 0.0
                        return result['response'], cost

                except Exception as e:
                    print(f"  ⚠️  Fallback Custom operator: {e}")

                print(f"  ⚠️  fallback")
                placeholder = f"[Fallback placeholder for problem: {problem[:80]}...]"
                return placeholder, 0.0

        return FallbackWorkflow


async def test_executor():
    """AFlow"""
    print("\n" + "=" * 60)
    print("🧪 AFlow")
    print("=" * 60)

    executor = AFlowExecutor(
        llm_config_path="config/aflow_llm.yaml",
        llm_model_name="gpt-oss-120b",
        timeout=60
    )

    test_workflow_code = """
import workspace.math.workflows.template.operator as operator
from scripts.async_llm import create_llm_instance
from scripts.evaluator import DatasetType

class Workflow:
    def __init__(self, name: str, llm_config, dataset: DatasetType):
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)

    async def __call__(self, problem: str):
        solution = await self.custom(input=problem, instruction="Solve this problem step by step and provide the final answer.")
        return solution['response'], self.llm.get_usage_summary()["total_cost"]
"""

    test_problem = "What is 15 + 27?"

    print(f"\n📝 : {test_problem}")

    answer, cost, metadata = await executor.execute_workflow(
        workflow_code=test_workflow_code,
        problem=test_problem,
        problem_type="math"
    )

    print(f"\n✅ :")
    print(f"  : {metadata['success']}")
    print(f"  : {answer}")
    print(f"  : ${cost:.6f}")
    print(f"  : {metadata['execution_time']:.2f}")


if __name__ == "__main__":
    asyncio.run(test_executor())
