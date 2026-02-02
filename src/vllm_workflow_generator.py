#!/usr/bin/env python3
"""
vLLM - vLLM APIFallback: transformers
"""
import asyncio
from typing import Dict, List, Optional, Tuple
import json
import ast
from pathlib import Path

try:  # pragma: no cover
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:  # pragma: no cover
    from openai import AsyncOpenAI  # type: ignore
except Exception:  # pragma: no cover
    AsyncOpenAI = None  # type: ignore

try:  # pragma: no cover
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
except Exception:  # pragma: no cover
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore

class VLLMWorkflowGenerator:
    """vLLM API

    1. vLLM APIAsyncOpenAIvLLM
    2. TransformersFallbacktransformers
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8003/v1",
        api_key: str = "EMPTY",
        model_name: str = "/path/to/Qwen2.5-7B-Instruct",
        max_concurrent: int = 6,
        operator_descriptions_path: Optional[str] = None,
        config: Optional[Dict] = None,
        use_vllm_api: bool = False,
        device: str = "cuda:0"
    ):
        """
        Args:
            base_url: vLLM
            api_key: APIvLLM
            model_name: /
            max_concurrent: 
            operator_descriptions_path: AFlow
            config: 
            use_vllm_api: vLLM APIFalsetransformers
            device: transformers
        """
        self.model_name = model_name
        self.max_concurrent = max_concurrent
        self.config = config or {}
        self.use_vllm_api = use_vllm_api
        self.device = device

        self.operator_descriptions = self._load_operator_descriptions(operator_descriptions_path)

        if use_vllm_api:
            if AsyncOpenAI is None:
                raise ImportError("Missing dependency: `openai` (required for vLLM API mode).")
            self.client = AsyncOpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=300.0,
                max_retries=2
            )
            self.semaphore = asyncio.Semaphore(max_concurrent)
            print(f"✅ vLLMAPI")
            print(f"  : {base_url}")
            print(f"  : {max_concurrent}")
        else:
            self.model = None
            self.tokenizer = None
            self._generation_lock = asyncio.Lock()
            print(f"✅ workflowTransformers")
            print(f"  : {model_name}")
            print(f"  : {device}")
            print(f"  ⚠️  GPUCUDA")

    def _load_operator_descriptions(self, descriptions_path: Optional[str]) -> Dict:
        """AFlow"""
        if descriptions_path and Path(descriptions_path).exists():
            with open(descriptions_path, 'r') as f:
                return json.load(f)

        return {
            "Custom": {
                "description": "Generates anything based on customized input and instruction.",
                "interface": "custom(input: str, instruction: str) -> dict with key 'response'"
            },
            "AnswerGenerate": {
                "description": "Generates step-by-step reasoning with thought process and final answer.",
                "interface": "answer_generate(input: str) -> dict with keys 'thought' and 'answer'"
            },
            "CustomCodeGenerate": {
                "description": "Generates code based on customized input and instruction.",
                "interface": "custom_code_generate(problem: str, entry_point: str, instruction: str) -> dict with key 'code'"
            },
            "Programmer": {
                "description": "Automatically writes and executes Python code, returns execution result.",
                "interface": "programmer(problem: str, analysis: str = 'None') -> dict with keys 'code' and 'output'"
            },
            "Test": {
                "description": "Tests code with test cases, reflects on errors and revises.",
                "interface": "test(problem: str, solution: str, entry_point: str, test_loop: int = 3) -> dict with keys 'result' and 'solution'"
            },
            "Format": {
                "description": "Extracts concise answer from verbose solution.",
                "interface": "format(problem: str, solution: str) -> dict with key 'solution'"
            },
            "Review": {
                "description": "Reviews solution correctness using critical thinking.",
                "interface": "review(problem: str, solution: str) -> dict with keys 'review_result' (bool) and 'feedback'"
            },
            "Revise": {
                "description": "Revises solution based on feedback.",
                "interface": "revise(problem: str, solution: str, feedback: str) -> dict with key 'solution'"
            },
            "ScEnsemble": {
                "description": "Uses self-consistency to select the most frequent solution.",
                "interface": "sc_ensemble(solutions: List[str], problem: str) -> dict with key 'response'"
            },
            "MdEnsemble": {
                "description": "Majority voting ensemble - shuffles and votes multiple times for stability.",
                "interface": "md_ensemble(solutions: List[str], problem: str) -> dict with key 'solution'"
            },
            "Decompose": {
                "description": "Breaks complex problems into smaller sub-problems (Least-to-Most). Use with Aggregate.",
                "interface": "decompose(problem: str) -> dict with keys 'sub_problems' and 'reasoning'"
            },
            "Verify": {
                "description": "Independently verifies if an answer is correct.",
                "interface": "verify(problem: str, answer: str) -> dict with keys 'is_correct', 'verification_steps', 'confidence'"
            },
            "Plan": {
                "description": "Creates a strategic plan before solving (Plan-and-Solve).",
                "interface": "plan(problem: str) -> dict with keys 'plan', 'approach', 'key_insights'"
            },
            "Aggregate": {
                "description": "Combines multiple sub-answers into final answer. Use after Decompose or parallel.",
                "interface": "aggregate(problem: str, sub_answers: List[str]) -> dict with keys 'aggregated_answer', 'synthesis'"
            }
        }

    def _build_generation_prompt(self, problem: str, problem_type: str) -> str:
        """ - 

        :
        1. XML
        2. emoji/LaTeX
        3. 
        4. 
        5. <output>
        """
        prompt = f"""<task>
Generate a DSL expression for the workflow to solve this problem.
Use as many operators as needed - simple problems may only need 2-3 operators, complex problems may need more.
</task>

<operators>
Custom: General reasoning, text generation, flexible input/output
AnswerGenerate: Step-by-step reasoning with thought process and final answer (useful for QA/logical reasoning)
Programmer: Write and execute Python code, returns code and execution output (useful for math/code)
Test: Run unit tests on code. CRITICAL: For CODE tasks, ALWAYS use Test after Programmer to verify correctness!
ScEnsemble: Self-consistency voting to select most frequent solution
Review: Check solution correctness. Often paired with Revise. Pattern: "Review ? Revise : done" (outputs feedback only, not solution)
Revise: Fix solution based on feedback
Decompose: Break complex problems into sub-problems (use with Aggregate for multi-step)
Verify: Independently verify answer correctness (extra check)
Plan: Create strategic plan before solving (Plan-and-Solve approach)
Aggregate: Combine sub-answers into final answer (use after Decompose or parallel)
</operators>

<syntax>
Single: Custom
Chain: Custom -> Programmer -> Custom
Parallel: [Custom, Custom, Custom] -> ScEnsemble
Conditional: Review ? Revise : done
Loop (single operator): (Revise) * 3
Loop (chain): (Custom -> Review -> Revise) * 3
</syntax>

<wrong_outputs>
WRONG: chart_with_upwards_trend -> Review (emoji text not allowed)
WRONG: \\boxed{{Programmer -> Custom}} (LaTeX not allowed)
WRONG: Revise * 3 (missing parentheses, must be (Revise) * 3)
WRONG: The workflow is: Custom -> Review (no explanation allowed)
WRONG: Based on the problem, I suggest Custom (no preamble allowed)
WRONG: Chain: Custom -> Programmer (no label prefix allowed)
WRONG: Parallel: [Custom, Custom] -> ScEnsemble (no label prefix allowed)
WRONG: Programmer (for CODE task, missing Test - MUST include Test!)
</wrong_outputs>

<good_examples>
GOOD: Plan -> Decompose -> [Programmer, Custom, ScEnsemble] -> Aggregate -> Verify
GOOD: Plan -> Programmer -> Test -> Review ? Revise : done -> Verify
GOOD: Decompose -> [Custom, Custom] -> Aggregate -> Verify ? Revise : done -> AnswerGenerate
GOOD: Plan -> [Programmer, Custom] -> ScEnsemble -> Review ? Revise : done -> Verify
GOOD: Programmer -> Test (minimal CODE workflow - Test is REQUIRED for code tasks)
GOOD: Programmer -> Test -> Review ? Revise : done (CODE workflow with error handling)
</good_examples>

<constraints>
- Output ONLY the DSL expression, nothing else
- Use operators as needed (2-3 for simple, more for complex problems)
- Use operators listed above to build effective workflows
- NO emojis or special Unicode characters
- NO LaTeX formatting (no \\boxed{{}}, no $$, no \\text{{}})
- NO explanations before or after the DSL
- NO phrases like "The answer is" or "The workflow is"
- NO prefix labels like "Chain:", "Parallel:", "DSL:"
- Single operator loop MUST use parentheses: (Custom) * 3, NOT Custom * 3
- CRITICAL: For CODE/code tasks, ALWAYS include Test after Programmer to run unit tests!
</constraints>

<problem type="{problem_type}">
{problem}
</problem>

DSL:"""
        return prompt

    async def generate_workflow(
        self,
        problem: str,
        problem_type: str = "math",
        temperature: float = 0.7,
        max_new_tokens: int = 2048,
        custom_prompt: Optional[str] = None
    ) -> Dict:
        """

        Returns:
            {
                "workflow_code": "Python",
                "valid": bool,
                "error": Optional[str],
                "metadata": {...}
            }
        """
        if self.use_vllm_api:
            return await self._generate_with_vllm_api(
                problem, problem_type, temperature, max_new_tokens, custom_prompt
            )
        else:
            return await self._generate_with_transformers(
                problem, problem_type, temperature, max_new_tokens, custom_prompt
            )

    async def _generate_with_vllm_api(
        self,
        problem: str,
        problem_type: str,
        temperature: float,
        max_tokens: int,
        custom_prompt: Optional[str]
    ) -> Dict:
        """vLLM API"""
        async with self.semaphore:
            try:
                prompt = custom_prompt or self._build_generation_prompt(problem, problem_type)

                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a workflow generation expert."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=self.config.get('top_p', 0.95),
                )

                generated_text = response.choices[0].message.content
                workflow_code, is_valid, error, dsl_info = self._parse_workflow_code(generated_text, problem_type)

                return {
                    "workflow_code": workflow_code,
                    "valid": is_valid,
                    "error": error,
                    "metadata": {
                        "tokens": response.usage.total_tokens if response.usage else 0,
                        "model": self.model_name,
                        "dsl_info": dsl_info
                    }
                }

            except Exception as e:
                return {
                    "workflow_code": "",
                    "valid": False,
                    "error": str(e),
                    "metadata": {
                        "dsl_info": self._analyze_dsl_quality("", is_fallback=True)
                    }
                }

    async def _generate_with_transformers(
        self,
        problem: str,
        problem_type: str,
        temperature: float,
        max_new_tokens: int,
        custom_prompt: Optional[str]
    ) -> Dict:
        """transformersGPU"""
        async with self._generation_lock:
            loop = asyncio.get_event_loop()

            def _sync_generate():
                """"""
                prompt = custom_prompt or self._build_generation_prompt(problem, problem_type)

                # Tokenize
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=self.config.get('top_p', 0.95),
                        top_k=self.config.get('top_k', 50),
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )

                generated_text = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:].long(),
                    skip_special_tokens=True
                )

                return generated_text

            try:
                generated_text = await loop.run_in_executor(None, _sync_generate)

                workflow_code, is_valid, error, dsl_info = self._parse_workflow_code(generated_text, problem_type)

                return {
                    "workflow_code": workflow_code,
                    "valid": is_valid,
                    "error": error,
                    "metadata": {
                        "problem": problem,
                        "problem_type": problem_type,
                        "temperature": temperature,
                        "dsl_info": dsl_info
                    }
                }
            except Exception as e:
                return {
                    "workflow_code": "",
                    "valid": False,
                    "error": str(e),
                    "metadata": {
                        "dsl_info": self._analyze_dsl_quality("", is_fallback=True)
                    }
                }

    def _analyze_dsl_quality(self, dsl_text: str, is_fallback: bool = False) -> Dict:
        """

        Graph-R1
        - is_fallback: workflow
        - num_operators: 
        - unique_operators: 
        - has_chain:  (->)
        - has_loop:  (*)
        - has_conditional:  (?)
        - has_parallel:  ([])
        - dsl_text: DSL

        Returns:
            dsl_info dict with quality metrics
        """
        import re

        valid_ops = ['Custom', 'Programmer', 'ScEnsemble', 'Review', 'Revise',
                     'AnswerGenerate', 'CustomCodeGenerate', 'Test', 'Format', 'MdEnsemble',
                     'Decompose', 'Verify', 'Plan', 'Aggregate']

        dsl_info = {
            'is_fallback': is_fallback,
            'num_operators': 1 if is_fallback else 0,
            'unique_operators': ['Custom'] if is_fallback else [],
            'has_chain': False,
            'has_loop': False,
            'has_conditional': False,
            'has_parallel': False,
            'dsl_text': dsl_text if dsl_text else 'Custom (default fallback)',
            'dsl_quality_score': 0.0
        }

        if is_fallback or not dsl_text:
            return dsl_info

        found_operators = []
        for op in valid_ops:
            matches = re.findall(rf'\b{op}\b', dsl_text)
            found_operators.extend(matches)

        dsl_info['num_operators'] = len(found_operators)
        dsl_info['unique_operators'] = list(set(found_operators))

        dsl_info['has_chain'] = '->' in dsl_text
        dsl_info['has_loop'] = '*' in dsl_text
        dsl_info['has_conditional'] = '?' in dsl_text and ':' in dsl_text
        dsl_info['has_parallel'] = '[' in dsl_text and ']' in dsl_text

        return dsl_info

    def _parse_workflow_code(self, generated_text: str, problem_type: str) -> Tuple[str, bool, Optional[str], Dict]:
        """

        DSL
        - : Custom
        - : Custom -> Programmer -> Custom
        - : [Custom, Custom, Custom] -> ScEnsemble
        - : Review ? Revise : done

        Returns:
            (workflow_code, is_valid, error, dsl_info)
        """
        import re

        text_clean = generated_text.strip()
        text_clean = re.sub(r'</?(output|dsl|workflow|answer)>', '', text_clean, flags=re.IGNORECASE)
        text_clean = re.sub(r'```\w*', '', text_clean)
        text_clean = text_clean.strip()

        first_line = text_clean.split('\n')[0].strip()

        valid_ops = ['Custom', 'Programmer', 'ScEnsemble', 'Review', 'Revise', 'AnswerGenerate',
                     'CustomCodeGenerate', 'Test', 'Format', 'MdEnsemble',
                     'Decompose', 'Verify', 'Plan', 'Aggregate']
        if any(op in first_line for op in valid_ops):
            dsl_text = re.sub(r'^[^A-Za-z\[]*', '', first_line)
            dsl_text = re.sub(r'[^A-Za-z\]>\-,\s\?\*\(\):]*$', '', dsl_text).strip()
            if dsl_text:
                print(f"  📝 DSL: {dsl_text}")
                generator = WorkflowCodeGenerator(problem_type)
                code, is_valid, error = generator.generate(dsl_text)
                if is_valid:
                    print(f"  ✅ DSL")
                    dsl_info = self._analyze_dsl_quality(dsl_text, is_fallback=False)
                    return code, True, None, dsl_info
                else:
                    print(f"  ⚠️ DSL: {error}")

        workflow_match = re.search(r'<workflow>\s*([\s\S]*?)\s*(?:</workflow>|$)', generated_text)
        if workflow_match:
            dsl_text = workflow_match.group(1).strip()
            print(f"  📝 XML DSL: {dsl_text}")
            generator = WorkflowCodeGenerator(problem_type)
            code, is_valid, error = generator.generate(dsl_text)
            if is_valid:
                print(f"  ✅ DSL")
                dsl_info = self._analyze_dsl_quality(dsl_text, is_fallback=False)
                return code, True, None, dsl_info
            else:
                print(f"  ⚠️ DSL: {error}")

        for line in text_clean.split('\n'):
            line = line.strip()
            if line and any(op in line for op in valid_ops):
                line = re.sub(r'^[^A-Za-z\[]*', '', line)
                line = re.sub(r'[^A-Za-z\]>\-,\s\?\*\(\):]*$', '', line)
                if line and '->' in line or '[' in line or line in valid_ops:
                    print(f"  📝 DSL: {line}")
                    generator = WorkflowCodeGenerator(problem_type)
                    code, is_valid, error = generator.generate(line)
                    if is_valid:
                        print(f"  ✅ DSL")
                        dsl_info = self._analyze_dsl_quality(line, is_fallback=False)
                        return code, True, None, dsl_info

        graph_code, prompt_code = self._extract_xml_workflow(generated_text)
        if graph_code:
            print(f"  📝 XML")
            code = graph_code.strip()
            if prompt_code:
                prompt_custom_code = prompt_code.strip()
            else:
                prompt_custom_code = self._get_default_prompt_custom(problem_type)
        else:
            print(f"  ⚠️ workflow")
            dsl_info = self._analyze_dsl_quality("", is_fallback=True)
            return self._get_default_workflow(problem_type), False, "No valid format detected", dsl_info

        if "TASK_PROMPT" not in code and prompt_custom_code:
            class_match = re.search(r'^class Workflow', code, re.MULTILINE)
            if class_match:
                code = prompt_custom_code + "\n\n" + code
            else:
                code = prompt_custom_code + "\n" + code

        code = self._validate_and_fix_workflow(code, problem_type)

        try:
            ast.parse(code)
            dsl_info = self._analyze_dsl_quality("XML-format workflow", is_fallback=False)
            dsl_info['is_xml_format'] = True
            return code, True, None, dsl_info
        except SyntaxError as e:
            dsl_info = self._analyze_dsl_quality("", is_fallback=True)
            return self._get_default_workflow(problem_type), False, f"Syntax error: {str(e)}", dsl_info

    def _extract_xml_workflow(self, text: str) -> Tuple[str, str]:
        """XMLgraphprompt

        Returns:
            (graph_code, prompt_code) - XML
        """
        import re

        graph_code = ""
        prompt_code = ""

        graph_match = re.search(r'<graph>\s*([\s\S]*?)\s*</graph>', text)
        if graph_match:
            graph_code = graph_match.group(1).strip()

        prompt_match = re.search(r'<prompt>\s*([\s\S]*?)\s*</prompt>', text)
        if prompt_match:
            prompt_code = prompt_match.group(1).strip()

        return graph_code, prompt_code

    def _parse_legacy_format(self, generated_text: str, problem_type: str) -> Tuple[str, str]:
        """Pythonclass"""
        import re

        code_start = generated_text.find("```python")
        if code_start == -1:
            code_start = generated_text.find("class Workflow:")
            if code_start == -1:
                return "", ""
            code = generated_text[code_start:]
        else:
            code_start += len("```python\n")
            code_end = generated_text.find("```", code_start)
            code = generated_text[code_start:code_end] if code_end != -1 else generated_text[code_start:]

        code = code.strip()

        prompt_custom_start = code.find("# === PROMPT_CUSTOM START ===")
        prompt_custom_end = code.find("# === PROMPT_CUSTOM END ===")

        prompt_custom_code = ""
        if prompt_custom_start != -1 and prompt_custom_end != -1:
            end_line_end = code.find("\n", prompt_custom_end)
            if end_line_end == -1:
                end_line_end = len(code)
            prompt_custom_code = code[prompt_custom_start:end_line_end + 1]
            code = code[:prompt_custom_start] + code[end_line_end + 1:]
        else:
            task_prompt_match = re.search(
                r'^(TASK_PROMPT\s*=\s*(?:"""[\s\S]*?"""|\'\'\' [\s\S]*?\'\'\'))',
                code,
                re.MULTILINE
            )
            if task_prompt_match:
                prompt_custom_code = task_prompt_match.group(1)
            else:
                prompt_custom_code = self._get_default_prompt_custom(problem_type)

        return code.strip(), prompt_custom_code

    def _get_default_prompt_custom(self, problem_type: str) -> str:
        """TASK_PROMPT"""
        if problem_type == "math":
            return '''TASK_PROMPT = """Solve this mathematical problem step by step.
Show your reasoning clearly and provide the final numerical answer.
Format: First explain your approach, then show calculations, finally state the answer."""'''
        elif problem_type == "code":
            return '''TASK_PROMPT = """Write a Python function to solve this problem.
Requirements:
1. The function should be efficient and handle edge cases
2. Include proper input validation
3. Return the correct type as specified"""'''
        else:
            return '''TASK_PROMPT = """Solve this problem carefully.
Provide a clear, structured answer with reasoning."""'''

    def _validate_and_fix_workflow(self, code: str, problem_type: str) -> str:
        """workflowoperator

        Args:
            code: workflow
            problem_type: 

        Returns:
        """
        import re

        initialized_ops = set()
        init_section = re.search(r'def __init__\([^)]+\):[\s\S]+?(?=\n    async def|\n    def|$)', code)
        if init_section:
            init_code = init_section.group(0)
            init_patterns = re.findall(r'self\.(\w+)\s*=\s*operator\.(\w+)\(', init_code)
            for attr_name, op_name in init_patterns:
                initialized_ops.add(attr_name)

        used_ops = set()
        call_section = re.search(r'async def __call__\([^)]+\):[\s\S]+', code)
        if call_section:
            call_code = call_section.group(0)
            used_patterns = re.findall(r'await self\.(\w+)\(', call_code)
            for op_name in used_patterns:
                used_ops.add(op_name)

        missing_ops = used_ops - initialized_ops

        if missing_ops:
            print(f"\n⚠️  operator: {missing_ops}")
            print(f"   : {initialized_ops}")
            print(f"   : {used_ops}")

            llm_init_match = re.search(r'(\s+)(self\.llm = create_llm_instance\([^)]+\))', code)
            if llm_init_match:
                indent = llm_init_match.group(1)
                llm_init_line = llm_init_match.group(2)

                missing_inits = []
                for op_name in sorted(missing_ops):
                    # answer_generate -> AnswerGenerate
                    # review -> Review
                    op_class_name = ''.join(word.capitalize() for word in op_name.split('_'))

                    valid_operators = [
                        'Custom', 'AnswerGenerate', 'CustomCodeGenerate',
                        'Programmer', 'Test', 'Format',
                        'Review', 'Revise', 'ScEnsemble', 'MdEnsemble'
                    ]
                    if op_class_name in valid_operators:
                        missing_inits.append(f"{indent}self.{op_name} = operator.{op_class_name}(self.llm)")

                if missing_inits:
                    insert_code = '\n' + '\n'.join(missing_inits)
                    code = code.replace(llm_init_line, llm_init_line + insert_code)
                    print(f"✅  {len(missing_inits)} operator")

        return code

    def _get_default_workflow(self, problem_type: str = "math") -> str:
        """ - TASK_PROMPT"""
        if problem_type == "math":
            task_prompt = '''"""Solve this mathematical problem step by step.
Show your complete reasoning process:
1. Identify what the problem is asking
2. List known information and variables
3. Apply relevant formulas or methods
4. Perform calculations carefully
5. State the final numerical answer clearly

IMPORTANT: Always verify your answer before providing it."""'''
        elif problem_type == "code":
            task_prompt = '''"""Write a Python function to solve this problem.
Requirements:
1. Handle all edge cases properly
2. Use efficient algorithms
3. Include proper input validation
4. Return the correct type as specified
5. Add brief comments for complex logic"""'''
        else:
            task_prompt = '''"""Solve this problem carefully and provide a clear answer.
Show your reasoning step by step."""'''

        return f"""# === PROMPT_CUSTOM START ===
TASK_PROMPT = {task_prompt}
# === PROMPT_CUSTOM END ===

import workspace.{problem_type}.workflows.template.operator as operator
from scripts.async_llm import create_llm_instance
from scripts.evaluator import DatasetType

class Workflow:
    def __init__(self, name: str, llm_config, dataset: DatasetType):
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)

    async def __call__(self, problem: str, entry_point: str = "solve", test: str = None, node_cache: dict = None):
        # entry_point used for code problems with Test operator
        solution = await self.custom(input=problem, instruction=TASK_PROMPT)
        return solution['response'], self.llm.get_usage_summary()["total_cost"]
"""

    async def generate_workflows_batch(
        self,
        problems: List[str],
        problem_types: List[str],
        temperatures: List[float],
        custom_prompts: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        GPU batch

        Args:
            problems: 
            problem_types: 
            temperatures: 
            custom_prompts: 

        Returns:
            
        """
        if self.use_vllm_api:
            tasks = []
            for i in range(len(problems)):
                task = self.generate_workflow(
                    problem=problems[i],
                    problem_type=problem_types[i],
                    temperature=temperatures[i],
                    custom_prompt=custom_prompts[i] if custom_prompts else None
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            processed_results = []
            for result in results:
                if isinstance(result, Exception):
                    processed_results.append({
                        "workflow_code": "",
                        "valid": False,
                        "error": str(result),
                        "metadata": {}
                    })
                else:
                    processed_results.append(result)

            return processed_results
        else:
            return await self._batch_generate_with_transformers(
                problems, problem_types, temperatures, custom_prompts
            )

    async def _batch_generate_with_transformers(
        self,
        problems: List[str],
        problem_types: List[str],
        temperatures: List[float],
        custom_prompts: Optional[List[str]]
    ) -> List[Dict]:
        """transformersGPU batch"""
        loop = asyncio.get_event_loop()

        MAX_BATCH_SIZE = 4

        def _sync_batch_generate(batch_prompts, batch_temp):
            """"""
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=3072
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.get('max_new_tokens', 2048),
                    temperature=batch_temp,
                    top_p=self.config.get('top_p', 0.95),
                    top_k=self.config.get('top_k', 50),
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            generated_texts = self.tokenizer.batch_decode(
                outputs[:, inputs['input_ids'].shape[1]:].long(),
                skip_special_tokens=True
            )

            del inputs, outputs
            torch.cuda.empty_cache()

            return generated_texts

        try:
            all_prompts = []
            for i in range(len(problems)):
                if custom_prompts and custom_prompts[i]:
                    prompt = custom_prompts[i]
                else:
                    prompt = self._build_generation_prompt(problems[i], problem_types[i])
                all_prompts.append(prompt)

            all_generated_texts = []
            for batch_start in range(0, len(all_prompts), MAX_BATCH_SIZE):
                batch_end = min(batch_start + MAX_BATCH_SIZE, len(all_prompts))
                batch_prompts = all_prompts[batch_start:batch_end]
                batch_temp = temperatures[batch_start]

                print(f"  🔧  {batch_start//MAX_BATCH_SIZE + 1}/{(len(all_prompts)-1)//MAX_BATCH_SIZE + 1} ({len(batch_prompts)})")

                if batch_start > 0:
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()

                batch_texts = await loop.run_in_executor(
                    None, _sync_batch_generate, batch_prompts, batch_temp
                )
                all_generated_texts.extend(batch_texts)

            results = []
            for i, generated_text in enumerate(all_generated_texts):
                workflow_code, is_valid, error, dsl_info = self._parse_workflow_code(
                    generated_text, problem_types[i]
                )
                if not workflow_code or len(workflow_code) < 10:
                    print(f"  ⚠️ [DEBUG] {i}: workflow_code! valid={is_valid}, error={error}")
                    print(f"      generated_text100: {generated_text[:100] if generated_text else 'EMPTY'}")
                results.append({
                    "workflow_code": workflow_code,
                    "valid": is_valid,
                    "error": error,
                    "metadata": {
                        "problem": problems[i],
                        "problem_type": problem_types[i],
                        "temperature": temperatures[i],
                        "dsl_info": dsl_info,
                        "raw_text": generated_text
                    }
                })

            return results

        except Exception as e:
            import traceback
            print(f"❌ Batch generation: {e}")
            traceback.print_exc()
            return [{
                "workflow_code": "",
                "valid": False,
                "error": str(e),
                "metadata": {}
            } for _ in problems]


# ============================================================================
# ============================================================================

class WorkflowDSLParser:
    """DSL

    :
    - : "Programmer -> Custom"
    - : "[Custom, Custom, Custom] -> ScEnsemble"
    - : "Programmer -> [Custom, Custom] -> ScEnsemble"
    """

    def __init__(self, debug: bool = False, problem_type: str = 'math'):
        """DSL

        Args:
            debug: 
            problem_type:  ('math'  'code')
        """
        self.debug = debug
        self.problem_type = problem_type

    VALID_OPERATORS = {
        'Custom', 'AnswerGenerate', 'CustomCodeGenerate',
        'Programmer', 'Test', 'Format',
        'Review', 'Revise', 'ScEnsemble', 'MdEnsemble',
        'Decompose', 'Verify', 'Plan', 'Aggregate'
    }

    OPERATOR_CORRECTIONS = {
        'Giver': 'Custom',
        'Generator': 'Custom',
        'Solver': 'Custom',
        'Thinker': 'Custom',
        'Reasoner': 'Custom',
        'Answer': 'Custom',
        'Coder': 'Programmer',
        'Code': 'Programmer',
        'Python': 'Programmer',
        'Execute': 'Programmer',
        'Calc': 'Programmer',
        'Calculator': 'Programmer',
        'Check': 'Review',
        'Verify': 'Review',
        'Validate': 'Review',
        'Fix': 'Revise',
        'Correct': 'Revise',
        'Improve': 'Revise',
        'Vote': 'ScEnsemble',
        'Ensemble': 'ScEnsemble',
        'Select': 'ScEnsemble',
        'Cust': 'Custom',
        'Prog': 'Programmer',
        'Rev': 'Review',
        'Sc': 'ScEnsemble',
        'custom': 'Custom',
        'programmer': 'Programmer',
        'review': 'Review',
        'revise': 'Revise',
        'scensemble': 'ScEnsemble',
        'test': 'Test',
        'format': 'Format',
        'decompose': 'Decompose',
        'verify': 'Verify',
        'plan': 'Plan',
        'aggregate': 'Aggregate',
        'Break': 'Decompose',
        'Split': 'Decompose',
        'Divide': 'Decompose',
        'Combine': 'Aggregate',
        'Merge': 'Aggregate',
        'Join': 'Aggregate',
        'Collect': 'Aggregate',
        'Confirm': 'Verify',
        'Check': 'Verify',
        'Strategy': 'Plan',
        'Think': 'Plan',
        'Approach': 'Plan',
    }

    OPERATOR_SIGNATURES = {
        'Custom': {
            'inputs': ['input', 'instruction'],
            'output': 'response',
            'output_type': 'str'
        },
        'CustomCodeGenerate': {
            'inputs': ['problem', 'entry_point', 'instruction'],
            'output': 'response',
            'output_type': 'str'
        },
        'Programmer': {
            'inputs': ['problem', 'analysis'],
            'output': 'output',
            'output_type': 'str'
        },
        'ScEnsemble': {
            'inputs': ['solutions', 'problem'],
            'output': 'response',
            'output_type': 'str',
            'accepts_list': True
        },
        'MdEnsemble': {
            'inputs': ['solutions', 'problem'],
            'output': 'solution',
            'output_type': 'str',
            'accepts_list': True
        },
        'Test': {
            'inputs': ['problem', 'solution', 'entry_point', 'test'],
            'output': 'solution',
            'output_type': 'str',
            'has_result': True
        },
        'Review': {
            'inputs': ['problem', 'solution'],
            'output': 'output',
            'output_type': 'str',
            'has_result': True
        },
        'Revise': {
            'inputs': ['problem', 'solution', 'feedback'],
            'output': 'solution',
            'output_type': 'str'
        },
        'Format': {
            'inputs': ['problem', 'solution'],
            'output': 'solution',
            'output_type': 'str'
        },
        'AnswerGenerate': {
            'inputs': ['input'],
            'output': 'answer',
            'output_type': 'str'
        },
        'Decompose': {
            'inputs': ['problem'],
            'output': 'sub_problems',
            'output_type': 'str',
            'produces_list': True
        },
        'Verify': {
            'inputs': ['problem', 'answer'],
            'output': 'answer',
            'output_type': 'str',
            'has_result': True
        },
        'Plan': {
            'inputs': ['problem'],
            'output': 'plan',
            'output_type': 'str'
        },
        'Aggregate': {
            'inputs': ['problem', 'sub_answers'],
            'output': 'aggregated_answer',
            'output_type': 'str',
            'accepts_list': True
        }
    }

    def _correct_operator_name(self, op_name: str) -> str:
        """

        :
        1. operator
        2. 
        3. 
        4. 
        5. Custom

        Args:
            op_name: operator

        Returns:
            operator
        """
        if op_name in self.VALID_OPERATORS:
            return op_name

        cleaned = ''.join(c for c in op_name if c.isalpha())

        if cleaned in self.VALID_OPERATORS:
            print(f"    Fix: '{op_name}' -> '{cleaned}' ()")
            return cleaned

        if op_name in self.OPERATOR_CORRECTIONS:
            corrected = self.OPERATOR_CORRECTIONS[op_name]
            print(f"    Fix: '{op_name}' -> '{corrected}' ()")
            return corrected

        if cleaned in self.OPERATOR_CORRECTIONS:
            corrected = self.OPERATOR_CORRECTIONS[cleaned]
            print(f"    Fix: '{op_name}' -> '{corrected}' ()")
            return corrected

        if len(cleaned) >= 2:
            for valid_op in self.VALID_OPERATORS:
                if valid_op.lower().startswith(cleaned.lower()):
                    print(f"    Fix: '{op_name}' -> '{valid_op}' ()")
                    return valid_op

        cleaned_lower = cleaned.lower()
        for valid_op in self.VALID_OPERATORS:
            if cleaned_lower in valid_op.lower() or valid_op.lower() in cleaned_lower:
                print(f"    Fix: '{op_name}' -> '{valid_op}' ()")
                return valid_op

        print(f"    Fix: '{op_name}' -> 'Custom' ()")
        return 'Custom'

    def _correct_dsl_operators(self, dsl_text: str) -> str:
        """

        Args:
            dsl_text: DSL

        Returns:
            DSL
        """
        import re

        words = re.findall(r'\b([A-Z][a-zA-Z\']*)\b', dsl_text)

        corrections_made = []
        for word in set(words):
            if word.lower() == 'done':
                continue
            corrected = self._correct_operator_name(word)
            if corrected != word:
                dsl_text = re.sub(r'\b' + re.escape(word) + r'\b', corrected, dsl_text)
                corrections_made.append(f"{word}->{corrected}")

        if corrections_made:
            print(f"    📝 DSL: {', '.join(corrections_made)}")

        return dsl_text

    def _clean_problem_content(self, dsl_text: str) -> str:
        """

        DSL:
        - "i)+3i(5-i) -> Programmer -> Custom"
        - "Final DSL: 5(3-i)+3i(5-i) -> Programmer"
        - "The answer is Programmer -> Custom"
        - "find_Element(Custom) -> Programmer -> Custom" 
        - "CheckIntegerWorkflow: Custom -> Programmer" 

        :
        1. Fix:  func(Op) -> ...
        2. Fix:  Label: Op -> ...
        3. operator
        4. operatorDSL
        5. 

        Args:
            dsl_text: DSL

        Returns:
            DSL
        """
        import re

        func_call_pattern = r'^[a-z_][a-zA-Z_0-9]*\(([A-Z][a-zA-Z]*)\)'
        func_match = re.match(func_call_pattern, dsl_text)
        if func_match:
            op_in_paren = func_match.group(1)
            if op_in_paren in self.VALID_OPERATORS:
                rest_match = re.search(r'\)\s*->\s*(.+)', dsl_text)
                if rest_match:
                    cleaned = rest_match.group(1).strip()
                    print(f"    Fix: : '{dsl_text[:40]}' -> '{cleaned[:50]}...'")
                    return cleaned

        label_pattern = r'^[a-zA-Z_][a-zA-Z_0-9]*:\s*'
        if re.match(label_pattern, dsl_text):
            cleaned = re.sub(label_pattern, '', dsl_text).strip()
            if cleaned and any(op in cleaned for op in self.VALID_OPERATORS):
                print(f"    Fix: : '{dsl_text[:40]}' -> '{cleaned[:50]}...'")
                return cleaned

        first_op_pos = len(dsl_text)
        first_op = None
        for op in self.VALID_OPERATORS:
            match = re.search(r'\b' + op + r'\b', dsl_text)
            if match and match.start() < first_op_pos:
                first_op_pos = match.start()
                first_op = op

        if first_op is None:
            return dsl_text

        if first_op_pos == 0:
            return dsl_text

        before_op = dsl_text[:first_op_pos].strip()

        valid_prefix_pattern = r'^[\[\(\s\n]*$'

        if re.match(valid_prefix_pattern, before_op):
            return dsl_text

        if '->' in before_op:
            parts = dsl_text.split('->')
            for i, part in enumerate(parts):
                part_stripped = part.strip()
                for op in self.VALID_OPERATORS:
                    if part_stripped.startswith(op):
                        cleaned = ' -> '.join(parts[i:])
                        print(f"    Fix: : '{before_op}...' -> '{cleaned[:50]}...'")
                        return cleaned

        cleaned = dsl_text[first_op_pos:]
        print(f"    Fix: : '{before_op}' -> '{cleaned[:50]}...'")
        return cleaned

    def _filter_non_operators(self, dsl_text: str) -> str:
        """

        DSL:
        - "Custom -> 5 -> Custom"
        - "Programmer -> 10 -> Review"
        - "DSL: 2 -> Custom"

        :  -> operator

        Args:
            dsl_text: DSL

        Returns:
            DSL
        """
        import re

        parts = [p.strip() for p in dsl_text.split('->')]
        filtered_parts = []

        for part in parts:
            part = part.strip()

            if not part:
                continue

            if part.lower() == 'done':
                filtered_parts.append(part)
                continue

            if part.startswith('[') and ']' in part:
                filtered_parts.append(part)
                continue

            if '?' in part or ':' in part:
                filtered_parts.append(part)
                continue

            if part.isdigit() or re.match(r'^[\d.]+$', part):
                continue

            if len(part) < 3 and '*' not in part:
                continue

            has_op = any(op in part for op in self.VALID_OPERATORS)
            if has_op or '*' in part:
                filtered_parts.append(part)

        result = ' -> '.join(filtered_parts)

        if result != dsl_text:
            if self.debug:
                print(f"    Fix: operator: '{dsl_text[:50]}' -> '{result[:50]}'")

        return result

    def _expand_loops(self, dsl_text: str) -> str:
        """
        🔧 :  * plan  * decompose 

        :
        - (A) * N → A -> A -> ... (N)
        - (A -> B) * N → A -> B -> A -> B -> ... (N)
        - N * A → A -> A -> ... (N) 
        - A * → A -> A -> A (3) 
        - (A) * plan → __PLAN_LOOP__A__ (Plan)
        - [A] * decompose → __DECOMPOSE_PARALLEL__A__ (Decompose)

        Args:
            dsl_text: DSL

        Returns:
            DSL
        """
        import re

        max_iterations = 200

        # ============================================================
        # ============================================================

        plan_loop_pattern = r'\(([^()]+)\)\s*\*\s*plan\b'
        match = re.search(plan_loop_pattern, dsl_text, re.IGNORECASE)
        if match:
            inner = match.group(1).strip()
            dsl_text = dsl_text[:match.start()] + f'__PLAN_LOOP__{inner}__' + dsl_text[match.end():]
            if self.debug:
                print(f"    🔧 : '({inner}) * plan' -> '__PLAN_LOOP__{inner}__'")

        decompose_parallel_pattern = r'\[([^\]]+)\]\s*\*\s*decompose\b'
        match = re.search(decompose_parallel_pattern, dsl_text, re.IGNORECASE)
        if match:
            inner = match.group(1).strip()
            dsl_text = dsl_text[:match.start()] + f'__DECOMPOSE_PARALLEL__{inner}__' + dsl_text[match.end():]
            if self.debug:
                print(f"    🔧 : '[{inner}] * decompose' -> '__DECOMPOSE_PARALLEL__{inner}__'")

        # ============================================================
        # ============================================================

        prefix_loop_pattern = r'(\d+)\s*\*\s*([A-Z][a-zA-Z]*)'
        iteration = 0
        while iteration < max_iterations:
            match = re.search(prefix_loop_pattern, dsl_text)
            if not match:
                break
            repeat_count = min(int(match.group(1)), 5)
            operator = match.group(2).strip()
            if operator in self.VALID_OPERATORS:
                expanded = ' -> '.join([operator] * repeat_count)
                dsl_text = dsl_text[:match.start()] + expanded + dsl_text[match.end():]
            else:
                break
            iteration += 1

        suffix_star_num_pattern = r'(?<![(\[])\b([A-Z][a-zA-Z]*)\s*\*\s*(\d+)'
        iteration = 0
        while iteration < max_iterations:
            match = re.search(suffix_star_num_pattern, dsl_text)
            if not match:
                break
            operator = match.group(1).strip()
            repeat_count = min(int(match.group(2)), 5)
            if operator in self.VALID_OPERATORS:
                expanded = ' -> '.join([operator] * repeat_count)
                dsl_text = dsl_text[:match.start()] + expanded + dsl_text[match.end():]
                if self.debug:
                    print(f"    Fix: '{operator} * {match.group(2)}' -> '{expanded}'")
            else:
                break
            iteration += 1

        suffix_star_pattern = r'([A-Z][a-zA-Z]*)\s*\*(?!\s*\d)'
        iteration = 0
        while iteration < max_iterations:
            match = re.search(suffix_star_pattern, dsl_text)
            if not match:
                break
            operator = match.group(1).strip()
            if operator in self.VALID_OPERATORS:
                expanded = ' -> '.join([operator] * 3)
                dsl_text = dsl_text[:match.start()] + expanded + dsl_text[match.end():]
            else:
                break
            iteration += 1

        paren_star_pattern = r'\(([^()]+)\)\s*\*(?!\s*\d)'
        iteration = 0
        while iteration < max_iterations:
            match = re.search(paren_star_pattern, dsl_text)
            if not match:
                break
            inner_content = match.group(1).strip()
            expanded = ' -> '.join([inner_content] * 3)
            dsl_text = dsl_text[:match.start()] + expanded + dsl_text[match.end():]
            iteration += 1

        loop_pattern = r'\(([^()]+)\)\s*\*\s*(\d+)'

        iteration = 0
        while iteration < max_iterations:
            match = re.search(loop_pattern, dsl_text)
            if not match:
                break

            inner_content = match.group(1).strip()
            repeat_count = int(match.group(2))

            repeat_count = min(repeat_count, 5)

            expanded = ' -> '.join([inner_content] * repeat_count)

            dsl_text = dsl_text[:match.start()] + expanded + dsl_text[match.end():]

            iteration += 1

        return dsl_text

    def parse(self, dsl_text: str) -> dict:
        """DSL

        Args:
            dsl_text: DSL "Programmer -> Custom"  "[Custom, Custom] -> ScEnsemble"

        Returns:
            {
                'valid': bool,
                'error': str or None,
                'stages': [
                    {
                        'type': 'single' | 'parallel',
                        'operators': ['Programmer'] | ['Custom', 'Custom', 'Custom'],
                    },
                    ...
                ]
            }
        """
        import re

        dsl_text = dsl_text.strip()

        if '</output>' in dsl_text or '<output>' in dsl_text:
            fragments = re.split(r'\s*</?\s*output\s*>\s*', dsl_text)
            for frag in fragments:
                frag = frag.strip()
                if frag and any(op in frag for op in self.VALID_OPERATORS):
                    dsl_text = frag
                    break

        dsl_text = re.sub(r'</?[a-zA-Z_][a-zA-Z0-9_]*/?>', '', dsl_text)
        dsl_text = re.sub(r'```\w*', '', dsl_text)
        dsl_text = re.sub(r'</?workflow>', '', dsl_text).strip()

        if not dsl_text:
            return {'valid': False, 'error': 'DSL', 'stages': []}

        dsl_text = self._clean_problem_content(dsl_text)

        if not dsl_text:
            return {'valid': False, 'error': 'DSL', 'stages': []}

        dsl_text = self._correct_dsl_operators(dsl_text)

        dsl_text = self._expand_loops(dsl_text)

        dsl_text = self._filter_non_operators(dsl_text)

        MAX_DSL_OPERATORS = 15
        parts = [p.strip() for p in dsl_text.split('->') if p.strip()]
        if len(parts) > MAX_DSL_OPERATORS:
            print(f"    Fix: DSL ({len(parts)} operators) {MAX_DSL_OPERATORS}")
            parts = parts[:MAX_DSL_OPERATORS]
            dsl_text = ' -> '.join(parts)

        parts = [p.strip() for p in dsl_text.split('->') if p.strip()]
        dedupe_parts = []
        for i, part in enumerate(parts):
            if i > 0 and not part.startswith('[') and not parts[i-1].startswith('['):
                if part == parts[i-1]:
                    continue
            dedupe_parts.append(part)
        if len(dedupe_parts) < len(parts):
            print(f"    Fix: : {len(parts)} -> {len(dedupe_parts)} operators")
            dsl_text = ' -> '.join(dedupe_parts)

        if self.problem_type == 'math' and 'Test' in dsl_text:
            parts = [p.strip() for p in dsl_text.split('->') if p.strip()]
            filtered_parts = []
            for part in parts:
                if part == 'Test':
                    continue
                if part.startswith('[') and part.endswith(']'):
                    inner = part[1:-1]
                    ops = [op.strip() for op in inner.split(',') if op.strip() != 'Test']
                    if ops:
                        part = '[' + ', '.join(ops) + ']'
                    else:
                        continue
                filtered_parts.append(part)
            if len(filtered_parts) < len(parts):
                print(f"    Fix: mathTest: {len(parts)} -> {len(filtered_parts)} operators")
                dsl_text = ' -> '.join(filtered_parts)

        has_valid_op = any(op in dsl_text for op in self.VALID_OPERATORS)
        if not has_valid_op:
            return {'valid': False, 'error': 'operator', 'stages': []}


        dsl_text = re.sub(r'->\s*done\s*$', '', dsl_text, flags=re.IGNORECASE).strip()

        stages = []

        parts = [p.strip() for p in dsl_text.split('->')]

        for part in parts:
            if not part:
                continue

            if part.lower() == 'done':
                continue

            # ============================================================
            # ============================================================

            plan_loop_match = re.match(r'^__PLAN_LOOP__(.+)__$', part)
            if plan_loop_match:
                body_op = plan_loop_match.group(1).strip()
                if body_op not in self.VALID_OPERATORS:
                    return {'valid': False, 'error': f'Planoperator: {body_op}', 'stages': []}
                stages.append({
                    'type': 'plan_loop',
                    'operators': [body_op],
                    'body_operator': body_op
                })
                continue

            decompose_parallel_match = re.match(r'^__DECOMPOSE_PARALLEL__(.+)__$', part)
            if decompose_parallel_match:
                body_op = decompose_parallel_match.group(1).strip()
                if body_op not in self.VALID_OPERATORS:
                    return {'valid': False, 'error': f'Decomposeoperator: {body_op}', 'stages': []}
                stages.append({
                    'type': 'decompose_parallel',
                    'operators': [body_op],
                    'body_operator': body_op
                })
                continue

            if part.startswith('[') and ']' in part:
                inner = part[1:part.rindex(']')].strip()
                operators = []
                for op in inner.split(','):
                    op = op.strip()
                    op = re.sub(r'[<>/\s]+$', '', op)
                    op = re.sub(r'^[<>/\s]+', '', op)
                    op = op.strip()
                    operators.append(op)

                for op in operators:
                    if op not in self.VALID_OPERATORS:
                        return {'valid': False, 'error': f'operator: {op}', 'stages': []}

                stages.append({
                    'type': 'parallel',
                    'operators': operators
                })

            elif '?' in part and ':' in part:
                cond_match = re.match(r'^(\w+)\s*\?\s*(\w+)\s*:\s*(\w+)$', part.strip())
                if cond_match:
                    condition_op, true_branch, false_branch = cond_match.groups()
                    if condition_op not in self.VALID_OPERATORS:
                        return {'valid': False, 'error': f'operator: {condition_op}', 'stages': []}
                    if true_branch.lower() != 'done' and true_branch not in self.VALID_OPERATORS:
                        return {'valid': False, 'error': f'operator: {true_branch}', 'stages': []}
                    if false_branch.lower() != 'done' and false_branch not in self.VALID_OPERATORS:
                        return {'valid': False, 'error': f'operator: {false_branch}', 'stages': []}

                    stages.append({
                        'type': 'conditional',
                        'condition_op': condition_op,
                        'true_branch': true_branch,
                        'false_branch': false_branch,
                        'operators': [condition_op, true_branch] if true_branch.lower() != 'done' else [condition_op]
                    })
                else:
                    return {'valid': False, 'error': f': {part}', 'stages': []}

            else:
                op = part.strip()
                op = re.sub(r'[<>/\s]+$', '', op)
                op = re.sub(r'^[<>/\s]+', '', op)
                op = op.strip()
                if op not in self.VALID_OPERATORS:
                    return {'valid': False, 'error': f'operator: {op}', 'stages': []}

                stages.append({
                    'type': 'single',
                    'operators': [op]
                })

        if not stages:
            return {'valid': False, 'error': 'operator', 'stages': []}

        return {'valid': True, 'error': None, 'stages': stages}


class WorkflowCodeGenerator:
    """DSLPython Workflow"""

    def __init__(self, problem_type: str = 'math'):
        self.problem_type = problem_type
        self.parser = WorkflowDSLParser(problem_type=problem_type)

    def generate(self, dsl_text: str) -> Tuple[str, bool, Optional[str]]:
        """DSLWorkflow

        Args:
            dsl_text: DSL

        Returns:
            (code, is_valid, error)
        """
        parsed = self.parser.parse(dsl_text)

        if not parsed['valid']:
            return self._get_default_code(), False, parsed['error']

        stages = parsed['stages']

        all_operators = set()
        for stage in stages:
            all_operators.update(stage['operators'])

        code = self._generate_workflow_code(stages, all_operators)

        try:
            ast.parse(code)
            return code, True, None
        except SyntaxError as e:
            return self._get_default_code(), False, f": {e}"

    def _generate_workflow_code(self, stages: List[dict], all_operators: set) -> str:
        """Workflow"""

        init_lines = []
        for op in sorted(all_operators):
            attr_name = self._to_snake_case(op)
            init_lines.append(f"        self.{attr_name} = operator.{op}(self.llm)")

        call_lines = self._generate_call_body(stages)

        code = f'''class Workflow:
    def __init__(self, name: str, llm_config, dataset):
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
{chr(10).join(init_lines)}

    async def __call__(self, problem: str, entry_point: str = None, test: str = None, node_cache: dict = None):
        """
        Auto-generated workflow from DSL
        """
{chr(10).join(call_lines)}
'''
        return code

    def _generate_call_body(
        self,
        stages: List[dict],
        prompts: Optional[dict] = None,
        stage_node_ids: Optional[List[List[str]]] = None,
    ) -> List[str]:
        """
        __call__
        -  solution  Custom/Programmer 
        -  feedback  Review
        - Revise  solution  feedback
        """
        lines = []
        prev_output = None
        prev_is_list = False

        last_solution_var = None
        last_feedback_var = None
        prev_op = None
        prev_raw_var = None
        programmer_code_var = None
        aggregate_output_var = None
        last_verify_raw_var = None
        last_review_raw_var = None

        if self.problem_type == 'code':
            lines.append("        code_for_eval = None")
            lines.append("        last_test_result = None")
            lines.append("        last_test_error_type = None")
            lines.append("        last_test_error = None")
            lines.append("        last_test_traceback = None")
            lines.append("        last_test_summary = None")
            lines.append("        last_test_backend = None")

        # ============================================================
        # Incremental execution cache (node_id -> (input_hash, raw_result))
        # ============================================================
        lines.append("        import hashlib")
        lines.append("        import json")
        lines.append("        def _wf_cache_serialize(obj):")
        lines.append("            try:")
        lines.append("                return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)")
        lines.append("            except Exception:")
        lines.append("                return str(obj)")
        lines.append("        def _wf_cache_hash(obj):")
        lines.append("            s = _wf_cache_serialize(obj)")
        lines.append("            return hashlib.sha256(s.encode('utf-8', 'ignore')).hexdigest()")
        lines.append("        async def _wf_cached(node_id, op_name, inputs, coro_factory):")
        lines.append("            if node_cache is None or not node_id:")
        lines.append("                return await coro_factory()")
        lines.append("            try:")
        lines.append("                h = _wf_cache_hash({'op': op_name, 'inputs': inputs})")
        lines.append("            except Exception:")
        lines.append("                h = _wf_cache_hash(str(inputs))")
        lines.append("            cached = node_cache.get(node_id)")
        lines.append("            if isinstance(cached, (tuple, list)) and len(cached) == 2 and cached[0] == h:")
        lines.append("                return cached[1]")
        lines.append("            result = await coro_factory()")
        lines.append("            node_cache[node_id] = (h, result)")
        lines.append("            return result")

        def _get_custom_prompt(op_name: str, node_id: Optional[str]) -> Optional[str]:
            """ prompts  custom prompt

             node_id  parallel  node_1_p0 
             operator / node_id 
            """
            if not prompts:
                return None
            if op_name == "Programmer":
                for nid, info in prompts.items():
                    if "PREVIOUS_ERROR" in str(info.get("prompt", "")):
                        print(f"[DEBUG] Found PREVIOUS_ERROR in prompts[{nid}]", flush=True)
            if node_id and node_id in prompts:
                result = prompts[node_id].get("prompt")
                if op_name == "Programmer":
                    print(f"[DEBUG] _get_custom_prompt: node_id={node_id}, has_error={'PREVIOUS_ERROR' in str(result)}", flush=True)
                return result
            for info in prompts.values():
                if info.get("operator") == op_name and info.get("prompt"):
                    return info.get("prompt")
            return None

        for i, stage in enumerate(stages):
            is_last = (i == len(stages) - 1)

            # ============================================================
            # ============================================================
            if stage['type'] == 'plan_loop':
                body_op = stage['body_operator']
                attr_name = self._to_snake_case(body_op)
                lines.append(f"        # Plan-driven dynamic loop: execute {body_op} for each step")
                lines.append(f"        plan_steps = raw_{i-1}.get('steps_list', []) if isinstance(raw_{i-1}, dict) else []")
                lines.append(f"        if not plan_steps:")
                lines.append(f"            # Fallback: parse plan text")
                lines.append(f"            plan_text = raw_{i-1}.get('plan', '') if isinstance(raw_{i-1}, dict) else str(raw_{i-1})")
                lines.append(f"            import re as _re_plan")
                lines.append(f"            _matches = _re_plan.findall(r'(?:Step\\s*)?(\\d+)[.:\\)]\\s*(.+?)(?=(?:Step\\s*)?\\d+[.:\\)]|$)', plan_text, _re_plan.DOTALL | _re_plan.IGNORECASE)")
                lines.append(f"            plan_steps = [m[1].strip() for m in _matches if m[1].strip()]")
                lines.append(f"        plan_results_{i} = []")
                lines.append(f"        for step_idx, step_content in enumerate(plan_steps):")
                lines.append(f"            step_result = await self.{attr_name}(input=problem, instruction=step_content)")
                lines.append(f"            plan_results_{i}.append(step_result.get('response', str(step_result)) if isinstance(step_result, dict) else str(step_result))")
                lines.append(f"        output_{i} = '\\n---\\n'.join(plan_results_{i})")
                lines.append(f"        solutions_{i} = plan_results_{i}")
                prev_output = f"solutions_{i}"
                prev_is_list = True
                last_solution_var = f"solutions_{i}"
                prev_op = body_op
                continue

            # ============================================================
            # ============================================================
            if stage['type'] == 'decompose_parallel':
                body_op = stage['body_operator']
                attr_name = self._to_snake_case(body_op)
                lines.append(f"        # Decompose-driven dynamic parallel: execute {body_op} for each sub-problem")
                lines.append(f"        sub_problems = raw_{i-1}.get('sub_problems_list', []) if isinstance(raw_{i-1}, dict) else []")
                lines.append(f"        if not sub_problems:")
                lines.append(f"            # Fallback: parse sub_problems text")
                lines.append(f"            sp_text = raw_{i-1}.get('sub_problems', '') if isinstance(raw_{i-1}, dict) else str(raw_{i-1})")
                lines.append(f"            import re as _re_decompose")
                lines.append(f"            _matches = _re_decompose.findall(r'(\\d+)[.:\\)]\\s*(.+?)(?=\\d+[.:\\)]|$)', sp_text, _re_decompose.DOTALL)")
                lines.append(f"            sub_problems = [m[1].strip() for m in _matches if m[1].strip()]")
                lines.append(f"        import asyncio")
                lines.append(f"        async def _solve_sub_{i}(sub_problem):")
                lines.append(f"            result = await self.{attr_name}(input=problem, instruction=sub_problem)")
                lines.append(f"            return result.get('response', str(result)) if isinstance(result, dict) else str(result)")
                lines.append(f"        _tasks_{i} = [_solve_sub_{i}(sp) for sp in sub_problems]")
                lines.append(f"        solutions_{i} = await asyncio.gather(*_tasks_{i})")
                lines.append(f"        output_{i} = list(solutions_{i})")
                prev_output = f"solutions_{i}"
                prev_is_list = True
                last_solution_var = f"solutions_{i}"
                prev_op = body_op
                continue

            if stage['type'] == 'parallel':
                ops = stage['operators']
                ops_desc = ", ".join(ops)

                lines.append(f"        #  {len(ops)}  operator: [{ops_desc}]")
                lines.append(f"        import asyncio")

                if prev_output:
                    input_param = prev_output
                else:
                    input_param = 'problem'

                tasks = []
                output_keys = []
                node_ids_for_stage = None
                if stage_node_ids and i < len(stage_node_ids):
                    node_ids_for_stage = stage_node_ids[i]
                if node_ids_for_stage and len(node_ids_for_stage) != len(ops):
                    node_ids_for_stage = None

                for j, op_name in enumerate(ops):
                    attr_name = self._to_snake_case(op_name)
                    node_id = node_ids_for_stage[j] if node_ids_for_stage else None
                    custom_prompt = _get_custom_prompt(op_name, node_id)
                    param_str = self._build_params(op_name, input_param, is_first=(i == 0), custom_prompt=custom_prompt)
                    cache_inputs = self._build_cache_inputs_expr(op_name, input_param, is_first=(i == 0), custom_prompt=custom_prompt)
                    cache_node_id = repr(node_id or f"stage_{i}_p{j}")
                    cache_op_name = repr(op_name)
                    tasks.append(
                        f"_wf_cached({cache_node_id}, {cache_op_name}, {cache_inputs}, "
                        f"lambda: self.{attr_name}({param_str}))"
                    )

                    sig = WorkflowDSLParser.OPERATOR_SIGNATURES.get(op_name, {})
                    output_key = sig.get('output', 'response')
                    if self.problem_type == 'code' and op_name == 'Programmer':
                        output_key = 'code'
                    output_keys.append(output_key)

                lines.append(f"        tasks = [{', '.join(tasks)}]")
                lines.append(f"        results_{i} = await asyncio.gather(*tasks)")
                lines.append(f"        output_keys_{i} = {output_keys!r}")
                lines.append(f"        solutions_{i} = [r.get(k, r.get('response', str(r))) for r, k in zip(results_{i}, output_keys_{i})]")
                if self.problem_type == 'code':
                    # Code: when a workflow ends with a parallel stage, avoid returning a Python list repr as "code".
                    # Pick one candidate that looks like code for evaluation/controller feedback.
                    lines.append(f"        if isinstance(solutions_{i}, list) and solutions_{i}:")
                    lines.append(f"            for _cand in reversed(solutions_{i}):")
                    lines.append(f"                _is_code = isinstance(_cand, str) and ('def ' in _cand or 'class ' in _cand or 'import ' in _cand)")
                    lines.append(f"                _is_step = any(p in str(_cand) for p in ['Step 1:', 'Step 2:', '1. ', '2. ', 'First,', 'To solve'])")
                    lines.append(f"                if _is_code and not _is_step:")
                    lines.append(f"                    code_for_eval = _cand")
                    lines.append(f"                    break")

                prev_output = f"solutions_{i}"
                prev_is_list = True
                last_solution_var = f"solutions_{i}"
                prev_op = None

            elif stage['type'] == 'conditional':
                condition_op = stage['condition_op']
                true_branch = stage['true_branch']
                false_branch = stage['false_branch']

                cond_attr = self._to_snake_case(condition_op)
                cond_sig = WorkflowDSLParser.OPERATOR_SIGNATURES.get(condition_op, {})

                cond_node_id = None
                if stage_node_ids and i < len(stage_node_ids) and stage_node_ids[i]:
                    cond_node_id = stage_node_ids[i][0]
                cond_custom_prompt = _get_custom_prompt(condition_op, cond_node_id)

                input_var = prev_output if prev_output else 'problem'
                if self.problem_type == 'code' and prev_op == 'Programmer' and condition_op in ('Review', 'Verify'):
                    input_var = f"code_{i-1}"
                    lines.append(f"        # Fix: {condition_op}  Programmer  code  solution")

                param_str = self._build_params(condition_op, input_var, is_first=(i == 0), custom_prompt=cond_custom_prompt)
                cache_inputs = self._build_cache_inputs_expr(condition_op, input_var, is_first=(i == 0), custom_prompt=cond_custom_prompt)
                cache_node_id = repr(cond_node_id or f"stage_{i}_cond")
                cache_op_name = repr(condition_op)

                lines.append(f"        #  -  {condition_op}")
                lines.append(
                    f"        raw_{i} = await _wf_cached({cache_node_id}, {cache_op_name}, {cache_inputs}, "
                    f"lambda: self.{cond_attr}({param_str}))"
                )

                lines.append(f"        #  {condition_op} ")
                if condition_op == 'Review':
                    # Review: {'review_result': bool, 'feedback': str}
                    lines.append(f"        _cond_passed_{i} = False")
                    lines.append(f"        if isinstance(raw_{i}, dict):")
                    lines.append(f"            _review_result = raw_{i}.get('review_result')")
                    lines.append(f"            if isinstance(_review_result, str):")
                    lines.append(f"                _cond_passed_{i} = _review_result.lower() == 'true'")
                    lines.append(f"            elif isinstance(_review_result, bool):")
                    lines.append(f"                _cond_passed_{i} = _review_result")
                    lines.append(f"        _cond_feedback_{i} = raw_{i}.get('feedback', '') if isinstance(raw_{i}, dict) else str(raw_{i})")
                elif condition_op == 'Verify':
                    # Verify: {'is_correct': bool, ...}
                    lines.append(f"        _cond_passed_{i} = False")
                    lines.append(f"        if isinstance(raw_{i}, dict):")
                    lines.append(f"            _verify_result = raw_{i}.get('is_correct')")
                    lines.append(f"            if isinstance(_verify_result, str):")
                    lines.append(f"                _cond_passed_{i} = _verify_result.lower() == 'true'")
                    lines.append(f"            elif isinstance(_verify_result, bool):")
                    lines.append(f"                _cond_passed_{i} = _verify_result")
                    lines.append(f"        _cond_feedback_{i} = raw_{i}.get('verification_steps', '') if isinstance(raw_{i}, dict) else str(raw_{i})")
                else:
                    lines.append(f"        _cond_passed_{i} = bool(raw_{i}.get('result', False)) if isinstance(raw_{i}, dict) else False")
                    lines.append(f"        _cond_feedback_{i} = str(raw_{i})")

                lines.append(f"        #  -  {condition_op}  {true_branch}")
                lines.append(f"        if not _cond_passed_{i}:")

                if true_branch.lower() != 'done':
                    true_attr = self._to_snake_case(true_branch)
                    true_node_id = None
                    if stage_node_ids and i < len(stage_node_ids) and len(stage_node_ids[i]) > 1:
                        true_node_id = stage_node_ids[i][1]
                    true_custom_prompt = _get_custom_prompt(true_branch, true_node_id)

                    if true_branch == 'Revise':
                        instruction_str_local = self._format_instruction_literal(true_custom_prompt, op=true_branch)
                        solution_var = last_solution_var if last_solution_var else input_var
                        lines.append(f"            #  {true_branch} solution  feedback")
                        true_cache_inputs = (
                            "{"
                            f"'problem': problem, 'solution': {solution_var}, 'feedback': _cond_feedback_{i}, "
                            f"'instruction': {instruction_str_local}"
                            "}"
                        )
                        true_cache_node_id = repr(true_node_id or f"stage_{i}_true")
                        true_cache_op_name = repr(true_branch)
                        lines.append(
                            f"            raw_{i}_true = await _wf_cached({true_cache_node_id}, {true_cache_op_name}, {true_cache_inputs}, "
                            f"lambda: self.{true_attr}(problem=problem, solution={solution_var}, feedback=_cond_feedback_{i}, instruction={instruction_str_local}))"
                        )
                    elif true_branch == 'Programmer':
                        instruction_str_local = self._format_instruction_literal(true_custom_prompt, op=true_branch)
                        solution_var = last_solution_var if last_solution_var else input_var
                        lines.append(f"            # Conditional:  {true_branch}")
                        lines.append(f"            _error_instruction_{i} = f'[PREVIOUS_ERROR]: {{str(_cond_feedback_{i})[:300]}}\\n\\nFix the code based on this error.\\n\\n' + {instruction_str_local}")
                        true_cache_inputs = (
                            "{"
                            f"'problem': problem, 'analysis': {solution_var}, "
                            f"'instruction': _error_instruction_{i}"
                            "}"
                        )
                        true_cache_node_id = repr(true_node_id or f"stage_{i}_true")
                        true_cache_op_name = repr(true_branch)
                        lines.append(
                            f"            raw_{i}_true = await _wf_cached({true_cache_node_id}, {true_cache_op_name}, {true_cache_inputs}, "
                            f"lambda: self.{true_attr}(problem=problem, analysis={solution_var}, instruction=_error_instruction_{i}))"
                        )
                    else:
                        true_param_str = self._build_params(true_branch, last_solution_var or input_var, is_first=False, custom_prompt=true_custom_prompt)
                        true_cache_inputs = self._build_cache_inputs_expr(true_branch, last_solution_var or input_var, is_first=False, custom_prompt=true_custom_prompt)
                        true_cache_node_id = repr(true_node_id or f"stage_{i}_true")
                        true_cache_op_name = repr(true_branch)
                        lines.append(
                            f"            raw_{i}_true = await _wf_cached({true_cache_node_id}, {true_cache_op_name}, {true_cache_inputs}, "
                            f"lambda: self.{true_attr}({true_param_str}))"
                        )

                    true_sig = WorkflowDSLParser.OPERATOR_SIGNATURES.get(true_branch, {})
                    true_output_key = true_sig.get('output', 'response')
                    lines.append(f"            output_{i} = raw_{i}_true.get('{true_output_key}', raw_{i}_true.get('response', str(raw_{i}_true)))")
                else:
                    lines.append(f"            # true_branch  done solution")
                    if last_solution_var:
                        lines.append(f"            output_{i} = {last_solution_var}")
                    else:
                        lines.append(f"            output_{i} = {input_var}")

                lines.append(f"        else:")

                if false_branch.lower() != 'done':
                    false_attr = self._to_snake_case(false_branch)
                    false_node_id = None
                    if stage_node_ids and i < len(stage_node_ids) and len(stage_node_ids[i]) > 2:
                        false_node_id = stage_node_ids[i][2]
                    false_custom_prompt = _get_custom_prompt(false_branch, false_node_id)
                    false_param_str = self._build_params(false_branch, last_solution_var or input_var, is_first=False, custom_prompt=false_custom_prompt)
                    false_cache_inputs = self._build_cache_inputs_expr(false_branch, last_solution_var or input_var, is_first=False, custom_prompt=false_custom_prompt)
                    false_cache_node_id = repr(false_node_id or f"stage_{i}_false")
                    false_cache_op_name = repr(false_branch)
                    lines.append(
                        f"            raw_{i}_false = await _wf_cached({false_cache_node_id}, {false_cache_op_name}, {false_cache_inputs}, "
                        f"lambda: self.{false_attr}({false_param_str}))"
                    )
                    false_sig = WorkflowDSLParser.OPERATOR_SIGNATURES.get(false_branch, {})
                    false_output_key = false_sig.get('output', 'response')
                    lines.append(f"            output_{i} = raw_{i}_false.get('{false_output_key}', raw_{i}_false.get('response', str(raw_{i}_false)))")
                else:
                    lines.append(f"            # {condition_op}  {true_branch}")
                    if last_solution_var:
                        lines.append(f"            output_{i} = {last_solution_var}")
                    else:
                        lines.append(f"            output_{i} = {input_var}")

                prev_output = f"output_{i}"
                prev_is_list = False
                last_solution_var = f"output_{i}"
                last_feedback_var = None
                prev_op = true_branch if true_branch.lower() != 'done' else condition_op

            else:
                op = stage['operators'][0]
                attr_name = self._to_snake_case(op)
                sig = WorkflowDSLParser.OPERATOR_SIGNATURES.get(op, {})
                node_id = None
                if stage_node_ids and i < len(stage_node_ids) and stage_node_ids[i]:
                    node_id = stage_node_ids[i][0]
                custom_prompt = _get_custom_prompt(op, node_id)

                input_var = prev_output if prev_output else 'problem'
                # Only for code tasks: checkers/debuggers must see the actual code.
                if self.problem_type == 'code' and prev_op == 'Programmer' and op in ('Review', 'Verify', 'Revise', 'Test'):
                    input_var = f"code_{i-1}"
                    lines.append(f"        # Fix: {op}  Programmer  code ( output)  solution")

                if prev_op == 'Review' and last_solution_var and op not in ('Revise',):
                    input_var = last_solution_var
                    lines.append(f"        # Fix: Review feedbacklast_solution")

                if op == 'Revise' and prev_op == 'Review' and last_solution_var and last_feedback_var:
                    instruction_str_local = self._format_instruction_literal(custom_prompt, op=op)
                    param_str = (
                        f"problem=problem, solution={last_solution_var}, feedback={last_feedback_var}, "
                        f"instruction={instruction_str_local}"
                    )
                    cache_inputs = (
                        "{"
                        f"'problem': problem, 'solution': {last_solution_var}, 'feedback': {last_feedback_var}, "
                        f"'instruction': {instruction_str_local}"
                        "}"
                    )
                elif op == 'Revise' and prev_op == 'Verify' and last_feedback_var:
                    instruction_str_local = self._format_instruction_literal(custom_prompt, op=op)
                    # Prefer verifier's original answer string as "solution" context (avoid passing full dict repr)
                    prev_raw = f"raw_{i-1}"
                    solution_expr = (
                        f"({prev_raw}.get('original_answer', {prev_raw}.get('answer', str({prev_raw}))) "
                        f"if isinstance({prev_raw}, dict) else str({input_var}))"
                    )
                    param_str = (
                        f"problem=problem, solution={solution_expr}, feedback={last_feedback_var}, "
                        f"instruction={instruction_str_local}"
                    )
                    cache_inputs = (
                        "{"
                        f"'problem': problem, 'solution': {solution_expr}, 'feedback': {last_feedback_var}, "
                        f"'instruction': {instruction_str_local}"
                        "}"
                    )
                elif prev_is_list and sig.get('accepts_list'):
                    param_str = f"solutions={prev_output}, problem=problem"
                    cache_inputs = f"{{'solutions': {prev_output}, 'problem': problem}}"
                elif prev_output:
                    param_str = self._build_params(op, input_var, is_first=False, custom_prompt=custom_prompt)
                    cache_inputs = self._build_cache_inputs_expr(op, input_var, is_first=False, custom_prompt=custom_prompt)
                else:
                    param_str = self._build_params(op, 'problem', is_first=True, custom_prompt=custom_prompt)
                    cache_inputs = self._build_cache_inputs_expr(op, 'problem', is_first=True, custom_prompt=custom_prompt)

                cache_node_id = repr(node_id or f"stage_{i}")
                cache_op_name = repr(op)
                lines.append(
                    f"        raw_{i} = await _wf_cached({cache_node_id}, {cache_op_name}, {cache_inputs}, "
                    f"lambda: self.{attr_name}({param_str}))"
                )

                if op == 'Programmer':
                    lines.append(f"        code_{i} = raw_{i}.get('code', '') if isinstance(raw_{i}, dict) else ''")
                    lines.append(f"        output_{i} = raw_{i}.get('output') or str(raw_{i}) if isinstance(raw_{i}, dict) else raw_{i}")
                    if self.problem_type == 'code':
                        lines.append(f"        code_for_eval = code_{i}")
                elif op == 'Test':
                    output_key = sig.get('output', 'response')
                    lines.append(f"        solution_{i} = raw_{i}.get('{output_key}', raw_{i}.get('response', str(raw_{i})))")

                    if self.problem_type == 'math':
                        lines.append(f"        # In math mode, Test forwards execution output, not code.")
                        lines.append(f"        math_output = raw_{i}.get('output') if isinstance(raw_{i}, dict) else None")
                        lines.append(f"        output_{i} = math_output if math_output is not None else solution_{i}")
                    else:
                        lines.append(f"        output_{i} = solution_{i}")
                    if self.problem_type == 'code':
                        lines.append(f"        code_for_eval = solution_{i}")
                        lines.append(f"        test_pass_{i} = bool(raw_{i}.get('result', False)) if isinstance(raw_{i}, dict) else False")
                        lines.append(f"        test_error_type_{i} = raw_{i}.get('error_type') if isinstance(raw_{i}, dict) else None")
                        lines.append(f"        test_error_{i} = raw_{i}.get('error') if isinstance(raw_{i}, dict) else None")
                        lines.append(f"        test_traceback_{i} = raw_{i}.get('traceback') if isinstance(raw_{i}, dict) else None")
                        lines.append(f"        test_backend_{i} = raw_{i}.get('backend') if isinstance(raw_{i}, dict) else None")
                        lines.append(
                            f"        test_summary_{i} = 'TEST_PASSED' if test_pass_{i} else "
                            f"('TEST_FAILED: ' + str(test_error_type_{i}) + ': ' + str(test_error_{i}))"
                        )
                        lines.append(f"        if (not test_pass_{i}) and test_traceback_{i}:")
                        lines.append(f"            test_summary_{i} = str(test_summary_{i}) + '\\n' + str(test_traceback_{i})")
                        lines.append(f"        last_test_result = test_pass_{i}")
                        lines.append(f"        last_test_error_type = test_error_type_{i}")
                        lines.append(f"        last_test_error = test_error_{i}")
                        lines.append(f"        last_test_traceback = test_traceback_{i}")
                        lines.append(f"        last_test_summary = test_summary_{i}")
                        lines.append(f"        last_test_backend = test_backend_{i}")
                        if is_last:
                            # Last stage is Test: expose summary (pass/fail + error) as output for controller feedback
                            lines.append(f"        output_{i} = test_summary_{i}")
                else:
                    output_key = sig.get('output', 'response')
                    if op == 'Plan':
                        lines.append(f"        # Plandictsteps_list")
                        lines.append(f"        output_{i} = raw_{i} if isinstance(raw_{i}, dict) else {{'plan': str(raw_{i})}}")
                    elif op == 'Decompose':
                        lines.append(f"        # Decomposedictsub_problems_list")
                        lines.append(f"        output_{i} = raw_{i} if isinstance(raw_{i}, dict) else {{'sub_problems': str(raw_{i})}}")
                    elif op == 'Verify':
                        lines.append(f"        # Verifydictis_correctoutput")
                        lines.append(f"        if isinstance(raw_{i}, dict):")
                        lines.append(f"            # Ensure a concise output string exists for display/extraction")
                        lines.append(f"            if 'output' not in raw_{i}:")
                        lines.append(f"                raw_{i}['output'] = raw_{i}.get('answer', raw_{i}.get('suggested_answer', raw_{i}.get('response', '')))")
                        lines.append(f"            output_{i} = raw_{i}")
                        lines.append(f"        else:")
                        lines.append(f"            output_{i} = {{'output': str(raw_{i}), 'answer': str(raw_{i}), 'is_correct': None, 'confidence': 'unknown'}}")
                    elif op == 'Review':
                        lines.append(f"        # Reviewdictreview_result, feedbackoutput")
                        lines.append(f"        if isinstance(raw_{i}, dict):")
                        lines.append(f"            if 'output' not in raw_{i}:")
                        lines.append(f"                raw_{i}['output'] = raw_{i}.get('feedback', '')")
                        lines.append(f"            output_{i} = raw_{i}")
                        lines.append(f"        else:")
                        lines.append(f"            output_{i} = {{'output': str(raw_{i}), 'review_result': None, 'feedback': str(raw_{i})}}")
                    else:
                        lines.append(f"        output_{i} = raw_{i}.get('{output_key}', raw_{i}.get('response', str(raw_{i})))")
                    if self.problem_type == 'code' and op in ('CustomCodeGenerate', 'Revise', 'ScEnsemble', 'MdEnsemble'):
                        lines.append(f"        code_for_eval = output_{i}")

                prev_output = f"output_{i}"
                prev_is_list = False

                if op == 'Review':
                    last_feedback_var = f"output_{i}"
                    last_review_raw_var = f"output_{i}"
                elif op == 'Verify':
                    last_feedback_var = f"(output_{i}.get('verification_steps', '') if isinstance(output_{i}, dict) else '')"
                    last_solution_var = f"output_{i}"
                    last_verify_raw_var = f"output_{i}"
                elif op == 'Decompose':
                    last_solution_var = f"output_{i}"
                    prev_is_list = True
                elif op == 'Plan':
                    last_solution_var = f"output_{i}"
                    last_feedback_var = None
                elif op == 'Aggregate':
                    last_solution_var = f"output_{i}"
                    last_feedback_var = None
                    aggregate_output_var = f"output_{i}"
                elif op == 'Programmer':
                    # Code: treat code as the "solution"; Non-code: treat execution output as the "solution".
                    last_solution_var = f"code_{i}" if self.problem_type == 'code' else f"output_{i}"
                    last_feedback_var = None
                    programmer_code_var = f"code_{i}"
                elif op == 'Test':
                    if self.problem_type == 'code':
                        last_solution_var = f"solution_{i}"
                    else:
                        last_solution_var = f"output_{i}"
                    last_feedback_var = None
                elif op in ('Custom', 'CustomCodeGenerate', 'Revise', 'Format', 'AnswerGenerate'):
                    last_solution_var = f"output_{i}"
                    last_feedback_var = None

                prev_op = op
                prev_raw_var = f"raw_{i}"

        final_return_var = prev_output
        if prev_op == 'Review' and last_solution_var:
            lines.append(f"        # Review: Review  checker last_solution  review_result ")
            final_return_var = last_solution_var

        if aggregate_output_var and prev_op != 'Aggregate':
            lines.append(f"        # Aggregate ")
            final_return_var = aggregate_output_var

        if self.problem_type == 'code':
            lines.append("        # Code: return structured payload for eval + controller feedback")
            lines.append(f"        _candidate = code_for_eval or str({final_return_var} or '')")
            lines.append("        # ")
            lines.append("        _is_code = any(ind in str(_candidate) for ind in ['def ', 'class ', 'import ', 'from ', 'return '])")
            lines.append("        _is_step_desc = any(p in str(_candidate) for p in ['Step 1:', 'Step 2:', '1. ', '2. ', 'First,', 'To solve'])")
            lines.append("        final_code = _candidate if (_is_code and not _is_step_desc) else (code_for_eval or '')")
            lines.append("        final_summary = last_test_summary if last_test_summary is not None else 'CODE_GENERATED'")
            lines.append(
                "        return {"
                "'output': str(final_summary), "
                "'code': final_code, "
                "'test_passed': last_test_result, "
                "'test_error_type': last_test_error_type, "
                "'test_error': last_test_error, "
                "'test_traceback': last_test_traceback, "
                "'test_backend': last_test_backend"
                "}, self.llm.get_usage_summary()['total_cost']"
            )
        else:
            if prev_op == 'Programmer':
                lines.append(f"        # Programmerdictcode")
                lines.append(f"        return {prev_raw_var}, self.llm.get_usage_summary()['total_cost']")
            elif programmer_code_var:
                lines.append(f"        # workflowProgrammercodedict")
                lines.append(f"        if isinstance({final_return_var}, dict):")
                lines.append(f"            _out = {final_return_var}")
                lines.append(f"            _out['code'] = {programmer_code_var}")
                lines.append(f"            return _out, self.llm.get_usage_summary()['total_cost']")
                if last_verify_raw_var:
                    lines.append(f"        # Verifyfeedback")
                    lines.append(f"        _verify_info = {last_verify_raw_var} if isinstance({last_verify_raw_var}, dict) else {{}}")
                    lines.append(f"        return {{")
                    lines.append(f"            'output': {final_return_var},")
                    lines.append(f"            'code': {programmer_code_var},")
                    lines.append(f"            'is_correct': _verify_info.get('is_correct'),")
                    lines.append(f"            'confidence': _verify_info.get('confidence', 'unknown'),")
                    lines.append(f"            'answer': _verify_info.get('answer', ''),")
                    lines.append(f"            'verification_steps': _verify_info.get('verification_steps', ''),")
                    lines.append(f"        }}, self.llm.get_usage_summary()['total_cost']")
                elif last_review_raw_var:
                    lines.append(f"        # Review")
                    lines.append(f"        _review_info = {last_review_raw_var} if isinstance({last_review_raw_var}, dict) else {{}}")
                    lines.append(f"        return {{")
                    lines.append(f"            'output': {final_return_var},")
                    lines.append(f"            'code': {programmer_code_var},")
                    lines.append(f"            'review_result': _review_info.get('review_result'),")
                    lines.append(f"            'feedback': _review_info.get('feedback', ''),")
                    lines.append(f"        }}, self.llm.get_usage_summary()['total_cost']")
                else:
                    lines.append(f"        return {{'output': {final_return_var}, 'code': {programmer_code_var}}}, self.llm.get_usage_summary()['total_cost']")
            elif last_review_raw_var:
                lines.append(f"        # Review")
                lines.append(f"        _review_info = {last_review_raw_var} if isinstance({last_review_raw_var}, dict) else {{}}")
                lines.append(f"        return {{")
                lines.append(f"            'output': {final_return_var},")
                lines.append(f"            'review_result': _review_info.get('review_result'),")
                lines.append(f"            'feedback': _review_info.get('feedback', ''),")
                lines.append(f"        }}, self.llm.get_usage_summary()['total_cost']")
            elif last_verify_raw_var and prev_op != 'Verify':
                lines.append(f"        # Verifyfeedback")
                lines.append(f"        _verify_info = {last_verify_raw_var} if isinstance({last_verify_raw_var}, dict) else {{}}")
                lines.append(f"        if isinstance({final_return_var}, dict):")
                lines.append(f"            _out = {final_return_var}")
                lines.append(f"            _out.setdefault('is_correct', _verify_info.get('is_correct'))")
                lines.append(f"            _out.setdefault('confidence', _verify_info.get('confidence', 'unknown'))")
                lines.append(f"            _out.setdefault('answer', _verify_info.get('answer', ''))")
                lines.append(f"            return _out, self.llm.get_usage_summary()['total_cost']")
                lines.append(f"        return {{")
                lines.append(f"            'output': {final_return_var},")
                lines.append(f"            'is_correct': _verify_info.get('is_correct'),")
                lines.append(f"            'confidence': _verify_info.get('confidence', 'unknown'),")
                lines.append(f"            'answer': _verify_info.get('answer', ''),")
                lines.append(f"        }}, self.llm.get_usage_summary()['total_cost']")
            else:
                lines.append(f"        return {final_return_var}, self.llm.get_usage_summary()['total_cost']")

        return lines

    def _format_instruction_literal(self, custom_prompt: Optional[str], op: Optional[str] = None) -> str:
        """Escape a custom prompt as a Python single-quoted string literal for generated code.
        """
        if not custom_prompt:
            return "''"

        op = op or ""
        qa_marker = "[QA_SHORT_ANSWER]"
        if qa_marker in custom_prompt and op in ("Custom", "AnswerGenerate", "Verify"):
            prefix = (
                "Read the problem carefully and answer directly. "
                "Do NOT output <think> tags or explanations.\n\n"
            )
        elif op == "Format":
            prefix = (
                "Only extract the final answer from the given solution. "
                "Do NOT re-solve, re-verify, or re-format beyond what appears.\n\n"
            )
        elif op in ("ScEnsemble", "MdEnsemble", "Aggregate"):
            prefix = (
                "Only select/aggregate from existing candidate answers. "
                "Do NOT re-solve the problem.\n\n"
            )
        elif op == "Verify":
            prefix = (
                "Verify independently and output the correct answer in the required format.\n\n"
            )
        else:
            prefix = "You need to read the problem carefully, then think step by step.\n\n"

        full_prompt = prefix + str(custom_prompt)
        escaped_prompt = (
            full_prompt
            .replace('\\', '\\\\')
            .replace("'", "\\'")
            .replace('"', '\\"')
            .replace('\n', '\\n')
        )
        return f"'{escaped_prompt}'"

    def _build_cache_inputs_expr(self, op: str, input_var: str, is_first: bool, custom_prompt: str = None) -> str:
        """Build a stable, JSON-serializable input object expression for node-level caching.

        The expression is embedded into generated code and hashed at runtime.
        """
        instruction_str = self._format_instruction_literal(custom_prompt, op=op)

        if op == 'Custom':
            return f"{{'input': {input_var}, 'instruction': {instruction_str}}}"
        if op == 'CustomCodeGenerate':
            if is_first:
                return f"{{'problem': {input_var}, 'entry_point': (entry_point or 'solve'), 'instruction': {instruction_str}}}"
            return f"{{'problem': problem, 'entry_point': (entry_point or 'solve'), 'instruction': {instruction_str}}}"
        if op == 'Programmer':
            if is_first:
                return f"{{'problem': {input_var}, 'analysis': 'None', 'instruction': {instruction_str}}}"
            return f"{{'problem': problem, 'analysis': {input_var}, 'instruction': {instruction_str}}}"
        if op == 'Test':
            return (
                f"{{'problem': problem, 'solution': {input_var}, 'entry_point': (entry_point or 'solve'), "
                f"'test': test, 'instruction': {instruction_str}}}"
            )
        if op == 'Review':
            return f"{{'problem': problem, 'solution': {input_var}, 'instruction': {instruction_str}}}"
        if op == 'Revise':
            return f"{{'problem': problem, 'solution': {input_var}, 'feedback': '', 'instruction': {instruction_str}}}"
        if op == 'Format':
            return f"{{'problem': problem, 'solution': {input_var}, 'instruction': {instruction_str}}}"
        if op == 'AnswerGenerate':
            return f"{{'input': f'Problem: {{problem}}\\n\\nPrevious result: {{{input_var}}}'}}"
        if op in ('ScEnsemble', 'MdEnsemble'):
            return f"{{'solutions': {input_var}, 'problem': problem}}"
        if op == 'Decompose':
            if is_first:
                return f"{{'problem': {input_var}}}"
            return "{'problem': problem}"
        if op == 'Verify':
            return f"{{'problem': problem, 'answer': {input_var}, 'instruction': {instruction_str}}}"
        if op == 'Plan':
            if is_first:
                return f"{{'problem': {input_var}}}"
            return "{'problem': problem}"
        if op == 'Aggregate':
            return f"{{'problem': problem, 'sub_answers': {input_var}}}"

        return f"{{'input': {input_var}, 'instruction': {instruction_str}}}"

    def _build_params(self, op: str, input_var: str, is_first: bool, custom_prompt: str = None) -> str:
        """operator

        Args:
            op: Operator
            input_var: 
            is_first: operator
            custom_prompt: prompt ()

        Returns:
        """
        instruction_str = self._format_instruction_literal(custom_prompt, op=op)

        if op == 'Custom':
            if is_first:
                return f"input={input_var}, instruction={instruction_str}"
            else:
                return f"input=f'Problem: {{problem}}\\n\\nPrevious result: {{{input_var}}}', instruction={instruction_str}"
        elif op == 'CustomCodeGenerate':
            if is_first:
                return f"problem={input_var}, entry_point=entry_point or 'solve', instruction={instruction_str}"
            else:
                return f"problem=problem, entry_point=entry_point or 'solve', instruction={instruction_str}"
        elif op == 'Programmer':
            if is_first:
                return f"problem={input_var}, analysis='None', instruction={instruction_str}"
            else:
                return f"problem=problem, analysis={input_var}, instruction={instruction_str}"
        elif op == 'Test':
            return f"problem=problem, solution={input_var}, entry_point=entry_point or 'solve', test=test, instruction={instruction_str}"
        elif op == 'Review':
            return f"problem=problem, solution={input_var}, instruction={instruction_str}"
        elif op == 'Revise':
            default_feedback = "If answer not in passage output unanswerable; otherwise copy exact span; no explanation"
            return f"problem=problem, solution={input_var}, feedback={repr(default_feedback)}, instruction={instruction_str}"
        elif op == 'Format':
            return f"problem=problem, solution={input_var}, instruction={instruction_str}"
        elif op == 'AnswerGenerate':
            return f"input=f'Problem: {{problem}}\\n\\nPrevious result: {{{input_var}}}', instruction={instruction_str}"
        elif op in ('ScEnsemble', 'MdEnsemble'):
            return f"solutions={input_var}, problem=problem"
        elif op == 'Decompose':
            if is_first:
                return f"problem={input_var}, instruction={instruction_str}"
            else:
                return f"problem=problem, instruction={instruction_str}"
        elif op == 'Verify':
            return f"problem=problem, answer={input_var}, instruction={instruction_str}"
        elif op == 'Plan':
            if is_first:
                return f"problem={input_var}, instruction={instruction_str}"
            else:
                return f"problem=problem, instruction={instruction_str}"
        elif op == 'Aggregate':
            return f"problem=problem, sub_answers={input_var}, instruction={instruction_str}"
        else:
            return f"input={input_var}, instruction={instruction_str}"

    def _to_snake_case(self, name: str) -> str:
        """CustomCodeGenerate -> custom_code_generate"""
        import re
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    def _get_default_code(self) -> str:
        """Workflow"""
        return '''class Workflow:
    def __init__(self, name: str, llm_config, dataset):
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)

    async def __call__(self, problem: str, entry_point: str = None, test: str = None, node_cache: dict = None):
        result = await self.custom(input=problem, instruction="")
        return result['response'], self.llm.get_usage_summary()['total_cost']
'''

    def _generate_workflow_code_with_custom_prompts(self, dsl: str, prompts: dict) -> str:
        """🔧  DSL  custom_prompts 


        Args:
            dsl: DSL  ( "Programmer")
            prompts:  custom_prompts {node_id: {"operator": str, "prompt": str}}

        Returns:
             workflow 
        """
        def _infer_stage_node_ids(stages: List[dict], node_prompts: dict) -> Optional[List[List[str]]]:
            """ stages  node_id parallel  node_k_p0/node_k_p1

             operator [Custom, Custom] prompt
             prompts  node_id  single/parallel/conditional parsed stages 
            """
            if not stages or not node_prompts:
                return None

            import re

            base_nums = sorted(
                {
                    int(m.group(1))
                    for node_id in node_prompts.keys()
                    if (m := re.match(r"^node_(\d+)", str(node_id)))
                }
            )
            if not base_nums:
                return None

            stage_cursor = 0
            stage_node_ids: List[List[str]] = []

            def _expect_single(node_id: str) -> bool:
                nonlocal stage_cursor
                if stage_cursor >= len(stages):
                    return False
                stage = stages[stage_cursor]
                if stage.get("type") != "single" or len(stage.get("operators", [])) != 1:
                    return False
                op_name = stage["operators"][0]
                if node_id in node_prompts and node_prompts[node_id].get("operator") != op_name:
                    return False
                stage_node_ids.append([node_id])
                stage_cursor += 1
                return True

            def _expect_parallel(child_node_ids: List[str]) -> bool:
                nonlocal stage_cursor
                if stage_cursor >= len(stages):
                    return False
                stage = stages[stage_cursor]
                ops = stage.get("operators", [])
                if stage.get("type") != "parallel" or len(ops) != len(child_node_ids):
                    return False
                for op_name, node_id in zip(ops, child_node_ids):
                    if node_id in node_prompts and node_prompts[node_id].get("operator") != op_name:
                        return False
                stage_node_ids.append(child_node_ids)
                stage_cursor += 1
                return True

            def _expect_conditional(cond_id: str, true_id: str, false_id: Optional[str] = None) -> bool:
                nonlocal stage_cursor
                if stage_cursor >= len(stages):
                    return False
                stage = stages[stage_cursor]
                if stage.get("type") != "conditional":
                    return False
                node_ids = [cond_id, true_id]
                if false_id:
                    node_ids.append(false_id)
                stage_node_ids.append(node_ids)
                stage_cursor += 1
                return True

            for base_num in base_nums:
                base_id = f"node_{base_num}"

                # Parallel: node_k_p0/node_k_p1/...
                parallel_children: List[Tuple[int, str]] = []
                for node_id in node_prompts.keys():
                    m = re.match(rf"^{re.escape(base_id)}_p(\d+)$", str(node_id))
                    if m:
                        parallel_children.append((int(m.group(1)), str(node_id)))
                if parallel_children:
                    children = [nid for _, nid in sorted(parallel_children, key=lambda x: x[0])]
                    if not _expect_parallel(children):
                        return None
                    continue

                cond_id = f"{base_id}_cond"
                true_id = f"{base_id}_true"
                false_id = f"{base_id}_false"
                if cond_id in node_prompts or true_id in node_prompts:
                    actual_false_id = false_id if false_id in node_prompts else None
                    if not _expect_conditional(cond_id, true_id, actual_false_id):
                        return None
                    continue

                if any(str(k).startswith(f"{base_id}_l") for k in node_prompts.keys()):
                    return None

                # Single: node_k
                if base_id in node_prompts:
                    if not _expect_single(base_id):
                        return None
                    continue


            if stage_cursor != len(stages):
                return None

            return stage_node_ids

        parsed = self.parser.parse(dsl)
        if not parsed['valid']:
            return self._get_default_code()

        stages = parsed['stages']

        all_operators = set()
        for stage in stages:
            all_operators.update(stage['operators'])

        stage_node_ids = _infer_stage_node_ids(stages, prompts)
        code = self._generate_workflow_code_with_prompts(
            stages, all_operators, prompts, stage_node_ids=stage_node_ids
        )

        try:
            ast.parse(code)
            return code
        except SyntaxError:
            return self._get_default_code()

    def generate_from_graph(self, graph, prompts: dict = None) -> Tuple[str, bool, Optional[str]]:
        """ WorkflowGraph  ( custom_prompt)

         -  WorkflowGraph  custom_prompt 

        Args:
            graph: WorkflowGraph 
            prompts:  {node_id: prompt}  prompt

        Returns:
            (code, is_valid, error)
        """
        if graph.is_empty():
            return self._get_default_code(), True, None

        dsl = graph.to_dsl()
        if not dsl:
            return self._get_default_code(), True, None

        all_prompts = graph.get_all_prompts()

        if prompts:
            all_prompts.update(prompts)

        parsed = self.parser.parse(dsl)
        if not parsed['valid']:
            return self._get_default_code(), False, parsed['error']

        stages = parsed['stages']

        all_operators = set()
        for stage in stages:
            all_operators.update(stage['operators'])

        stage_node_ids = None
        try:
            from src.interactive.workflow_graph import NodeType  # type: ignore
            candidate: List[List[str]] = []
            supported = True
            for node in getattr(graph, "nodes", []):
                if node.node_type == NodeType.OPERATOR:
                    candidate.append([node.id])
                elif node.node_type == NodeType.PARALLEL:
                    candidate.append([c.id for c in node.children])
                else:
                    supported = False
                    break
            if supported and len(candidate) == len(stages):
                stage_node_ids = candidate
        except Exception:
            stage_node_ids = None

        code = self._generate_workflow_code_with_prompts(
            stages, all_operators, all_prompts, stage_node_ids=stage_node_ids
        )

        try:
            ast.parse(code)
            return code, True, None
        except SyntaxError as e:
            return self._get_default_code(), False, f": {e}"

    def _generate_workflow_code_with_prompts(
        self,
        stages: List[dict],
        all_operators: set,
        prompts: dict,
        stage_node_ids: Optional[List[List[str]]] = None,
    ) -> str:
        """ prompt  Workflow 

        Args:
            stages: 
            all_operators:  operator
            prompts: {node_id: {"operator": str, "prompt": str}} 

        Returns:
        """
        init_lines = []
        for op in sorted(all_operators):
            attr_name = self._to_snake_case(op)
            init_lines.append(f"        self.{attr_name} = operator.{op}(self.llm)")

        call_lines = self._generate_call_body(stages, prompts=prompts, stage_node_ids=stage_node_ids)

        code = f'''class Workflow:
    def __init__(self, name: str, llm_config, dataset):
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
{chr(10).join(init_lines)}

    async def __call__(self, problem: str, entry_point: str = None, test: str = None, node_cache: dict = None):
        """
        Auto-generated workflow with custom prompts
        """
{chr(10).join(call_lines)}
'''
        return code

    def _generate_call_body_with_prompts(
        self,
        stages: List[dict],
        prompts: dict
    ) -> List[str]:
        """ _generate_call_body"""
        return self._generate_call_body(stages, prompts=prompts)
