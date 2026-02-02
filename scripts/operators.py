#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Desc: AFlow-compatible operators module
# Provides workflow operators with Formatter abstraction and ProcessPoolExecutor

import asyncio
import concurrent.futures
import multiprocessing
import os
import signal
import random
import sys
import traceback
from collections import Counter
from typing import Dict, List, Tuple, Optional, Any
import logging

from tenacity import retry, stop_after_attempt, wait_fixed

OPERATOR_DEBUG = True
logger = logging.getLogger("operators")
logger.setLevel(logging.WARNING if not OPERATOR_DEBUG else logging.DEBUG)

for noisy_logger in ['httpx', 'httpcore', 'openai', 'openai._base_client', 'matplotlib', 'matplotlib.font_manager']:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

from scripts.formatter import (
    BaseFormatter,
    FormatError,
    XmlFormatter,
    TextFormatter,
    CodeFormatter
)
from scripts.operator_analysis import (
    AnswerGenerateOp,
    CodeGenerateOp,
    FormatOp,
    GenerateOp,
    MdEnsembleOp,
    ReflectionTestOp,
    ReviewOp,
    ReviseOp,
    ScEnsembleOp,
    DecomposeOp,
    VerifyOp,
    PlanOp,
    AggregateOp,
)
from scripts.prompts.prompt import (
    ANSWER_GENERATION_PROMPT,
    FORMAT_PROMPT,
    MD_ENSEMBLE_PROMPT,
    PYTHON_CODE_VERIFIER_PROMPT,
    REFLECTION_ON_PUBLIC_TEST_PROMPT,
    REVIEW_PROMPT,
    REVISE_PROMPT,
    SC_ENSEMBLE_PROMPT,
    DECOMPOSE_PROMPT,
    VERIFY_PROMPT,
    PLAN_PROMPT,
    AGGREGATE_PROMPT,
)
from scripts.utils.sanitize import sanitize, DISALLOWED_IMPORTS


class Operator:
    """Base class for all operators with Formatter support"""

    def __init__(self, llm, name: str):
        self.name = name
        self.llm = llm

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    async def _fill_node(self, op_class, prompt, mode=None, **extra_kwargs):
        """Core method to call LLM with optional formatting
        """
        formatter = self._create_formatter(op_class, mode, **extra_kwargs)

        if OPERATOR_DEBUG:
            prompt_preview = prompt[:500] + "..." if len(prompt) > 500 else prompt
            logger.debug(f"[{self.name}] LLM (mode={mode}):")
            logger.debug(f"  prompt500: {prompt_preview}")

        try:
            if formatter:
                # Use formatter for structured responses
                response = await self.llm.call_with_format(prompt, formatter)
            else:
                # Direct call without formatting
                response = await self.llm(prompt)

            if OPERATOR_DEBUG:
                response_str = str(response)
                response_preview = response_str[:500] + "..." if len(response_str) > 500 else response_str
                logger.debug(f"[{self.name}] LLM:")
                logger.debug(f"  response500: {response_preview}")

            # Normalize response format
            if isinstance(response, dict):
                return response
            else:
                return {"response": response}
        except FormatError as e:
            print(f"Format error in {self.name}: {str(e)}")
            if OPERATOR_DEBUG:
                logger.error(f"[{self.name}] FormatError: {str(e)}")
            return {"error": str(e)}

    def _create_formatter(self, op_class, mode=None, **extra_kwargs) -> Optional[BaseFormatter]:
        """Create appropriate formatter based on operation class and mode"""
        if mode == "xml_fill":
            return XmlFormatter.from_model(op_class)
        elif mode == "code_fill":
            function_name = extra_kwargs.get("function_name")
            return CodeFormatter(function_name=function_name)
        elif mode == "single_fill":
            return TextFormatter()
        else:
            return None


def _ensure_str(value) -> str:
    """list/dict
    """
    if isinstance(value, dict):
        for key in ['output', 'answer', 'response', 'aggregated_answer', 'solution', 'result']:
            if key in value:
                extracted = value[key]
                if extracted is not None and str(extracted).strip():
                    return _ensure_str(extracted)
        for v in value.values():
            if v is not None and str(v).strip():
                return str(v)
        return ""

    if isinstance(value, list):
        import re
        from collections import Counter

        clean_values = [str(x).strip() for x in value if x is not None and str(x).strip()]
        if not clean_values:
            return ""

        numeric_values = []
        for v in clean_values:
            match = re.match(r'^[+-]?\d+(\.\d+)?$', v.strip())
            if match:
                numeric_values.append(v.strip())

        if numeric_values:
            most_common = Counter(numeric_values).most_common(1)
            if most_common:
                return most_common[0][0]

        if len(clean_values) == 1:
            return clean_values[0]

        return "\n---\n".join(clean_values)

    return str(value) if value is not None else ""


class Custom(Operator):
    """Custom operator - most flexible, generates anything based on instruction"""

    def __init__(self, llm, name: str = "Custom"):
        super().__init__(llm, name)

    async def __call__(self, input: str, instruction: str) -> Dict[str, str]:
        input = _ensure_str(input)
        instruction = _ensure_str(instruction) if instruction else ""

        if OPERATOR_DEBUG:
            input_preview = input[:200] + "..." if len(input) > 200 else input
            logger.debug(f"[Custom] :")
            logger.debug(f"  input: {input_preview}")
            logger.debug(f"  instruction: {instruction[:100] if instruction else 'None'}")

        if instruction and input:
            prompt = f"{instruction}\n\n[PREVIOUS RESULT]\n{input}"
        elif instruction:
            prompt = instruction
        else:
            prompt = input
        response = await self._fill_node(GenerateOp, prompt, mode="single_fill")

        if OPERATOR_DEBUG:
            resp_str = str(response.get('response', ''))[:200]
            logger.debug(f"[Custom] : {resp_str}")

        return response


class AnswerGenerate(Operator):
    """Generates step-by-step reasoning with thought and final answer"""

    def __init__(self, llm, name: str = "AnswerGenerate"):
        super().__init__(llm, name)

    async def __call__(self, input: str, instruction: str = "") -> Dict[str, str]:
        """Support dynamic prompts: instruction"""
        input = _ensure_str(input)

        if OPERATOR_DEBUG:
            input_preview = input[:200] + "..." if len(input) > 200 else input
            logger.debug(f"[AnswerGenerate] : {input_preview}")

        prompt = ANSWER_GENERATION_PROMPT.format(input=input)
        response = await self._fill_node(AnswerGenerateOp, prompt, mode="xml_fill")

        if OPERATOR_DEBUG:
            answer = str(response.get('answer', ''))[:200]
            thought = str(response.get('thought', ''))[:100]
            logger.debug(f"[AnswerGenerate] : thought={thought}, answer={answer}")

        return response


class CustomCodeGenerate(Operator):
    """Generates code based on customized input and instruction"""

    def __init__(self, llm, name: str = "CustomCodeGenerate"):
        super().__init__(llm, name)

    async def __call__(self, problem: str, entry_point: str, instruction: str) -> Dict[str, str]:
        prompt = instruction + problem
        response = await self._fill_node(
            GenerateOp, prompt, mode="code_fill", function_name=entry_point
        )
        return response


class ScEnsemble(Operator):
    """
    Self-Consistency Ensemble - selects most consistent solution
    Paper: Self-Consistency Improves Chain of Thought Reasoning in Language Models
    Link: https://arxiv.org/abs/2203.11171
    """

    def __init__(self, llm, name: str = "ScEnsemble"):
        super().__init__(llm, name)

    async def __call__(self, solutions: List[str], problem: str) -> Dict[str, str]:
        if isinstance(solutions, str):
            logger.warning(f"[ScEnsemble] ")
            solutions = [solutions]
        elif not isinstance(solutions, list):
            logger.warning(f"[ScEnsemble]  {type(solutions).__name__}")
            solutions = [str(solutions)]

        solutions = [str(x) for x in solutions if x is not None and str(x).strip()]

        if not solutions:
            return {"response": ""}
        if len(solutions) == 1:
            return {"response": solutions[0]}

        max_solution_chars = int(os.environ.get("SC_ENSEMBLE_MAX_SOLUTION_CHARS", "2000") or 2000)

        # Create answer mapping (A, B, C, ...)
        answer_mapping = {}
        solution_text = ""
        for index, solution in enumerate(solutions):
            answer_mapping[chr(65 + index)] = index
            solution_str = str(solution)
            if max_solution_chars > 0 and len(solution_str) > max_solution_chars:
                head = max(800, max_solution_chars // 2)
                tail = max(400, max_solution_chars - head - 40)
                if head + tail + 40 > max_solution_chars:
                    tail = max(0, max_solution_chars - head - 40)
                if tail > 0:
                    solution_str = solution_str[:head] + "\n...[truncated]...\n" + solution_str[-tail:]
                else:
                    solution_str = solution_str[:max_solution_chars] + "\n...[truncated]...\n"

            solution_text += f"{chr(65 + index)}: \n{solution_str}\n\n\n"

        prompt = SC_ENSEMBLE_PROMPT.format(question=problem, solutions=solution_text)
        response = await self._fill_node(ScEnsembleOp, prompt, mode="xml_fill")

        answer = response.get("solution_letter", "A")
        answer = answer.strip().upper()

        # Get first letter if multiple characters
        if len(answer) > 0:
            answer = answer[0]

        if answer in answer_mapping:
            return {"response": solutions[answer_mapping[answer]]}
        return {"response": solutions[0]}  # Fallback to first


def _preprocess_code_solve_conflict(code: str) -> str:
    """Preprocessing:  solve() 

    : gpt-4o-mini  `def solve():` 
     `solve(expr, x)`  sympy.solve()
     TypeError: solve() takes 0 positional arguments but 2 were given

    :
    1.  `from sympy import solve`  sympy_solve
    2.  `solve(`  `sympy_solve(`
    3.  `def solve():` 
    4.  `.solve(` 
    """
    import re

    # Pattern: from sympy import solve
    code = re.sub(r'from\s+sympy\s+import\s+solve\s*$', '# solve import removed (use sympy_solve)', code, flags=re.MULTILINE)

    # Pattern: from sympy import ..., solve, ...
    # Replace ", solve" or "solve, " in import statements
    def remove_solve_from_import(match):
        line = match.group(0)
        # Remove ", solve" or "solve, " from the import
        line = re.sub(r',\s*solve(?=\s*[,\)]|\s*$)', '', line)
        line = re.sub(r'solve\s*,\s*', '', line)
        return line

    code = re.sub(r'from\s+sympy\s+import\s+[^#\n]+', remove_solve_from_import, code)


    def replace_solve_calls(match):
        prefix = match.group(1)
        if prefix and (prefix.endswith('def ') or prefix.endswith('.')):
            return match.group(0)
        return prefix + 'sympy_solve('

    code = re.sub(r'(^|[^.\w])solve\((?!\s*\))', replace_solve_calls, code, flags=re.MULTILINE)

    return code


def _preprocess_code_safe_abs(code: str) -> str:
    """Safe Abs:  Abs 

    : SymPy  Abs() :
    "solving Abs(x) when the argument is not real or imaginary"

    :  safe_abs  Abs  safe_abs
    """
    if 'Abs(' not in code:
        return code

    safe_abs_code = '''
def _safe_abs(x):
    from sympy import Abs, im, re as sympy_re, sqrt, simplify, S
    try:
        result = Abs(x)
        try:
            float(result.evalf())
        except:
            pass
        return result
    except:
        try:
            r = sympy_re(x)
            i = im(x)
            if i == S.Zero:
                return Abs(r)
            return simplify(sqrt(r**2 + i**2))
        except:
            return Abs(x)

'''

    import re

    def replace_abs_calls(match):
        prefix = match.group(1)
        if prefix and (prefix[-1] == '.' or prefix[-1].isalnum() or prefix[-1] == '_'):
            return match.group(0)
        return prefix + '_safe_abs('

    modified_code = re.sub(r'(^|[^.\w])Abs\(', replace_abs_calls, code, flags=re.MULTILINE)

    if modified_code != code:
        return safe_abs_code + modified_code

    return code


def run_code(code: str) -> Tuple[str, str]:
    """Execute Python code in isolated context (called in separate process)
    -  gpt-4o-mini  import 
    -  PYTHON_CODE_VERIFIER_PROMPT 
    """
    try:
        # ============================================
        # ============================================
        import os
        import sys

        if not os.environ.get('DISPLAY'):
            os.environ['DISPLAY'] = ':99'
        import math
        import re
        import json
        import random
        import string
        import datetime
        import collections
        import itertools
        import functools
        import statistics
        import heapq
        import bisect
        import copy
        import decimal
        import fractions
        from operator import itemgetter, attrgetter
        from typing import List, Dict, Set, Tuple, Optional, Any, Union, Callable

        try:
            import numpy as np
        except ImportError:
            np = None
        try:
            import scipy
        except ImportError:
            scipy = None
        try:
            import sympy
        except ImportError:
            sympy = None

        try:
            import sklearn
        except ImportError:
            sklearn = None

        try:
            import tensorflow as tf
            from tensorflow import keras
        except ImportError:
            tf = None
            keras = None

        # NLP
        try:
            import nltk
        except ImportError:
            nltk = None

        try:
            import pandas as pd
        except ImportError:
            pd = None

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            matplotlib = None
            plt = None
        try:
            import seaborn as sns
        except ImportError:
            sns = None

        try:
            from prettytable import PrettyTable
        except ImportError:
            PrettyTable = None
        try:
            import pytesseract
        except ImportError:
            pytesseract = None

        import pathlib
        from pathlib import Path
        import shutil
        import urllib
        import urllib.parse
        import urllib.request
        import http
        import http.client
        import multiprocessing
        import hashlib
        import base64
        import csv
        import glob as glob_module
        import struct
        import threading
        import tempfile
        import io
        import pickle
        import gzip
        import zipfile
        import tarfile
        import time as time_module
        import traceback as traceback_module

        try:
            import django
        except ImportError:
            django = None

        try:
            import requests
        except ImportError:
            requests = None

        # ============================================
        # ============================================
        global_namespace = {
            # Execution context helpers
            '__name__': '__main__',
            '__file__': 'solution.py',
            'os': os,
            'sys': sys,
            'math': math,
            're': re,
            'json': json,
            'random': random,
            'string': string,
            'datetime': datetime,
            'collections': collections,
            'itertools': itertools,
            'combinations': itertools.combinations,
            'permutations': itertools.permutations,
            'product': itertools.product,
            'chain': itertools.chain,
            'groupby': itertools.groupby,
            'functools': functools,
            'statistics': statistics,
            'heapq': heapq,
            'bisect': bisect,
            'copy': copy,
            'decimal': decimal,
            'fractions': fractions,
            'itemgetter': itemgetter,
            'attrgetter': attrgetter,
            # typing
            'List': List,
            'Dict': Dict,
            'Set': Set,
            'Tuple': Tuple,
            'Optional': Optional,
            'Any': Any,
            'Union': Union,
            'Callable': Callable,
            'np': np,
            'numpy': np,
            'scipy': scipy,
            'sympy': sympy,
            'sympy_solve': sympy.solve,
            'sklearn': sklearn,
            'tf': tf,
            'tensorflow': tf,
            'keras': keras,
            # NLP
            'nltk': nltk,
            'pd': pd,
            'pandas': pd,
            'plt': plt,
            'matplotlib': matplotlib,
            'sns': sns,
            'seaborn': sns,
            'PrettyTable': PrettyTable,
            'prettytable': PrettyTable,
            'pytesseract': pytesseract,
            'pathlib': pathlib,
            'Path': Path,
            'shutil': shutil,
            'urllib': urllib,
            'http': http,
            'multiprocessing': multiprocessing,
            'hashlib': hashlib,
            'base64': base64,
            'csv': csv,
            'glob': glob_module,
            'struct': struct,
            'threading': threading,
            'tempfile': tempfile,
            'io': io,
            'pickle': pickle,
            'gzip': gzip,
            'zipfile': zipfile,
            'tarfile': tarfile,
            'time': time_module,
            'traceback': traceback_module,
            'django': django,
            'requests': requests,
        }

        # Check for prohibited imports
        for lib in DISALLOWED_IMPORTS:
            if f"import {lib}" in code or f"from {lib}" in code:
                return "Error", f"Prohibited import: {lib}"

        original_recursion_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(500)

        code = _preprocess_code_solve_conflict(code)

        # code = _preprocess_code_safe_abs(code)

        try:
            # Execute the code
            exec(code, global_namespace)
        finally:
            sys.setrecursionlimit(original_recursion_limit)

        # Look for solve function
        if "solve" in global_namespace and callable(global_namespace["solve"]):
            sys.setrecursionlimit(500)
            try:
                result = global_namespace["solve"]()
                return "Success", str(result)
            finally:
                sys.setrecursionlimit(original_recursion_limit)
        else:
            return "Error", "Function 'solve' not found"

    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb_str = traceback.format_exception(exc_type, exc_value, exc_traceback)
        full_tb = ''.join(tb_str)
        if len(full_tb) > 2000:
            full_tb = full_tb[:1000] + "\n... [truncated] ...\n" + full_tb[-500:]
        return "Error", f"Execution error: {str(e)}\n{full_tb}"


def _run_code_in_process(code: str, result_queue: multiprocessing.Queue):
    """
    Queue
    """
    try:
        status, output = run_code(code)
        result_queue.put((status, output))
    except Exception as e:
        result_queue.put(("Error", f"Process error: {str(e)}"))


class Programmer(Operator):
    """
    Programmer operator - generates and executes Python code
    SIGKILL
    """

    def __init__(self, llm, name: str = "Programmer"):
        super().__init__(llm, name)

    async def exec_code(self, code: str, timeout: int = 30) -> Tuple[str, str]:
        """
        Execute code asynchronously with timeout
        """
        result_queue = multiprocessing.Queue()

        process = multiprocessing.Process(
            target=_run_code_in_process,
            args=(code, result_queue)
        )

        try:
            process.start()

            loop = asyncio.get_running_loop()

            async def wait_for_result():
                """"""
                while True:
                    if not result_queue.empty():
                        return result_queue.get_nowait()
                    if not process.is_alive():
                        if not result_queue.empty():
                            return result_queue.get_nowait()
                        return ("Error", "Process exited without result")
                    await asyncio.sleep(0.1)

            result = await asyncio.wait_for(wait_for_result(), timeout=timeout)
            return result

        except asyncio.TimeoutError:
            if process.is_alive():
                process.terminate()
                process.join(timeout=0.5)

                if process.is_alive():
                    process.kill()
                    process.join(timeout=0.5)

                if process.is_alive():
                    try:
                        os.kill(process.pid, signal.SIGKILL)
                    except (ProcessLookupError, OSError):
                        pass

            return "Error", f"Code execution timed out after {timeout}s (process killed)"

        except Exception as e:
            if process.is_alive():
                process.terminate()
                process.join(timeout=0.5)
                if process.is_alive():
                    process.kill()
            return "Error", f"Unknown error: {str(e)}"

        finally:
            if process.is_alive():
                process.terminate()
                process.join(timeout=0.1)
            try:
                result_queue.close()
            except:
                pass

    def _detect_function_signature(self, problem: str) -> Optional[str]:
        """Signature detection: problem

        :
        1. def func_name(params):  def func_name(params) -> return_type:
        2.  task_func, solve 

        Returns:
             "def task_func(df):"None
        """
        import re
        if not problem:
            return None

        # Prefer top-level def lines (start-of-line) to avoid class methods in scaffolding.
        pattern = re.compile(
            r'(?m)^(def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*(?:->\s*[^:]+)?\s*:)'  # signature + name
        )

        any_signature: Optional[str] = None
        for match in pattern.finditer(problem):
            signature = match.group(1).strip()
            func_name = match.group(2).strip()

            # Track first signature as fallback.
            if any_signature is None:
                any_signature = signature

            # Skip default / unsafe picks.
            if func_name == "solve":
                continue
            if re.match(r"^__.*__$", func_name):  # __init__, __str__, ...
                continue

            if OPERATOR_DEBUG:
                logger.debug(f"[SigDetect] Detected function signature: {signature}")
            return signature

        # Fallback: if we only saw dunder/solve, still return the first signature (better than None).
        if any_signature:
            if OPERATOR_DEBUG:
                logger.debug(f"[SigDetect] Detected function signature (fallback): {any_signature}")
            return any_signature

        return None

    async def code_generate(self, problem: str, analysis: str, feedback: str, mode: str, instruction: str = ""):
        """Generate code using LLM

        v19:  instruction  Qwen  custom_prompt 
        """
        detected_signature = self._detect_function_signature(problem)

        safe_instruction = instruction
        if instruction and instruction.strip():
            instruction_lower = instruction.lower()
            danger_patterns = [
                "describe", "explain", "outline", "step by step", "step-by-step",
                "list the steps", "write steps", "algorithm steps"
            ]
            is_dangerous = any(pattern in instruction_lower for pattern in danger_patterns)
            if is_dangerous:
                safe_instruction = (
                    f"{instruction}\n\n"
                    "[OVERRIDE] You MUST output executable Python code, NOT step descriptions. "
                    "Do NOT write 'Step 1: ...'. Write actual Python code with def/return."
                )
                if OPERATOR_DEBUG:
                    logger.debug(f"[Safety] Detected dangerous instruction, added code enforcement")

        enhanced_problem = problem
        if safe_instruction and safe_instruction.strip():
            enhanced_problem = (
                f"{problem}\n\n[Controller Guidance]:\n{safe_instruction}\n"
            )

        if detected_signature:
            signature_instruction = f"\n[REQUIRED SIGNATURE] Use exactly: {detected_signature}\n"
            enhanced_problem = signature_instruction + enhanced_problem
            if OPERATOR_DEBUG:
                logger.debug(f"[SigDetect] Injected signature requirement: {detected_signature}")

        prompt = PYTHON_CODE_VERIFIER_PROMPT.format(
            problem=enhanced_problem,
            analysis=analysis,
            feedback=feedback
        )
        response = await self._fill_node(
            CodeGenerateOp, prompt, mode, function_name="solve"
        )
        return response

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    async def __call__(self, problem: str, analysis: str = "None", instruction: str = "") -> Dict[str, str]:
        """Generate and execute code with retry logic

        v19:  instruction  Qwen  custom_prompt
        """
        problem = _ensure_str(problem)
        analysis = _ensure_str(analysis) if analysis and analysis != "None" else "None"
        instruction = _ensure_str(instruction) if instruction else ""

        if OPERATOR_DEBUG:
            problem_preview = problem[:200] + "..." if len(problem) > 200 else problem
            logger.debug(f"[Programmer] :")
            logger.debug(f"  problem: {problem_preview}")
            logger.debug(f"  analysis: {analysis[:100] if analysis else 'None'}")
            logger.debug(f"  instruction: {instruction[:100] if instruction else 'None'}")  # v19

        code = None
        output = None
        feedback = ""

        for i in range(3):
            code_response = await self.code_generate(problem, analysis, feedback, mode="code_fill", instruction=instruction)
            code = code_response.get("code") or code_response.get("response")

            if not code:
                if OPERATOR_DEBUG:
                    logger.debug(f"[Programmer] : No code generated")
                return {"code": "", "output": "No code generated"}

            status, output = await self.exec_code(code)

            if status == "Success":
                if OPERATOR_DEBUG:
                    output_preview = str(output)[:200]
                    logger.debug(f"[Programmer] : {output_preview}")
                return {"code": code, "output": output}
            else:
                print(f"Execution error on attempt {i + 1}: {output}")
                feedback = (
                    f"\nThe result of the error from the code you wrote in the previous round:\n"
                    f"Code: {code}\n\nStatus: {status}, {output}"
                )

        if OPERATOR_DEBUG:
            logger.debug(f"[Programmer] (3): output={str(output)[:200]}")

        return {"code": code, "output": output}


class Test(Operator):
    """Test operator - tests code with test cases and reflects on errors"""

    def __init__(self, llm, name: str = "Test"):
        super().__init__(llm, name)

    def exec_code(self, solution: str, test_code: str, timeout: int = 30) -> str:
        """
        Execute solution with test code
        """
        def _run_test_code(code_str, result_queue):
            """"""
            try:
                global_namespace = {}
                exec(code_str, global_namespace)
                result_queue.put("no error")
            except AssertionError as e:
                result_queue.put(f"AssertionError: {str(e)}")
            except Exception as e:
                result_queue.put(f"ExecutionError: {str(e)}")

        full_code = f"{solution}\n\n{test_code}"
        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=_run_test_code,
            args=(full_code, result_queue)
        )

        try:
            process.start()
            process.join(timeout=timeout)

            if process.is_alive():
                process.terminate()
                process.join(timeout=0.5)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=0.5)
                return f"ExecutionError: Code execution timed out after {timeout}s"

            if not result_queue.empty():
                return result_queue.get_nowait()
            return "ExecutionError: Process exited without result"

        except Exception as e:
            if process.is_alive():
                process.terminate()
                process.join(timeout=0.5)
            return f"ExecutionError: {str(e)}"
        finally:
            try:
                result_queue.close()
            except:
                pass

    async def __call__(
        self,
        problem: str,
        solution: str,
        entry_point: str,
        test: str = None,
        test_loop: int = 3,
        instruction: str = "",
    ) -> Dict[str, Any]:
        """Test solution and reflect/revise if needed"""
        problem = _ensure_str(problem)
        solution = _ensure_str(solution)
        entry_point = _ensure_str(entry_point) or "solve"
        test = _ensure_str(test) if test else ""
        instruction = _ensure_str(instruction) if instruction else ""

        # Official-style tests if provided (HumanEval / BigCodeBench)
        if test:
            try:
                from src.code_execution import run_code_check
            except Exception as e:
                # Fallback to simple smoke test if backend unavailable
                if OPERATOR_DEBUG:
                    logger.debug(f"[Test] code_execution import failed: {e}")
                test = ""

        # Simple smoke test fallback - try to execute the solution
        test_code = f"# Testing {entry_point}\nresult = {entry_point}()"

        last_error_type: str = ""
        last_error_msg: str = ""
        last_traceback: str = ""

        for _ in range(test_loop):
            if test:
                check_res = run_code_check(
                    solution=solution,
                    test=test,
                    entry_point=entry_point,
                    prompt=problem,
                    problem=problem,
                    timeout=30,
                )
                if check_res.passed:
                    return {
                        "result": True,
                        "solution": solution,
                        "error_type": None,
                        "error": "",
                        "traceback": "",
                        "backend": getattr(check_res, "backend", "unknown"),
                    }

                last_error_type = str(check_res.error_type or "Error")
                last_error_msg = str(check_res.error or "")
                last_traceback = ""
                if isinstance(getattr(check_res, "meta", None), dict):
                    last_traceback = str(check_res.meta.get("traceback") or "")

                result = f"{last_error_type}: {last_error_msg}".strip()
                if last_traceback:
                    result = result + "\n" + last_traceback
            else:
                result = self.exec_code(solution, test_code)

            if result == "no error":
                return {
                    "result": True,
                    "solution": solution,
                    "error_type": None,
                    "error": "",
                    "traceback": "",
                    "backend": "smoke",
                }

            # Reflect and revise
            prompt = REFLECTION_ON_PUBLIC_TEST_PROMPT.format(
                problem=problem,
                solution=solution,
                exec_pass=f"executed unsuccessfully, error: {result}",
                test_fail=result,
            )
            if instruction and instruction.strip():
                prompt = (
                    "[Controller Guidance]\n"
                    + instruction.strip()
                    + "\n\n"
                    + prompt
                )
            response = await self._fill_node(ReflectionTestOp, prompt, mode="code_fill")
            solution = response.get("response", solution)

        # Final check
        backend_name = "smoke"
        if test:
            check_res = run_code_check(
                solution=solution,
                test=test,
                entry_point=entry_point,
                prompt=problem,
                problem=problem,
                timeout=30,
            )
            backend_name = getattr(check_res, "backend", "unknown")
            if check_res.passed:
                return {
                    "result": True,
                    "solution": solution,
                    "error_type": None,
                    "error": "",
                    "traceback": "",
                    "backend": backend_name,
                }

            last_error_type = str(check_res.error_type or "Error")
            last_error_msg = str(check_res.error or "")
            last_traceback = ""
            if isinstance(getattr(check_res, "meta", None), dict):
                last_traceback = str(check_res.meta.get("traceback") or "")
        else:
            result = self.exec_code(solution, test_code)
            if result == "no error":
                return {
                    "result": True,
                    "solution": solution,
                    "error_type": None,
                    "error": "",
                    "traceback": "",
                    "backend": "smoke",
                }
            last_error_type = "ExecutionError"
            last_error_msg = str(result)
            last_traceback = ""

        return {
            "result": False,
            "solution": solution,
            "error_type": last_error_type or "Error",
            "error": last_error_msg or "",
            "traceback": last_traceback or "",
            "backend": backend_name,
        }


class Format(Operator):
    """Format operator - extracts concise answer from solution

    Supports dynamic instruction like other operators (Verify, Review, etc.)
    """

    def __init__(self, llm, name: str = "Format"):
        super().__init__(llm, name)

    async def __call__(self, problem: str, solution: str, instruction: str = "", mode: str = None) -> Dict[str, str]:
        problem = _ensure_str(problem)
        solution = _ensure_str(solution)
        instruction = _ensure_str(instruction) if instruction else ""

        prompt = FORMAT_PROMPT.format(problem_description=problem, solution=solution)

        if instruction and instruction.strip():
            prompt = (
                "[Controller Guidance]\n"
                + instruction.strip()
                + "\n\n"
                + prompt
            )

        response = await self._fill_node(FormatOp, prompt, mode)
        return response


class Review(Operator):
    """Review operator - reviews solution correctness using critical thinking"""

    def __init__(self, llm, name: str = "Review"):
        super().__init__(llm, name)

    async def __call__(self, problem: str, solution: str, instruction: str = "", mode: str = None) -> Dict[str, Any]:
        if not isinstance(solution, str):
            solution = str(solution)
        if not isinstance(problem, str):
            problem = str(problem)
        instruction = _ensure_str(instruction) if instruction else ""

        if OPERATOR_DEBUG:
            problem_preview = problem[:150] + "..." if len(problem) > 150 else problem
            solution_preview = solution[:150] + "..." if len(solution) > 150 else solution
            logger.debug(f"[Review] :")
            logger.debug(f"  problem: {problem_preview}")
            logger.debug(f"  solution: {solution_preview}")

        prompt = REVIEW_PROMPT.format(problem=problem, solution=solution)
        if instruction and instruction.strip():
            prompt = (
                "[Controller Guidance]\n"
                + instruction.strip()
                + "\n\n"
                + prompt
            )
        response = await self._fill_node(ReviewOp, prompt, mode="xml_fill")

        # Handle boolean parsing from XML
        review_result = response.get("review_result", False)
        if isinstance(review_result, str):
            review_result = review_result.lower() in ("true", "yes", "1")

        result = {
            "output": solution,
            "review_result": review_result,
            "feedback": response.get("feedback", "")
        }

        if OPERATOR_DEBUG:
            logger.debug(f"[Review] : result={review_result}, feedback={result['feedback'][:100]}")

        return result


class Revise(Operator):
    """Revise operator - revises solution based on feedback"""

    def __init__(self, llm, name: str = "Revise"):
        super().__init__(llm, name)

    async def __call__(
        self,
        problem: str,
        solution: str,
        feedback: str,
        instruction: str = "",
        mode: str = None
    ) -> Dict[str, str]:
        if not isinstance(solution, str):
            solution = str(solution)
        if not isinstance(problem, str):
            problem = str(problem)
        if not isinstance(feedback, str):
            feedback = str(feedback)
        instruction = _ensure_str(instruction) if instruction else ""

        if OPERATOR_DEBUG:
            problem_preview = problem[:100] + "..." if len(problem) > 100 else problem
            solution_preview = solution[:100] + "..." if len(solution) > 100 else solution
            feedback_preview = feedback[:100] + "..." if len(feedback) > 100 else feedback
            logger.debug(f"[Revise] :")
            logger.debug(f"  problem: {problem_preview}")
            logger.debug(f"  solution: {solution_preview}")
            logger.debug(f"  feedback: {feedback_preview}")

        prompt = REVISE_PROMPT.format(
            problem=problem,
            solution=solution,
            feedback=feedback
        )
        if instruction and instruction.strip():
            prompt = (
                "[Controller Guidance]\n"
                + instruction.strip()
                + "\n\n"
                + prompt
            )
        response = await self._fill_node(ReviseOp, prompt, mode="xml_fill")

        if OPERATOR_DEBUG:
            sol = str(response.get('solution', response.get('response', '')))[:200]
            logger.debug(f"[Revise] : {sol}")

        return response


class MdEnsemble(Operator):
    """
    Majority voting ensemble - shuffles and votes multiple times
    Paper: Can Generalist Foundation Models Outcompete Special-Purpose Tuning?
    Link: https://arxiv.org/abs/2311.16452
    """

    def __init__(self, llm, name: str = "MdEnsemble", vote_count: int = 5):
        super().__init__(llm, name)
        self.vote_count = vote_count

    @staticmethod
    def shuffle_answers(solutions: List[str]) -> Tuple[List[str], Dict[str, int]]:
        """Shuffle solutions and create mapping"""
        shuffled_solutions = solutions.copy()
        random.shuffle(shuffled_solutions)
        answer_mapping = {
            chr(65 + i): solutions.index(sol)
            for i, sol in enumerate(shuffled_solutions)
        }
        return shuffled_solutions, answer_mapping

    async def __call__(self, solutions: List[str], problem: str, mode: str = None) -> Dict[str, str]:
        if not solutions:
            return {"solution": ""}
        if len(solutions) == 1:
            return {"solution": solutions[0]}

        all_responses = []

        for _ in range(self.vote_count):
            shuffled_solutions, answer_mapping = self.shuffle_answers(solutions)

            solution_text = ""
            for index, solution in enumerate(shuffled_solutions):
                solution_text += f"{chr(65 + index)}: \n{str(solution)}\n\n\n"

            prompt = MD_ENSEMBLE_PROMPT.format(solutions=solution_text, question=problem)
            response = await self._fill_node(MdEnsembleOp, prompt, mode="xml_fill")

            answer = response.get("solution_letter", "A")
            answer = answer.strip().upper()

            if len(answer) > 0:
                answer = answer[0]

            if answer in answer_mapping:
                original_index = answer_mapping[answer]
                all_responses.append(original_index)

        if not all_responses:
            return {"solution": solutions[0]}

        most_frequent_index = Counter(all_responses).most_common(1)[0][0]
        return {"solution": solutions[most_frequent_index]}


# ============================================
# ============================================

class Decompose(Operator):
    """
    Decompose operator - breaks complex problems into smaller sub-problems
    Based on Least-to-Most Prompting technique
    Paper: https://arxiv.org/abs/2205.10625
    """

    def __init__(self, llm, name: str = "Decompose"):
        super().__init__(llm, name)

    def _parse_sub_problems(self, text: str) -> List[str]:
        """ Decompose """
        import re
        if not text:
            return []
        pattern = r'(\d+)[.:\)]\s*(.+?)(?=\d+[.:\)]|$)'
        matches = re.findall(pattern, text, re.DOTALL)
        return [m[1].strip() for m in matches if m[1].strip()]

    async def __call__(self, problem: str, instruction: str = "") -> Dict[str, str]:
        """Decompose a problem into sub-problems
        """
        if OPERATOR_DEBUG:
            problem_preview = problem[:200] + "..." if len(problem) > 200 else problem
            logger.debug(f"[Decompose] : {problem_preview}")

        prompt = DECOMPOSE_PROMPT.format(problem=problem)
        if instruction and instruction.strip():
            prompt = (
                "[Controller Guidance]\n"
                + instruction.strip()
                + "\n\n"
                + prompt
            )
        response = await self._fill_node(DecomposeOp, prompt, mode="xml_fill")

        sub_problems_text = response.get('sub_problems', '')
        response['sub_problems_list'] = self._parse_sub_problems(sub_problems_text)

        if OPERATOR_DEBUG:
            sub_problems = str(response.get('sub_problems', ''))[:200]
            logger.debug(f"[Decompose] : {sub_problems}")

        return response


class Verify(Operator):
    """
    Verify operator - independently verifies if an answer is correct
    Based on Self-Verification technique
    """

    def __init__(self, llm, name: str = "Verify"):
        super().__init__(llm, name)

    async def __call__(self, problem: str, answer: str, instruction: str = "") -> Dict[str, Any]:
        """Verify if an answer is correct"""
        problem = _ensure_str(problem)
        answer = _ensure_str(answer)
        instruction = _ensure_str(instruction) if instruction else ""

        if OPERATOR_DEBUG:
            problem_preview = problem[:150] + "..." if len(problem) > 150 else problem
            answer_preview = answer[:150] + "..." if len(answer) > 150 else answer
            logger.debug(f"[Verify] :")
            logger.debug(f"  problem: {problem_preview}")
            logger.debug(f"  answer: {answer_preview}")

        prompt = VERIFY_PROMPT.format(problem=problem, answer=answer)
        if instruction and instruction.strip():
            prompt = (
                "[Controller Guidance]\n"
                + instruction.strip()
                + "\n\n"
                + prompt
            )
        response = await self._fill_node(VerifyOp, prompt, mode="xml_fill")

        # Parse boolean from string if needed
        is_correct = response.get("is_correct", False)
        if isinstance(is_correct, str):
            is_correct = is_correct.lower() in ("true", "yes", "1")

        suggested_answer = response.get("answer", None)
        confidence = response.get("confidence", "medium")
        verification_steps = response.get("verification_steps", "")

        verify_valid = True
        my_calc_answer = None
        conflict = False
        needs_arbitration = False

        import re as _re_self_check
        calc_match = _re_self_check.search(
            r'MY\s+CALCULATED\s+ANSWER:\s*([^\n]+)',
            str(verification_steps),
            _re_self_check.IGNORECASE
        )
        if calc_match:
            my_calc_answer = calc_match.group(1).strip()
            my_calc_answer = my_calc_answer.rstrip('.')

        def _normalize_answer(s):
            """LaTeX"""
            s = str(s).strip().lower()
            s = _re_self_check.sub(r'\\\(|\\\)|\$|\\begin\{[^}]*\}|\\end\{[^}]*\}', '', s)
            s = _re_self_check.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'\1/\2', s)
            s = _re_self_check.sub(r'\\sqrt\{([^}]*)\}', r'sqrt(\1)', s)
            s = _re_self_check.sub(r'\\(left|right|cdot|times|text\{[^}]*\})', '', s)
            s = _re_self_check.sub(r'\s+', ' ', s).strip()
            s = s.replace(', ', ',').replace(' ,', ',')
            return s

        def _try_fraction_to_float(s):
            """float"""
            s = str(s).strip()
            s = _re_self_check.sub(r'[^\d./\-]', '', s)
            if '/' in s:
                parts = s.split('/')
                if len(parts) == 2:
                    try:
                        return float(parts[0]) / float(parts[1])
                    except:
                        pass
            return float(s)

        def _are_answers_equivalent(a, b):
            """"""
            a_norm = _normalize_answer(a)
            b_norm = _normalize_answer(b)
            if a_norm == b_norm:
                return True
            try:
                a_num = _try_fraction_to_float(a_norm)
                b_num = _try_fraction_to_float(b_norm)
                if a_num == 0 and b_num == 0:
                    return True
                if a_num != 0:
                    rel_err = abs(a_num - b_num) / abs(a_num)
                    if rel_err < 0.01:
                        return True
            except:
                pass
            a_alphanum = _re_self_check.sub(r'[^a-z0-9]', '', a_norm)
            b_alphanum = _re_self_check.sub(r'[^a-z0-9]', '', b_norm)
            if a_alphanum == b_alphanum and a_alphanum:
                return True
            return False

        if my_calc_answer and suggested_answer:
            if not _are_answers_equivalent(my_calc_answer, suggested_answer):
                verify_valid = False
                confidence = 'low'
                if OPERATOR_DEBUG:
                    logger.debug(f"  ⚠️ SelfConsistency: MY CALCULATED={my_calc_answer}, <answer>={suggested_answer}")

        if suggested_answer and answer:
            if not _are_answers_equivalent(answer, suggested_answer):
                conflict = True

        def _is_short_answer(ans):
            """"""
            if not ans:
                return False
            ans_str = str(ans).strip()
            if len(ans_str) > 80:
                return False
            if '\n' in ans_str:
                return False
            long_patterns = ['Step ', 'To solve', 'Because', 'First,', '1. ', '2. ', 'The solution']
            return not any(p in ans_str for p in long_patterns)

        import re as _re_local
        def _is_clean_numeric(ans):
            if not ans:
                return False
            ans_str = str(ans).strip()
            return bool(_re_local.match(r'^[+-]?\d+(\.\d+)?(/\d+)?$', ans_str))

        def _safe_to_float(s):
            """float'1/2', '-5/2'"""
            s = str(s).strip()
            if '/' in s and not s.startswith('http'):
                try:
                    parts = s.split('/')
                    if len(parts) == 2:
                        return float(parts[0].strip()) / float(parts[1].strip())
                except:
                    pass
            return float(s)

        final_answer = answer
        original_answer = answer
        answer_overwritten = False


        should_overwrite = False
        overwrite_reason = ""

        if conflict and _is_clean_numeric(answer):
            needs_arbitration = True

        if not is_correct and suggested_answer and confidence == "high" and verify_valid:
            orig_str = str(answer).strip()
            sugg_str = str(suggested_answer).strip()

            if _is_clean_numeric(answer):
                orig_num = _safe_to_float(str(answer).strip())
                sugg_num = None
                try:
                    sugg_num = _safe_to_float(str(suggested_answer).strip())
                except:
                    pass

                if orig_num < 0 and sugg_num is not None and sugg_num > 0:
                    should_overwrite = True
                    overwrite_reason = f"{orig_num}{sugg_num}"
                else:
                    should_overwrite = False
                    overwrite_reason = ""
            elif any(kw in orig_str.lower() for kw in ['step', 'calculate', 'therefore', 'thus']):
                import re
                numbers = re.findall(r'[-+]?\d*\.?\d+', orig_str)
                if numbers:
                    extracted = numbers[-1]
                    if extracted == sugg_str:
                        should_overwrite = True
                        overwrite_reason = f"'{extracted}'"
                    else:
                        should_overwrite = False
                        overwrite_reason = f"'{extracted}''{sugg_str}'"
                else:
                    should_overwrite = False
                    overwrite_reason = ""
            elif (
                not orig_str
                or orig_str.lower() in ['none', 'null', '', 'error', 'no code generated']
                or 'execution error' in orig_str.lower()
                or 'traceback' in orig_str.lower()
                or "function 'solve' not found" in orig_str.lower()
                or 'prohibited import' in orig_str.lower()
                or 'timed out' in orig_str.lower()
                or 'timeout' in orig_str.lower()
            ):
                should_overwrite = True
                overwrite_reason = "Verify"
            else:
                should_overwrite = False
                overwrite_reason = ""

        if should_overwrite:
            final_answer = suggested_answer
            answer_overwritten = True
            if OPERATOR_DEBUG:
                logger.debug(f"  🔄 Overwrite:  - {overwrite_reason}")
        elif not is_correct and suggested_answer and OPERATOR_DEBUG:
            logger.debug(f"  🔒 Overwrite:  - {overwrite_reason}")

        result = {
            "is_correct": is_correct,
            "verification_steps": response.get("verification_steps", ""),
            "confidence": confidence,
            "answer": final_answer,
            "output": final_answer,
            "original_answer": original_answer,
            "suggested_answer": suggested_answer,
            "answer_overwritten": answer_overwritten,
            "verify_valid": verify_valid,
            "conflict": conflict,
            "needs_arbitration": needs_arbitration,
            "my_calculated_answer": my_calc_answer
        }

        if OPERATOR_DEBUG:
            status_icon = "✅" if is_correct else "❌"
            answer_preview = str(original_answer)[:100] + "..." if len(str(original_answer)) > 100 else str(original_answer)
            suggested_preview = str(suggested_answer)[:100] + "..." if suggested_answer and len(str(suggested_answer)) > 100 else str(suggested_answer)
            logger.debug(f"[Verify] {status_icon} :")
            logger.debug(f"  is_correct: {is_correct}")
            logger.debug(f"  confidence: {confidence}")
            logger.debug(f"  : {answer_preview}")
            if suggested_answer and str(suggested_answer) != str(original_answer):
                logger.debug(f"  LLM: {suggested_preview}")
            if answer_overwritten:
                final_preview = str(final_answer)[:100] + "..." if len(str(final_answer)) > 100 else str(final_answer)
                logger.debug(f"  🔄 : {final_preview}")
            elif not is_correct:
                logger.debug(f"  ⚠️ ")
            steps = response.get("verification_steps", "")
            if steps:
                steps_preview = str(steps)[:300] + "..." if len(str(steps)) > 300 else str(steps)
                logger.debug(f"  : {steps_preview}")

        return result


class Plan(Operator):
    """
    Plan operator - creates a strategic plan for solving a problem
    Based on Plan-and-Solve technique
    Paper: https://arxiv.org/abs/2305.04091
    """

    def __init__(self, llm, name: str = "Plan"):
        super().__init__(llm, name)

    def _parse_steps(self, plan_text: str) -> List[str]:
        """ Plan """
        import re
        if not plan_text:
            return []
        lines = plan_text.strip().split('\n')
        steps = []
        current_step = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            step_match = re.match(r'^(?:Step\s*)?(\d+)[.:\)]\s*(.*)$', line, re.IGNORECASE)
            if step_match:
                if current_step:
                    steps.append(current_step.strip())
                current_step = step_match.group(2)
            elif current_step is not None:
                current_step += ' ' + line
        if current_step:
            steps.append(current_step.strip())
        return steps

    async def __call__(self, problem: str, instruction: str = "") -> Dict[str, str]:
        """Create a plan for solving the problem
        """
        problem = _ensure_str(problem)

        if OPERATOR_DEBUG:
            problem_preview = problem[:200] + "..." if len(problem) > 200 else problem
            logger.debug(f"[Plan] : {problem_preview}")

        prompt = PLAN_PROMPT.format(problem=problem)
        if instruction and instruction.strip():
            prompt = (
                "[Controller Guidance]\n"
                + instruction.strip()
                + "\n\n"
                + prompt
            )
        response = await self._fill_node(PlanOp, prompt, mode="xml_fill")

        plan_text = response.get('plan', '')
        response['steps_list'] = self._parse_steps(plan_text)

        if OPERATOR_DEBUG:
            plan = str(response.get('plan', ''))[:200]
            approach = str(response.get('approach', ''))[:100]
            logger.debug(f"[Plan] : approach={approach}, plan={plan}")

        return response


class Aggregate(Operator):
    """
    Aggregate operator - combines multiple sub-answers into a final answer
    Used together with Decompose for multi-step reasoning
    """

    def __init__(self, llm, name: str = "Aggregate"):
        super().__init__(llm, name)

    async def __call__(self, problem: str, sub_answers: List[str] = None, solutions: List[str] = None, instruction: str = "", **kwargs) -> Dict[str, str]:
        """
        Aggregate sub-answers into a final answer
        - sub_answers: 
        - solutions:  ()
        """
        if sub_answers is None and solutions is not None:
            sub_answers = solutions
        if sub_answers is None:
            sub_answers = kwargs.get('answers', kwargs.get('results', []))

        if not isinstance(sub_answers, list):
            sub_answers = [str(sub_answers)] if sub_answers else []

        import re as _re_agg
        from collections import Counter as _Counter_agg

        normalized = []
        for ans in sub_answers:
            if isinstance(ans, dict):
                val = ans.get('output') or ans.get('answer') or ans.get('aggregated_answer') or ans.get('response') or ans.get('result')
                normalized.append(str(val).strip() if val else "")
            else:
                normalized.append(str(ans).strip())

        numeric_vals = []
        for v in normalized:
            if not v:
                continue
            match = _re_agg.match(r'^[+-]?\d+(\.\d+)?$', v.strip())
            if match:
                numeric_vals.append(v.strip())

        if OPERATOR_DEBUG:
            problem_preview = problem[:150] + "..." if len(problem) > 150 else problem
            logger.debug(f"[Aggregate] :")
            logger.debug(f"  problem: {problem_preview}")
            logger.debug(f"  sub_answers count: {len(sub_answers)}")
            logger.debug(f"  normalized values: {normalized[:5]}..." if len(normalized) > 5 else f"  normalized values: {normalized}")
            logger.debug(f"  numeric_vals: {numeric_vals}")

        if len(numeric_vals) >= 2:
            if len(set(numeric_vals)) == 1:
                result = {"aggregated_answer": numeric_vals[0], "selection_reason": f"All {len(numeric_vals)} candidates agree"}
                if OPERATOR_DEBUG:
                    logger.debug(f"[Aggregate] 🎯 FastPath:  {len(numeric_vals)}  = {numeric_vals[0]}")
                return result
            else:
                most_common = _Counter_agg(numeric_vals).most_common(1)
                if most_common and most_common[0][1] > 1:
                    selected = most_common[0][0]
                    votes = most_common[0][1]
                    result = {"aggregated_answer": selected, "selection_reason": f"Majority voting: {votes}/{len(numeric_vals)} votes for {selected}"}
                    if OPERATOR_DEBUG:
                        logger.debug(f"[Aggregate] 🎯 FastPath:  {selected} ({votes}/{len(numeric_vals)})")
                    return result
        elif len(numeric_vals) == 1 and OPERATOR_DEBUG:
            logger.debug(f"[Aggregate] ⚠️ FastPath: 1 {numeric_vals[0]}LLM")

        # Format sub-answers for the prompt
        formatted_sub_answers = ""
        for i, ans in enumerate(sub_answers, 1):
            formatted_sub_answers += f"Candidate {i}: {ans}\n\n"

        prompt = AGGREGATE_PROMPT.format(
            problem=problem,
            sub_answers=formatted_sub_answers
        )
        if instruction and instruction.strip():
            prompt = (
                "[Controller Guidance - CRITICAL]\n"
                + instruction.strip()
                + "\n\n"
                + prompt
            )
        response = await self._fill_node(AggregateOp, prompt, mode="xml_fill")

        if OPERATOR_DEBUG:
            agg_answer = str(response.get('aggregated_answer', ''))[:200]
            logger.debug(f"[Aggregate] LLM: {agg_answer}")

        return response


# Export all operators
__all__ = [
    'Operator',
    'Custom',
    'AnswerGenerate',
    'CustomCodeGenerate',
    'ScEnsemble',
    'Programmer',
    'Test',
    'Format',
    'Review',
    'Revise',
    'MdEnsemble',
    'run_code',
    'Decompose',
    'Verify',
    'Plan',
    'Aggregate'
]
