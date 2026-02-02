#!/usr/bin/env python3
"""
Code execution backend for HumanEval / BigCodeBench.

Design goals:
- Run official-style tests (HumanEval: check(candidate), BigCodeBench: unittest.TestCases).
- Prefer BigCodeBench official environment (Docker) for dependency completeness.
- Keep the host Python environment dependency-free (stdlib only).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Optional, Dict, Any


DEFAULT_BACKEND = os.environ.get("COLAB_GRPO_CODE_EVAL_BACKEND", "auto").lower()  # auto|docker|local
DEFAULT_DOCKER_IMAGE = os.environ.get(
    "COLAB_GRPO_CODE_EVAL_DOCKER_IMAGE",
    "colab-grpo/bigcodebench-eval:latest",
)


@dataclass
class CodeCheckResult:
    passed: bool
    backend: str
    error_type: Optional[str] = None
    error: Optional[str] = None
    stdout: str = ""
    stderr: str = ""
    meta: Optional[Dict[str, Any]] = None


def extract_python_code(text: str) -> str:
    """Extract Python code from a model response.

    Heuristics:
    - Prefer fenced code blocks; pick the *last* syntactically valid Python block when possible.
    - Otherwise, fall back to the first "code-looking" line (import/def/class/@) to the end.
    """
    if not text:
        return ""

    s = str(text).strip()
    if not s:
        return ""

    # 0) Some pipelines accidentally serialize code as a list of strings, e.g.:
    # - "['def f():\\n  ...']" (Python repr)
    # - "[\"def f():\\n  ...\"]" (JSON)
    # Unwrap these early so the normal heuristics can find the real code.
    if s.startswith("[") and s.endswith("]"):
        unwrapped = None
        # JSON first
        try:
            loaded = json.loads(s)
            if isinstance(loaded, list) and all(isinstance(x, str) for x in loaded):
                unwrapped = "\n".join(x.strip("\n") for x in loaded if x is not None)
        except Exception:
            pass
        if unwrapped is None:
            # Python literal repr
            try:
                import ast
                loaded = ast.literal_eval(s)
                if isinstance(loaded, list) and all(isinstance(x, str) for x in loaded):
                    unwrapped = "\n".join(x.strip("\n") for x in loaded if x is not None)
            except Exception:
                pass
        if isinstance(unwrapped, str) and unwrapped.strip():
            s = unwrapped.strip()

    def _is_valid_python(code: str) -> bool:
        try:
            import ast
            ast.parse(code)
            return True
        except Exception:
            return False

    # 1) Fenced blocks: ```python ... ``` or ``` ... ```
    blocks = []
    for m in re.finditer(r"```(?:python)?\s*\n?(.*?)```", s, re.DOTALL | re.IGNORECASE):
        block = (m.group(1) or "").strip()
        if block:
            blocks.append(block)

    if blocks:
        # Prefer blocks that look like solutions.
        def _priority(code: str) -> tuple:
            has_def = ("def " in code) or ("class " in code)
            has_import = ("import " in code) or ("from " in code)
            valid = _is_valid_python(code)
            # Sort: valid > has_def > has_import > length
            return (1 if valid else 0, 1 if has_def else 0, 1 if has_import else 0, len(code))

        blocks_sorted = sorted(blocks, key=_priority)
        return blocks_sorted[-1]

    # 2) No fences: try to extract from first code-looking line.
    code_start = re.search(r"(?m)^(?:from\s+\S+\s+import\s+|import\s+\S+|def\s+\w+\s*\(|class\s+\w+\s*\(|@)", s)
    if code_start:
        candidate = s[code_start.start():].strip()
        # Common tail separators; keep code above them.
        for sep in ("\nExplanation:", "\nEXPLANATION:", "\n# Explanation", "\nNotes:", "\nNOTE:"):
            if sep in candidate:
                candidate = candidate.split(sep, 1)[0].rstrip()
        return candidate.strip() or s

    return s


def is_bigcodebench_test(test: str) -> bool:
    t = str(test or "")
    return ("class TestCases" in t) or ("unittest.TestCase" in t)


def is_humaneval_test(test: str) -> bool:
    return "def check(candidate)" in str(test or "")


def extract_code_prompt_from_problem(problem: str) -> str:
    """Extract BigCodeBench code_prompt from problem field if present."""
    if not problem:
        return ""

    # ```python ... ``` or ``` ... ```
    match = re.search(r"```(?:python)?\s*\n(.*?)```", problem, re.DOTALL)
    if match:
        code_prompt = match.group(1).strip()
        if "def " in code_prompt:
            return code_prompt

    if "starting with:" in problem.lower():
        parts = re.split(r"starting with:\s*", problem, flags=re.IGNORECASE)
        if len(parts) > 1:
            code_part = parts[1]
            match = re.search(r"```(?:python)?\s*\n(.*?)```", code_part, re.DOTALL)
            if match:
                return match.group(1).strip()

    return ""


def infer_entry_point_from_test(test: str) -> Optional[str]:
    """Infer entry point from test code (covers BigCodeBench task_func + HumanEval candidate)."""
    t = str(test or "")

    if "task_func(" in t:
        return "task_func"

    match = re.search(r"candidate\s*=\s*([a-zA-Z_][a-zA-Z0-9_]*)", t)
    if match:
        return match.group(1)

    match = re.search(r"result\s*=\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", t)
    if match:
        return match.group(1)

    builtin_funcs = {
        "True", "False", "None",
        "len", "str", "int", "float", "list", "dict", "set", "tuple",
        "abs", "sum", "max", "min", "sorted", "type", "isinstance", "print",
    }

    # Common style: `assert (func(x)) == y` (leading parentheses).
    match = re.search(r"assert\s*\(?\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", t)
    if match and match.group(1) not in builtin_funcs:
        return match.group(1)

    match = re.search(r"self\.assert\w+\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", t)
    if match and match.group(1) not in builtin_funcs:
        return match.group(1)

    return None


def _docker_available() -> bool:
    return shutil.which("docker") is not None


_DOCKER_IMAGE_CACHE: Dict[str, bool] = {}


def _docker_image_available(image: str) -> bool:
    if not image:
        return False
    if image in _DOCKER_IMAGE_CACHE:
        return _DOCKER_IMAGE_CACHE[image]
    if not _docker_available():
        _DOCKER_IMAGE_CACHE[image] = False
        return False
    try:
        subprocess.run(
            ["docker", "image", "inspect", image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=10,
        )
        _DOCKER_IMAGE_CACHE[image] = True
    except Exception:
        _DOCKER_IMAGE_CACHE[image] = False
    return _DOCKER_IMAGE_CACHE[image]


def _write_runner_files(
    workdir: str,
    solution: str,
    test: str,
    entry_point: str,
    prompt: str,
) -> str:
    os.makedirs(workdir, exist_ok=True)
    try:
        # Docker runs as a non-host user; allow write for pycache/temp files.
        os.chmod(workdir, 0o777)
    except Exception:
        pass
    with open(os.path.join(workdir, "solution.py"), "w", encoding="utf-8") as f:
        f.write(solution)
    try:
        os.chmod(os.path.join(workdir, "solution.py"), 0o644)
    except Exception:
        pass
    with open(os.path.join(workdir, "test_code.py"), "w", encoding="utf-8") as f:
        f.write(test)
    try:
        os.chmod(os.path.join(workdir, "test_code.py"), 0o644)
    except Exception:
        pass
    with open(os.path.join(workdir, "prompt.txt"), "w", encoding="utf-8") as f:
        f.write(prompt or "")
    try:
        os.chmod(os.path.join(workdir, "prompt.txt"), 0o644)
    except Exception:
        pass

    runner = r"""
import json
import traceback
import unittest
import io
import re
from typing import List, Dict, Tuple, Optional, Any

def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def _transform_asserts(test_code: str) -> str:
    '''TestError v8:  assert func()==expected '''
    lines = test_code.split('\n')
    new_lines = []
    for line in lines:
        stripped = line.strip()
        match = re.match(r'^assert\s+(.+?)\s*==\s*(.+)$', stripped)
        if match:
            actual_expr = match.group(1).strip()
            expected_expr = match.group(2).strip()
            indent = line[:len(line) - len(line.lstrip())]
            new_line = f'''{indent}_actual = {actual_expr}; _expected = {expected_expr}; assert _actual == _expected, f"Expected {{_expected}} (type: {{type(_expected).__name__}}), got {{_actual}} (type: {{type(_actual).__name__}})"'''
            new_lines.append(new_line)
        else:
            new_lines.append(line)
    return '\n'.join(new_lines)

def main():
    solution = _read("solution.py")
    test = _read("test_code.py")
    prompt = _read("prompt.txt")
    entry_point = _read("entry_point.txt").strip()

    global_dict = {
        "math": __import__("math"),
        "hashlib": __import__("hashlib"),
        "re": __import__("re"),
        "List": List,
        "Dict": Dict,
        "Tuple": Tuple,
        "Optional": Optional,
        "Any": Any,
        "unittest": unittest,
    }

    is_bigcodebench = ("class TestCases" in test) or ("unittest.TestCase" in test)
    is_humaneval = ("def check(candidate)" in test)

    try:
        if is_bigcodebench:
            # Exec with stable filenames so tracebacks can be mapped back to solution/test.
            exec(compile(solution, "solution.py", "exec"), global_dict)
            exec(compile(test, "test_code.py", "exec"), global_dict)
            if "TestCases" not in global_dict:
                print(json.dumps({"passed": False, "error_type": "NameError", "error": "TestCases not defined"}))
                return
            suite = unittest.TestLoader().loadTestsFromTestCase(global_dict["TestCases"])
            stream = io.StringIO()
            runner = unittest.TextTestRunner(stream=stream, verbosity=0)
            result = runner.run(suite)
            passed = (result.testsRun > 0) and (len(result.failures) == 0) and (len(result.errors) == 0)
            if passed:
                print(json.dumps({"passed": True}))
            else:
                summary = stream.getvalue()
                print(json.dumps({
                    "passed": False,
                    "error_type": "UnitTestFailure",
                    "error": summary[-2000:],
                }))
            return

        if is_humaneval:
            # HumanEval official-style: prompt + completion + test + check(candidate)
            # Always include prompt if provided, because it may contain helper defs/imports.
            if prompt:
                exec(compile(prompt, "prompt.py", "exec"), global_dict)
            exec(compile(solution, "solution.py", "exec"), global_dict)
            exec(compile(test, "test_code.py", "exec"), global_dict)

	            if "check" not in global_dict:
	                print(json.dumps({"passed": False, "error_type": "NameError", "error": "check(candidate) not defined"}))
	                return
	            cand = global_dict.get(entry_point)
	            if cand is None:
	                print(json.dumps({"passed": False, "error_type": "NameError", "error": f"entry_point not found: {entry_point}"}))
	                return
	            global_dict["check"](cand)
	            print(json.dumps({"passed": True}))
	            return

        # Generic: exec solution, then exec test.
        test_transformed = _transform_asserts(test)
        exec(compile(solution, "solution.py", "exec"), global_dict)
        exec(compile(test_transformed, "test_code.py", "exec"), global_dict)
        if "check" in global_dict and entry_point:
            global_dict["check"](global_dict.get(entry_point))
        print(json.dumps({"passed": True}))
        return

    except AssertionError as e:
        tb = traceback.format_exc(limit=8)
        error_msg = str(e)
        if not error_msg:
            for line in tb.splitlines():
                if 'assert ' in line.lower():
                    error_msg = line.strip()
                    break
        print(json.dumps({
            "passed": False,
            "error_type": "AssertionError",
            "error": error_msg,
            "traceback": tb,
        }))
        return
    except Exception as e:
        print(json.dumps({
            "passed": False,
            "error_type": type(e).__name__,
            "error": str(e),
            "traceback": traceback.format_exc(limit=8),
        }))
        return

	if __name__ == "__main__":
	    main()
    """

    runner = runner.replace("\t", "")

    runner_path = os.path.join(workdir, "runner.py")
    with open(runner_path, "w", encoding="utf-8") as f:
        f.write(runner)
    try:
        os.chmod(runner_path, 0o644)
    except Exception:
        pass
    with open(os.path.join(workdir, "entry_point.txt"), "w", encoding="utf-8") as f:
        f.write(entry_point or "")
    try:
        os.chmod(os.path.join(workdir, "entry_point.txt"), 0o644)
    except Exception:
        pass
    return runner_path


def _run_runner_local(workdir: str, runner_path: str, timeout: int) -> CodeCheckResult:
    try:
        proc = subprocess.run(
            [sys.executable, runner_path],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()
        result = _parse_runner_output(stdout, stderr)
        result.backend = "local"
        return result
    except subprocess.TimeoutExpired as e:
        stdout = (getattr(e, "stdout", "") or "")
        stderr = (getattr(e, "stderr", "") or "")
        return CodeCheckResult(
            passed=False,
            backend="local",
            error_type="Timeout",
            error=f"Runner timed out after {timeout}s",
            stdout=str(stdout),
            stderr=str(stderr),
            meta={"timeout": int(timeout)},
        )


def _run_runner_docker(workdir: str, timeout: int, docker_image: str) -> CodeCheckResult:
    cmd = [
        "docker", "run", "--rm",
        "--network", "none",
        "-e", "PYTHONDONTWRITEBYTECODE=1",
        "-v", f"{workdir}:/app",
        "-w", "/app",
        "--entrypoint", "python3",
        docker_image,
        "runner.py",
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()
        result = _parse_runner_output(stdout, stderr)
        result.backend = "docker"
        result.stdout = stdout
        result.stderr = stderr
        return result
    except subprocess.TimeoutExpired as e:
        stdout = (getattr(e, "stdout", "") or "")
        stderr = (getattr(e, "stderr", "") or "")
        return CodeCheckResult(
            passed=False,
            backend="docker",
            error_type="Timeout",
            error=f"Docker runner timed out after {timeout}s",
            stdout=str(stdout),
            stderr=str(stderr),
            meta={"timeout": int(timeout), "docker_image": str(docker_image or "")},
        )


def _parse_runner_output(stdout: str, stderr: str) -> CodeCheckResult:
    text = stdout.strip().splitlines()[-1].strip() if stdout.strip() else ""
    try:
        payload = json.loads(text)
        passed = bool(payload.get("passed", False))
        return CodeCheckResult(
            passed=passed,
            backend="local",
            error_type=payload.get("error_type"),
            error=payload.get("error"),
            stdout=stdout,
            stderr=stderr,
            meta=payload,
        )
    except Exception:
        # If runner didn't output JSON, treat as failure and include raw stdout/stderr.
        return CodeCheckResult(
            passed=False,
            backend="local",
            error_type="RunnerError",
            error=(stderr or stdout)[:500],
            stdout=stdout,
            stderr=stderr,
            meta=None,
        )


def _should_try_docker_fallback(result: CodeCheckResult) -> bool:
    if result.passed:
        return False
    et = (result.error_type or "").lower()
    msg = (result.error or "").lower()
    if "modulenotfounderror" in et or "importerror" in et:
        return True
    if "no module named" in msg:
        return True
    return False


def run_code_check(
    solution: str,
    test: str,
    entry_point: Optional[str] = None,
    prompt: str = "",
    problem: str = "",
    source: str = "",
    timeout: int = 15,
    backend: str = DEFAULT_BACKEND,
    docker_image: str = DEFAULT_DOCKER_IMAGE,
    code_prompt: str = "",
) -> CodeCheckResult:
    """
    Run code tests with official-ish semantics.

    - BigCodeBench: supports calibrated mode via code_prompt (prefix + 'pass' + completion).
    - HumanEval: prompt + completion + test + check(entry_point).
    """
    if not solution or not test:
        return CodeCheckResult(passed=False, backend="none", error_type="ValueError", error="Missing solution/test")

    solution_code = extract_python_code(solution)
    test_code = str(test).strip()

    entry_point_final = (entry_point or "").strip() or infer_entry_point_from_test(test_code) or ""

    # BigCodeBench calibrated mode: code_prompt + "\n    pass\n" + completion
    is_bcb = ("bigcodebench" in (source or "").lower()) or is_bigcodebench_test(test_code)
    if is_bcb:
        cp = code_prompt or extract_code_prompt_from_problem(problem)
        if cp:
            if entry_point_final and re.search(rf"\bdef\s+{re.escape(entry_point_final)}\s*\(", solution_code):
                pass
            else:
                solution_code = cp + "\n    pass\n" + solution_code

    is_he = ("humaneval" in (source or "").lower()) or is_humaneval_test(test_code)
    prompt_for_humaneval = prompt if is_he else ""

    effective_backend = (backend or "auto").lower()
    if effective_backend not in {"auto", "docker", "local"}:
        effective_backend = "auto"

    if effective_backend == "auto" and is_bcb:
        # BigCodeBench strongly prefers docker for dependency completeness.
        effective_backend = "docker" if _docker_image_available(docker_image) else "local"

    with tempfile.TemporaryDirectory(prefix="code_eval_") as tmpdir:
        runner_path = _write_runner_files(
            tmpdir,
            solution_code,
            test_code,
            entry_point_final,
            prompt_for_humaneval,
        )

        local_result = None
        if effective_backend in {"auto", "local"}:
            local_result = _run_runner_local(tmpdir, runner_path, timeout=timeout)
            if effective_backend == "local":
                return local_result

        if effective_backend in {"auto", "docker"}:
            if not _docker_image_available(docker_image):
                if local_result is not None:
                    return local_result
                return CodeCheckResult(
                    passed=False,
                    backend="docker",
                    error_type="DockerImageMissing",
                    error=f"Docker image not found: {docker_image}",
                )

            if local_result is None or _should_try_docker_fallback(local_result) or is_bcb:
                return _run_runner_docker(tmpdir, timeout=timeout, docker_image=docker_image)

        return local_result or CodeCheckResult(
            passed=False,
            backend="auto",
            error_type="Unknown",
            error="No backend executed",
        )


def run_stdin_stdout_check(
    solution: str,
    inputs: list,
    outputs: list,
    timeout: int = 15,
    lenient: bool = False,
) -> CodeCheckResult:
    """
    Run stdin/stdout tests for APPS-style problems.

    Args:
        solution: The Python code to test
        inputs: List of input strings (stdin)
        outputs: List of expected output strings (stdout)
        timeout: Timeout per test case in seconds
        lenient: If True, use lenient comparison (strip trailing spaces, float tolerance)

    Returns:
        CodeCheckResult with pass/fail status
    """
    if not solution:
        return CodeCheckResult(passed=False, backend="local", error_type="ValueError", error="Missing solution")

    if not inputs or not outputs:
        return CodeCheckResult(passed=False, backend="local", error_type="ValueError", error="Missing test cases")

    solution_code = extract_python_code(solution)

    solution_code = re.sub(r'^```(?:python)?\s*\n?', '', solution_code)
    solution_code = re.sub(r'\n?```\s*$', '', solution_code)
    solution_code = solution_code.strip()

    auto_imports = []
    if 'sys.' in solution_code and 'import sys' not in solution_code:
        auto_imports.append('import sys')
    if 'math.' in solution_code and 'import math' not in solution_code:
        auto_imports.append('import math')
    if 'collections.' in solution_code and 'import collections' not in solution_code:
        auto_imports.append('import collections')
    if 'itertools.' in solution_code and 'import itertools' not in solution_code:
        auto_imports.append('import itertools')
    if 'heapq.' in solution_code and 'import heapq' not in solution_code:
        auto_imports.append('import heapq')
    if 'bisect.' in solution_code and 'import bisect' not in solution_code:
        auto_imports.append('import bisect')
    if 're.' in solution_code and 'import re' not in solution_code:
        auto_imports.append('import re')
    if 'functools.' in solution_code and 'import functools' not in solution_code:
        auto_imports.append('import functools')
    if 'defaultdict' in solution_code and 'from collections import' not in solution_code and 'import collections' not in solution_code:
        auto_imports.append('from collections import defaultdict, deque, Counter')
    if 'deque' in solution_code and 'from collections import' not in solution_code and 'import collections' not in solution_code:
        auto_imports.append('from collections import defaultdict, deque, Counter')

    if auto_imports:
        auto_imports = list(dict.fromkeys(auto_imports))
        solution_code = '\n'.join(auto_imports) + '\n\n' + solution_code

    solution_code = re.sub(r'input\s*=\s*sys\.stdin\.read(?!\()', 'input = sys.stdin.read()', solution_code)

    solution_code = re.sub(r'input\(\)\.', 'input.', solution_code)
    if 'input = sys.stdin.read()' in solution_code:
        solution_code = re.sub(r'(?<!= )input\(\)', 'input', solution_code)

    if 'def solve(' in solution_code or 'def solve()' in solution_code:
        solve_uses_return = False
        in_solve_func = False
        for line in solution_code.split('\n'):
            stripped = line.strip()
            if stripped.startswith('def solve(') or stripped.startswith('def solve()'):
                in_solve_func = True
            elif in_solve_func and stripped.startswith('def '):
                in_solve_func = False
            elif in_solve_func and stripped.startswith('return ') and not stripped.startswith('return None'):
                solve_uses_return = True
                break

        lines = solution_code.strip().split('\n')
        has_solve_call = False
        for line in lines[-5:]:
            stripped = line.strip()
            if stripped == 'solve()' or stripped == 'print(solve())' or stripped.startswith('solve(') or 'solve()' in stripped:
                has_solve_call = True
                break

        if not has_solve_call:
            if solve_uses_return:
                solution_code = solution_code.rstrip() + '\n\nprint(solve())\n'
            else:
                solution_code = solution_code.rstrip() + '\n\nsolve()\n'

    max_tests = min(len(inputs), len(outputs), 10)
    test_inputs = inputs[:max_tests]
    test_outputs = outputs[:max_tests]

    passed_count = 0
    failed_tests = []

    for i, (inp, expected_out) in enumerate(zip(test_inputs, test_outputs)):
        try:
            with tempfile.TemporaryDirectory(prefix="apps_eval_") as tmpdir:
                solution_path = os.path.join(tmpdir, "solution.py")
                with open(solution_path, "w", encoding="utf-8") as f:
                    f.write(solution_code)

                proc = subprocess.run(
                    [sys.executable, solution_path],
                    input=inp,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=tmpdir,
                )

                actual_out = proc.stdout.strip()
                expected_out_clean = expected_out.strip()

                if lenient:
                    def normalize_output(s):
                        """"""
                        lines = s.strip().split('\n')
                        lines = [line.rstrip() for line in lines]
                        return '\n'.join(lines)

                    def compare_values(expected, actual):
                        """"""
                        expected = expected.strip()
                        actual = actual.strip()
                        if expected == actual:
                            return True
                        try:
                            exp_float = float(expected)
                            act_float = float(actual)
                            if abs(exp_float - act_float) < 1e-9:
                                return True
                            if abs(exp_float) > 1e-9 and abs((exp_float - act_float) / exp_float) < 1e-6:
                                return True
                        except (ValueError, ZeroDivisionError):
                            pass
                        return False

                    actual_normalized = normalize_output(actual_out)
                    expected_normalized = normalize_output(expected_out_clean)

                    if actual_normalized == expected_normalized:
                        passed_count += 1
                    else:
                        actual_lines = actual_normalized.split('\n')
                        expected_lines = expected_normalized.split('\n')
                        if len(actual_lines) == len(expected_lines):
                            all_match = True
                            for a_line, e_line in zip(actual_lines, expected_lines):
                                a_words = a_line.split()
                                e_words = e_line.split()
                                if len(a_words) != len(e_words):
                                    all_match = False
                                    break
                                for a_word, e_word in zip(a_words, e_words):
                                    if not compare_values(e_word, a_word):
                                        all_match = False
                                        break
                                if not all_match:
                                    break
                            if all_match:
                                passed_count += 1
                            else:
                                failed_tests.append({
                                    "test_idx": i,
                                    "input": inp[:100] + "..." if len(inp) > 100 else inp,
                                    "expected": expected_out_clean[:100] + "..." if len(expected_out_clean) > 100 else expected_out_clean,
                                    "actual": actual_out[:100] + "..." if len(actual_out) > 100 else actual_out,
                                    "stderr": proc.stderr[:200] if proc.stderr else None,
                                })
                        else:
                            failed_tests.append({
                                "test_idx": i,
                                "input": inp[:100] + "..." if len(inp) > 100 else inp,
                                "expected": expected_out_clean[:100] + "..." if len(expected_out_clean) > 100 else expected_out_clean,
                                "actual": actual_out[:100] + "..." if len(actual_out) > 100 else actual_out,
                                "stderr": proc.stderr[:200] if proc.stderr else None,
                            })
                else:
                    def smart_compare(expected, actual):
                        """"""
                        expected = expected.strip()
                        actual = actual.strip()
                        if expected == actual:
                            return True
                        exp_lines = expected.split('\n')
                        act_lines = actual.split('\n')
                        if len(exp_lines) != len(act_lines):
                            return False
                        for exp_line, act_line in zip(exp_lines, act_lines):
                            exp_line = exp_line.strip()
                            act_line = act_line.strip()
                            if exp_line == act_line:
                                continue
                            try:
                                exp_float = float(exp_line)
                                act_float = float(act_line)
                                if abs(exp_float - act_float) < 1e-9:
                                    continue
                                if abs(exp_float) > 1e-15 and abs((exp_float - act_float) / exp_float) < 1e-6:
                                    continue
                                return False
                            except (ValueError, ZeroDivisionError):
                                exp_words = exp_line.split()
                                act_words = act_line.split()
                                if len(exp_words) != len(act_words):
                                    return False
                                for exp_w, act_w in zip(exp_words, act_words):
                                    if exp_w == act_w:
                                        continue
                                    try:
                                        exp_f = float(exp_w)
                                        act_f = float(act_w)
                                        if abs(exp_f - act_f) < 1e-9:
                                            continue
                                        if abs(exp_f) > 1e-15 and abs((exp_f - act_f) / exp_f) < 1e-6:
                                            continue
                                        return False
                                    except (ValueError, ZeroDivisionError):
                                        return False
                        return True

                    def order_insensitive_compare(expected, actual):
                        """

                        :
                        1. : 
                        2. :  spell  "c a"  "a c" 
                        """
                        expected = expected.strip()
                        actual = actual.strip()
                        exp_lines = expected.split('\n')
                        act_lines = actual.split('\n')

                        if len(exp_lines) != len(act_lines):
                            return False

                        if len(exp_lines) <= 1:
                            return False

                        if exp_lines[0].strip() != act_lines[0].strip():
                            return False

                        def normalize_line(line):
                            parts = line.strip().split()
                            return tuple(sorted(parts))

                        exp_set = set(normalize_line(line) for line in exp_lines[1:])
                        act_set = set(normalize_line(line) for line in act_lines[1:])

                        return exp_set == act_set

                    if smart_compare(expected_out_clean, actual_out) or order_insensitive_compare(expected_out_clean, actual_out):
                        passed_count += 1
                    else:
                        failed_tests.append({
                            "test_idx": i,
                            "input": inp[:100] + "..." if len(inp) > 100 else inp,
                            "expected": expected_out_clean[:100] + "..." if len(expected_out_clean) > 100 else expected_out_clean,
                            "actual": actual_out[:100] + "..." if len(actual_out) > 100 else actual_out,
                            "stderr": proc.stderr[:200] if proc.stderr else None,
                        })

        except subprocess.TimeoutExpired:
            failed_tests.append({
                "test_idx": i,
                "input": inp[:50] + "..." if len(inp) > 50 else inp,
                "error": f"Timeout after {timeout}s",
            })
        except Exception as e:
            failed_tests.append({
                "test_idx": i,
                "input": inp[:50] + "..." if len(inp) > 50 else inp,
                "error": str(e)[:200],
            })

    all_passed = (passed_count == max_tests)

    if all_passed:
        return CodeCheckResult(
            passed=True,
            backend="local",
            meta={"passed_tests": passed_count, "total_tests": max_tests},
        )
    else:
        error_msg = f"Passed {passed_count}/{max_tests} tests."
        if failed_tests:
            first_fail = failed_tests[0]
            if "error" in first_fail:
                error_msg += f" First failure: {first_fail['error']}"
            else:
                error_msg += f" First failure: Expected '{first_fail['expected']}', got '{first_fail['actual']}'"

        return CodeCheckResult(
            passed=False,
            backend="local",
            error_type="TestFailure",
            error=error_msg,
            meta={
                "passed_tests": passed_count,
                "total_tests": max_tests,
                "failed_tests": failed_tests[:3],
            },
        )
