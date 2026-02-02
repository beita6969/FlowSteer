#!/usr/bin/env python3
"""
Code - PythonSympy
"""
import ast
import sys
import io
import traceback
import contextlib
import signal
from typing import Tuple, Dict, Any, Optional
import subprocess
import tempfile
import os


class CodeExecutor:
    """
    PythonSympy
    """

    def __init__(self, timeout: int = 10):
        """
        Args:
            timeout: 
        """
        self.timeout = timeout

    def safe_execute_code(
        self,
        code: str,
        test_code: Optional[str] = None,
        entry_point: Optional[str] = None
    ) -> Tuple[bool, str, Dict]:
        """
        Sympy

        Args:
            code: 
            test_code: 
            entry_point: 

        Returns:
            (success, output, metadata)
        """
        metadata = {
            'method': None,
            'error_type': None,
            'sympy_error': False
        }

        success, output = self._execute_restricted(code, test_code, entry_point)
        metadata['method'] = 'restricted'

        if success:
            return success, output, metadata

        if 'cannot determine truth value' in output.lower():
            metadata['sympy_error'] = True
            fixed_code = self._fix_sympy_errors(code)
            success, output = self._execute_restricted(fixed_code, test_code, entry_point)
            metadata['method'] = 'sympy_fixed'

            if success:
                return success, output, metadata

        success, output = self._execute_subprocess(code, test_code, entry_point)
        metadata['method'] = 'subprocess'

        if not success:
            metadata['error_type'] = self._classify_error(output)

        return success, output, metadata

    def _execute_restricted(
        self,
        code: str,
        test_code: Optional[str] = None,
        entry_point: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        """
        safe_globals = {
            '__builtins__': {
                'int': int, 'float': float, 'str': str, 'bool': bool,
                'list': list, 'dict': dict, 'set': set, 'tuple': tuple,
                'len': len, 'range': range, 'enumerate': enumerate,
                'zip': zip, 'map': map, 'filter': filter,
                'min': min, 'max': max, 'sum': sum, 'abs': abs,
                'sorted': sorted, 'reversed': reversed,
                'print': print, 'input': input,
                'pow': pow, 'round': round, 'divmod': divmod,
                'ord': ord, 'chr': chr, 'hex': hex, 'bin': bin,
                'isinstance': isinstance, 'type': type,
                'Exception': Exception, 'ValueError': ValueError,
                'TypeError': TypeError, 'IndexError': IndexError,
                'KeyError': KeyError,
            },
            '__name__': '__main__',
        }

        try:
            import math
            safe_globals['math'] = math
        except ImportError:
            pass

        try:
            import collections
            safe_globals['collections'] = collections
        except ImportError:
            pass

        try:
            import itertools
            safe_globals['itertools'] = itertools
        except ImportError:
            pass

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        captured_output = io.StringIO()
        sys.stdout = captured_output
        sys.stderr = captured_output

        try:
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Code execution exceeded {self.timeout} seconds")

            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.timeout)

            exec(code, safe_globals)

            if test_code:
                exec(test_code, safe_globals)

            if entry_point and entry_point in safe_globals:
                func = safe_globals[entry_point]
                if callable(func):
                    test_cases = [
                        (),
                        (0,), (1,), (5,),
                        ([],), ([1, 2, 3],),
                        ('',), ('test',),
                    ]

                    for args in test_cases:
                        try:
                            result = func(*args)
                            print(f"{entry_point}{args} = {result}")
                            break
                        except Exception:
                            continue

            signal.alarm(0)

            output = captured_output.getvalue()
            return True, output

        except TimeoutError as e:
            signal.alarm(0)
            return False, str(e)

        except Exception as e:
            signal.alarm(0)
            error_msg = f"{type(e).__name__}: {str(e)}\n"
            error_msg += traceback.format_exc()
            return False, error_msg

        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def _execute_subprocess(
        self,
        code: str,
        test_code: Optional[str] = None,
        entry_point: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)

            if test_code:
                f.write("\n\n# Tests\n")
                f.write(test_code)

            if entry_point:
                f.write(f"\n\n# Test entry point\n")
                f.write(f"if __name__ == '__main__':\n")
                f.write(f"    if '{entry_point}' in globals():\n")
                f.write(f"        func = {entry_point}\n")
                f.write(f"        try:\n")
                f.write(f"            print(f'{entry_point}(5) = {{func(5)}}')\n")
                f.write(f"        except Exception as e:\n")
                f.write(f"            print(f'Error calling {entry_point}: {{e}}')\n")

            temp_file = f.name

        try:
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            if result.returncode == 0:
                return True, result.stdout
            else:
                error_output = result.stderr or result.stdout
                return False, error_output

        except subprocess.TimeoutExpired:
            return False, f"Code execution exceeded {self.timeout} seconds"

        except Exception as e:
            return False, f"Subprocess execution error: {e}"

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def _fix_sympy_errors(self, code: str) -> str:
        """
        Sympy
        """
        fixed_code = code

        sympy_patterns = [
            (r'if\s+(\w+)\s*([<>]=?)\s*(\w+)', r'if float(\1) \2 float(\3)'),
            (r'if\s+(\w+)\s*==\s*(\w+)', r'if abs(float(\1) - float(\3)) < 1e-9'),
            (r'while\s+(\w+)\s*([<>]=?)\s*(\w+)', r'while float(\1) \2 float(\3)'),
        ]

        import re
        for pattern, replacement in sympy_patterns:
            fixed_code = re.sub(pattern, replacement, fixed_code)

        if 'sympy' in fixed_code.lower():
            helper = '''
# Sympy helper functions
def safe_compare(a, b, op='=='):
    """Safely compare Sympy expressions"""
    try:
        if hasattr(a, 'evalf'):
            a = float(a.evalf())
        if hasattr(b, 'evalf'):
            b = float(b.evalf())
        a, b = float(a), float(b)

        if op == '==':
            return abs(a - b) < 1e-9
        elif op == '<':
            return a < b
        elif op == '<=':
            return a <= b
        elif op == '>':
            return a > b
        elif op == '>=':
            return a >= b
        else:
            return False
    except:
        return False

'''
            fixed_code = helper + fixed_code

        return fixed_code

    def _classify_error(self, error_msg: str) -> str:
        """
        """
        error_lower = error_msg.lower()

        if 'cannot determine truth value' in error_lower:
            return 'sympy_relational'
        elif 'syntaxerror' in error_lower:
            return 'syntax'
        elif 'nameerror' in error_lower:
            return 'undefined_variable'
        elif 'typeerror' in error_lower:
            return 'type_error'
        elif 'indexerror' in error_lower:
            return 'index_out_of_bounds'
        elif 'keyerror' in error_lower:
            return 'missing_key'
        elif 'zerodivision' in error_lower:
            return 'division_by_zero'
        elif 'timeout' in error_lower:
            return 'timeout'
        else:
            return 'unknown'


def test_code_executor():
    """"""
    executor = CodeExecutor(timeout=5)

    print("="*60)
    print("1: ")
    print("="*60)

    code1 = '''
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)
'''

    success, output, metadata = executor.safe_execute_code(code1, entry_point='factorial')
    print(f": {success}")
    print(f": {output}")
    print(f": {metadata}")

    print("\n" + "="*60)
    print("2: Sympy")
    print("="*60)

    code2 = '''
import sympy as sp
x = sp.Symbol('x')
y = x**2 + 2*x + 1
# This would cause "cannot determine truth value" error
if y > 0:
    print("Positive")
'''

    success, output, metadata = executor.safe_execute_code(code2)
    print(f": {success}")
    print(f": {output[:200]}...")
    print(f": {metadata}")

    print("\n" + "="*60)
    print("3: ")
    print("="*60)

    code3 = '''
while True:
    pass  # Infinite loop
'''

    success, output, metadata = executor.safe_execute_code(code3)
    print(f": {success}")
    print(f": {output}")
    print(f": {metadata}")


if __name__ == "__main__":
    test_code_executor()
