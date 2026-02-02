#!/usr/bin/env python3
"""
SymPy - 
"""
import re
from typing import Tuple


class SymPyCodeFixer:
    """
    SymPy API
    """

    @staticmethod
    def fix_code(code: str) -> Tuple[str, bool, list]:
        """
        SymPy

        Args:
            code: 

        Returns:
            (fixed_code, was_modified, fixes_applied)
        """
        original_code = code
        fixes_applied = []

        if 'IntervalSet' in code or 'interval_set' in code.lower():
            code = re.sub(
                r'sp\.calculus\.util\.IntervalSet',
                'sp.Union',
                code
            )
            code = re.sub(
                r'IntervalSet\(',
                'Union(',
                code
            )
            if code != original_code:
                fixes_applied.append('IntervalSet → Union')

        code = re.sub(
            r'sp\.Interval\([^)]+\)\s*\+\s*sp\.Interval\([^)]+\)',
            lambda m: m.group(0).replace(' + ', ' | '),
            code
        )
        if 'Interval' in code and ' + ' in code:
            fixes_applied.append('Interval + → Interval |')

        if 'sympy' in code.lower() and ('if ' in code or 'while ' in code):
            helper = '''\n# SymPy safe comparison helper\ndef _sp_bool(expr):\n    """Safely convert SymPy expression to bool"""\n    try:\n        if hasattr(expr, 'evalf'):\n            return bool(expr.evalf())\n        return bool(expr)\n    except:\n        return False\n\n'''

            if '# SymPy safe comparison helper' not in code:
                code = helper + code
                fixes_applied.append('Added safe comparison helper')

        replacements = [
            ('sp.solve_univariate_inequality', 'sp.solve'),
            ('sp.calculus.singularities', 'sp.singularities'),
        ]

        for old, new in replacements:
            if old in code:
                code = code.replace(old, new)
                fixes_applied.append(f'{old} → {new}')

        if 'import sympy' in code and 'sp.' in code:
            if 'import sympy as sp' not in code:
                code = code.replace('import sympy', 'import sympy as sp')
                fixes_applied.append('Fixed sympy import alias')

        was_modified = (code != original_code)
        return code, was_modified, fixes_applied

    @staticmethod
    def add_safe_execution_wrapper(code: str) -> str:
        """

        Args:
            code: 

        Returns:
        """
        wrapper = '''\ntry:\n{indented_code}\nexcept Exception as e:\n    print(f"Execution error: {{e}}")\n    import traceback\n    traceback.print_exc()\n'''

        indented_lines = ['    ' + line for line in code.split('\n')]
        indented_code = '\n'.join(indented_lines)

        return wrapper.format(indented_code=indented_code)

    @staticmethod
    def validate_code_safety(code: str) -> Tuple[bool, list]:
        """

        Args:
            code: 

        Returns:
            (is_safe, warnings)
        """
        warnings = []

        dangerous_patterns = [
            (r'\bexec\(', 'Uses exec()'),
            (r'\beval\(', 'Uses eval()'),
            (r'\b__import__\(', 'Uses __import__()'),
            (r'\bopen\([^)]*["\']w', 'Opens file for writing'),
            (r'\bos\.system\(', 'Uses os.system()'),
            (r'\bsubprocess\.', 'Uses subprocess'),
        ]

        for pattern, warning in dangerous_patterns:
            if re.search(pattern, code):
                warnings.append(warning)

        is_safe = len(warnings) == 0
        return is_safe, warnings


def test_sympy_fixer():
    """SymPy"""
    fixer = SymPyCodeFixer()

    code1 = '''
import sympy as sp
result = sp.calculus.util.IntervalSet(-sp.oo, 2)
print(result)
'''
    fixed1, modified1, fixes1 = fixer.fix_code(code1)
    print("1 - IntervalSet:")
    print(f"  : {modified1}")
    print(f"  : {fixes1}")
    print(f"  : {fixed1[:100]}...\n")

    code2 = '''
import sympy as sp
interval = sp.Interval(-2, 2) + sp.Interval(3, 5)
print(interval)
'''
    fixed2, modified2, fixes2 = fixer.fix_code(code2)
    print("2 - Interval:")
    print(f"  : {modified2}")
    print(f"  : {fixes2}\n")

    code3 = '''
import os
os.system("rm -rf /")
'''
    is_safe, warnings = fixer.validate_code_safety(code3)
    print("3 - :")
    print(f"  : {is_safe}")
    print(f"  : {warnings}")


if __name__ == "__main__":
    test_sympy_fixer()
