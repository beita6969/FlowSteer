#!/usr/bin/env python3
"""
 - 
"""
import ast
import re
from typing import Tuple, Dict, List


class WorkflowValidator:
    """
    RL

    1. 
    2. 
    3. 
    4. 
    """

    def __init__(self):
        self.valid_operators = [
            'Custom', 'AnswerGenerate', 'Programmer', 'ScEnsemble',
            'Test', 'Review', 'Revise', 'CustomCodeGenerate',
            'Format', 'MdEnsemble', 'Decompose', 'Verify'
        ]

        self.operator_requirements = {
            'Custom': ['input', 'instruction'],
            'AnswerGenerate': ['input'],
            'Programmer': ['problem', 'analysis'],
            'ScEnsemble': ['solutions', 'problem'],
            'Test': ['problem', 'solution', 'entry_point'],
            'Review': ['problem', 'solution'],
            'Revise': ['problem', 'solution', 'feedback'],
            'CustomCodeGenerate': ['problem', 'entry_point', 'instruction'],
            'Format': ['problem', 'solution'],
            'MdEnsemble': ['solutions', 'problem'],
            'Decompose': ['problem'],
            'Verify': ['problem', 'answer']
        }

    def validate_workflow_code(self, code: str, problem_type: str = 'math') -> Tuple[bool, str, Dict]:
        """

        Args:
            code: Python
            problem_type:  (math/code/qa)

        Returns:
            (is_valid, error_message, validation_details)
        """
        validation_details = {
            'syntax_valid': False,
            'has_workflow_class': False,
            'has_call_method': False,
            'has_return': False,
            'operators_valid': False,
            'async_calls_valid': False,
            'warnings': []
        }

        try:
            tree = ast.parse(code)
            validation_details['syntax_valid'] = True
        except SyntaxError as e:
            return False, f": {e}", validation_details

        has_workflow_class = any(
            isinstance(node, ast.ClassDef) and node.name == 'Workflow'
            for node in ast.walk(tree)
        )
        validation_details['has_workflow_class'] = has_workflow_class
        if not has_workflow_class:
            return False, "Workflow", validation_details

        has_call_method = self._has_call_method(tree)
        validation_details['has_call_method'] = has_call_method
        if not has_call_method:
            return False, "async def __call__", validation_details

        has_return = self._has_return_in_call(tree)
        validation_details['has_return'] = has_return
        if not has_return:
            return False, "__call__return", validation_details

        operator_issues = self._check_operators(code)
        if operator_issues:
            validation_details['operators_valid'] = False
            validation_details['warnings'].extend(operator_issues)
        else:
            validation_details['operators_valid'] = True

        async_issues = self._check_async_calls(code)
        if async_issues:
            validation_details['async_calls_valid'] = False
            validation_details['warnings'].extend(async_issues)
        else:
            validation_details['async_calls_valid'] = True

        if problem_type == 'code':
            code_issues = self._check_code_workflow(tree, code)
            if code_issues:
                validation_details['warnings'].extend(code_issues)

        if validation_details['warnings']:
            warning_msg = '; '.join(validation_details['warnings'])
            return True, f": {warning_msg}", validation_details

        return True, "", validation_details

    def _has_call_method(self, tree: ast.AST) -> bool:
        """__call__"""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'Workflow':
                for item in node.body:
                    if isinstance(item, ast.AsyncFunctionDef) and item.name == '__call__':
                        return True
        return False

    def _has_return_in_call(self, tree: ast.AST) -> bool:
        """__call__return"""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'Workflow':
                for item in node.body:
                    if isinstance(item, ast.AsyncFunctionDef) and item.name == '__call__':
                        for stmt in ast.walk(item):
                            if isinstance(stmt, ast.Return):
                                return True
        return False

    def _check_operators(self, code: str) -> List[str]:
        """"""
        issues = []

        lowercase_pattern = r'operator\.([a-z][a-zA-Z_]*?)\('
        lowercase_matches = re.findall(lowercase_pattern, code)
        for match in lowercase_matches:
            issues.append(f"PascalCase: operator.{match} -> operator.{match.capitalize()}")

        operator_pattern = r'operator\.([A-Z][a-zA-Z_]*?)\('
        operator_matches = re.findall(operator_pattern, code)
        for op in operator_matches:
            if op not in self.valid_operators:
                issues.append(f": {op}")

        if 'self.test' in code:
            test_pattern = r'self\.test\([^)]*\)'
            test_calls = re.findall(test_pattern, code)
            for call in test_calls:
                if not all(param in call for param in ['problem', 'solution', 'entry_point']):
                    issues.append("Test: problem, solution, entry_point")

        return issues

    def _check_async_calls(self, code: str) -> List[str]:
        """"""
        issues = []

        operator_call_pattern = r'(self\.[a-z_]+)\([^)]*\)'
        calls = re.findall(operator_call_pattern, code)

        for call in calls:
            if call in ['self.llm', 'self.name', 'self.dataset']:
                continue

            if f'await {call}' not in code:
                issues.append(f"await: {call}")

        return issues

    def _check_code_workflow(self, tree: ast.AST, code: str) -> List[str]:
        """Code"""
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'Workflow':
                for item in node.body:
                    if isinstance(item, ast.AsyncFunctionDef) and item.name == '__call__':
                        args = [arg.arg for arg in item.args.args]
                        if 'entry_point' not in args:
                            issues.append("Code__call__entry_point")

        return issues

    def fix_common_issues(self, code: str) -> str:
        """

        Args:
            code: 

        Returns:
        """
        fixed_code = code

        lowercase_pattern = r'operator\.([a-z][a-zA-Z_]*?)\('
        def fix_case(match):
            name = match.group(1)
            if name == 'custom':
                return 'operator.Custom('
            elif name == 'answergenerae' or name == 'answer_generate':
                return 'operator.AnswerGenerate('
            elif name == 'programmer':
                return 'operator.Programmer('
            elif name == 'test':
                return 'operator.Test('
            elif name == 'review':
                return 'operator.Review('
            elif name == 'revise':
                return 'operator.Revise('
            elif name.startswith('sc'):
                return 'operator.ScEnsemble('
            else:
                return f'operator.{name.capitalize()}('

        fixed_code = re.sub(lowercase_pattern, fix_case, fixed_code)

        call_pattern = r'^(\s*)(self\.(?:custom|answer_generate|programmer|test|review|revise|sc_ensemble)\([^)]*\))'
        lines = fixed_code.split('\n')
        fixed_lines = []

        for line in lines:
            if re.match(call_pattern, line) and 'await' not in line:
                line = re.sub(call_pattern, r'\1await \2', line)
            fixed_lines.append(line)

        fixed_code = '\n'.join(fixed_lines)

        if 'self.test' in fixed_code and 'entry_point' not in fixed_code:
            test_pattern = r'self\.test\(([^)]+)\)'
            def add_entry_point(match):
                params = match.group(1)
                if 'entry_point' not in params:
                    return f'self.test({params}, entry_point=entry_point)'
                return match.group(0)

            fixed_code = re.sub(test_pattern, add_entry_point, fixed_code)

        return fixed_code


def test_validator():
    """"""
    validator = WorkflowValidator()

    good_code = '''
import operator
from scripts.async_llm import create_llm_instance

class Workflow:
    def __init__(self, name, llm_config, dataset):
        self.name = name
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)

    async def __call__(self, problem):
        result = await self.custom(input=problem, instruction="Solve")
        return result['response'], self.llm.get_usage_summary()["total_cost"]
'''

    bad_code = '''
class Workflow:
    def __init__(self, name, llm_config, dataset):
        self.custom = operator.custom(self.llm)

    async def __call__(self, problem):
        result = self.custom(input=problem)
'''

    print(":")
    valid, msg, details = validator.validate_workflow_code(good_code)
    print(f"  : {valid}, : {msg}")

    print("\n:")
    valid, msg, details = validator.validate_workflow_code(bad_code)
    print(f"  : {valid}, : {msg}")

    print("\n:")
    fixed = validator.fix_common_issues(bad_code)
    print(":")
    print(fixed)


if __name__ == "__main__":
    test_validator()
