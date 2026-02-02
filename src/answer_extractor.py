#!/usr/bin/env python3
"""
 - ground truth
"""
import re
import json
import os
from typing import Any, Optional, Tuple

class AnswerExtractor:
    """"""

    def __init__(self, use_llm_fallback: bool = True, llm_client=None):
        """
        Args:
            use_llm_fallback: LLM
            llm_client: LLM
        """
        self.use_llm_fallback = use_llm_fallback
        self.llm_client = llm_client

    def extract_answer(self, text: str, problem_type: str, is_ground_truth: bool = False) -> str:
        """

        Args:
            text: 
            problem_type:  (math/code/qa)
            is_ground_truth: ground truth

        Returns:
        """
        if not text:
            return ""

        if problem_type == "math":
            return self._extract_math_answer(text, is_ground_truth)
        elif problem_type == "code":
            return self._extract_code_answer(text, is_ground_truth)
        elif problem_type == "qa":
            return self._extract_qa_answer(text, is_ground_truth)
        else:
            return str(text).strip()

    def _extract_math_answer(self, text: str, is_ground_truth: bool) -> str:
        """
         - 

        AgentFlow:
        1. <answer>
        2. \boxed{}LaTeX
        3. GSM8K
        4. "Final Answer"
        5. ground_truth: LLM
        6. 
        """
        text = str(text).strip()

        # Normalize common math symbols early (helps short answers like "6π", "20\\%", etc.)
        # NOTE: keep this conservative; full normalization is handled later.
        if '％' in text:
            text = text.replace('％', '%')
        if 'π' in text:
            text = text.replace('π', 'pi')
        if '\\%' in text:
            text = text.replace('\\%', '%')
        if '\\pi' in text:
            text = text.replace('\\pi', 'pi')

        # Quick path: very short, single-token math answers (percent / pi / plain numbers)
        compact = re.sub(r'\s+', '', text)
        if compact:
            # percent like 20%
            if re.fullmatch(r'[-+]?\d+(?:\.\d+)?%', compact):
                return self._clean_math_answer(compact)
            # pi like 6pi / pi
            if re.fullmatch(r'[-+]?\d+(?:\.\d+)?pi', compact, flags=re.IGNORECASE) or compact.lower() == 'pi':
                return self._clean_math_answer(compact)

        if re.fullmatch(r'[-+]?\d{1,3}(?:,\s*\d{3})+(?:\.\d+)?', text):
            return self._clean_math_answer(text)

        if self._looks_like_numeric_tuple(text) or self._looks_like_numeric_list(text):
            normalized = self._normalize_numeric_list_or_tuple(text)
            if normalized:
                return normalized

        answer_matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
        if answer_matches:
            answer_text = answer_matches[-1].strip()
            return self._clean_math_answer(answer_text)

        if "Revised Solution:" in text or "Based on the feedback" in text:
            clean_text = re.sub(r'Revised Solution:.*?(?=\\boxed|\d|$)', '', text, flags=re.DOTALL)
            clean_text = re.sub(r'Based on the feedback[^\\]*(?=\\boxed|$)', '', clean_text, flags=re.DOTALL)
            if clean_text.strip():
                text = clean_text

        boxed = self._extract_boxed(text)
        if boxed:
            if not boxed or boxed.strip() == '':
                boxed = None
            elif any(keyword in boxed for keyword in ['def ', 'return ', 'import ', 'class ', 'if __name__', 'print(', 'for ', 'while ', 'elif ', ':\n', 'await ', 'async ']):
                print(f"  ⚠️ \\boxed{{}}...")
                executed_answer = self._execute_code_and_extract_answer(boxed, 'math')
                if executed_answer:
                    print(f"  ✅ : {executed_answer}")
                    return executed_answer
                code_answer = self._extract_answer_from_code_block(boxed)
                if code_answer and not any(kw in str(code_answer) for kw in ['def ', 'import ', 'class ']):
                    print(f"  ✅ : {code_answer}")
                    return self._clean_math_answer(code_answer)
                print(f"  ❌ ")
                boxed = None
            elif '```python' in boxed or boxed.startswith('```'):
                executed_answer = self._execute_code_and_extract_answer(boxed, 'math')
                if executed_answer:
                    return executed_answer

                code_answer = self._extract_answer_from_code_block(boxed)
                if code_answer:
                    return code_answer
                boxed = None
            elif boxed.startswith('Error:') or 'Traceback' in boxed or 'SyntaxError' in boxed:
                boxed = None
            elif 'Based on the feedback' in boxed or 'Revised Solution' in boxed:
                boxed = None
            else:
                return self._clean_math_answer(boxed)

        gsm8k_match = re.search(r'####\s*([-+]?\d+(?:\.\d+)?)', text)
        if gsm8k_match:
            return self._clean_math_answer(gsm8k_match.group(1))

        final_answer_patterns = [
            r"(?:the\s+final\s+answer\s+is)[:]*\s*([^\n]+)",
            r"(?:Final\s+Answer|)[:]*\s*([^\n]+)",
            r"(?:The\s+answer\s+is)[:]*\s*([^\n]+)",
            r"(?:Therefore|Thus|Hence)[,:]*\s*([^\n]+)",
        ]
        for pattern in final_answer_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                candidate = str(matches[-1]).strip()
                candidate = re.sub(r'[\s\.\u3002]+$', '', candidate)
                cleaned = self._clean_math_answer(candidate)

                if cleaned and re.search(r'[A-Za-z]{3,}', cleaned):
                    numbers = self._extract_all_numbers(candidate)
                    if numbers:
                        return str(numbers[-1])
                    continue

                if cleaned:
                    return cleaned

        if self.use_llm_fallback and self._get_llm_client() is not None:
            has_calculations = text.count('=') >= 2 or len(re.findall(r'\d+', text)) > 6
            long_or_messy = len(text) > 500 or ("```" in text) or ("<think" in text)
            if is_ground_truth and has_calculations:
                llm_result = self._llm_extract_math_ground_truth(text)
                if llm_result and llm_result != text:
                    return llm_result
            if (not is_ground_truth) and (long_or_messy or has_calculations):
                llm_result = self._llm_extract_math(text)
                if llm_result and llm_result != text:
                    return self._clean_math_answer(llm_result)

        if '\\frac' in text or '\\(' in text:
            frac_matches = re.findall(r'\\frac\{([^}]+)\}\{([^}]+)\}', text)
            if frac_matches:
                numerator, denominator = frac_matches[-1]
                numerator = str(numerator).strip()
                denominator = str(denominator).strip()
                try:
                    float(numerator)
                    float(denominator)
                    return f"{numerator}/{denominator}"
                except ValueError:
                    pass

            cleaned_latex = self._clean_math_answer(text)
            if cleaned_latex and (re.match(r'^-?\d+\.?\d*$', cleaned_latex) or
                                  re.match(r'^-?\d+/\d+$', cleaned_latex)):
                return cleaned_latex

        has_variables = bool(re.search(r'[a-zA-Z]', text))
        has_operators = bool(re.search(r'[+\-*/\^]', text))
        if has_variables and has_operators:
            words = re.findall(r"[A-Za-z]{2,}", text)
            looks_like_sentence = len(words) >= 4 and len(text.split()) >= 6
            if not looks_like_sentence:
                return self._clean_math_answer(text)

        list_match = re.fullmatch(r'[-+\d,\s.]+', text.strip())
        if list_match and ',' in text:
            normalized = re.sub(r'\s+', '', text.strip())
            if normalized:
                return normalized

        if is_ground_truth:
            numbers = self._extract_all_numbers(text)
            if numbers:
                return str(numbers[-1])
        else:
            clean_text = text
            if not re.fullmatch(r'\s*[\(\[][^)\]]*[\)\]]\s*', text):
                clean_text = re.sub(r'\([^)]*\)', '', text)
            clean_numbers = self._extract_all_numbers(clean_text)
            if clean_numbers:
                short_and_multi = (len(str(text)) <= 80 and len(clean_numbers) >= 2)
                if short_and_multi:
                    return str(clean_numbers[-1])
                return str(clean_numbers[-1])
            numbers = self._extract_all_numbers(text)
            if numbers:
                short_and_multi = (len(str(text)) <= 80 and len(numbers) >= 2)
                if short_and_multi:
                    return str(numbers[-1])
                return str(numbers[-1])

        if 'Based on the feedback' in text or 'Revised Solution' in text or '```python' in text:
            return ""
        cleaned = re.sub(r'[^\d\-+./]', ' ', text).strip()
        if cleaned:
            nums = re.findall(r'-?\d+\.?\d*', cleaned)
            if nums:
                if len(str(text)) <= 80 and len(nums) >= 2:
                    return nums[-1]
                return nums[-1]
        return ""

    def _extract_code_answer(self, text: str, is_ground_truth: bool) -> str:
        """

        Code:
        - prediction: 
        - ground_truth: 
        - : test_result metadata

        1. ```python...``` AST
        2. def 
        3. ground truth
        """
        text = str(text).strip()

        text = re.sub(r'```python\s*```', '', text)
        text = re.sub(r'```\s*```', '', text)
        text = text.replace('No code provided', '').replace('No code', '')

        code_blocks = re.findall(r'```(?:python)?\s*\n?([^`]+)```', text)
        if code_blocks:
            for block in reversed(code_blocks):
                block = block.strip()
                if self._validate_code_syntax(block):
                    return block
            return code_blocks[-1].strip()

        func_pattern = r'(def\s+\w+\s*\([^)]*\)[^:]*:[\s\S]+?)(?=\n(?:def\s|class\s|$))'
        funcs = re.findall(func_pattern, text)
        if funcs:
            first_func = funcs[0].strip()
            if self._validate_code_syntax(first_func):
                return first_func
            return first_func

        if is_ground_truth:
            return text

        if self.use_llm_fallback and self._get_llm_client() is not None:
            return self._llm_extract_code(text)

        return text

    def _validate_code_syntax(self, code: str) -> bool:
        """

        Returns:
            True if valid Python syntax, False otherwise
        """
        try:
            import ast
            ast.parse(code)
            return True
        except SyntaxError:
            return False
        except Exception:
            return False

    def _extract_qa_answer(self, text: str, is_ground_truth: bool) -> str:
        """
        QA
        - : 
        - : 
        - : A/B/C/D/E
        """
        text = str(text).strip()

        answer_matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
        if answer_matches:
            answer_text = answer_matches[-1].strip()
            option_normalized = self._normalize_option_answer(answer_text)
            if option_normalized:
                return option_normalized
            return self._normalize_qa_answer(answer_text)

        option_answer = self._normalize_option_answer(text)
        if option_answer:
            return option_answer

        answer_patterns = [
            r"(?:Answer|)[:]*\s*([^\n.]+)",
            r"(?:The answer is)[:]*\s*([^\n.]+)",
            r"(?:Final answer|Therefore)[:]*\s*([^\n.]+)",
        ]

        for pattern in answer_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                answer_text = match.group(1).strip()
                answer_text = re.sub(r'^[\*\#\s]+|[\*\#\s]+$', '', answer_text).strip()
                option_normalized = self._normalize_option_answer(answer_text)
                if option_normalized:
                    return option_normalized
                if self._looks_like_number_string(answer_text):
                    numbers = self._extract_all_numbers(answer_text)
                    if numbers:
                        return str(numbers[-1])
                return self._normalize_qa_answer(answer_text)

        lines = [l.strip() for l in text.splitlines() if l.strip()]
        tail = lines[-1] if lines else text
        if self._looks_like_number_string(tail):
            numbers = self._extract_all_numbers(tail)
            if numbers:
                return str(numbers[-1])

        if not is_ground_truth:
            quoted = self._extract_last_quoted_span(text)
            if quoted:
                option_normalized = self._normalize_option_answer(quoted)
                if option_normalized:
                    return option_normalized
                return self._normalize_qa_answer(self._postprocess_qa_candidate(quoted))

            copula_tail = self._extract_after_copula(text)
            if copula_tail:
                option_normalized = self._normalize_option_answer(copula_tail)
                if option_normalized:
                    return option_normalized
                return self._normalize_qa_answer(self._postprocess_qa_candidate(copula_tail))

            approx_num = self._extract_leading_approx_number(text)
            if approx_num:
                return approx_num

        if len(lines) >= 2 and not is_ground_truth:
            normalized = self._normalize_qa_answer(lines[-1])
        else:
            sentences = [s.strip() for s in re.split(r'(?<!\d)[.!?]\s+', text) if s.strip()]
            normalized = self._normalize_qa_answer(sentences[-1] if sentences else text)

        if len(normalized.split()) > 50 and not is_ground_truth:
            sentences = text.split('.')
            if len(sentences) > 2:
                key_text = sentences[-2] + '.' + sentences[-1]
                normalized = self._normalize_qa_answer(key_text)

        if not is_ground_truth and self.use_llm_fallback and self._get_llm_client() is not None:
            token_count = len(normalized.split())
            digit_count = len(re.findall(r'\d', text))
            if len(text) > 500 or token_count > 12 or (digit_count >= 2 and token_count > 4):
                llm_result = self._llm_extract_qa(text)
                if llm_result:
                    option_normalized = self._normalize_option_answer(llm_result)
                    if option_normalized:
                        return option_normalized
                    return self._normalize_qa_answer(llm_result)

        return normalized

    def _extract_last_quoted_span(self, text: str) -> Optional[str]:
        """QA

        prediction“”
        """
        if not text:
            return None
        s = str(text)

        spans = re.findall(r'["“”](.+?)["“”]', s)
        if not spans:
            return None

        candidate = spans[-1].strip()
        if not candidate:
            return None

        if len(candidate) > 120:
            return None

        return candidate

    def _extract_after_copula(self, text: str) -> Optional[str]:
        """ 'is/was/are/were' QA"""
        if not text:
            return None
        s = str(text)

        matches = []
        for m in re.finditer(r'\b(?:is|was|are|were)\b\s+(.+)', s, flags=re.IGNORECASE):
            tail = (m.group(1) or "").strip()
            if tail:
                matches.append(tail)

        if not matches:
            return None

        candidate = matches[-1]

        candidate = re.split(r'[,;\n]', candidate, maxsplit=1)[0].strip()

        candidate = re.sub(r'^(?:the|a|an)\s+', '', candidate, flags=re.IGNORECASE).strip()
        candidate = re.sub(r'^(?:approximately|about|around|roughly|nearly|almost)\s+', '', candidate, flags=re.IGNORECASE).strip()

        candidate = re.sub(r'[\s\.\u3002]+$', '', candidate).strip()

        return candidate or None

    def _postprocess_qa_candidate(self, candidate: str) -> str:
        """ground-truth"""
        s = str(candidate or "").strip()
        if not s:
            return ""

        s = s.strip('"\'' '“”')

        per_match = re.search(
            r'([-+]?\d+(?:,\d{3})*(?:\.\d+)?)\s+[^\n]*?\bper\b\s*(?:1,?000|1000|100)',
            s,
            flags=re.IGNORECASE,
        )
        if per_match:
            return per_match.group(1).replace(',', '')

        return s

    def _extract_leading_approx_number(self, text: str) -> Optional[str]:
        """ 'Approximately 25 ...' prediction"""
        if not text:
            return None
        s = str(text).strip()
        m = re.match(
            r'^(?:approximately|about|around|roughly|nearly|almost)\s+([-+]?\d+(?:,\d{3})*(?:\.\d+)?)\b',
            s,
            flags=re.IGNORECASE,
        )
        if not m:
            return None
        return m.group(1).replace(',', '') or None

    def _normalize_option_answer(self, text: str) -> Optional[str]:
        """

        - "A" → "A"
        - "A." → "A"
        - "A. ream" → "A"
        - "ream" () → None
        - "Option A" → "A"
        - "(A)" → "A"
        """
        text = text.strip()

        if len(text) == 1 and text.upper() in 'ABCDE':
            return text.upper()

        match = re.fullmatch(r'[\(\[]?\s*([A-E])\s*[\)\]]?\s*[\.:]?\s*', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        match = re.match(r'^[\(\[]?\s*([A-E])\s*[\)\]]?\s*[\.:]\s+\S+', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        match = re.search(r'(?:Option|)\s*([A-E])\b', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        return None

    def _execute_code_and_extract_answer(self, code_block: str, problem_type: str) -> Optional[str]:
        """

        Args:
            code_block: 
            problem_type: math

        Returns:
            None
        """
        if problem_type != "math":
            return None

        import subprocess
        import tempfile
        import os

        code = re.sub(r'^```python\n?', '', code_block)
        code = re.sub(r'```$', '', code)
        code = code.strip()

        dangerous_keywords = ['os.system', 'subprocess', 'eval', 'exec', 'open', '__import__', 'rm ', 'del ']
        if any(kw in code for kw in dangerous_keywords):
            return None

        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                if 'print(' not in code:
                    lines = code.split('\n')
                    last_var = None
                    for line in reversed(lines):
                        line = line.strip()
                        if '=' in line and not line.startswith('#'):
                            var_name = line.split('=')[0].strip()
                            if var_name.isidentifier():
                                last_var = var_name
                                break

                    if last_var:
                        code += f'\nprint({last_var})'

                f.write(code)
                temp_path = f.name

            result = subprocess.run(
                ['python3', temp_path],
                capture_output=True,
                text=True,
                timeout=5
            )

            os.unlink(temp_path)

            if result.returncode == 0 and result.stdout:
                output = result.stdout.strip()
                if output:
                    last_line = output.split('\n')[-1].strip()
                    try:
                        if '/' in last_line:
                            parts = last_line.split('/')
                            float(parts[0])
                            float(parts[1])
                            return last_line
                        else:
                            num = float(last_line)
                            return str(int(num) if num == int(num) else num)
                    except:
                        return last_line

            return None

        except subprocess.TimeoutExpired:
            try:
                os.unlink(temp_path)
            except:
                pass
            return None
        except Exception:
            try:
                os.unlink(temp_path)
            except:
                pass
            return None

    def _extract_answer_from_code_block(self, code_block: str) -> Optional[str]:
        """

        1. print
        2. return
        3. 

         _execute_code_and_extract_answer
        """
        code_block = code_block.strip()

        code_block = re.sub(r'^```python\n?', '', code_block)
        code_block = re.sub(r'```$', '', code_block)

        print_pattern = r'print\(([^)]+)\)'
        print_matches = re.findall(print_pattern, code_block)
        if print_matches:
            last_print = print_matches[-1].strip()
            if last_print.isidentifier():
                var_pattern = rf'{last_print}\s*=\s*(.+)'
                var_match = re.search(var_pattern, code_block)
                if var_match:
                    return var_match.group(1).strip()
            return last_print

        return_pattern = r'return\s+(.+?)\s*(?:\n|$)'
        return_matches = re.findall(return_pattern, code_block)
        if return_matches:
            return return_matches[-1].strip()

        assignment_lines = [line for line in code_block.split('\n') if '=' in line and not line.strip().startswith('#')]
        if assignment_lines:
            last_assignment = assignment_lines[-1]
            if '=' in last_assignment:
                value = last_assignment.split('=', 1)[1].strip()
                return value

        return None

    def _extract_boxed(self, text: str) -> Optional[str]:
        """\boxed{}"""
        pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
        matches = re.findall(pattern, str(text or ""), flags=re.DOTALL)
        if matches:
            return matches[-1].strip()
        return None

    def _extract_all_numbers(self, text: str) -> list:
        """

        : 
        """
        numbers: list[str] = []

        s = str(text or "")

        # 1) Fractions first (keep exact form)
        fraction_pattern = r'-?\d+/\d+'
        fraction_matches = re.findall(fraction_pattern, s)
        for frac in fraction_matches:
            numbers.append(frac)

        # 2) Regular numbers (optionally with thousands separators) and scientific notation.
        # IMPORTANT: require at least one comma group for the comma-variant to avoid splitting plain 4+ digit ints
        # like "14400" into ["144", "00"].
        number_pattern = r'[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?'
        for m in re.findall(number_pattern, s):
            # Skip pieces already belonging to fractions (e.g., "5/18" would also match "5" and "18")
            if any(m in frac for frac in fraction_matches):
                continue
            numbers.append(m.replace(',', ''))

        return numbers

    def _clean_math_answer(self, answer: str) -> str:
        """

        : 
        """
        answer = str(answer).strip()

        if answer.startswith('i') and len(answer) > 1 and answer[1:].replace('.', '', 1).replace('/', '').isdigit():
            answer = answer[1:]

        # Normalize common symbols
        answer = answer.replace('％', '%')
        answer = answer.replace('π', 'pi')
        answer = answer.replace('\\%', '%')
        answer = answer.replace('\\pi', 'pi')

        answer = re.sub(r'\\boxed\{(.+?)\}', r'\1', answer)
        answer = re.sub(r'\\frac\{(.+?)\}\{(.+?)\}', r'\1/\2', answer)  # \frac{a}{b} → a/b
        # If \\text{} is used alongside digits, treat it as annotation and drop it (e.g., 9\\text{ values} -> 9).
        if re.search(r'\d', answer):
            answer = re.sub(r'\\text\{(.+?)\}', '', answer)
        else:
            answer = re.sub(r'\\text\{(.+?)\}', r'\1', answer)
        answer = re.sub(r'\\\(|\\\)|\\\[|\\\]', '', answer)

        units = ['grams', 'gram', 'g', 'kg', 'meters', 'meter', 'm', 'cm',
                 'seconds', 'second', 's', 'minutes', 'minute', 'min',
                 'dollars', 'dollar', '$', '', '', '', 'km', 'hours', 'hour']

        for unit in units:
            answer = re.sub(rf'\s*{re.escape(unit)}\b', '', answer, flags=re.IGNORECASE)

        if self._looks_like_numeric_tuple(answer) or self._looks_like_numeric_list(answer):
            normalized = self._normalize_numeric_list_or_tuple(answer)
            if normalized:
                answer = normalized
        else:
            answer = re.sub(r'[,\s]+', '', answer)

        base_match = re.fullmatch(r'([0-9A-Za-z_]+)_\{?(\d{1,2})\}?', answer)
        if base_match:
            digits_part, base_part = base_match.group(1), base_match.group(2)
            try:
                base = int(base_part)
                if 2 <= base <= 36:
                    digits_clean = digits_part.replace('_', '')
                    value = int(digits_clean, base)
                    return str(value)
            except Exception:
                pass

        try:
            if '/' in answer:
                parts = answer.split('/')
                if len(parts) == 2:
                    try:
                        numerator = float(parts[0])
                        denominator = float(parts[1])

                        if denominator == 1:
                            return str(int(numerator) if numerator == int(numerator) else numerator)

                        from math import gcd
                        if numerator == int(numerator) and denominator == int(denominator):
                            g = gcd(int(abs(numerator)), int(abs(denominator)))
                            if g > 1:
                                numerator /= g
                                denominator /= g
                            return f"{int(numerator)}/{int(denominator)}"

                        return answer
                    except:
                        return answer

            if '%' in answer:
                return str(float(answer.replace('%', '')) / 100)

            num = float(answer)
            if num == int(num):
                return str(int(num))
            return str(num)
        except:
            return answer

    def _normalize_qa_answer(self, text: str) -> str:
        """QA"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = ' '.join(text.split())
        return text.strip()

    def _looks_like_number_string(self, text: str) -> bool:
        """“”/"""
        if text is None:
            return False
        s = str(text).strip()
        if not s:
            return False
        s = re.sub(r'^[\(\[]\s*', '', s)
        s = re.sub(r'\s*[\)\]]$', '', s)
        s = s.replace('$', '').replace(',', '').replace(' ', '')
        return bool(re.fullmatch(r'[-+]?(?:\d+(?:\.\d+)?|\d+/\d+)(?:[eE][-+]?\d+)?%?', s))

    def _looks_like_numeric_tuple(self, text: str) -> bool:
        s = str(text or "").strip()
        return bool(re.fullmatch(r'[\(\[]\s*[-+]?\d+(?:\.\d+)?\s*(?:,\s*[-+]?\d+(?:\.\d+)?\s*)+[\)\]]', s))

    def _looks_like_numeric_list(self, text: str) -> bool:
        s = str(text or "").strip()
        if re.fullmatch(r'[-+]?\d{1,3}(?:,\s*\d{3})+(?:\.\d+)?', s):
            return False
        return bool(re.fullmatch(r'[-+]?\d+(?:\.\d+)?(?:\s*,\s*[-+]?\d+(?:\.\d+)?)+', s))

    def _normalize_numeric_list_or_tuple(self, text: str) -> Optional[str]:
        """/ \\( \\) """
        if text is None:
            return None
        s = str(text).strip()
        s = re.sub(r'\\\(|\\\)|\\\[|\\\]', '', s).strip()
        s = re.sub(r'\s+', '', s)
        return s or None

    def _get_llm_client(self):
        """LLM llm_client OpenAI """
        if self.llm_client is not None:
            return self.llm_client

        if not self.use_llm_fallback:
            return None

        flag = str(os.environ.get("COLAB_GRPO_ANSWER_EXTRACTOR_USE_OPENAI", "")).strip().lower()
        if flag in {"0", "false", "no"}:
            return None

        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            return None

        try:
            from openai import OpenAI  # type: ignore
        except Exception:
            return None

        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
        model = os.environ.get("COLAB_GRPO_ANSWER_EXTRACTOR_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"

        class _OpenAIChatWrapper:
            def __init__(self, client, model_name: str):
                self._client = client
                self._model = model_name

            def generate(self, prompt: str, max_tokens: int = 200, temperature: float = 0.0) -> str:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": "You are a precise answer extractor. Return only what is asked."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                content = resp.choices[0].message.content
                return (content or "").strip()

        try:
            client = OpenAI(base_url=base_url, api_key=api_key)
            self.llm_client = _OpenAIChatWrapper(client, model)
            return self.llm_client
        except Exception:
            return None

    def _llm_extract_math(self, text: str) -> str:
        """LLMprediction"""
        if not self.llm_client:
            return text

        prompt = f"""Extract ONLY the final numerical answer from this math solution.
Return JUST the number, no explanation.

Solution: {text[:1000]}

Final answer (number only):"""

        try:
            response = self.llm_client.generate(prompt, max_tokens=20, temperature=0)
            answer = response.strip()
            float(answer.replace('/', '.').replace(',', ''))
            return answer
        except:
            return text

    def _llm_extract_math_ground_truth(self, text: str) -> str:
        """LLMground truthAgentFlow

        prompt:
        1. ""
        2. ""
        3. vs
        """
        if not self.llm_client:
            return text

        prompt = f"""You are extracting the FINAL ANSWER from a mathematical solution text.

**Instructions:**
1. **Ignore intermediate calculations** - Focus only on the concluding answer
2. **Look for concluding statements** like "So the answer is...", "Therefore...", "The result is..."
3. **Extract the final numeric value** - Return JUST the number

**Text:**
{text[:800]}

**Output Format:**
- Return ONLY the final numerical answer
- No explanation, no intermediate values
- If multiple numbers exist, return the one from the final conclusion

**Final Answer (number only):**"""

        try:
            response = self.llm_client.generate(prompt, max_tokens=30, temperature=0)
            answer = response.strip()
            if '/' in answer:
                parts = answer.split('/')
                float(parts[0])
                float(parts[1])
            else:
                float(answer.replace(',', ''))
            return answer
        except:
            return text

    def _llm_extract_code(self, text: str) -> str:
        """LLM"""
        if not self.llm_client:
            return text

        prompt = f"""Extract ONLY the Python function code from this text.
Return JUST the code, no explanation.

Text: {text[:1000]}

Code:"""

        try:
            response = self.llm_client.generate(prompt, max_tokens=500, temperature=0)
            if 'def ' in response:
                return response.strip()
            return text
        except:
            return text

    def _llm_extract_qa(self, text: str) -> str:
        """LLMQA"""
        if not self.llm_client:
            return text

        prompt = f"""Extract ONLY the final short answer from this QA response.
Return JUST the answer phrase, no explanation.

QA response:
{text[:1200]}

Final answer:"""

        try:
            response = self.llm_client.generate(prompt, max_tokens=60, temperature=0)
            return (response or "").strip()
        except:
            return text


def test_extractor():
    """"""
    extractor = AnswerExtractor(use_llm_fallback=False)

    test_cases = [
        # Math cases
        {
            "text": "The probability is $\\frac{1}{27}$. So the answer is \\boxed{\\frac{8}{9}}",
            "type": "math",
            "expected": "0.8888888888888888"
        },
        {
            "text": "After calculating, we get 586 grams",
            "type": "math",
            "expected": "586"
        },
        {
            "text": "Therefore, the final answer is 42.",
            "type": "math",
            "expected": "42"
        },
        # Code cases
        {
            "text": "```python\ndef solve(n):\n    return n * 2\n```",
            "type": "code",
            "expected": "def solve(n):\n    return n * 2"
        },
        # QA cases
        {
            "text": "The capital of France is Paris.",
            "type": "qa",
            "expected": "the capital of france is paris"
        },
    ]

    print("=" * 60)
    print("🧪 ")
    print("=" * 60)

    for i, case in enumerate(test_cases, 1):
        result = extractor.extract_answer(case["text"], case["type"])
        print(f"\nTest {i} ({case['type']}):")
        print(f"  : {case['text'][:50]}...")
        print(f"  : {result}")
        print(f"  : {case['expected']}")
        print(f"  ✅ " if result == case["expected"] else f"  ❌ ")


if __name__ == "__main__":
    test_extractor()
