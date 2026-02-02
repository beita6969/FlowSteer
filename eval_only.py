#!/usr/bin/env python3
"""Evaluation Script for FlowSteer

This script evaluates trained models using the AFlow executor.

Features:
- Supports vLLM inference with workflow execution
- Evaluates on multiple problem types (math, code, QA)
- Computes accuracy metrics

Usage:
    python eval_only.py --num-samples 100

Note:
    Use --checkpoint to specify a LoRA checkpoint for vLLM
"""

import os
import sys
import json
import time
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

from src.answer_extractor import AnswerExtractor

def llm_extract_answer(text: str, vllm_url: str = 'http://localhost:8002/v1') -> str:
    """vLLMQwen3-8B"""
    import requests
    if not text or len(text) < 5:
        return text

    messages = [
        {"role": "system", "content": "1/2(x,y)"},
        {"role": "user", "content": f"(//):\n\n{text[:800]}"}
    ]

    try:
        response = requests.post(
            f'{vllm_url}/chat/completions',
            json={
                'model': 'lora230',
                'messages': messages,
                'max_tokens': 50,
                'temperature': 0.1
            },
            timeout=15
        )
        if response.status_code == 200:
            result = response.json()
            answer = result['choices'][0]['message']['content'].strip()
            if '<think>' in answer:
                answer = answer.split('</think>')[-1].strip()
            answer = answer.strip('., \n')
            if answer:
                return answer
    except:
        pass
    return ''

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
try:
    import torch
except:
    torch = None
try:
    import numpy as np
except Exception:
    np = None

WORKFLOW_LOG_FILE = None

def init_workflow_log(output_path: str = None):
    """workflow"""
    global WORKFLOW_LOG_FILE
    if output_path:
        log_path = output_path.replace('.json', '_workflows.log')
    else:
        log_path = f"eval_results/workflow_log_{time.strftime('%Y%m%d_%H%M%S')}.log"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    WORKFLOW_LOG_FILE = open(log_path, 'w', encoding='utf-8')
    WORKFLOW_LOG_FILE.write(f"# Workflow Log - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    WORKFLOW_LOG_FILE.write("="*80 + "\n")
    WORKFLOW_LOG_FILE.flush()
    logger.info(f"📝 Workflow log: {log_path}")
    return log_path

def log_workflow(
    idx: int,
    problem: str,
    dsl: str,
    answer: str,
    is_correct: bool,
    rounds: int,
    ground_truth: str = "",
    operator_trace: list = None,
    problem_type: str = "qa"
):
    """workflowoperator trace

    Args:
        idx: 
        problem: 
        dsl: DSL
        answer: 
        is_correct: 
        rounds: 
        ground_truth: 
        operator_trace: operatordict:
            {'round': int, 'operator': str, 'input': str, 'output': str, 'feedback': str}
        problem_type: 
    """
    global WORKFLOW_LOG_FILE
    logger.debug(f"[LOG_WORKFLOW] idx={idx}, is_correct={is_correct}, WORKFLOW_LOG_FILE={'NONE' if WORKFLOW_LOG_FILE is None else 'OPEN'}")
    if WORKFLOW_LOG_FILE is None:
        logger.warning(f"[LOG_WORKFLOW] SKIPPED idx={idx}: WORKFLOW_LOG_FILE is None")
        return

    status = "✓" if is_correct else "✗"
    WORKFLOW_LOG_FILE.write(f"\n{'='*80}\n")
    WORKFLOW_LOG_FILE.write(f"[{idx+1}] {status} (rounds={rounds}, type={problem_type})\n")
    WORKFLOW_LOG_FILE.write(f"{'='*80}\n")

    WORKFLOW_LOG_FILE.write(f"\n## Problem:\n{problem[:500]}\n")
    WORKFLOW_LOG_FILE.write(f"\n## Ground Truth:\n{ground_truth[:200]}\n")
    WORKFLOW_LOG_FILE.write(f"\n## Predicted Answer:\n{str(answer)[:500]}\n")

    WORKFLOW_LOG_FILE.write(f"\n## Final DSL:\n{dsl}\n")

    if operator_trace:
        WORKFLOW_LOG_FILE.write(f"\n## Operator Trace ({len(operator_trace)} steps):\n")
        for step in operator_trace:
            round_num = step.get('round', '?')
            op_name = step.get('operator', 'Unknown')
            WORKFLOW_LOG_FILE.write(f"\n### Round {round_num}: {op_name}\n")

            op_input = step.get('input', '')[:300]
            if op_input:
                WORKFLOW_LOG_FILE.write(f"Input: {op_input}...\n")

            op_output = step.get('output', '')[:500]
            if op_output:
                WORKFLOW_LOG_FILE.write(f"Output: {op_output}...\n")

            # Feedback
            feedback = step.get('feedback', '')
            if feedback:
                WORKFLOW_LOG_FILE.write(f"Feedback: {feedback[:300]}\n")

            if 'verify_result' in step:
                vr = step['verify_result']
                WORKFLOW_LOG_FILE.write(f"Verify Result: is_correct={vr.get('is_correct')}, suggested_answer={vr.get('suggested_answer', 'N/A')[:100]}\n")

    WORKFLOW_LOG_FILE.write(f"\n{'-'*80}\n")
    WORKFLOW_LOG_FILE.flush()

def close_workflow_log():
    """workflow"""
    global WORKFLOW_LOG_FILE
    if WORKFLOW_LOG_FILE:
        WORKFLOW_LOG_FILE.close()
        WORKFLOW_LOG_FILE = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_dataset(data_path: str, num_samples: Optional[int] = None) -> List[dict]:
    """"""
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))

    if num_samples and num_samples < len(data):
        source_to_items = defaultdict(list)
        for item in data:
            source_to_items[item.get('source', 'unknown')].append(item)

        samples_per_source = max(1, num_samples // len(source_to_items))
        sampled = []
        np.random.seed(42)
        for source, items in sorted(source_to_items.items()):
            if len(items) >= samples_per_source:
                selected = np.random.choice(len(items), samples_per_source, replace=False)
                sampled.extend([items[i] for i in selected])
            else:
                sampled.extend(items)
        data = sampled[:num_samples]

    return data


def create_vllm_generate_fn(config: dict):
    """vLLM API"""
    gen_config = config.get('generation_config', {}) or {}
    base_url = config.get('vllm_base_url', 'http://localhost:8003/v1')
    lora_adapter_name = config.get('lora_adapter_name', None) or None
    base_model_name = config.get('vllm_served_model_name', 'Qwen3-8B')

    try:
        from openai import OpenAI
        import threading

        _thread_local = threading.local()

        def _get_client() -> OpenAI:
            client = getattr(_thread_local, "client", None)
            if client is None:
                client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=300.0)
                _thread_local.client = client
            return client

        def generate(prompt: str, max_tokens: int = 2048, disable_thinking: bool = False) -> str:
            """ - train_interactive.py"""
            import re
            try:
                enable_thinking = bool(gen_config.get('enable_thinking', True)) and not disable_thinking

                context_limit = int(gen_config.get("vllm_context_limit", 16384))
                estimated_input_tokens = max(1, len(prompt) // 3)
                reserve = 128

                if disable_thinking:
                    desired_max_tokens = int(gen_config.get("vllm_prompt_max_tokens", 256))
                else:
                    desired_max_tokens = int(gen_config.get("vllm_action_max_tokens", 512))

                available = max(16, context_limit - estimated_input_tokens - reserve)
                actual_max_tokens = max(16, min(desired_max_tokens, available))

                client = _get_client()
                model_name = lora_adapter_name or base_model_name

                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=gen_config.get('temperature', 0.6),
                    top_p=gen_config.get('top_p', 0.95),
                    max_tokens=actual_max_tokens,
                    extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
                )
                result = resp.choices[0].message.content or ""

                if '<think>' in result.lower():
                    think_match = re.search(r'<think>(.*?)</think>', result, re.DOTALL | re.IGNORECASE)
                    if think_match:
                        result_after_think = re.sub(r'<think>.*?</think>\s*', '', result, flags=re.DOTALL | re.IGNORECASE)
                        if result_after_think.strip():
                            result = result_after_think

                if not disable_thinking:
                    has_action = re.search(r'[<\[]action[>\]]', str(result), re.IGNORECASE) is not None
                    if not has_action:
                        repair_max = int(gen_config.get("vllm_repair_max_tokens", 128))
                        repair_max = max(16, min(repair_max, available))
                        repair_prompt = (
                            prompt
                            + "\n\nIMPORTANT: Output EXACTLY ONE XML <action>...</action> now. "
                            + "No <think> tags. No extra text."
                        )
                        try:
                            repair_resp = client.chat.completions.create(
                                model=model_name,
                                messages=[{"role": "user", "content": repair_prompt}],
                                temperature=gen_config.get('temperature', 0.6),
                                max_tokens=repair_max,
                                top_p=gen_config.get('top_p', 0.95),
                                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                            )
                            repair_result = repair_resp.choices[0].message.content or ""
                            if re.search(r'[<\[]action[>\]]', repair_result, re.IGNORECASE):
                                result = repair_result
                        except Exception as repair_e:
                            logger.warning(f"Repair generation failed: {repair_e}")

                return result
            except Exception as e:
                logger.warning(f"vLLM generation error: {e}")
                return ""

        return generate

    except Exception:
        import requests

        def generate(prompt: str, max_tokens: int = 2048, disable_thinking: bool = False) -> str:
            try:
                extra_body = {}
                if gen_config.get('enable_thinking', True) and not disable_thinking:
                    extra_body['chat_template_kwargs'] = {'enable_thinking': True}

                model_name = lora_adapter_name or base_model_name
                response = requests.post(
                    f"{base_url}/completions",
                    json={
                        "model": model_name,
                        "prompt": prompt,
                        "max_tokens": max_tokens,
                        "temperature": gen_config.get('temperature', 0.6),
                        "top_p": gen_config.get('top_p', 0.95),
                        "top_k": gen_config.get('top_k', 20),
                        **({"extra_body": extra_body} if extra_body else {}),
                    },
                    timeout=120,
                )
                response.raise_for_status()
                return response.json()['choices'][0]['text']
            except Exception as e:
                logger.warning(f"vLLM generation error: {e}")
                return ""

        return generate


def run_single_problem(
    idx: int,
    item: dict,
    generate_fn,
    aflow_executor,
    config: dict,
    prompt_builder,
):
    """"""
    import re
    import string
    from src.interactive import (
        create_env, create_aflow_executor_wrapper,
        Trajectory, TurnRecord,
    )
    from src.interactive.workflow_env import EnvState

    def normalize_answer(s: str) -> str:
        """"""
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        def white_space_fix(text):
            return ' '.join(text.split())
        def remove_punc(text):
            return ''.join(ch for ch in text if ch not in string.punctuation)
        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(str(s)))))

    def compute_f1(pred: str, truth: str) -> float:
        """F1"""
        pred_tokens = normalize_answer(pred).split()
        truth_tokens = normalize_answer(truth).split()
        if not pred_tokens or not truth_tokens:
            return 0.0
        common = set(pred_tokens) & set(truth_tokens)
        if not common:
            return 0.0
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(truth_tokens)
        return 2 * precision * recall / (precision + recall)

    def extract_boxed(text: str) -> Optional[str]:
        """\\boxed{} - BoxedExtract: """
        text = str(text or "")
        start = text.find('\\boxed{')
        if start == -1:
            return None

        brace_start = start + len('\\boxed{')
        depth = 1
        i = brace_start

        while i < len(text) and depth > 0:
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
            i += 1

        if depth == 0:
            return text[brace_start:i-1].strip()
        return None

    def extract_all_boxed(text: str) -> list:
        """SafeAbs: \\boxed{} - """
        results = []
        text = str(text or "")
        pos = 0
        while True:
            start = text.find('\\boxed{', pos)
            if start == -1:
                break
            brace_start = start + len('\\boxed{')
            depth = 1
            i = brace_start
            while i < len(text) and depth > 0:
                if text[i] == '{':
                    depth += 1
                elif text[i] == '}':
                    depth -= 1
                i += 1
            if depth == 0:
                results.append(text[brace_start:i-1].strip())
            pos = i
        return results

    def parse_latex_fraction(text: str) -> Optional[float]:
        """LaTeX"""
        # \frac{a}{b} or \dfrac{a}{b}
        frac_match = re.search(r'\\d?frac\{([^}]+)\}\{([^}]+)\}', text)
        if frac_match:
            try:
                numer = float(frac_match.group(1).strip())
                denom = float(frac_match.group(2).strip())
                if denom != 0:
                    return numer / denom
            except:
                pass
        return None

    def parse_latex_number(text: str) -> Optional[float]:
        """LaTeX"""
        text = text.strip()
        neg = False
        if text.startswith('-'):
            neg = True
            text = text[1:].strip()

        frac_match = re.search(r'\\d?frac\{([^}]+)\}\{([^}]+)\}', text)
        if frac_match:
            try:
                numer = float(frac_match.group(1).strip())
                denom = float(frac_match.group(2).strip())
                if denom != 0:
                    val = numer / denom
                    return -val if neg else val
            except:
                pass

        try:
            val = float(re.sub(r'[^\d.]', '', text))
            return -val if neg else val
        except:
            pass

        return None

    def parse_python_complex(text: str) -> Optional[tuple]:
        """Python(real, imag)"""
        # (20+0j), (-1-5j), (3+4j)
        match = re.match(r'\(?([\d.\-]+)([+-][\d.]+)j\)?', text.strip())
        if match:
            try:
                real = float(match.group(1))
                imag = float(match.group(2))
                return (real, imag)
            except:
                pass
        match2 = re.match(r'\(?([\d.\-]+)\+0j\)?', text.strip())
        if match2:
            try:
                return (float(match2.group(1)), 0.0)
            except:
                pass
        return None

    def parse_latex_complex(text: str) -> Optional[tuple]:
        """LaTeX(real, imag)"""
        # -1 - 5i, 3 + 4i, -3 + 6i
        text = text.replace(' ', '')
        match = re.match(r'([\d.\-]*)([+-][\d.]+)i', text)
        if match:
            try:
                real_str = match.group(1)
                real = float(real_str) if real_str and real_str != '-' else (0.0 if real_str != '-' else -0.0)
                imag = float(match.group(2))
                return (real, imag)
            except:
                pass
        if 'i' not in text:
            try:
                return (float(text), 0.0)
            except:
                pass
        return None

    def parse_vector(text: str) -> Optional[list]:
        """/"""
        if text.startswith('[') and text.endswith(']'):
            try:
                import ast
                vals = ast.literal_eval(text)
                if isinstance(vals, (list, tuple)):
                    return [float(v) for v in vals]
            except:
                pass
        # LaTeX pmatrix: \begin{pmatrix} 6 \\ -15 \end{pmatrix}
        pmat_match = re.search(r'\\begin\{pmatrix\}(.+?)\\end\{pmatrix\}', text, re.DOTALL)
        if pmat_match:
            content = pmat_match.group(1)
            parts = re.split(r'\\\\', content)
            try:
                return [float(p.strip()) for p in parts if p.strip()]
            except:
                pass
        return None

    def parse_coordinate(text: str) -> Optional[tuple]:
        """"""
        # Python tuple: (-0.25, -2.0)
        if '(' in text and ')' in text and '\\' not in text:
            try:
                import ast
                vals = ast.literal_eval(text.replace('j', '*1j'))
                if isinstance(vals, tuple) and len(vals) == 2:
                    return tuple(float(v.real if hasattr(v, 'real') else v) for v in vals)
            except:
                pass
        # LaTeX: \left( -\frac{1}{4}, -2 \right)
        coord_match = re.search(r'\\left\s*\((.+?),(.+?)\\right\s*\)', text)
        if coord_match:
            x_str, y_str = coord_match.group(1).strip(), coord_match.group(2).strip()
            x = parse_latex_number(x_str)
            y = parse_latex_number(y_str)
            if x is not None and y is not None:
                return (x, y)
        return None

    def math_equal(pred: str, truth: str, tolerance: float = 1e-4) -> bool:
        """StructOutput: 
        """
        pred = str(pred).strip()
        truth = str(truth).strip()

        # ============================================
        # ============================================
        def preprocess_answer(s: str) -> str:
            """PlanOutput: """
            s = str(s).strip()

            if '=' in s and not s.startswith('='):
                parts = s.rsplit('=', 1)
                if len(parts) == 2:
                    right_side = parts[1].strip()
                    if right_side and not re.match(r'^[a-zA-Z_]\w*$', right_side):
                        s = right_side

            if s.startswith('[') and s.endswith(']'):
                inner = s[1:-1].strip()
                if re.match(r'^-?[\d.,\s\-]+$', inner):
                    s = inner
            if s.startswith('{') and s.endswith('}'):
                inner = s[1:-1].strip()
                if re.match(r'^-?[\d.,\s\-]+$', inner):
                    s = inner

            str_tuple_match = re.match(r"^\(\s*['\"](.+?)['\"](?:\s*,\s*['\"](.+?)['\"])*\s*\)$", s)
            if str_tuple_match:
                values = re.findall(r"['\"]([^'\"]+)['\"]", s)
                s = '(' + ', '.join(values) + ')'

            s = re.sub(r'\+0[jJiI]\b', '', s)
            s = re.sub(r'([\d.]+)\s*\*\s*[IiJj]', r'\1i', s)
            s = s.replace('j', 'i').replace('J', 'i').replace('I', 'i')
            s = re.sub(r'\s+([+-])\s*(\d)', r'\1\2', s)
            if s.startswith('(') and s.endswith(')') and 'i' in s:
                inner = s[1:-1].strip()
                if re.match(r'^-?[\d.]+[+-][\d.]*i$', inner):
                    s = inner

            s = re.sub(r'(\w)\*\*(-?\d+)', r'\1^\2', s)

            s = re.sub(r'(\d)\*([a-zA-Z])', r'\1\2', s)
            s = re.sub(r'\)\*([a-zA-Z])', r')\1', s)

            return s.strip()

        pred = preprocess_answer(pred)
        truth = preprocess_answer(truth)

        if normalize_answer(pred) == normalize_answer(truth):
            return True

        try:
            pred_num = float(re.sub(r'[^\d.\-]', '', pred))
            truth_num = float(re.sub(r'[^\d.\-]', '', truth))
            if abs(pred_num - truth_num) < tolerance:
                return True
        except:
            pass

        truth_frac = parse_latex_fraction(truth)
        if truth_frac is not None:
            try:
                pred_num = float(pred)
                if abs(pred_num - truth_frac) < tolerance:
                    return True
            except:
                pass

        pred_frac = parse_latex_fraction(pred)
        if pred_frac is not None:
            try:
                truth_num = float(truth)
                if abs(pred_frac - truth_num) < tolerance:
                    return True
            except:
                pass

        pred_complex = parse_python_complex(pred)
        truth_complex = parse_latex_complex(truth)
        if pred_complex and truth_complex:
            if abs(pred_complex[0] - truth_complex[0]) < tolerance and abs(pred_complex[1] - truth_complex[1]) < tolerance:
                return True
        if pred_complex and pred_complex[1] == 0:
            try:
                truth_num = float(re.sub(r'[^\d.\-]', '', truth))
                if abs(pred_complex[0] - truth_num) < tolerance:
                    return True
            except:
                pass

        pred_vec = parse_vector(pred)
        truth_vec = parse_vector(truth)
        if pred_vec and truth_vec and len(pred_vec) == len(truth_vec):
            if all(abs(p - t) < tolerance for p, t in zip(pred_vec, truth_vec)):
                return True

        pred_coord = parse_coordinate(pred)
        truth_coord = parse_coordinate(truth)
        if pred_coord and truth_coord:
            if abs(pred_coord[0] - truth_coord[0]) < tolerance and abs(pred_coord[1] - truth_coord[1]) < tolerance:
                return True

        if '/' in pred and not '\\' in pred:
            try:
                parts = pred.split('/')
                if len(parts) == 2:
                    pred_val = float(parts[0]) / float(parts[1])
                    if truth_frac is not None and abs(pred_val - truth_frac) < tolerance:
                        return True
            except:
                pass

        import math
        if '\\pi' in truth:
            try:
                truth_with_pi = truth.replace('\\pi', str(math.pi))
                pi_frac_match = re.search(r'\\d?frac\{.*?\\pi.*?\}\{([^}]+)\}', truth)
                if pi_frac_match:
                    denom = float(pi_frac_match.group(1).strip())
                    truth_val = math.pi / denom
                    pred_num = float(pred)
                    if abs(pred_num - truth_val) < 0.01:
                        return True
                npi_match = re.match(r'([\d.]+)\s*\\pi', truth)
                if npi_match:
                    coef = float(npi_match.group(1))
                    truth_val = coef * math.pi
                    pred_num = float(pred)
                    if abs(pred_num - truth_val) < 0.01:
                        return True
            except:
                pass

        if '\\sqrt' in truth:
            try:
                def eval_sqrt_expr(expr):
                    expr = re.sub(r'\\sqrt\{(\d+)\}', r'math.sqrt(\1)', expr)
                    expr = re.sub(r'\\d?frac\{([^}]+)\}\{([^}]+)\}', r'((\1)/(\2))', expr)
                    expr = re.sub(r'\\[a-zA-Z]+', '', expr)
                    return eval(expr)
                truth_val = eval_sqrt_expr(truth)
                pred_num = float(pred)
                if abs(pred_num - truth_val) < 0.01:
                    return True
            except:
                pass

        if pred.startswith('[') and pred.endswith(']'):
            try:
                import ast
                pred_list = ast.literal_eval(pred)
                truth_parts = [p.strip() for p in truth.split(',')]
                truth_list = [float(p) for p in truth_parts]
                if len(pred_list) == len(truth_list):
                    if all(abs(float(p) - t) < tolerance for p, t in zip(pred_list, truth_list)):
                        return True
            except:
                pass

        if ',' in pred and ',' in truth:
            try:
                pred_parts = sorted([float(p.strip()) for p in pred.split(',')])
                truth_parts = sorted([float(p.strip()) for p in truth.split(',')])
                if len(pred_parts) == len(truth_parts):
                    if all(abs(p - t) < tolerance for p, t in zip(pred_parts, truth_parts)):
                        return True
            except:
                pass

        base_match = re.match(r'^([0-9A-Fa-f]+)_\{?(\d{1,2})\}?$', truth)
        if base_match:
            digits, base_str = base_match.groups()
            try:
                base = int(base_str)
                if 2 <= base <= 36:
                    truth_decimal = int(digits, base)
                    pred_num = int(pred)
                    if pred_num == truth_decimal:
                        return True
                    if pred == digits:
                        return True
            except:
                pass

        try:
            pred_num = float(pred)
            truth_clean = re.sub(r'[^\d.\-]', '', truth)
            if truth_clean:
                truth_num = float(truth_clean)
                if abs(pred_num - truth_num) < 0.02:
                    return True
        except:
            pass

        return False

    def check_correctness(pred: str, truth: str, ptype: str) -> tuple:
        """"""
        ptype = str(ptype).lower().strip()
        if ptype == 'math':
            answer_extractor = AnswerExtractor(use_llm_fallback=False)

            boxed_truth = extract_boxed(truth)
            if boxed_truth:
                truth = boxed_truth
            else:
                extracted_truth = answer_extractor.extract_answer(truth, 'math', is_ground_truth=True)
                if extracted_truth:
                    truth = extracted_truth

            boxed_pred = extract_boxed(pred)
            if boxed_pred:
                pred = boxed_pred
            else:
                extracted_pred = answer_extractor.extract_answer(pred, 'math', is_ground_truth=False)
                if extracted_pred:
                    pred = extracted_pred

            if math_equal(pred, truth):
                return 1.0, True

            original_pred, original_truth = pred, truth
            if len(str(pred)) > 20 or len(str(truth)) > 20:
                if len(str(pred)) > 20:
                    llm_pred = llm_extract_answer(str(pred))
                    if llm_pred and llm_pred != pred:
                        if math_equal(llm_pred, truth):
                            return 1.0, True
                        pred = llm_pred

                if len(str(truth)) > 20:
                    llm_truth = llm_extract_answer(str(truth))
                    if llm_truth and llm_truth != truth:
                        if math_equal(pred, llm_truth):
                            return 1.0, True
                        if math_equal(original_pred, llm_truth):
                            return 1.0, True

            return 0.0, False
        elif ptype == 'qa':
            f1 = compute_f1(pred, truth)
            return f1, f1 >= 0.5
        elif ptype == 'code':
            if 'PASS' in str(pred).upper() or 'passed' in str(pred).lower():
                return 1.0, True
            return 0.0, False
        else:
            if normalize_answer(pred) == normalize_answer(truth):
                return 1.0, True
            return compute_f1(pred, truth), False

    def sanitize_feedback(fb: str) -> str:
        """"""
        if len(fb) > 500:
            return fb[:500] + "..."
        return fb

    problem = item.get('problem', item.get('question', ''))
    raw_type = item.get('problem_type', 'qa')
    problem_type = str(raw_type).lower().strip()
    ground_truth = item.get('ground_truth', item.get('answer', ''))
    source = item.get('source', 'unknown')
    meta = item.get('meta', {})

    interactive_cfg = config.get('interactive_grpo', {}) or {}
    max_rounds = int(interactive_cfg.get('max_rounds', 20))
    finish_cfg = interactive_cfg.get('finish_constraints', {}) or {}

    executor = create_aflow_executor_wrapper(aflow_executor, problem_type)

    executor_kwargs = {}
    if problem_type == 'code':
        test = meta.get('test')
        entry_point = meta.get('entry_point')
        if test:
            executor_kwargs['test'] = test
        if entry_point:
            executor_kwargs['entry_point'] = entry_point

    env = create_env(
        problem=problem,
        problem_type=problem_type,
        executor=executor,
        executor_kwargs=executor_kwargs if executor_kwargs else None,
        max_rounds=max_rounds,
        execute_each_step=True,
        finish_min_total_operators=finish_cfg.get('min_total_operators', 1),
        finish_require_checker=finish_cfg.get('require_checker', False),
        finish_require_structure=finish_cfg.get('require_structure', False),
        use_custom_prompts_in_execution=True,
    )

    prompt = prompt_builder.build_initial_prompt(
        problem=problem,
        problem_type=problem_type,
    )

    rounds_used = 0
    final_answer: Any = ""
    final_dsl = ""

    for round_idx in range(max_rounds):
        rounds_used = round_idx + 1

        is_awaiting_prompt = hasattr(env, 'env_state') and env.env_state == EnvState.AWAITING_PROMPT
        response = generate_fn(prompt, disable_thinking=is_awaiting_prompt)

        feedback, success_list, active = env.step(response)
        done = not active

        if hasattr(env, 'final_answer') and getattr(env, 'final_answer'):
            final_answer = getattr(env, 'final_answer')
        if hasattr(env, 'last_execution_result') and getattr(env, 'last_execution_result') is not None:
            final_answer = getattr(env, 'last_execution_result')

        if done:
            final_dsl = env.get_dsl()
            if hasattr(env, 'final_answer'):
                final_answer = env.final_answer
            elif hasattr(env, 'last_execution_result'):
                final_answer = env.last_execution_result
            break

        if hasattr(env, 'env_state') and env.env_state == EnvState.AWAITING_PROMPT:
            operator_name = env.pending_operator or "Custom"
            prompt = prompt_builder.build_prompt_request(
                problem=problem,
                operator_name=operator_name,
                operator_description="",
                context=getattr(env, "pending_prompt_context", "") or "",
                problem_type=problem_type,
            )
        else:
            stats = env.graph.get_statistics() if hasattr(env, 'graph') else {}
            prompt = prompt_builder.build_continuation_prompt(
                problem=problem,
                current_dsl=env.get_dsl(),
                total_operators=stats.get('total_operators', 0),
                unique_types=stats.get('unique_types', 0),
                round_number=round_idx + 2,
                max_rounds=max_rounds,
                node_ids=[n.id for n in env.graph.nodes] if hasattr(env, 'graph') else [],
                last_success=all(success_list) if success_list else True,
                last_result=sanitize_feedback(feedback),
                last_message="",
                problem_type=problem_type,
            )

    if not final_dsl:
        final_dsl = env.get_dsl() if hasattr(env, 'get_dsl') else ""

    pred_obj: Any = final_answer
    if isinstance(pred_obj, dict):
        pred_obj = (
            pred_obj.get('output')
            or pred_obj.get('answer')
            or pred_obj.get('response')
            or pred_obj.get('aggregated_answer')
            or pred_obj.get('solution')
            or pred_obj.get('result')
            or pred_obj
        )
        if isinstance(pred_obj, dict):
            for key in ['output', 'answer', 'response', 'aggregated_answer', 'solution', 'result']:
                if key in pred_obj and pred_obj[key]:
                    pred_obj = pred_obj[key]
                    break
    elif isinstance(pred_obj, list) and pred_obj:
        pred_obj = ", ".join(str(x) for x in pred_obj)

    score, is_correct = check_correctness(str(pred_obj), str(ground_truth), problem_type)

    return {
        'idx': idx,
        'source': source,
        'problem_type': problem_type,
        'problem': problem[:100],
        'ground_truth': str(ground_truth)[:50],
        'final_answer': str(pred_obj)[:200],
        'final_dsl': final_dsl,
        'rounds_used': rounds_used,
        'score': score,
        'is_correct': is_correct,
    }


def run_vectorized_evaluation(
    config: dict,
    data: List[dict],
    generate_fn,
    aflow_executor,
    prompt_builder,
    num_workers: int = 32,
):
    """
     Vectorized Rollout  ( train_interactive.py)
    - 
    -  active  action ( vLLM  dynamic batching)
    -  env.step
    """
    import re
    import string
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from src.interactive import (
        create_env, create_aflow_executor_wrapper,
        Trajectory, TurnRecord,
    )
    from src.interactive.workflow_env import EnvState

    def normalize_answer(s: str) -> str:
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        def white_space_fix(text):
            return ' '.join(text.split())
        def remove_punc(text):
            return ''.join(ch for ch in text if ch not in string.punctuation)
        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(str(s)))))

    def extract_boxed(text: str) -> Optional[str]:
        """\\boxed{} - BoxedExtract: """
        text = str(text or "")
        start = text.find('\\boxed{')
        if start == -1:
            return None

        brace_start = start + len('\\boxed{')
        depth = 1
        i = brace_start

        while i < len(text) and depth > 0:
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
            i += 1

        if depth == 0:
            return text[brace_start:i-1].strip()
        return None

    def extract_all_boxed(text: str) -> list:
        """SafeAbs: \\boxed{} - """
        results = []
        text = str(text or "")
        pos = 0
        while True:
            start = text.find('\\boxed{', pos)
            if start == -1:
                break
            brace_start = start + len('\\boxed{')
            depth = 1
            i = brace_start
            while i < len(text) and depth > 0:
                if text[i] == '{':
                    depth += 1
                elif text[i] == '}':
                    depth -= 1
                i += 1
            if depth == 0:
                results.append(text[brace_start:i-1].strip())
            pos = i
        return results

    def compute_f1(pred: str, truth: str) -> float:
        pred_tokens = normalize_answer(pred).split()
        truth_tokens = normalize_answer(truth).split()
        if not pred_tokens or not truth_tokens:
            return 0.0
        common = set(pred_tokens) & set(truth_tokens)
        if not common:
            return 0.0
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(truth_tokens)
        return 2 * precision * recall / (precision + recall)

    def parse_latex_fraction(text: str) -> Optional[float]:
        """LaTeX - LaTeX: , LaTeX: , LaTeX: """
        import re

        text = text.strip()
        negative = False
        if text.startswith('-'):
            negative = True
            text = text[1:].strip()

        def eval_expr(s):
            s = s.strip()
            sqrt_match = re.match(r'^\\sqrt\{(\d+)\}$', s)
            if sqrt_match:
                return math.sqrt(float(sqrt_match.group(1)))
            return float(s)

        frac_match = re.search(r'\\d?frac\{(.+?)\}\{([^}]+)\}$', text)
        if frac_match:
            num_str, den_str = frac_match.group(1).strip(), frac_match.group(2).strip()
            try:
                num_val = eval_expr(num_str)
                den_val = eval_expr(den_str)
                if den_val != 0:
                    val = num_val / den_val
                    return -val if negative else val
            except:
                pass

        simple_frac_match = re.search(r'\\d?frac(\d)(\d)', text)
        if simple_frac_match:
            try:
                num = float(simple_frac_match.group(1))
                den = float(simple_frac_match.group(2))
                val = num / den
                return -val if negative else val
            except:
                pass

        slash_match = re.search(r'(-?\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)', text)
        if slash_match:
            try:
                val = float(slash_match.group(1)) / float(slash_match.group(2))
                return -val if negative else val
            except:
                pass

        return None

    def _extract_clean_answer(pred_obj):
        """CompleteAnswer:  - Format

        :
        1. Plan: {'approach': ...} -> 
        2. : ['1/3', 'To determine...'] -> 
        3. : 'To find the remainder...' -> 
        """
        if pred_obj is None:
            return ""

        if isinstance(pred_obj, dict):
            for key in ['output', 'answer', 'response', 'result', 'value']:
                if key in pred_obj and pred_obj[key]:
                    val = pred_obj[key]
                    if not isinstance(val, dict):
                        return _extract_clean_answer(val)
            return str(pred_obj)

        if isinstance(pred_obj, list):
            if not pred_obj:
                return ""
            if len(pred_obj) > 1:
                return ", ".join(str(x).strip() for x in pred_obj if x)
            first = pred_obj[0]
            if isinstance(first, (int, float)):
                return str(first)
            if isinstance(first, str) and len(first) < 50:
                return first.strip()
            return _extract_clean_answer(first)

        pred_str = str(pred_obj).strip()

        text_patterns = [
            r'^To (?:find|determine|calculate|solve)',
            r'^(?:|||)',
            r'^The (?:answer|result|value|solution)',
            r'^We (?:can|need|have|first)',
        ]
        is_explanation = any(re.match(p, pred_str, re.IGNORECASE) for p in text_patterns)

        if is_explanation and len(pred_str) > 100:
            result_match = re.search(r'(?:remainder|result|answer|value)\s*(?:is|of|=|:)?\s*([\-\d]+(?:\.\d+)?)', pred_str, re.IGNORECASE)
            if result_match:
                return result_match.group(1).rstrip('.')
            nums = re.findall(r'(?:^|\s)([\-+]?\d+(?:\.\d+)?)(?:\s|$|[,.])', pred_str)
            if nums:
                return nums[-1].rstrip('.')

        return pred_str

    def math_equal_vectorized(pred: str, truth: str, tolerance: float = 1e-3) -> bool:
        """Vectorized:  (vectorized)
        """
        import math
        pred = str(pred).strip()
        truth = str(truth).strip()

        # ============================================
        # ============================================
        def preprocess_answer(s: str) -> str:
            """PlanOutput: """
            s = str(s).strip()

            if '=' in s and not s.startswith('='):
                parts = s.rsplit('=', 1)
                if len(parts) == 2:
                    right_side = parts[1].strip()
                    if right_side and not re.match(r'^[a-zA-Z_]\w*$', right_side):
                        s = right_side

            if s.startswith('[') and s.endswith(']'):
                inner = s[1:-1].strip()
                if re.match(r'^-?[\d.,\s\-]+$', inner):
                    s = inner
            if s.startswith('{') and s.endswith('}'):
                inner = s[1:-1].strip()
                if re.match(r'^-?[\d.,\s\-]+$', inner):
                    s = inner

            str_tuple_match = re.match(r"^\(\s*['\"](.+?)['\"](?:\s*,\s*['\"](.+?)['\"])*\s*\)$", s)
            if str_tuple_match:
                values = re.findall(r"['\"]([^'\"]+)['\"]", s)
                s = '(' + ', '.join(values) + ')'

            s = re.sub(r'\+0[jJiI]\b', '', s)
            s = re.sub(r'([\d.]+)\s*\*\s*[IiJj]', r'\1i', s)  # 5*I -> 5i
            s = s.replace('j', 'i').replace('J', 'i').replace('I', 'i')  # j/J/I -> i
            s = re.sub(r'\s+([+-])\s*(\d)', r'\1\2', s)
            if s.startswith('(') and s.endswith(')') and 'i' in s:
                inner = s[1:-1].strip()
                if re.match(r'^-?[\d.]+[+-][\d.]*i$', inner):
                    s = inner

            s = re.sub(r'(\w)\*\*(-?\d+)', r'\1^\2', s)

            s = re.sub(r'(\d)\*([a-zA-Z])', r'\1\2', s)  # 5*x -> 5x
            s = re.sub(r'\)\*([a-zA-Z])', r')\1', s)  # sqrt(3)*pi -> sqrt(3)pi

            return s.strip()

        pred = preprocess_answer(pred)
        truth = preprocess_answer(truth)

        try:
            from sympy import simplify, N
            from sympy.parsing.latex import parse_latex
            from sympy.parsing.sympy_parser import parse_expr

            def try_parse(s):
                """"""
                s = str(s).strip()
                s_normalized = s.replace('π', 'pi')  # Unicode π -> pi
                s_normalized = re.sub(r'\bpi\s*/\s*(\d+)', r'\\frac{\\pi}{\1}', s_normalized)
                s_normalized = re.sub(r'(\d*)\*?sqrt\((\d+)\)', r'\1\\sqrt{\2}', s_normalized)
                s_normalized = re.sub(r'(?<!\\)\bpi\b', r'\\pi', s_normalized)
                try:
                    return parse_latex(s_normalized)
                except:
                    pass
                try:
                    return parse_latex(s)
                except:
                    pass
                try:
                    return parse_expr(s.replace('π', 'pi'))
                except:
                    pass
                return None

            pred_expr = try_parse(pred)
            truth_expr = try_parse(truth)

            if pred_expr is not None and truth_expr is not None:
                try:
                    if simplify(pred_expr - truth_expr) == 0:
                        return True
                except:
                    pass
                try:
                    pred_val = float(N(pred_expr))
                    truth_val = float(N(truth_expr))
                    if abs(pred_val - truth_val) < tolerance:
                        return True
                except:
                    pass

            if truth_expr is not None:
                try:
                    pred_num = float(pred)
                    truth_val = float(N(truth_expr))
                    if abs(pred_num - truth_val) < tolerance:
                        return True
                except:
                    pass

            if pred_expr is not None:
                try:
                    pred_val = float(N(pred_expr))
                    truth_num = float(truth)
                    if abs(pred_val - truth_num) < tolerance:
                        return True
                except:
                    pass
        except ImportError:
            pass

        def parse_simple_fraction(s: str) -> Optional[float]:
            """Fraction:  a/b (LaTeX)"""
            s = str(s).strip()
            frac_match = re.match(r'^(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)$', s)
            if frac_match:
                try:
                    numer = float(frac_match.group(1))
                    denom = float(frac_match.group(2))
                    if denom != 0:
                        return numer / denom
                except:
                    pass
            return None

        pred_frac = parse_simple_fraction(pred)
        truth_frac = parse_simple_fraction(truth)

        if pred_frac is not None and truth_frac is not None:
            if abs(pred_frac - truth_frac) < tolerance:
                return True

        if pred_frac is not None:
            try:
                truth_num = float(truth)
                if abs(pred_frac - truth_num) < tolerance:
                    return True
            except:
                pass

        if truth_frac is not None:
            try:
                pred_num = float(pred)
                if abs(pred_num - truth_frac) < tolerance:
                    return True
            except:
                pass

        pred_normalized = pred.replace('j', 'i').strip('()').strip()
        truth_normalized = truth.replace('j', 'i').strip('()').strip()

        def parse_complex(s: str) -> Optional[complex]:
            s = s.replace(' ', '').replace('j', 'i')
            s = s.strip('()')
            try:
                s_py = s.replace('i', 'j')
                return complex(s_py)
            except:
                pass
            m = re.match(r'^([+-]?[\d.]+)?\s*([+-])\s*([\d.]+)?i$', s)
            if m:
                real = float(m.group(1) or 0)
                sign = 1 if m.group(2) == '+' else -1
                imag = float(m.group(3) or 1)
                return complex(real, sign * imag)
            m = re.match(r'^([+-]?[\d.]+)?i$', s)
            if m:
                imag = float(m.group(1) or 1)
                return complex(0, imag)
            try:
                return complex(float(s), 0)
            except:
                pass
            return None

        pred_complex = parse_complex(pred)
        truth_complex = parse_complex(truth)
        if pred_complex is not None and truth_complex is not None:
            if abs(pred_complex - truth_complex) < 0.01:
                return True

        if normalize_answer(pred_normalized) == normalize_answer(truth_normalized):
            return True

        try:
            pred_num = float(re.sub(r'[^\d.\-]', '', pred))
            truth_num = float(re.sub(r'[^\d.\-]', '', truth))
            if abs(pred_num - truth_num) < tolerance:
                return True
            if abs(truth_num) > 1e-6:
                rel_error = abs(pred_num - truth_num) / abs(truth_num)
                if rel_error < 0.01:
                    return True
        except:
            pass

        truth_frac = parse_latex_fraction(truth)
        pred_frac = parse_latex_fraction(pred)

        if truth_frac is not None and pred_frac is not None:
            if abs(pred_frac - truth_frac) < tolerance:
                return True

        if truth_frac is not None:
            try:
                pred_num = float(pred)
                if abs(pred_num - truth_frac) < tolerance:
                    return True
            except:
                pass
        if pred_frac is not None:
            try:
                truth_num = float(truth)
                if abs(pred_frac - truth_num) < tolerance:
                    return True
            except:
                pass

        def parse_numeric_expr(s: str) -> Optional[float]:
            """: √π"""
            s = s.strip()
            frac = parse_latex_fraction(s)
            if frac is not None:
                return frac
            sqrt_match = re.search(r'^(-?)(\d*)\s*\\?sqrt\{?(\d+)\}?$', s)
            if sqrt_match:
                sign = -1 if sqrt_match.group(1) == '-' else 1
                coef = float(sqrt_match.group(2)) if sqrt_match.group(2) else 1.0
                radicand = float(sqrt_match.group(3))
                return sign * coef * math.sqrt(radicand)
            try:
                return float(s)
            except:
                return None

        def parse_coordinate_tuple(s: str) -> Optional[tuple]:
            s = s.strip()
            s = re.sub(r'\\left\s*|\\right\s*', '', s)
            s = s.strip('()')
            parts = re.split(r'\s*,\s*', s)
            if len(parts) >= 2:
                coords = []
                for p in parts:
                    p = p.strip()
                    val = parse_numeric_expr(p)
                    if val is not None:
                        coords.append(val)
                    else:
                        return None
                return tuple(coords)
            return None

        pred_coords = parse_coordinate_tuple(pred)
        truth_coords = parse_coordinate_tuple(truth)
        if pred_coords is not None and truth_coords is not None:
            if len(pred_coords) == len(truth_coords):
                if all(abs(p - t) < 0.01 for p, t in zip(pred_coords, truth_coords)):
                    return True

        def parse_vector(s: str) -> Optional[list]:
            s = s.strip()
            if s.startswith('(') and s.endswith(')'):
                try:
                    import ast
                    result = ast.literal_eval(s)
                    if isinstance(result, tuple):
                        return [float(x) if not isinstance(x, complex) else x for x in result]
                except:
                    pass
            if s.startswith('[') and s.endswith(']'):
                try:
                    import ast
                    return list(ast.literal_eval(s))
                except:
                    pass
            pmat_match = re.search(r'\\begin\{[pbvBV]?matrix\}(.+?)\\end\{[pbvBV]?matrix\}', s, re.DOTALL)
            if pmat_match:
                content = pmat_match.group(1).strip()
                elements = re.split(r'\\\\|&', content)
                vals = []
                for e in elements:
                    e = e.strip()
                    if not e:
                        continue
                    val = parse_numeric_expr(e)
                    if val is not None:
                        vals.append(val)
                    else:
                        return None
                return vals if vals else None
            return None

        pred_vec = parse_vector(pred)
        truth_vec = parse_vector(truth)
        if pred_vec is not None and truth_vec is not None:
            if len(pred_vec) == len(truth_vec):
                if all(abs(p - t) < 0.01 for p, t in zip(pred_vec, truth_vec)):
                    return True

        def parse_pi_expr(s: str) -> Optional[float]:
            """Pi: π"""
            s = s.strip().replace('π', 'pi')
            try:
                frac_match = re.search(r'\\d?frac\{.*?(?:\\pi|pi).*?\}\{([^}]+)\}', s)
                if frac_match:
                    denom = float(frac_match.group(1).strip())
                    return math.pi / denom
                simple_frac = re.match(r'^(?:\\?pi|π)\s*/\s*(\d+(?:\.\d+)?)$', s)
                if simple_frac:
                    return math.pi / float(simple_frac.group(1))
                npi_match = re.match(r'^([\d.]+)\s*\*?\s*(?:\\?pi|π)$', s)
                if npi_match:
                    return float(npi_match.group(1)) * math.pi
                if re.match(r'^(?:\\?pi|π)$', s):
                    return math.pi
                return None
            except:
                return None

        pred_pi = parse_pi_expr(pred)
        truth_pi = parse_pi_expr(truth)
        if pred_pi is not None and truth_pi is not None:
            if abs(pred_pi - truth_pi) < 0.01:
                return True
        if truth_pi is not None:
            try:
                pred_num = float(pred)
                if abs(pred_num - truth_pi) < 0.01:
                    return True
            except:
                pass
        if pred_pi is not None:
            try:
                truth_num = float(truth)
                if abs(pred_pi - truth_num) < 0.01:
                    return True
            except:
                pass

        if '\\sqrt' in truth:
            try:
                def eval_sqrt_expr(expr):
                    expr = re.sub(r'\\sqrt\{(\d+)\}', r'math.sqrt(\1)', expr)
                    expr = re.sub(r'\\d?frac\{([^}]+)\}\{([^}]+)\}', r'((\1)/(\2))', expr)
                    expr = re.sub(r'\\[a-zA-Z]+', '', expr)
                    return eval(expr)
                truth_val = eval_sqrt_expr(truth)
                pred_num = float(pred)
                if abs(pred_num - truth_val) < 0.01:
                    return True
            except:
                pass

        base_match = re.match(r'^(\d+)_(\d+)$', truth)
        if base_match:
            truth_digits = base_match.group(1)
            if pred.strip() == truth_digits:
                return True

        def extract_text_content(s: str) -> str:
            """ \text{...} """
            text_match = re.match(r'^\\text\{([^}]+)\}$', s.strip())
            if text_match:
                return text_match.group(1).strip()
            return s.strip()

        pred_text = extract_text_content(pred)
        truth_text = extract_text_content(truth)
        if pred_text.upper() == truth_text.upper():
            return True

        degree_match = re.match(r'^([\d.]+)\s*\^?\\?circ$', truth)
        if degree_match:
            truth_num = degree_match.group(1)
            if pred.strip() == truth_num:
                return True

        if '\\sqrt' in truth or 'sqrt' in pred.lower():
            try:
                def eval_sqrt_approx(expr):
                    expr = re.sub(r'\\sqrt\{(\d+)\}', r'math.sqrt(\1)', expr)
                    expr = re.sub(r'\\d?frac\{([^}]+)\}\{([^}]+)\}', r'((\1)/(\2))', expr)
                    expr = re.sub(r'\\[a-zA-Z]+', '', expr)
                    expr = re.sub(r'\s+', '', expr)
                    return eval(expr)
                truth_val = eval_sqrt_approx(truth)
                pred_num = float(pred)
                if abs(pred_num - truth_val) < 0.001:
                    return True
            except:
                pass

        if ',' in truth and not truth.startswith('(') and not truth.startswith('['):
            try:
                truth_parts = [x.strip() for x in truth.split(',')]
                truth_nums = [float(x) for x in truth_parts]
                if pred.startswith('[') and pred.endswith(']'):
                    import ast
                    pred_list = ast.literal_eval(pred)
                    if isinstance(pred_list, list):
                        pred_set = list(dict.fromkeys([round(x, 6) for x in pred_list]))
                        truth_set = [round(x, 6) for x in truth_nums]
                        if sorted(pred_set) == sorted(truth_set):
                            return True
            except:
                pass

        if (',' in truth) or (',' in pred):
            try:
                def split_top_level_commas(s: str) -> list:
                    s = s.strip().strip('{}').strip('[]')
                    parts, buf, depth_paren, depth_brack, depth_brace = [], [], 0, 0, 0
                    for ch in s:
                        if ch == '(':
                            depth_paren += 1
                        elif ch == ')':
                            depth_paren = max(0, depth_paren - 1)
                        elif ch == '[':
                            depth_brack += 1
                        elif ch == ']':
                            depth_brack = max(0, depth_brack - 1)
                        elif ch == '{':
                            depth_brace += 1
                        elif ch == '}':
                            depth_brace = max(0, depth_brace - 1)

                        if ch == ',' and depth_paren == 0 and depth_brack == 0 and depth_brace == 0:
                            parts.append(''.join(buf).strip())
                            buf = []
                        else:
                            buf.append(ch)
                    if buf:
                        parts.append(''.join(buf).strip())
                    return [p for p in parts if p]

                truth_parts = split_top_level_commas(truth)
                pred_parts = split_top_level_commas(pred)

                def get_part_values(parts):
                    values = []
                    unmatched = []

                    def eval_latex_expr(s: str):
                        s = s.strip()
                        m = re.match(r'^(-?\d+(?:\.\d+)?)\s*([+-])\s*\\?sqrt\{?(\d+)\}?$', s)
                        if m:
                            a = float(m.group(1))
                            sign = 1 if m.group(2) == '+' else -1
                            b = float(m.group(3))
                            return a + sign * math.sqrt(b)
                        m = re.match(r'^\\?sqrt\{?(\d+)\}?\s*([+-])\s*(-?\d+(?:\.\d+)?)$', s)
                        if m:
                            b = float(m.group(1))
                            sign = 1 if m.group(2) == '+' else -1
                            a = float(m.group(3))
                            return math.sqrt(b) + sign * a
                        return None

                    for p in parts:
                        c = None
                        try:
                            c = parse_complex(p)
                        except:
                            c = None
                        if c is not None:
                            values.append(c)
                            continue

                        try:
                            v = eval_latex_expr(p)
                            if v is not None:
                                values.append(complex(float(v), 0.0))
                                continue
                        except:
                            pass

                        try:
                            v2 = parse_numeric_expr(p)
                            if v2 is not None:
                                values.append(complex(float(v2), 0.0))
                                continue
                        except:
                            pass

                        try:
                            values.append(complex(float(p), 0.0))
                            continue
                        except:
                            pass

                        unmatched.append(p.strip())

                    return values, unmatched

                truth_vals, truth_unmatched = get_part_values(truth_parts)
                pred_vals, pred_unmatched = get_part_values(pred_parts)

                if truth_vals and pred_vals and len(truth_vals) == len(pred_vals) and not truth_unmatched and not pred_unmatched:
                    def is_close(c1: complex, c2: complex, tol=0.01):
                        return abs(c1 - c2) < tol

                    matched = [False] * len(truth_vals)
                    for pv in pred_vals:
                        found = False
                        for i, tv in enumerate(truth_vals):
                            if not matched[i] and is_close(pv, tv):
                                matched[i] = True
                                found = True
                                break
                        if not found:
                            break
                    if all(matched):
                        return True

                if not truth_vals and not pred_vals and sorted(truth_unmatched) == sorted(pred_unmatched):
                    return True

            except:
                pass

        if pred.startswith('[') and pred.endswith(']'):
            try:
                import ast
                pred_list = ast.literal_eval(pred)
                if isinstance(pred_list, list) and len(pred_list) == 2:
                    a, b = pred_list
                    poly_match = re.match(r'^(-?\d+)x\s*([+-])\s*(\d+)$', truth.replace(' ', ''))
                    if poly_match:
                        truth_a = int(poly_match.group(1))
                        sign = 1 if poly_match.group(2) == '+' else -1
                        truth_b = sign * int(poly_match.group(3))
                        if abs(a - truth_a) < 0.01 and abs(b - truth_b) < 0.01:
                            return True
            except:
                pass

        if pred.startswith('[') and pred.endswith(']'):
            try:
                import ast
                pred_list = ast.literal_eval(pred)
                if isinstance(pred_list, list):
                    truth_complex = parse_complex(truth)
                    if truth_complex is not None:
                        for item in pred_list:
                            item_complex = parse_complex(str(item))
                            if item_complex is not None:
                                if abs(item_complex - truth_complex) < 0.01:
                                    return True
            except:
                pass

        if pred.startswith('[') or pred.startswith('{') or pred.startswith("'"):
            try:
                import ast
                try:
                    pred_parsed = ast.literal_eval(pred)
                    if isinstance(pred_parsed, (list, tuple)) and len(pred_parsed) > 0:
                        first_item = str(pred_parsed[0]).strip()
                        if math_equal_vectorized(first_item, truth, tolerance):
                            return True
                    elif isinstance(pred_parsed, dict):
                        for key in ['answer', 'result', 'output', 'value']:
                            if key in pred_parsed:
                                if math_equal_vectorized(str(pred_parsed[key]), truth, tolerance):
                                    return True
                except:
                    pass
            except:
                pass

        return False

    def check_correctness(pred: str, truth: str, ptype: str) -> tuple:
        ptype = str(ptype).lower().strip()
        if ptype == 'math':
            boxed_truth = extract_boxed(truth)
            if boxed_truth:
                truth = boxed_truth

            boxed_pred = extract_boxed(pred)
            if boxed_pred:
                pred = boxed_pred
            if math_equal_vectorized(pred, truth):
                return 1.0, True
            return 0.0, False
        elif ptype == 'qa':
            f1 = compute_f1(pred, truth)
            return f1, f1 >= 0.5
        elif ptype == 'code':
            if 'PASS' in str(pred).upper() or 'passed' in str(pred).lower():
                return 1.0, True
            return 0.0, False
        else:
            if normalize_answer(pred) == normalize_answer(truth):
                return 1.0, True
            return compute_f1(pred, truth), False

    results = []
    stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'score_sum': 0.0, 'rounds_sum': 0})

    interactive_cfg = config.get('interactive_grpo', {}) or {}
    max_rounds = int(interactive_cfg.get('max_rounds', 20))
    finish_cfg = interactive_cfg.get('finish_constraints', {}) or {}

    logger.info(f"🚀 Starting VECTORIZED evaluation: {len(data)} problems, {num_workers} workers, max_rounds={max_rounds}")
    start_time = time.time()

    envs = []
    prompts = []
    item_data = []
    active_mask = []

    for i, item in enumerate(data):
        problem = item.get('problem', item.get('question', ''))
        problem_type = str(item.get('problem_type', 'qa')).lower().strip()
        meta = item.get('meta', {})

        executor = create_aflow_executor_wrapper(aflow_executor, problem_type)
        executor_kwargs = {}
        if problem_type == 'code':
            test = meta.get('test')
            entry_point = meta.get('entry_point')
            if test:
                executor_kwargs['test'] = test
            if entry_point:
                executor_kwargs['entry_point'] = entry_point

        env = create_env(
            problem=problem,
            problem_type=problem_type,
            executor=executor,
            executor_kwargs=executor_kwargs if executor_kwargs else None,
            max_rounds=max_rounds,
            execute_each_step=True,
            finish_min_total_operators=finish_cfg.get('min_total_operators', 1),
            finish_require_checker=finish_cfg.get('require_checker', False),
            finish_require_structure=finish_cfg.get('require_structure', False),
            use_custom_prompts_in_execution=True,
        )
        envs.append(env)

        prompt = prompt_builder.build_initial_prompt(
            problem=problem,
            problem_type=problem_type,
        )
        prompts.append(prompt)
        item_data.append(item)
        active_mask.append(True)

    final_answers = ["" for _ in data]
    final_dsls = ["" for _ in data]
    rounds_used = [0 for _ in data]
    operator_traces = [[] for _ in data]

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        for round_idx in range(max_rounds):
            active_indices = [i for i, a in enumerate(active_mask) if a]
            if not active_indices:
                break

            logger.info(f"  Round {round_idx+1}/{max_rounds}: {len(active_indices)} active problems")

            gen_futures = {}
            for idx in active_indices:
                env = envs[idx]
                is_awaiting_prompt = hasattr(env, 'env_state') and env.env_state == EnvState.AWAITING_PROMPT
                fut = pool.submit(generate_fn, prompts[idx], disable_thinking=is_awaiting_prompt)
                gen_futures[fut] = idx

            responses = {}
            for fut in as_completed(gen_futures):
                idx = gen_futures[fut]
                try:
                    responses[idx] = fut.result()
                except Exception as e:
                    logger.warning(f"[WARNING] vLLM gen failed for idx={idx}: {e}")
                    responses[idx] = "<action>add</action><operator>Custom</operator><prompt>error</prompt>"

            def execute_step(idx):
                env = envs[idx]
                response = responses.get(idx, "")

                pre_operator = getattr(env, 'pending_operator', None)
                pre_prompt = getattr(env, 'pending_prompt_context', '')

                feedback, success_list, still_active = env.step(response)
                done = not still_active

                answer = ""
                if hasattr(env, 'final_answer') and getattr(env, 'final_answer'):
                    answer = getattr(env, 'final_answer')
                if hasattr(env, 'last_execution_result') and getattr(env, 'last_execution_result') is not None:
                    answer = getattr(env, 'last_execution_result')

                operator_info = {
                    'round': round_idx + 1,
                    'operator': pre_operator or 'Unknown',
                    'input': str(pre_prompt)[:500] if pre_prompt else '',
                    'output': str(answer)[:500] if answer else '',
                    'feedback': str(feedback)[:500] if feedback else '',
                    'response': response[:300] if response else '',
                }

                if pre_operator == 'Verify' and hasattr(env, 'last_execution_result'):
                    exec_result = env.last_execution_result
                    if isinstance(exec_result, dict):
                        operator_info['verify_result'] = {
                            'is_correct': exec_result.get('is_correct'),
                            'suggested_answer': str(exec_result.get('suggested_answer', ''))[:200],
                            'confidence': exec_result.get('confidence', ''),
                        }

                return {
                    'idx': idx,
                    'done': done,
                    'answer': answer,
                    'dsl': env.get_dsl() if done else "",
                    'feedback': feedback,
                    'success_list': success_list,
                    'env': env,
                    'operator_info': operator_info,
                }

            step_futures = {pool.submit(execute_step, idx): idx for idx in active_indices}
            step_results = {}
            for fut in as_completed(step_futures):
                idx = step_futures[fut]
                try:
                    step_results[idx] = fut.result()
                except Exception as e:
                    logger.warning(f"[WARNING] env.step failed for idx={idx}: {e}")
                    step_results[idx] = {'idx': idx, 'done': True, 'answer': '', 'dsl': '', 'feedback': str(e), 'success_list': []}

            for idx in active_indices:
                res = step_results.get(idx, {})
                rounds_used[idx] = round_idx + 1

                if res.get('operator_info'):
                    operator_traces[idx].append(res['operator_info'])

                if res.get('answer'):
                    final_answers[idx] = res['answer']
                if res.get('dsl'):
                    final_dsls[idx] = res['dsl']

                if res.get('done'):
                    active_mask[idx] = False
                    env = envs[idx]
                    if hasattr(env, 'final_answer') and env.final_answer:
                        final_answers[idx] = env.final_answer
                    elif hasattr(env, 'last_execution_result') and env.last_execution_result is not None:
                        final_answers[idx] = env.last_execution_result
                    if not final_dsls[idx]:
                        final_dsls[idx] = env.get_dsl()
                else:
                    env = envs[idx]
                    item = item_data[idx]
                    problem = item.get('problem', item.get('question', ''))
                    problem_type = str(item.get('problem_type', 'qa')).lower().strip()
                    feedback = res.get('feedback', '')
                    success_list = res.get('success_list', [])

                    if hasattr(env, 'env_state') and env.env_state == EnvState.AWAITING_PROMPT:
                        operator_name = env.pending_operator or "Custom"
                        prompts[idx] = prompt_builder.build_prompt_request(
                            problem=problem,
                            operator_name=operator_name,
                            operator_description="",
                            context=getattr(env, "pending_prompt_context", "") or "",
                            problem_type=problem_type,
                        )
                    else:
                        env_stats = env.graph.get_statistics() if hasattr(env, 'graph') else {}
                        prompts[idx] = prompt_builder.build_continuation_prompt(
                            problem=problem,
                            current_dsl=env.get_dsl(),
                            total_operators=env_stats.get('total_operators', 0),
                            unique_types=env_stats.get('unique_types', 0),
                            round_number=round_idx + 2,
                            max_rounds=max_rounds,
                            node_ids=[n.id for n in env.graph.nodes] if hasattr(env, 'graph') else [],
                            last_success=all(success_list) if success_list else True,
                            last_result=feedback[:500] if feedback else "",
                            last_message="",
                            problem_type=problem_type,
                        )
                        if feedback and '[NEXT]' in feedback:
                            print(f"[DEBUG] idx={idx}, feedback[:200]={feedback[:200]}")

            completed = sum(1 for a in active_mask if not a)
            if completed > 0 and (round_idx + 1) % 5 == 0:
                elapsed = time.time() - start_time
                logger.info(f"    Progress: {completed}/{len(data)} completed, {len(active_indices)} active, {elapsed:.1f}s elapsed")

    for i, item in enumerate(data):
        problem_type = str(item.get('problem_type', 'qa')).lower().strip()
        ground_truth = item.get('ground_truth', item.get('answer', ''))
        source = item.get('source', 'unknown')

        pred_obj = final_answers[i]
        pred_obj = _extract_clean_answer(pred_obj)

        score, is_correct = check_correctness(str(pred_obj), str(ground_truth), problem_type)

        boxed_truth = extract_boxed(str(ground_truth))
        boxed_pred = extract_boxed(str(pred_obj))
        if not is_correct:
            logger.warning(f"❌ WRONG [{i+1}]: pred='{str(pred_obj)[:80]}' vs boxed_truth='{boxed_truth}' (raw_truth[:50]='{str(ground_truth)[:50]}')")
        else:
            logger.debug(f"✓ CORRECT [{i+1}]: pred='{str(pred_obj)[:50]}' vs boxed_truth='{boxed_truth}'")

        result = {
            'idx': i,
            'source': source,
            'problem_type': problem_type,
            'problem': item.get('problem', '')[:100],
            'ground_truth': str(ground_truth)[:50],
            'final_answer': str(pred_obj)[:200],
            'final_dsl': final_dsls[i],
            'rounds_used': rounds_used[i],
            'score': score,
            'is_correct': is_correct,
        }
        results.append(result)

        log_workflow(
            idx=i,
            problem=item.get('problem', '')[:500],
            dsl=final_dsls[i],
            answer=str(pred_obj)[:500],
            is_correct=is_correct,
            rounds=rounds_used[i],
            ground_truth=str(ground_truth)[:200],
            operator_trace=operator_traces[i],
            problem_type=problem_type
        )

        for key in [source, problem_type, 'overall']:
            stats[key]['total'] += 1
            stats[key]['score_sum'] += score
            stats[key]['rounds_sum'] += rounds_used[i]
            if is_correct:
                stats[key]['correct'] += 1

        logger.debug(f"  [{i+1}/{len(data)}] {source}/{problem_type}: is_correct={is_correct}")

    elapsed = time.time() - start_time
    overall = stats.get('overall', {})
    acc = overall.get('correct', 0) / overall.get('total', 1) * 100
    logger.info(f"✅ Vectorized evaluation done: {overall.get('total')} samples, {acc:.1f}% accuracy, {elapsed:.1f}s")

    return results, dict(stats)


def run_evaluation(
    config: dict,
    data: List[dict],
    generate_fn,
    aflow_executor,
    prompt_builder,
    num_workers: int = 16,
):
    """ ()"""
    results = []
    stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'score_sum': 0.0, 'rounds_sum': 0})

    logger.info(f"Starting parallel evaluation with {num_workers} workers...")
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                run_single_problem, i, item, generate_fn, aflow_executor, config, prompt_builder
            ): i
            for i, item in enumerate(data)
        }

        completed = 0
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                idx = futures[future]
                logger.warning(f"Error processing item {idx}: {e}")
                result = {
                    'idx': idx,
                    'source': data[idx].get('source', 'unknown'),
                    'problem_type': data[idx].get('problem_type', 'unknown'),
                    'score': 0.0,
                    'is_correct': False,
                    'error': str(e),
                }

            results.append(result)

            source = result.get('source', 'unknown')
            ptype = result.get('problem_type', 'unknown')
            score = result.get('score', 0.0)
            is_correct = result.get('is_correct', False)
            rounds_used = result.get('rounds_used', 0)

            for key in [source, ptype, 'overall']:
                stats[key]['total'] += 1
                stats[key]['score_sum'] += score
                stats[key]['rounds_sum'] += rounds_used
                if is_correct:
                    stats[key]['correct'] += 1

            completed += 1
            if completed % 5 == 0 or completed == len(data):
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                overall_acc = stats['overall']['correct'] / stats['overall']['total'] * 100
                avg_rounds = stats['overall']['rounds_sum'] / stats['overall']['total']
                logger.info(
                    f"Progress: {completed}/{len(data)} ({rate:.2f}/s) | "
                    f"Acc: {overall_acc:.1f}% | Avg Rounds: {avg_rounds:.1f}"
                )

    return results, dict(stats)


def print_results(stats: dict, elapsed: float):
    """"""
    print("\n" + "=" * 70)
    print("📊 EVALUATION RESULTS (AFlow Executor: 4o-mini)")
    print("=" * 70)

    print("\n📁 By Source:")
    print("-" * 60)
    print(f"  {'Source':<12} {'Correct':<10} {'Accuracy':<10} {'Avg Score':<12} {'Avg Rounds'}")
    print("-" * 60)
    for source in ['gsm8k', 'math', 'hotpotqa', 'squad_v2', 'humaneval', 'mbpp']:
        if source in stats:
            s = stats[source]
            acc = s['correct'] / s['total'] * 100 if s['total'] > 0 else 0
            avg_score = s['score_sum'] / s['total'] if s['total'] > 0 else 0
            avg_rounds = s['rounds_sum'] / s['total'] if s['total'] > 0 else 0
            print(f"  {source:<12} {s['correct']:>4}/{s['total']:<4} {acc:>6.1f}%    {avg_score:>6.3f}       {avg_rounds:>5.1f}")

    print("\n📝 By Problem Type:")
    print("-" * 60)
    for ptype in ['math', 'qa', 'code']:
        if ptype in stats:
            s = stats[ptype]
            acc = s['correct'] / s['total'] * 100 if s['total'] > 0 else 0
            avg_score = s['score_sum'] / s['total'] if s['total'] > 0 else 0
            avg_rounds = s['rounds_sum'] / s['total'] if s['total'] > 0 else 0
            print(f"  {ptype:<12} {s['correct']:>4}/{s['total']:<4} {acc:>6.1f}%    {avg_score:>6.3f}       {avg_rounds:>5.1f}")

    print("\n" + "=" * 70)
    print("🎯 OVERALL RESULTS:")
    print("-" * 60)
    if 'overall' in stats:
        s = stats['overall']
        acc = s['correct'] / s['total'] * 100 if s['total'] > 0 else 0
        avg_score = s['score_sum'] / s['total'] if s['total'] > 0 else 0
        avg_rounds = s['rounds_sum'] / s['total'] if s['total'] > 0 else 0
        print(f"  Total Samples: {s['total']}")
        print(f"  Correct: {s['correct']} ({acc:.1f}%)")
        print(f"  Average Score: {avg_score:.3f}")
        print(f"  Average Rounds: {avg_rounds:.1f}")
        print(f"  Time Elapsed: {elapsed:.1f}s ({s['total']/elapsed:.2f} samples/s)")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Pure Inference Evaluation with AFlow")
    parser.add_argument('--config', type=str, default='config/eval_interactive.yaml')
    parser.add_argument('--checkpoint', type=str, default=None, help='(vLLM) LoRA checkpoint path for record/diagnosis')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'], help='Which split to evaluate (default: test)')
    parser.add_argument('--data', type=str, default=None, help='Dataset jsonl path (override config)')
    parser.add_argument('--vllm-base-url', type=str, default=None, help='Override vLLM base_url, e.g. http://localhost:8002/v1')
    parser.add_argument('--vllm-model', type=str, default=None, help='Override primary model name (e.g. lora_adapter)')
    parser.add_argument('--num-samples', type=int, default=None, help='Number of samples to evaluate')
    parser.add_argument('--workers', type=int, default=16, help='Number of parallel workers')
    parser.add_argument('--vectorized', action='store_true', help='Use true vectorized rollout (recommended)')
    parser.add_argument('--output', type=str, default=None, help='Output file for results')
    args = parser.parse_args()

    config = load_config(args.config)
    logger.info(f"Config loaded: {args.config}")

    if args.vllm_base_url:
        config['vllm_base_url'] = args.vllm_base_url
    if args.vllm_model:
        config['lora_adapter_name'] = args.vllm_model

    if args.checkpoint:
        logger.warning(
            f"--checkpoint={args.checkpoint} will NOT be auto-loaded into vLLM; "
            f"please ensure your vLLM server is started with this checkpoint/adapter."
        )

    data_path = args.data or config.get(f'{args.split}_dataset') or config.get('test_dataset') or config.get('train_dataset')
    logger.info(f"Loading dataset from {data_path}")
    data = load_dataset(data_path, args.num_samples)
    logger.info(f"Loaded {len(data)} samples")

    source_counts = defaultdict(int)
    for item in data:
        source_counts[item.get('source', 'unknown')] += 1
    logger.info(f"Data distribution: {dict(sorted(source_counts.items()))}")

    vllm_base_url = str(config.get('vllm_base_url', '')).rstrip('/')
    try:
        import urllib.request

        with urllib.request.urlopen(f"{vllm_base_url}/models", timeout=5) as resp:
            resp.read(256)
        logger.info(f"vLLM reachable: {vllm_base_url}")
    except Exception as e:
        raise RuntimeError(
            f"vLLM not reachable at {vllm_base_url} (config={args.config}). "
            f"Override with --vllm-base-url. Root error: {e}"
        )

    logger.info("Using vLLM API for Qwen3-8B inference")
    generate_fn = create_vllm_generate_fn(config)

    logger.info("Creating AFlow executor (gptoss120b / 4o-mini)...")
    from src.aflow_executor import AFlowExecutor
    aflow_executor = AFlowExecutor(
        llm_config_path=config.get('aflow_config_path', 'config/aflow_llm.yaml'),
        llm_model_name=config.get('aflow_executor_model', 'gptoss120b'),
        timeout=int(config.get('execution_timeout', 600) or 600),
    )
    logger.info("✅ AFlow executor created")

    from src.interactive import create_prompt_builder
    prompt_builder = create_prompt_builder(compact=False)

    init_workflow_log(args.output)

    start_time = time.time()
    if args.vectorized:
        logger.info("🚀 Using VECTORIZED rollout mode (true parallel)")
        results, stats = run_vectorized_evaluation(
            config=config,
            data=data,
            generate_fn=generate_fn,
            aflow_executor=aflow_executor,
            prompt_builder=prompt_builder,
            num_workers=args.workers,
        )
    else:
        logger.info("Using problem-level parallel mode")
        results, stats = run_evaluation(
            config=config,
            data=data,
            generate_fn=generate_fn,
            aflow_executor=aflow_executor,
            prompt_builder=prompt_builder,
            num_workers=args.workers,
        )
    elapsed = time.time() - start_time

    print_results(stats, elapsed)

    if args.output:
        output_data = {
            'config': args.config,
            'checkpoint': args.checkpoint,
            'num_samples': len(data),
            'elapsed_seconds': elapsed,
            'stats': {k: dict(v) for k, v in stats.items()},
            'results': results,
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {args.output}")

        errors = [r for r in results if not r.get('is_correct', True)]
        if errors:
            errors_file = args.output.replace('results.json', 'errors.json')
            with open(errors_file, 'w') as f:
                json.dump(errors, f, indent=2, ensure_ascii=False)
            logger.info(f"Found {len(errors)} errors, saved to {errors_file}")
            print(f"\n{'='*60}")
            print(f"ERROR ANALYSIS: {len(errors)} incorrect answers")
            print('='*60)
            for e in errors:
                print(f"\nSample {e.get('idx')}: {e.get('problem', '')[:80]}...")
                print(f"  Model: {e.get('final_answer', 'N/A')[:50]}")
                print(f"  Gold:  {e.get('ground_truth', 'N/A')}")
                print(f"  DSL:   {e.get('final_dsl', 'N/A')}")

    close_workflow_log()


if __name__ == '__main__':
    main()
