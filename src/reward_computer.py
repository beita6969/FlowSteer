#!/usr/bin/env python3
"""
 - P0/P1/P2

:
P0-1: 5 (0/0.2/0.4/0.7/1.0)
P0-3:  + 
P0-4: 
P1-2: Judge
P2-1: LLM Judge max_tokens200800reasoningtokencontent
"""
import os

# os.environ.pop('http_proxy', None)
# os.environ.pop('https_proxy', None)
# os.environ.pop('HTTP_PROXY', None)
# os.environ.pop('HTTPS_PROXY', None)
# os.environ['no_proxy'] = 'localhost,127.0.0.1'

import sys
import re
import threading
import time
import json
import random
import multiprocessing
from multiprocessing import Process, Queue
from typing import Any, Dict, Optional, List, Tuple
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from .answer_extractor import AnswerExtractor
    from .judge_prompt_loader import JudgePromptLoader
    from .unified_evaluator import UnifiedEvaluator
except ImportError:
    from answer_extractor import AnswerExtractor
    from judge_prompt_loader import JudgePromptLoader
    from unified_evaluator import UnifiedEvaluator

_unified_evaluator = None
def _get_unified_evaluator():
    global _unified_evaluator
    if _unified_evaluator is None:
        _unified_evaluator = UnifiedEvaluator()
    return _unified_evaluator


class RewardComputer:
    """
    P0/P1

    :
    1. 5 (0/0.2/0.4/0.7/1.0) - 
    2.  - +
    3.  - Code
    4.  - boxed//
    5. Judge - 
    6. QAF1 - 
    """

    def __init__(
        self,
        reward_weights: Optional[Dict[str, float]] = None,
        use_answer_extractor: bool = True,
        use_llm_judge: bool = False,
        llm_config: Optional[Dict] = None,
        debug_logging: bool = False
    ):
        """
        Args:
            reward_weights: 
            use_answer_extractor: 
            use_llm_judge: LLM Judge
            llm_config: LLMLLM Judge
            debug_logging: 
        """
        self.reward_weights = reward_weights or {
            "correctness": 1.0
        }

        self.debug_logging = debug_logging

        self.use_answer_extractor = use_answer_extractor
        if use_answer_extractor:
            self.extractor = AnswerExtractor(use_llm_fallback=False)
        else:
            self.extractor = None

        self.use_llm_judge = use_llm_judge
        self.llm_judge_client = None
        self.judge_prompt_loader = None
        if use_llm_judge:
            self._init_llm_judge_client(llm_config)
            try:
                self.judge_prompt_loader = JudgePromptLoader()
                stats = self.judge_prompt_loader.get_stats()
                print(f"  ✅ Judge Prompt")
                print(f"      {stats['total_datasets']} ")
                print(f"     : {', '.join(stats['enabled_datasets'][:5])}...")
            except Exception as e:
                print(f"  ⚠️  Judge Prompt: {e}")
                print(f"     Prompt")
                self.judge_prompt_loader = None

        print(f"✅ 10")
        print(f"  : 5 [0, 0.2, 0.4, 0.7, 1.0] (P0)")
        print(f"  : {'' if use_answer_extractor else ''}")
        print(f"  LLM Judge: {' (GPT OSS 120B @ port 8002)' if use_llm_judge else ''}")
        print(f"  : {'' if debug_logging else ''}")
        print(f"  :  (P0)")

        self.judge_log_dir = Path("logs/judge_samples")
        self.judge_log_dir.mkdir(parents=True, exist_ok=True)
        self.judge_log_file = self.judge_log_dir / f"judge_samples_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"

        self.eval_stats = {
            'total_evaluations': 0,
            'llm_judge_success': 0,
            'llm_judge_parse_failures': 0,
            'llm_judge_api_failures': 0,
            'correct_predictions': 0,
            'incorrect_predictions': 0
        }

    def _init_llm_judge_client(self, llm_config: Optional[Dict]):
        """LLM JudgeOpenAI API"""
        try:
            from openai import OpenAI

            default_config = {
                "base_url": "https://api.openai.com/v1",
                "api_key": os.environ.get('OPENAI_API_KEY', 'sk-dummy'),
                "model_name": "gpt-4o-mini"
            }

            config = llm_config or default_config

            self.llm_judge_client = OpenAI(
                base_url=config.get("base_url", default_config["base_url"]),
                api_key=config.get("api_key", default_config["api_key"])
            )
            self.llm_judge_model = config.get("model_name", default_config["model_name"])

            print(f"  ✅ LLM Judge")
            print(f"     : {self.llm_judge_model}")
            print(f"     URL: {config.get('base_url', default_config['base_url'])}")
        except Exception as e:
            print(f"  ⚠️  LLM Judge: {e}")
            self.use_llm_judge = False
            self.llm_judge_client = None

    def _llm_judge_compare(
        self,
        problem: str,
        prediction: str,
        ground_truth: str,
        problem_type: str,
        source: Optional[str] = None
    ) -> bool:
        """
        LLM JudgePrompt

        Args:
            problem: 
            prediction: 
            ground_truth: Ground truth
            problem_type: 
            source: 'gsm8k', 'math', 'hotpotqa'

        Returns:
            bool: TrueFalse
        """
        self.eval_stats['total_evaluations'] += 1

        if not self.llm_judge_client:
            if self.debug_logging:
                print("⚠️  LLM Judge")
            self.eval_stats['llm_judge_api_failures'] += 1
            return False

        if self.judge_prompt_loader:
            query_prompt_template = self.judge_prompt_loader.get_judge_prompt(
                source=source,
                problem_type=problem_type
            )
            query_prompt = query_prompt_template.replace('{{problem}}', problem)
            query_prompt = query_prompt.replace('{{prediction}}', prediction)
            query_prompt = query_prompt.replace('{{ground_truth}}', ground_truth)
            if self.debug_logging:
                print(f"  📋 Prompt: source={source}")
        else:
            query_prompt = self._get_legacy_prompt(problem, prediction, ground_truth)
            if self.debug_logging:
                print(f"  📋 Prompt (Fallback)")


        try:
            for attempt in range(2):
                response = self.llm_judge_client.chat.completions.create(
                    model=self.llm_judge_model,
                    messages=[
                        {"role": "system", "content": "You are a precise answer equivalence evaluator."},
                        {"role": "user", "content": query_prompt}
                    ],
                    temperature=0.0,
                    max_tokens=800
                )

                content = response.choices[0].message.content
                if content is None:
                    if attempt == 0:
                        if self.debug_logging:
                            print(f"⚠️  LLM Judge...")
                        self.eval_stats['llm_judge_api_failures'] += 1
                        continue
                    else:
                        if self.debug_logging:
                            print(f"⚠️  LLM JudgefallbackFalse")
                        self.eval_stats['llm_judge_api_failures'] += 1
                        return False

                result_text = content.strip()
                break

            import re
            # 1. <true_false>True</true_false>
            # 2. <true_false>: True
            # 3. **true_false**: True
            # 4. true_false: True

            true_false_match = re.search(
                r'<true_false>\s*(True|False)\s*</true_false>',
                result_text,
                re.IGNORECASE
            )

            if not true_false_match:
                true_false_match = re.search(
                    r'<true_false>\s*:\s*(True|False)',
                    result_text,
                    re.IGNORECASE
                )

            if not true_false_match:
                true_false_match = re.search(
                    r'\*\*true_false\*\*\s*:?\s*(True|False)',
                    result_text,
                    re.IGNORECASE
                )

            if not true_false_match:
                true_false_match = re.search(
                    r'true_false\s*:?\s*(True|False)',
                    result_text,
                    re.IGNORECASE
                )

            if not true_false_match:
                last_200_chars = result_text[-200:]
                true_false_match = re.search(
                    r'\b(True|False)\b',
                    last_200_chars,
                    re.IGNORECASE
                )

            if true_false_match:
                verdict = true_false_match.group(1).lower() == "true"
                self.eval_stats['llm_judge_success'] += 1

                if verdict:
                    self.eval_stats['correct_predictions'] += 1
                else:
                    self.eval_stats['incorrect_predictions'] += 1

                if self.debug_logging:
                    import random
                    if random.random() < 0.2:
                        print(f"\n🤖 LLM Judge ({problem_type}):")
                        print(f"  : {problem[:60]}...")
                        print(f"  : {str(prediction)[:60]}...")
                        print(f"  : {str(ground_truth)[:60]}...")
                        print(f"  : {verdict}")
                        print(f"  LLM: {result_text[:150]}...")

                return verdict
            else:
                self.eval_stats['llm_judge_parse_failures'] += 1
                if self.debug_logging:
                    print(f"⚠️  LLM Judge5")
                    print(f"  : {result_text}")
                    print(f"  : {problem[:100]}")
                    print(f"  : {str(prediction)[:100]}")
                    print(f"  : {str(ground_truth)[:100]}")
                return False

        except Exception as e:
            self.eval_stats['llm_judge_api_failures'] += 1
            if self.debug_logging:
                print(f"⚠️  LLM Judge: {e}")
                import traceback
                traceback.print_exc()
            return False

    def _get_legacy_prompt(self, problem: str, prediction: str, ground_truth: str) -> str:
        """Prompt"""
        return f"""You are a precise mathematical and logical equivalence evaluator. Your task is to determine if the Model Response contains an answer equivalent to the Ground Truth.

**Step 1: Extract the Final Answer**
From the Model Response, extract ONLY the final answer, ignoring all reasoning steps, explanations, and intermediate calculations.

Look for answers in these formats (in order of priority):
1. Inside `\\boxed{{...}}` LaTeX notation
2. After phrases like "The answer is", "Therefore", "So", "Thus", "Final answer:"
3. In `<answer>...</answer>` tags
4. The last number, expression, or entity mentioned

**Step 2: Extract from Ground Truth**
Similarly extract the final answer from Ground Truth, which may contain:
- Step-by-step solutions (extract only the final result)
- Multiple numbers (take the last/final one)
- Explanatory text (ignore and find the answer)

**Step 3: Normalize Both Answers**
Before comparing, normalize both answers:
- **Numbers:** Convert to same format (0.5 == 1/2 == 50%)
- **Units/Currency:** Ignore ($30 == 30, 10 meters == 10)
- **Formatting:** Ignore spaces, case, punctuation
- **LaTeX:** Interpret mathematical meaning (\\frac{{1}}{{2}} == 0.5)

**Step 4: Compare Equivalence**
Answers are equivalent if:
- **Math:** Numerically/algebraically equal (even if different forms)
- **Text:** Same entity/concept (ignore synonyms, case)
- **Precision:** Allow reasonable rounding (42.0 == 42)

**Examples of CORRECT equivalence:**
- "1/2" == "0.5" ✓
- "$30" == "30" ✓
- "\\boxed{{42}}" == "42" ✓
- "x^2+2x+1" == "(x+1)^2" ✓ (algebraically equivalent)
- "10 meters" == "10" ✓

**Examples of INCORRECT equivalence:**
- "John Smith" == "Jane Doe" ✗ (different entities)
- "42" == "43" ✗ (different numbers)
- "Paris" == "London" ✗ (different locations)

**Inputs:**
Question: {problem}
Model Response: {prediction}
Ground Truth: {ground_truth}

**Required Output Format:**
<analysis>Your reasoning in 1-2 sentences</analysis>
<true_false>True or False</true_false>

Be LENIENT with formatting differences but STRICT with factual/numerical differences.
"""

    def compute_reward(
        self,
        problem: str,
        prediction: Any,
        ground_truth: Any,
        problem_type: str = "math",
        metadata: Optional[Dict] = None,
        test: Optional[str] = None,
        entry_point: Optional[str] = None,
        source: Optional[str] = None
    ) -> float:
        """
         - P0: 5

        :
        - 1.0: 
        - 0.7:  (<5%, >80%)
        - 0.4:  (, >50%)
        - 0.2:  (, >20%)
        - 0.0: 

        Args:
            source: 'gsm8k', 'math', 'hotpotqa'- Judge Prompt

        Returns:
            reward: 0.0 / 0.2 / 0.4 / 0.7 / 1.0
        """
        metadata = metadata or {}

        if isinstance(prediction, str):
            if prediction.startswith('[REVIEW_OUTPUT]'):
                print(f"  ⚠️ ReviewOp:  Review operator ")
                print(f"     prediction: {prediction}")
                if metadata is not None:
                    metadata['correctness_score'] = 0.2
                    metadata['used_llm_judge'] = False
                    metadata['is_correct'] = False
                    metadata['reward_level'] = '🟠 Review (0.2)'
                    metadata['review_output'] = True
                return 0.2

            if prediction.startswith('[VERIFY_OUTPUT]'):
                print(f"  ⚠️ VerifyOp:  Verify operator ")
                print(f"     prediction: {prediction}")
                if metadata is not None:
                    metadata['correctness_score'] = 0.2
                    metadata['used_llm_judge'] = False
                    metadata['is_correct'] = False
                    metadata['reward_level'] = '🟠 Verify (0.2)'
                    metadata['verify_output'] = True
                return 0.2

            if prediction.startswith('[TEST_PASSED]') or prediction.startswith('[TEST_FAILED]'):
                print(f"  ⚠️ VerifyOp:  Test operator ")
                print(f"     prediction: {prediction}")
                if metadata is not None:
                    metadata['correctness_score'] = 0.2
                    metadata['used_llm_judge'] = False
                    metadata['is_correct'] = False
                    metadata['reward_level'] = '🟠 Test (0.2)'
                    metadata['test_output'] = True
                return 0.2

        if self.debug_logging:
            print(f"\n📊  ({problem_type}, source={source}):")
            print(f"  : {str(problem)[:100]}...")
            print(f"  : {str(prediction)[:100]}...")
            print(f"  : {str(ground_truth)[:100]}...")

        if problem_type == "code":
            reward = self._compute_code_reward(problem, prediction, ground_truth, test, entry_point)
        elif problem_type == "math":
            reward = self._compute_math_reward(problem, prediction, ground_truth, source)
        elif problem_type == "qa":
            reward = self._compute_qa_reward(problem, prediction, ground_truth, source)
        else:
            reward = self._compute_general_reward(prediction, ground_truth)

        if reward >= 0.9:
            self.eval_stats['correct_predictions'] += 1
        else:
            self.eval_stats['incorrect_predictions'] += 1

        if metadata is not None:
            metadata['correctness_score'] = reward
            metadata['used_llm_judge'] = self.use_llm_judge
            metadata['is_correct'] = reward >= 0.9
            metadata['reward_level'] = self._get_reward_level(reward)

        if self.debug_logging:
            level = self._get_reward_level(reward)
            print(f"  : {level}")
            print(f"  : {reward:.2f}")

        return reward


    def compute_dsl_quality_reward(self, dsl_info: Dict) -> float:
        """

        DSL
        - 0.35: fallback
        - 0.20: operators (>=2)
        - 0.15:  (->)
        - 0.10: loop *  conditional ?
        - 0.10: operator (>=3operator)
        - 0.10:  ([])

        Returns:
            float: DSL [0.0, 1.0]
        """
        if not dsl_info:
            return 0.0

        score = 0.0

        if not dsl_info.get('is_fallback', True):
            score += 0.35

        num_ops = dsl_info.get('num_operators', 0)
        if num_ops >= 2:
            score += 0.20
        elif num_ops == 1 and not dsl_info.get('is_fallback', True):
            score += 0.10

        if dsl_info.get('has_chain', False):
            score += 0.15

        if dsl_info.get('has_loop', False) or dsl_info.get('has_conditional', False):
            score += 0.10

        unique_ops = dsl_info.get('unique_operators', [])
        if len(unique_ops) >= 3:
            score += 0.10
        elif len(unique_ops) == 2:
            score += 0.05

        if dsl_info.get('has_parallel', False):
            score += 0.10

        return min(score, 1.0)

    def compute_reward_with_conditional_activation(
        self,
        problem: str,
        prediction: Any,
        ground_truth: Any,
        problem_type: str = "math",
        metadata: Optional[Dict] = None,
        test: Optional[str] = None,
        entry_point: Optional[str] = None,
        source: Optional[str] = None,
        dsl_info: Optional[Dict] = None
    ) -> Tuple[float, Dict]:
        """

        : R_total = base + R_dsl + 𝕀{R_dsl >= threshold} · R_correctness

        
        - base = -0.3: 
        - threshold = 0.5: DSL
        -  = 0.7: 

        
        1. fallbackworkflow
        2. DSL
        3. DSL + 

        Returns:
            (total_reward, reward_breakdown)
        """
        dsl_quality_reward = self.compute_dsl_quality_reward(dsl_info) if dsl_info else 0.0

        correctness_reward = self.compute_reward(
            problem=problem,
            prediction=prediction,
            ground_truth=ground_truth,
            problem_type=problem_type,
            metadata=metadata,
            test=test,
            entry_point=entry_point,
            source=source
        )

        base_penalty = -0.3
        dsl_threshold = 0.5
        correctness_weight = 0.7

        total_reward = base_penalty + dsl_quality_reward

        activation = 1.0 if dsl_quality_reward >= dsl_threshold else 0.0
        total_reward += activation * correctness_weight * correctness_reward

        total_reward = max(min(total_reward, 1.0), -1.0)

        reward_breakdown = {
            'total_reward': total_reward,
            'base_penalty': base_penalty,
            'dsl_quality_reward': dsl_quality_reward,
            'correctness_reward': correctness_reward,
            'activation': activation,
            'dsl_threshold': dsl_threshold,
            'is_activated': activation > 0,
            'dsl_info': dsl_info
        }

        if self.debug_logging:
            print(f"\n🔥 ConditionalReward:")
            print(f"  DSL: {dsl_quality_reward:.3f} (: {dsl_threshold})")
            print(f"  : {correctness_reward:.3f}")
            print(f"  : {'' if activation > 0 else ''}")
            print(f"  : {total_reward:.3f}")
            if dsl_info:
                print(f"  DSL: {dsl_info.get('dsl_text', 'N/A')[:50]}")
                print(f"  Fallback: {dsl_info.get('is_fallback', True)}")

        return total_reward, reward_breakdown

    def _get_reward_level(self, reward: float) -> str:
        """"""
        if reward >= 0.9:
            return "✅  (1.0)"
        elif reward >= 0.6:
            return "🟡  (0.7)"
        elif reward >= 0.35:
            return "🟠  (0.4)"
        elif reward >= 0.15:
            return "🔴  (0.2)"
        else:
            return "❌  (0.0)"

    def _is_correct(
        self,
        prediction: Any,
        ground_truth: Any,
        problem_type: str
    ) -> bool:
        """
         ()
        
        Returns:
            bool: True if correct, False otherwise
        """
        if prediction is None:
            return False

        if problem_type == "math":
            return self._is_math_correct(prediction, ground_truth)
        elif problem_type == "code":
            # Fallback for code if no test cases provided (should generally not happen if trained correctly)
            return False 
        elif problem_type == "qa":
            return self._is_qa_correct(prediction, ground_truth)
        else:
            return self._is_general_correct(prediction, ground_truth)


    def _compute_math_reward(self, problem: str, prediction: Any, ground_truth: Any, source: Optional[str]) -> float:
        """
        P0: Math5

        :
        - 1.0: 
        - 0.7:  (<5%)
        - 0.4:  (<50%)
        - 0.2:  (boxed)
        - 0.0: 
        """
        import re

        if prediction is None:
            return 0.0

        pred_str = str(prediction).strip()
        gt_str = str(ground_truth).strip()

        skip_code_detection = False
        llm_extracted = False

        needs_extraction = (
            len(pred_str) > 50 or
            '**' in pred_str or
            '```' in pred_str or
            '\\boxed' in pred_str or  # LaTeX boxed
            'Step' in pred_str or
            'Therefore' in pred_str or 'Thus' in pred_str or 'Hence' in pred_str or
            'Answer' in pred_str
        )

        if needs_extraction and self.use_llm_judge:
            if self.debug_logging:
                print(f"  🔧 AnswerExtract: LLM...")
            extracted = self._extract_answer_with_llm(pred_str, problem, "math")
            if extracted:
                if self.debug_logging:
                    print(f"  ✅ AnswerExtract: LLM: {extracted[:50]}...")
                pred_str = extracted
                skip_code_detection = True
                llm_extracted = True
            else:
                if self.debug_logging:
                    print(f"  🔄 AnswerExtract: LLMboxed...")
                boxed_answer = self._extract_boxed_robust(pred_str)
                invalid_boxed_prefixes = ['**Approach', '**Step', '**Solution', '**Analysis', '**Method', '**Let', 'Approach', 'Step']
                boxed_is_invalid = boxed_answer and any(boxed_answer.strip().startswith(p) for p in invalid_boxed_prefixes)

                if boxed_answer and not boxed_is_invalid and len(boxed_answer) < 100:
                    if self.debug_logging:
                        print(f"  ✅ AnswerExtract: boxed: {boxed_answer[:50]}...")
                    pred_str = boxed_answer
                    skip_code_detection = True
                else:
                    if self.debug_logging:
                        print(f"  ⚠️  AnswerExtract: ")
                    skip_code_detection = True

        if not skip_code_detection:
            strict_code_patterns = [
                r'\bimport\s+[a-zA-Z_][a-zA-Z0-9_]*',      # import module
                r'\bfrom\s+[a-zA-Z_][a-zA-Z0-9_.]*\s+import',  # from xxx import
                r'\bdef\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(',    # def func(
                r'\bclass\s+[a-zA-Z_][a-zA-Z0-9_]*\s*[:\(]',
                r'\bfor\s+[a-zA-Z_][a-zA-Z0-9_]*\s+in\s+',
                r'\bwhile\s+[a-zA-Z_][a-zA-Z0-9_]*\s*[:<>=!]',
                r'if\s+__name__\s*==',                      # if __name__ ==
                r'print\s*\([^)]+\)',                       # print(xxx)
            ]
            pred_lower = pred_str.lower()
            is_code = any(re.search(pattern, pred_lower) for pattern in strict_code_patterns)

            if is_code:
                if self.debug_logging:
                    print(f"  ⚠️  CodeDetect: (0.2)")
                return 0.2

        invalid_patterns = [
            r'^\s*\{\s*\}\s*$',
            r'^\s*\[\s*\]\s*$',
            r'^\s*$',
            r'^\s*None\s*$',         # None
            r'^\s*null\s*$',         # null
            r'^\s*undefined\s*$',    # undefined
        ]
        pred_stripped = pred_str.strip()
        is_invalid_answer = any(re.match(p, pred_stripped, re.IGNORECASE) for p in invalid_patterns)

        if is_invalid_answer:
            if self.debug_logging:
                print(f"  ⚠️  InvalidPattern:  '{pred_stripped[:20]}...', LLM Judge")
            return 0.0

        gt_needs_extraction = (
            len(gt_str) > 100 or
            'Let ' in gt_str or
            'We ' in gt_str or
            'There are' in gt_str or
            'The ' in gt_str[:20] or
            '<<' in gt_str or
            gt_str.count('.') > 2
        )

        if gt_needs_extraction and self.use_llm_judge:
            if self.debug_logging:
                print(f"  🔧 TruthExtract: ...")
            gt_extracted = self._extract_answer_with_llm(gt_str, problem, "math")
            if gt_extracted:
                if self.debug_logging:
                    print(f"  ✅ TruthExtract: : {gt_extracted[:50]}...")
                gt_str = gt_extracted
            else:
                if '####' in gt_str:
                    gt_str = gt_str.split('####')[-1].strip()
                    if self.debug_logging:
                        print(f"  ✅ TruthExtract: GSM8K: {gt_str[:50]}...")
                else:
                    boxed = self._extract_boxed_robust(gt_str)
                    if boxed and len(boxed) < 100:
                        gt_str = boxed
                        if self.debug_logging:
                            print(f"  ✅ TruthExtract: Boxed: {gt_str[:50]}...")

        if self.use_llm_judge:
            is_correct = self._llm_judge_compare(
                problem=problem,
                prediction=pred_str,
                ground_truth=gt_str,
                problem_type="math",
                source=source
            )
            if is_correct:
                return 1.0

        try:
            evaluator = _get_unified_evaluator()
            if evaluator.math_equal(pred_str, gt_str):
                return 1.0
        except Exception:
            pass

        pred_answer = self._extract_math_answer(pred_str)
        gt_answer = self._extract_math_answer(gt_str)

        if pred_answer is None:
            return 0.0

        if gt_answer is None:
            if gt_str.lower() in pred_str.lower():
                return 1.0
            return 0.0

        try:
            pred_num = self._parse_number_robust(pred_answer)
            gt_num = self._parse_number_robust(gt_answer)

            if pred_num is not None and gt_num is not None:
                import math

                if source == 'gsm8k':
                    tolerance = 1e-6
                else:
                    tolerance = 1e-3

                if math.isclose(pred_num, gt_num, abs_tol=tolerance):
                    return 1.0

                abs_error = abs(pred_num - gt_num)
                if abs_error <= tolerance:
                    return 1.0

                if abs(gt_num) > 1e-6:
                    rel_error = abs_error / abs(gt_num)
                    if rel_error < 0.01:
                        return 1.0
                    elif rel_error < 0.05:
                        return 0.7
                    elif rel_error < 0.10:
                        return 0.4
                    else:
                        return 0.0
                else:
                    if abs_error < 0.01:
                        return 0.7
                    elif abs_error < 0.05:
                        return 0.4
                    else:
                        return 0.0
        except:
            pass

        if pred_answer.lower() == gt_answer.lower():
            return 1.0

        return 0.2

    def _compute_code_reward(self, problem: Optional[str], prediction: Any, ground_truth: Any,
                             test: Optional[str], entry_point: Optional[str]) -> float:
        """
        P0: Code + 
        P6: HumanEval(problem=, prediction=)

        :
        - 1.0: 
        - 0.7: >80%
        - 0.4: >50%
        - 0.2: >20%
        - 0.0: 
        """
        if isinstance(test, list):
            test = '\n'.join(test)
            if self.debug_logging:
                print(f"  🔬 [CODE DEBUG] TestConvert: Converted test_cases list to string ({len(test)} chars)")

        if self.debug_logging:
            print(f"  🔬 [CODE DEBUG] prediction type: {type(prediction).__name__}")
            pred_str = str(prediction)
            print(f"  🔬 [CODE DEBUG] prediction[:300]: {pred_str[:300]}")
            print(f"  🔬 [CODE DEBUG] entry_point: {entry_point}")
            print(f"  🔬 [CODE DEBUG] test exists: {bool(test)}")

        if prediction is None:
            return 0.0

        solution = str(prediction).strip()
        if not solution:
            return 0.0

        if solution.startswith("{") and "'code'" in solution:
            try:
                import ast
                parsed = ast.literal_eval(solution)
                if isinstance(parsed, dict) and 'code' in parsed:
                    solution = parsed['code']
                    if self.debug_logging:
                        print(f"  🔬 [CODE DEBUG] Extracted code from dict string")
            except:
                pass

        # Sanitize solution (remove markdown blocks if any)
        if "```python" in solution:
            try:
                solution = solution.split("```python")[1].split("```")[0]
                if self.debug_logging:
                    print(f"  🔬 [CODE DEBUG] Removed ```python blocks")
            except:
                pass
        elif "```" in solution:
            try:
                solution = solution.split("```")[1].split("```")[0]
                if self.debug_logging:
                    print(f"  🔬 [CODE DEBUG] Removed ``` blocks")
            except:
                pass

        solution = self._sanitize_code(solution, entry_point)

        if entry_point and problem:
            has_def_in_solution = f"def {entry_point}" in solution
            has_def_in_problem = f"def {entry_point}" in str(problem)

            if not has_def_in_solution and has_def_in_problem:
                import re
                existing_func_match = re.search(r'^def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', solution, re.MULTILINE)

                if existing_func_match:
                    existing_func_name = existing_func_match.group(1)
                    if existing_func_name != entry_point:
                        solution = re.sub(
                            rf'\bdef\s+{re.escape(existing_func_name)}\s*\(',
                            f'def {entry_point}(',
                            solution,
                            count=1
                        )
                        if self.debug_logging:
                            print(f"  🔬 [CODE DEBUG] P6-Bug2: Renamed '{existing_func_name}' -> '{entry_point}' (merge)")
                else:
                    problem_str = str(problem)
                    signature_match = re.search(rf'(def\s+{re.escape(entry_point)}\s*\([^)]*\)\s*(?:->\s*[^:]+)?\s*:)', problem_str)
                    if signature_match:
                        func_signature = signature_match.group(1)
                        body_lines = solution.split('\n')
                        indented_body = []
                        for line in body_lines:
                            if line.strip():
                                if not line.startswith('    ') and not line.startswith('\t'):
                                    indented_body.append('    ' + line)
                                else:
                                    indented_body.append(line)
                            else:
                                indented_body.append(line)
                        solution = func_signature + '\n' + '\n'.join(indented_body)
                        if self.debug_logging:
                            print(f"  🔬 [CODE DEBUG] P6: Merged function signature from problem")
                            print(f"  🔬 [CODE DEBUG] P6: merged solution[:200]: {solution[:200]}")

        if self.debug_logging:
            print(f"  🔬 [CODE DEBUG] cleaned solution[:300]: {solution[:300]}")
            if entry_point:
                if f"def {entry_point}" in solution:
                    print(f"  🔬 [CODE DEBUG] ✅ entry_point '{entry_point}' found in solution")
                else:
                    print(f"  🔬 [CODE DEBUG] ❌ entry_point '{entry_point}' NOT found in solution")

        if entry_point and f"def {entry_point}" not in solution:
            import re
            defined_funcs = re.findall(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', solution)
            if defined_funcs:
                actual_func = None
                for func in defined_funcs:
                    if not func.startswith('_'):
                        actual_func = func
                        break
                if not actual_func:
                    actual_func = defined_funcs[0]

                if actual_func != entry_point:
                    old_def = f"def {actual_func}"
                    new_def = f"def {entry_point}"
                    solution = solution.replace(old_def, new_def, 1)
                    solution = re.sub(rf'\b{re.escape(actual_func)}\s*\(', f'{entry_point}(', solution)
                    if self.debug_logging:
                        print(f"  🔬 [CODE DEBUG] AnswerExtract: Renamed function '{actual_func}' -> '{entry_point}'")

        if not entry_point and test:
            import re
            match = re.search(r'assert\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', test)
            if not match:
                match = re.search(r'=\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', test)
            if not match:
                match = re.search(r'self\.assert\w+\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', test)
            if match:
                entry_point = match.group(1)
                if self.debug_logging:
                    print(f"  🔬 [CODE DEBUG] EntryPoint: Extracted entry_point from test_cases: {entry_point}")

        if not test or not entry_point:
            if self.use_llm_judge and ground_truth:
                is_equivalent = self._llm_judge_compare(
                    problem=str(problem) if problem else "",
                    prediction=solution,
                    ground_truth=str(ground_truth),
                    problem_type="code",
                    source="code_llm_judge"
                )
                if is_equivalent is True:
                    try:
                        compile(solution, '<string>', 'exec')
                        return 1.0
                    except:
                        return 0.4
                elif is_equivalent is False:
                    try:
                        compile(solution, '<string>', 'exec')
                        return 0.2
                    except:
                        return 0.0

            try:
                compile(solution, '<string>', 'exec')
                return 0.2
            except:
                return 0.0

        pass_rate = self._execute_code_isolated(solution, test, entry_point)

        if pass_rate >= 1.0:
            return 1.0
        elif pass_rate >= 0.8:
            return 0.7
        elif pass_rate >= 0.5:
            return 0.4
        elif pass_rate >= 0.2:
            return 0.2
        else:
            try:
                compile(solution, '<string>', 'exec')
                return 0.2
            except:
                return 0.0

    def _execute_code_isolated(self, solution: str, test: str, entry_point: str, timeout: int = 15) -> float:
        """
        P0: 
        P7: 15AFlow (10)

        Returns:
            pass_rate:  [0.0, 1.0]
        """
        def run_tests_in_process(solution: str, test: str, entry_point: str, result_queue: Queue):
            """"""
            try:
                global_dict = {
                    "math": __import__("math"),
                    "hashlib": __import__("hashlib"),
                    "re": __import__("re"),
                    "sys": __import__("sys"),
                    "List": List,
                    "Dict": Dict,
                    "Tuple": Tuple,
                    "Optional": Optional,
                    "Any": Any,
                }

                HUMANEVAL_HELPERS = {
                    'decode_cyclic': '''
def encode_cyclic(s: str):
    groups = [s[(3 * i):min((3 * i + 3), len(s))] for i in range((len(s) + 2) // 3)]
    groups = [(group[1:] + group[0]) if len(group) == 3 else group for group in groups]
    return "".join(groups)
''',
                    'decode_shift': '''
def encode_shift(s: str):
    return "".join([chr(((ord(ch) + 5 - ord("a")) % 26) + ord("a")) for ch in s])
''',
                    'find_zero': '''
def poly(xs: list, x: float):
    return sum([coeff * x ** i for i, coeff in enumerate(xs)])
'''
                }

                if entry_point in HUMANEVAL_HELPERS:
                    helper_code = HUMANEVAL_HELPERS[entry_point]
                    exec(helper_code, global_dict)

                exec(solution, global_dict)

                if entry_point not in global_dict:
                    defined_funcs = [name for name in global_dict
                                    if callable(global_dict.get(name))
                                    and not name.startswith('_')
                                    and name not in ('math', 'hashlib', 're', 'sys', 'List', 'Dict', 'Tuple', 'Optional', 'Any')]

                    matched_func = None

                    entry_lower = entry_point.lower()
                    for func_name in defined_funcs:
                        if func_name.lower() == entry_lower:
                            matched_func = func_name
                            break

                    if not matched_func and defined_funcs:
                        def edit_distance(s1, s2):
                            """"""
                            s1, s2 = s1.lower(), s2.lower()
                            if len(s1) < len(s2):
                                s1, s2 = s2, s1
                            if len(s2) == 0:
                                return len(s1)
                            prev_row = range(len(s2) + 1)
                            for i, c1 in enumerate(s1):
                                curr_row = [i + 1]
                                for j, c2 in enumerate(s2):
                                    insertions = prev_row[j + 1] + 1
                                    deletions = curr_row[j] + 1
                                    substitutions = prev_row[j] + (c1 != c2)
                                    curr_row.append(min(insertions, deletions, substitutions))
                                prev_row = curr_row
                            return prev_row[-1]

                        min_dist = float('inf')
                        for func_name in defined_funcs:
                            dist = edit_distance(entry_point, func_name)
                            if dist < min_dist and dist <= 3:
                                min_dist = dist
                                matched_func = func_name

                    if matched_func:
                        global_dict[entry_point] = global_dict[matched_func]
                    else:
                        result_queue.put({'pass_rate': 0.0, 'error': f'entry_point {entry_point} not found, defined: {defined_funcs}'})
                        return

                try:
                    exec(test, global_dict)

                    if "check" in global_dict:
                        check_func = global_dict["check"]
                        check_func(global_dict[entry_point])

                    result_queue.put({'pass_rate': 1.0, 'error': None})

                except AssertionError as e:
                    result_queue.put({'pass_rate': 0.5, 'error': f'AssertionError: {e}'})

                except Exception as e:
                    result_queue.put({'pass_rate': 0.0, 'error': f'{type(e).__name__}: {e}'})

            except SyntaxError as e:
                result_queue.put({'pass_rate': 0.0, 'error': f'SyntaxError: {e}'})
            except Exception as e:
                result_queue.put({'pass_rate': 0.0, 'error': f'{type(e).__name__}: {e}'})

        result_queue = multiprocessing.Queue()

        process = multiprocessing.Process(
            target=run_tests_in_process,
            args=(solution, test, entry_point, result_queue)
        )

        try:
            process.start()
            process.join(timeout=timeout)

            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
                if process.is_alive():
                    process.kill()
                if self.debug_logging:
                    print(f"  ⏱️  ({timeout}s)")
                return 0.2

            if not result_queue.empty():
                result = result_queue.get_nowait()
                if self.debug_logging and result.get('error'):
                    print(f"  🔧 : {result.get('error', 'unknown')[:50]}")
                return result.get('pass_rate', 0.0)
            else:
                return 0.0

        except Exception as e:
            if self.debug_logging:
                print(f"  ⚠️ : {e}")
            return 0.0
        finally:
            if process.is_alive():
                process.terminate()

    def _sanitize_code(self, code: str, entry_point: Optional[str] = None) -> str:
        """
        P7: AFlow scripts/utils/sanitize.py

        :
        1. 
        2. AST
        3. entry_point

        Args:
            code: 
            entry_point: 

        Returns:
            
        """
        import ast

        if not code or not code.strip():
            return code

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code

        imports = []
        definitions = []  # (name, code, dependencies)

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.append(ast.unparse(node))
            elif isinstance(node, ast.FunctionDef):
                deps = self._get_dependencies(node)
                definitions.append((node.name, ast.unparse(node), deps))
            elif isinstance(node, ast.ClassDef):
                deps = self._get_dependencies(node)
                definitions.append((node.name, ast.unparse(node), deps))
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        deps = self._get_dependencies(node)
                        definitions.append((target.id, ast.unparse(node), deps))

        if not entry_point:
            return code

        entry_exists = any(name == entry_point for name, _, _ in definitions)
        if not entry_exists:
            return code

        needed = self._find_reachable(entry_point, definitions)

        result_parts = imports[:]
        for name, code_str, _ in definitions:
            if name in needed:
                result_parts.append(code_str)

        return '\n'.join(result_parts)

    def _get_dependencies(self, node: 'ast.AST') -> set:
        """AST"""
        import ast
        deps = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                deps.add(child.id)
        return deps

    def _find_reachable(self, entry_point: str, definitions: list) -> set:
        """entry_point"""
        dep_map = {name: deps for name, _, deps in definitions}

        visited = set()
        queue = [entry_point]

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            if current in dep_map:
                for dep in dep_map[current]:
                    if dep not in visited and dep in dep_map:
                        queue.append(dep)

        return visited

    def _compute_qa_reward(self, problem: str, prediction: Any, ground_truth: Any, source: Optional[str]) -> float:
        """
        P1: QA - SQuAD/TriviaQA

         (SQuAD):
        1. Exact Match (EM): 
        2. F1 Score: TokenF1
        3. : 
        4. : 
        5. LLM Judge: 

        :
        - 1.0: EM=1  F1>=0.8    LLM
        - 0.7: F1>=0.5  
        - 0.4: F1>=0.3
        - 0.2: F1>=0.1 ()
        - 0.0: 
        """
        import re

        if prediction is None:
            return 0.0

        pred_str = str(prediction).strip()
        gt_str = str(ground_truth).strip()

        if not pred_str:
            return 0.0

        needs_extraction = (
            len(pred_str) > 50 or
            '**' in pred_str or
            'Step' in pred_str or
            'Answer' in pred_str or
            'Explanation' in pred_str or
            'Therefore' in pred_str or 'Thus' in pred_str
        )

        if needs_extraction and self.use_llm_judge:
            if self.debug_logging:
                print(f"  🔧 AnswerExtract: QALLM...")
            extracted = self._extract_answer_with_llm(pred_str, problem, "qa")
            if extracted:
                if self.debug_logging:
                    print(f"  ✅ AnswerExtract: LLM: {extracted[:50]}...")
                pred_str = extracted
            else:
                if self.debug_logging:
                    print(f"  🔄 AnswerExtract: LLMAnswer...")
                answer_match = re.search(r'\*\*Answer[:\*]*\s*[:\-–—]*\s*(.+?)(?:\n\n|\*\*|$)', pred_str, re.IGNORECASE | re.DOTALL)
                if answer_match:
                    local_extracted = answer_match.group(1).strip()
                    local_extracted = re.sub(r'^[\*\#\-–—:]+|[\*\#\-–—:]+$', '', local_extracted).strip()
                    if local_extracted and len(local_extracted) < 300:
                        if self.debug_logging:
                            print(f"  ✅ AnswerExtract: : {local_extracted[:50]}...")
                        pred_str = local_extracted

        if self.use_llm_judge:
            is_correct = self._llm_judge_compare(
                problem=problem,
                prediction=pred_str,
                ground_truth=gt_str,
                problem_type="qa",
                source=source
            )
            if is_correct:
                return 1.0

        pred_normalized = self._normalize_answer_squad(pred_str)
        gt_normalized = self._normalize_answer_squad(gt_str)

        # 3. Exact Match (EM)
        if pred_normalized == gt_normalized:
            return 1.0

        if self._check_numeric_equivalence(pred_str, gt_str):
            return 1.0

        if self._check_containment(pred_normalized, gt_normalized):
            return 0.7

        f1 = self._compute_f1_score_squad(pred_normalized, gt_normalized)

        if f1 >= 0.8:
            return 1.0
        elif f1 >= 0.5:
            return 0.7
        elif f1 >= 0.3:
            return 0.4
        elif f1 >= 0.1:
            return 0.2
        else:
            return 0.0

    def _normalize_answer_squad(self, text: str) -> str:
        """
        SQuAD
        : https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py
        """
        import string
        import re

        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(text))))

    def _check_numeric_equivalence(self, pred: str, gt: str) -> bool:
        """
        

        :
        - "4" vs "four" vs "4 cylinders"
        - "1990" vs "in 1990" vs "the year 1990"
        - "$100" vs "100 dollars" vs "100"
        """
        number_words = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
            'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
            'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
            'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
            'eighteen': 18, 'nineteen': 19, 'twenty': 20, 'thirty': 30,
            'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
            'eighty': 80, 'ninety': 90, 'hundred': 100, 'thousand': 1000,
            'million': 1000000, 'billion': 1000000000
        }

        def extract_number(text: str) -> Optional[float]:
            text_lower = text.lower().strip()

            num_match = re.search(r'-?\d+\.?\d*', text_lower)
            if num_match:
                try:
                    return float(num_match.group())
                except:
                    pass

            for word, num in number_words.items():
                if word in text_lower:
                    return float(num)

            return None

        pred_num = extract_number(pred)
        gt_num = extract_number(gt)

        if pred_num is not None and gt_num is not None:
            if pred_num == gt_num:
                return True
            if gt_num != 0 and abs(pred_num - gt_num) / abs(gt_num) < 0.01:
                return True

        return False

    def _check_containment(self, pred: str, gt: str) -> bool:
        """
         ()

        1: gt (pred)
        2: gt (gtpred)
        3:  ( "watch"  "pocketwatch" )
        """
        if len(pred) < 2 or len(gt) < 2:
            return False

        if pred in gt or gt in pred:
            shorter = pred if len(pred) < len(gt) else gt
            longer = gt if len(pred) < len(gt) else pred

            if len(shorter) >= len(longer) * 0.3:
                return True

        pred_words = pred.split()
        gt_words = gt.split()

        for pw in pred_words:
            if len(pw) >= 3:
                for gw in gt_words:
                    if pw in gw and len(pw) >= len(gw) * 0.4:
                        return True
                    if gw in pw and len(gw) >= len(pw) * 0.4:
                        return True

        return False

    def _compute_f1_score_squad(self, pred: str, gt: str) -> float:
        """
        SQuADF1
        : https://rajpurkar.github.io/SQuAD-explorer/
        """
        from collections import Counter

        pred_tokens = pred.split()
        gt_tokens = gt.split()

        if len(gt_tokens) == 0:
            return 1.0 if len(pred_tokens) == 0 else 0.0
        if len(pred_tokens) == 0:
            return 0.0

        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            return 0.0

        precision = num_same / len(pred_tokens)
        recall = num_same / len(gt_tokens)
        f1 = 2 * precision * recall / (precision + recall)

        return f1

    def _normalize_answer(self, text: str) -> str:
        """"""
        import string
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = ' '.join(text.split())
        return text

    def _compute_f1_score(self, pred: str, gt: str) -> float:
        """P1: tokenF1Counterset"""
        from collections import Counter

        pred_tokens = Counter(pred.split())
        gt_tokens = Counter(gt.split())

        if sum(gt_tokens.values()) == 0:
            return 1.0 if sum(pred_tokens.values()) == 0 else 0.0

        if sum(pred_tokens.values()) == 0:
            return 0.0

        common = pred_tokens & gt_tokens
        num_same = sum(common.values())

        if num_same == 0:
            return 0.0

        precision = num_same / sum(pred_tokens.values())
        recall = num_same / sum(gt_tokens.values())
        f1 = 2 * precision * recall / (precision + recall)

        return f1

    def _compute_general_reward(self, prediction: Any, ground_truth: Any) -> float:
        """"""
        if prediction is None:
            return 0.0

        pred_str = str(prediction).strip().lower()
        gt_str = str(ground_truth).strip().lower()

        if pred_str == gt_str:
            return 1.0
        elif gt_str in pred_str:
            return 0.7
        elif self._compute_f1_score(pred_str, gt_str) > 0.5:
            return 0.4
        else:
            return 0.0

    def _extract_answer_with_llm(self, explanatory_text: str, problem: str, problem_type: str) -> Optional[str]:
        """

        Args:
            explanatory_text:  **Step 1...** 
            problem: 
            problem_type:  (math/qa)

        Returns:
            None
        """
        if not self.llm_judge_client:
            return None

        if problem_type == "math":
            extract_prompt = f"""Extract ONLY the final numeric answer from the following solution text.

Problem: {problem[:200]}

Solution Text:
{explanatory_text[:1000]}

IMPORTANT: Return ONLY the final answer (a number, fraction, or simple expression like "42", "5/6", "2x+1").
Do NOT include any explanations, steps, or formatting. Just the answer value.

<answer>YOUR_ANSWER_HERE</answer>"""
        else:  # qa
            extract_prompt = f"""Extract ONLY the direct answer from the following response.

Question: {problem[:200]}

Response:
{explanatory_text[:1000]}

IMPORTANT: Return ONLY the direct answer (a name, place, date, number, or short phrase).
Do NOT include any explanations or reasoning. Just the answer.

<answer>YOUR_ANSWER_HERE</answer>"""

        try:
            response = self.llm_judge_client.chat.completions.create(
                model=self.llm_judge_model,
                messages=[
                    {"role": "system", "content": "You are a precise answer extractor. Extract only the final answer."},
                    {"role": "user", "content": extract_prompt}
                ],
                temperature=0.0,
                max_tokens=100
            )

            content = response.choices[0].message.content
            if content is None:
                return None

            result = content.strip()

            import re
            answer_match = re.search(r'<answer>\s*(.+?)\s*</answer>', result, re.IGNORECASE | re.DOTALL)
            if answer_match:
                extracted = answer_match.group(1).strip()
                extracted = re.sub(r'^[\*\#]+|[\*\#]+$', '', extracted).strip()
                if extracted and len(extracted) < 200:
                    return extracted

            lines = [l.strip() for l in result.split('\n') if l.strip()]
            if lines:
                last_line = lines[-1]
                last_line = re.sub(r'^[\*\#]+|[\*\#]+$', '', last_line).strip()
                if last_line and len(last_line) < 200:
                    return last_line

            return None

        except Exception as e:
            if self.debug_logging:
                print(f"  ⚠️  ExtractFail : {e}")
            return None

    def _extract_math_answer(self, text: str) -> Optional[str]:
        """
        P0-4: 

        :
        - boxed: \\boxed{{a \\choose b}}
        - : 5/324
        - : 50%
        - : 1.5e-3
        """
        if not text:
            return None

        boxed = self._extract_boxed_robust(text)
        if boxed:
            code_leak_keywords = ['def ', 'return ', 'import ', 'class ', 'if __name__', 'async def ']
            if any(kw in boxed for kw in code_leak_keywords):
                pass
            elif not boxed.strip():
                pass
            elif boxed.startswith('Error:') or 'Traceback' in boxed or 'SyntaxError' in boxed:
                pass
            else:
                return boxed

        answer_patterns = [
            r'[:]+\s*([\d\./\-]+)',
            r'[Tt]he answer is[:\s]+([\d\./\-]+)',
            r'[Tt]herefore[,\s]+([\d\./\-]+)',
            r'[Ss]o[,\s]+([\d\./\-]+)',
            r'=\s*([\d\./\-]+)\s*$',
        ]

        for pattern in answer_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()

        numbers = self._extract_numbers(text)
        if numbers:
            return str(numbers[-1])

        if len(text) < 50:
            return text.strip()

        return None

    def _extract_boxed_robust(self, text: str) -> Optional[str]:
        """
        P0-4: boxed
        """
        pattern = r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}'
        matches = re.findall(pattern, text, re.DOTALL)

        def is_valid_boxed_content(content: str) -> bool:
            """AnswerExtract: boxed"""
            if not content:
                return False
            content_stripped = content.strip()
            invalid_prefixes = [
                '**Approach', '**Step', '**Solution', '**Analysis',
                '**Method', '**Let', '**Define', '**Given',
                'Approach', 'Step 1', 'Solution:', 'Let ',
            ]
            for prefix in invalid_prefixes:
                if content_stripped.startswith(prefix):
                    return False
            if len(content_stripped) > 200:
                if not any(tex in content for tex in ['\\frac', '\\sqrt', '\\sum', '\\int']):
                    return False
            return True

        if matches:
            for match in reversed(matches):
                if is_valid_boxed_content(match):
                    return match.strip()
            return matches[-1].strip()

        simple_match = re.search(r'\\boxed\{([^}]+)\}', text)
        if simple_match:
            content = simple_match.group(1).strip()
            if is_valid_boxed_content(content):
                return content
            return content

        return None

    def _normalize_float(self, num: float, precision: int = 9) -> float:
        """

        :
        1. 
        2. 0
        3. 
        """
        if num is None:
            return None

        import math
        if math.isnan(num) or math.isinf(num):
            return num

        rounded = round(num, precision)

        if abs(rounded - round(rounded)) < 1e-9:
            return float(round(rounded))

        return rounded

    def _parse_number_robust(self, text: str) -> Optional[float]:
        """
        P0-4: 

        :
        - : 5/324
        - : 50% -> 0.5
        - : 1.5e-3
        - : 1,234,567
        - : 8.200000000001 -> 8.2
        """
        if not text:
            return None

        text = text.strip()

        text = text.replace(',', '')

        if '%' in text:
            try:
                num_str = text.replace('%', '').strip()
                result = float(num_str) / 100.0
                return self._normalize_float(result)
            except:
                pass

        if '/' in text:
            try:
                parts = text.split('/')
                if len(parts) == 2:
                    result = float(parts[0].strip()) / float(parts[1].strip())
                    return self._normalize_float(result)
            except:
                pass

        try:
            result = float(text)
            return self._normalize_float(result)
        except:
            pass

        match = re.search(r'-?\d+\.?\d*(?:[eE][+-]?\d+)?', text)
        if match:
            try:
                result = float(match.group())
                return self._normalize_float(result)
            except:
                pass

        return None


    def _is_math_correct(self, prediction: str, ground_truth: str) -> bool:
        """

        :
        - 
        - 
        -  5/324 vs 0.0154...
        - LaTeX \\frac{1}{2} vs 0.5
        - sympy
        """
        try:
            evaluator = _get_unified_evaluator()

            return evaluator.math_equal(prediction, ground_truth)

        except Exception as e:
            try:
                pred_str = str(prediction).strip()
                gt_str = str(ground_truth).strip()

                if pred_str == gt_str:
                    return True

                def parse_number(s: str) -> float:
                    if '/' in s:
                        parts = s.split('/')
                        return float(parts[0]) / float(parts[1])
                    return float(s)

                pred_num = parse_number(pred_str)
                gt_num = parse_number(gt_str)
                return abs(pred_num - gt_num) < 1e-4
            except:
                return False

    class TimeoutError(Exception):
        pass

    def run_with_timeout(self, func, args, timeout):
        result = []
        stop_event = threading.Event()

        def target():
            try:
                result.append(func(*args))
            except Exception as e:
                result.append(e)
            finally:
                stop_event.set()

        thread = threading.Thread(target=target)
        thread.start()
        is_timeout = not stop_event.wait(timeout)

        if is_timeout:
            raise self.TimeoutError("Function execution timed out")

        if not result:
            return None
        if isinstance(result[0], Exception):
            raise result[0]
        return result[0]

    def _check_code_solution(self, solution: str, test: str, entry_point: str) -> bool:
        """
        Use execution to check if the code solution is correct.
        Inspired by AFlow's evaluation mechanism.
        """
        if not solution or not test or not entry_point:
            return False

        # Sanitize solution (remove markdown blocks if any)
        if "```python" in solution:
            solution = solution.split("```python")[1].split("```")[0]
        elif "```" in solution:
            solution = solution.split("```")[1].split("```")[0]
        
        try:
            global_dict = {
                "math": __import__("math"),
                "hashlib": __import__("hashlib"),
                "re": __import__("re"),
                "List": List,
                "Dict": Dict,
                "Tuple": Tuple,
                "Optional": Optional,
                "Any": Any,
            }

            # Execute the solution code
            exec(solution, global_dict)

            if entry_point not in global_dict:
                # Try to find if there is a 'solve' function or similar if entry_point is missing
                # But for HumanEval/MBPP, entry_point is strict.
                # If it's a full script, maybe we shouldn't fail immediately, but for now strict is better.
                return False

            # Execute the test code
            # The test code usually contains a 'check' function or assertions
            exec(test, global_dict)

            # Check if 'check' function exists (common in HumanEval)
            if "check" in global_dict:
                check = global_dict["check"]
                try:
                    # Run the check function with timeout
                    self.run_with_timeout(check, (global_dict[entry_point],), 5) # 5 seconds timeout
                    return True
                except Exception as e:
                    if self.debug_logging:
                        print(f"Code execution check failed: {e}")
                    return False
            else:
                # If no check function, assume the test code runs assertions directly
                # If exec(test) didn't raise exception, it might be correct
                return True

        except Exception as e:
            if self.debug_logging:
                print(f"Code execution error: {e}")
            return False

    def _is_code_correct(self, prediction: str, ground_truth: str, test: Optional[str] = None, entry_point: Optional[str] = None) -> bool:
        """"""
        import re

        prediction = str(prediction)
        if '```' in prediction:
            code_blocks = re.findall(r'```(?:python)?\s*\n?([^`]+)```', prediction)
            if code_blocks:
                prediction = code_blocks[-1].strip()
            else:
                prediction = re.sub(r'^```(?:python)?\n?', '', prediction)
                prediction = re.sub(r'```$', '', prediction)
                prediction = prediction.strip()

        # Prioritize execution-based checking if test cases are available
        if test and entry_point:
            return self._check_code_solution(prediction, test, entry_point)

        # Fallback to string matching if execution is not possible
        try:
            pred_str = str(prediction).strip()
            gt_str = str(ground_truth).strip()

            if not pred_str:
                return False

            if pred_str.lower() == gt_str.lower():
                return True

            if gt_str.lower() in pred_str.lower():
                return True

            return False

        except Exception:
            return False

    def _is_qa_correct(self, prediction: str, ground_truth: str) -> bool:
        """QA"""
        try:
            pred_str = str(prediction).strip().lower()
            gt_str = str(ground_truth).strip().lower()

            if pred_str == gt_str:
                return True

            if gt_str in pred_str or pred_str in gt_str:
                return True

            from collections import Counter
            pred_tokens = Counter(pred_str.split())
            gt_tokens = Counter(gt_str.split())

            if sum(gt_tokens.values()) == 0:
                return False

            common = pred_tokens & gt_tokens
            overlap_ratio = sum(common.values()) / sum(gt_tokens.values())
            return overlap_ratio > 0.8

        except Exception:
            return False

    def _is_general_correct(self, prediction: str, ground_truth: str) -> bool:
        """"""
        try:
            pred_str = str(prediction).strip().lower()
            gt_str = str(ground_truth).strip().lower()

            return pred_str == gt_str or gt_str in pred_str

        except Exception:
            return False

    def _compute_correctness_reward(
        self,
        prediction: Any,
        ground_truth: Any,
        problem_type: str
    ) -> float:
        """
        
        
        Returns:
            reward: 1.0 or 0.0
        """
        # This function is kept for compatibility but compute_reward should be used
        # We map the binary 0/1 back to whatever range was expected if needed, 
        # but here we simply return 1.0 or 0.0 as requested.
        
        if prediction is None:
            return 0.0

        is_correct = False
        if problem_type == "math":
            is_correct = self._is_math_correct(prediction, ground_truth)
        elif problem_type == "code":
            # Without test cases here, we fall back to string matching which is weak
            is_correct = self._is_code_correct(prediction, ground_truth)
        elif problem_type == "qa":
            is_correct = self._is_qa_correct(prediction, ground_truth)
        else:
            is_correct = self._is_general_correct(prediction, ground_truth)
            
        return 1.0 if is_correct else 0.0

    def _extract_boxed(self, text: str) -> Optional[str]:
        """\\boxed{}(ROLL)"""
        match = re.search(r'\\boxed\{([^}]+)\}', text)
        if match:
            return match.group(1).strip()
        return None

    def _extract_numbers(self, text: str) -> list:
        """( + )"""
        numbers = []

        # Method 1: Numeric extraction (existing)
        pattern = r'-?\d+\.?\d*(?:[eE][+-]?\d+)?'
        matches = re.findall(pattern, text)
        for m in matches:
            if m:
                try:
                    numbers.append(float(m))
                except:
                    pass

        # Method 2: Word-to-number recognition (NEW - fixes ~15-20% QA errors)
        # Aligns with SQuAD/HotpotQA standards for text-based answers
        word_to_num = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
            'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
            'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
            'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
            'eighteen': 18, 'nineteen': 19, 'twenty': 20, 'thirty': 30,
            'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
            'eighty': 80, 'ninety': 90, 'hundred': 100, 'thousand': 1000
        }

        text_lower = text.lower()
        for word, num in word_to_num.items():
            if word in text_lower:
                numbers.append(float(num))

        return numbers

    def _extract_function_names(self, code: str) -> list:
        """"""
        pattern = r'def\s+(\w+)\s*\('
        matches = re.findall(pattern, code)
        return matches

    def _compute_efficiency_reward(self, cost: float) -> float:
        return 0.0

    def _compute_simplicity_reward(
        self,
        execution_time: float,
        num_operators: int = 1
    ) -> float:
        return 0.0

    def _compute_format_reward(self, response: str, problem_type: str) -> float:
        return 0.0

    def _compute_repetition_penalty(self, response: str, ngram_size: int = 3) -> float:
        return 0.0

    def print_eval_stats(self):
        """
        
        """
        stats = self.eval_stats
        total = stats['total_evaluations']

        if total == 0:
            print("\n📊 : ")
            return

        print(f"\n📊  (: {total} ):")
        print(f"  ✅ LLM Judge: {stats['llm_judge_success']} ({stats['llm_judge_success']/total*100:.1f}%)")
        print(f"  ⚠️  : {stats['llm_judge_parse_failures']} ({stats['llm_judge_parse_failures']/total*100:.1f}%)")
        print(f"  ❌ API: {stats['llm_judge_api_failures']} ({stats['llm_judge_api_failures']/total*100:.1f}%)")
        print(f"\n  :")
        print(f"    : {stats['correct_predictions']} ({stats['correct_predictions']/total*100:.1f}%)")
        print(f"    : {stats['incorrect_predictions']} ({stats['incorrect_predictions']/total*100:.1f}%)")

        judged = stats['correct_predictions'] + stats['incorrect_predictions']
        if judged > 0:
            accuracy = stats['correct_predictions'] / judged * 100
            print(f"\n  🎯 : {accuracy:.1f}% ({judged})")

    def reset_eval_stats(self):
        """"""
        self.eval_stats = {
            'total_evaluations': 0,
            'llm_judge_success': 0,
            'llm_judge_parse_failures': 0,
            'llm_judge_api_failures': 0,
            'correct_predictions': 0,
            'incorrect_predictions': 0
        }
        print("🔄 ")


def test_reward_computer():
    """"""
    print("\n" + "=" * 60)
    print("🧪 ")
    print("=" * 60)

    computer = RewardComputer()

    test_cases = [
        {
            "name": " - +",
            "problem": "What is 15 + 27?",
            "prediction": "<think>Let me calculate: 15 + 27 = 42</think><answer>\\boxed{42}</answer>",
            "ground_truth": "42",
            "problem_type": "math",
            "metadata": {"cost": 0.002, "execution_time": 3.5}
        },
        {
            "name": " - ",
            "problem": "Write a function to square a number",
            "prediction": "def square(x):\n    return x * x",
            "ground_truth": "def square(x):\n    return x * x",
            "problem_type": "code",
            "test": "check = lambda func: func(2) == 4",
            "entry_point": "square",
            "metadata": {"cost": 0.003, "execution_time": 5.0}
        }
    ]

    for case in test_cases:
        reward = computer.compute_reward(
            problem=case["problem"],
            prediction=case["prediction"],
            ground_truth=case["ground_truth"],
            problem_type=case["problem_type"],
            metadata=case["metadata"],
            test=case.get("test"),
            entry_point=case.get("entry_point")
        )

        print(f"\n📝 {case['name']}")
        print(f"  : {case['prediction'][:60]}...")
        print(f"  : {case['ground_truth']}")
        print(f"  : {reward:.2f}")


if __name__ == "__main__":
    test_reward_computer()
