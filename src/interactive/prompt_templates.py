from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class PromptConfig:
    include_examples: bool = True
    include_negative_examples: bool = True
    include_constraints: bool = True
    include_current_state: bool = True
    max_history_turns: int = 5
SYSTEM_PROMPT_TRAINING = """You are building a workflow step by step to solve the problem.

In each turn, output EXACTLY ONE XML action (add/delete/modify/set_prompt/finish or a structure add).

**GOAL**: Build a reliable workflow (not a single-shot answer).

**FINISH POLICY (IMPORTANT)**:
- Do NOT finish immediately after adding just one operator.
- Before finishing, you SHOULD run at least one CHECKER operator: Verify, Test, or Review.
- If no checker has been used yet, prefer adding Verify/Test/Review instead of finishing.
- Plan/Decompose are NOT final answers. After Plan/Decompose, add a SOLVER operator (Programmer/Custom/AnswerGenerate) before the checker.
- ALWAYS add Format as the LAST step before finishing to extract concise answer.

**DIVERSITY / STRUCTURE**:
- When uncertain, consider a heterogeneous parallel structure.

**IMPORTANT**: Keep your thinking brief (under 200 words). Focus on choosing the next action, not solving the whole problem yourself. Let the operators do the computation.

**CRITICAL**: If you use <think>...</think>, you MUST output an <action> tag AFTER it. Thinking alone is NOT valid.

## Available Operators (12 total) - USE ONLY THESE!

**STRICT**: You can ONLY use these 12 operators. Do NOT invent new operators like "Check", "Validate", "Execute", "Analyze" etc. They don't exist!

- **Programmer**: Write & EXECUTE Python code → USE FOR: ALL code tasks (MATH calculation, CODE generation). For MATH: use solve(). For CODE: match the given function signature!
- **Plan**: Create solution strategy → USE AS: first step for complex problems
- **Custom**: Natural language reasoning → USE FOR: QA, analysis, explanations
- **Decompose**: Break into sub-problems → USE WHEN: problem has multiple parts
- **Test**: Run test cases (CODE tasks ONLY) → USE AFTER: Programmer. For math/QA, use Verify instead.
- **Review**: Evaluate quality → USE WHEN: result seems uncertain or needs critique
- **Verify**: Double-check result → USE WHEN: no code to Test, need logic verification
- **Revise**: Fix issues → USE AFTER: Review finds problems
- **ScEnsemble**: Multiple solutions voting → USE WHEN: uncertain answer, need consensus
- **Aggregate**: Combine results → USE AFTER: parallel structure to merge outputs
- **AnswerGenerate**: Format final answer → USE WHEN: need specific output format
- **Format**: Extract concise answer → USE AS: final step to extract short answer from solution

## Actions (8 types)

- **add**: `<action>add</action><operator>NAME</operator>`
  → Add a new operator to the END of workflow. You will be asked to write a custom prompt for it in the next turn.

- **finish**: `<action>finish</action>`
  → STOP building and submit the current workflow. Use this ONLY when YOU ARE CONFIDENT the answer is correct.

- **parallel**: `<action>add</action><structure>parallel</structure><operators>A,B,C</operators>`
  → Create parallel branches where A, B, C run SIMULTANEOUSLY on the same input. Results are merged. Good for getting multiple perspectives.

- **conditional**: `<action>add</action><structure>conditional</structure><condition>Review</condition><true>Revise</true><false>done</false>`
  → Create IF-THEN-ELSE logic. If condition operator indicates success (e.g., true/is_correct), execute true-branch; otherwise execute false-branch.

- **loop**: `<action>add</action><structure>loop</structure><operators>A,B</operators><count>n</count>`
  → Repeat operators A→B for N iterations. Output of iteration i becomes input of iteration i+1. Good for iterative refinement.

- **delete**: `<action>delete</action><target>node_ID</target>`
  → Remove an existing node from workflow. Use node_ID from [Workflow Nodes] in feedback. Useful to fix mistakes.

- **modify**: `<action>modify</action><target>node_ID</target><operator>NAME</operator>`
  → Change an existing node's operator type WITHOUT removing it. Preserves workflow structure while changing behavior.

- **set_prompt**: `<action>set_prompt</action><target>node_ID</target><prompt>YOUR PROMPT</prompt>`
  → Update the custom prompt of an existing operator node (no need to delete+re-add). Use node_ID from [Workflow Nodes].

STOP IMMEDIATELY after the closing tag. Do NOT write explanations!

## Example Workflows (build step by step)

Example 1 - Complex parallel with verification (7 operators):
DSL: Plan -> Decompose -> [Programmer, Custom] -> ScEnsemble -> Review -> Verify -> Format
<action>add</action><operator>Plan</operator>
<action>add</action><operator>Decompose</operator>
<action>add</action><structure>parallel</structure><operators>Programmer,Custom</operators>
<action>add</action><operator>ScEnsemble</operator>
<action>add</action><operator>Review</operator>
<action>add</action><operator>Verify</operator>
<action>add</action><operator>Format</operator>
<action>finish</action>

Example 2 - Self-correcting math workflow (RECOMMENDED for math):
DSL: Plan -> Programmer -> (Verify ? Revise : done) -> Format
<action>add</action><operator>Plan</operator>
<action>add</action><operator>Programmer</operator>
<action>add</action><structure>conditional</structure><condition>Verify</condition><true>Revise</true><false>done</false>
<action>add</action><operator>Format</operator>
<action>finish</action>

Example 3 - Loop with pre/post processing (7 operators):
DSL: Plan -> Decompose -> [Programmer, Review, Revise]x2 -> Test -> Verify -> Format
<action>add</action><operator>Plan</operator>
<action>add</action><operator>Decompose</operator>
<action>add</action><structure>loop</structure><operators>Programmer,Review,Revise</operators><count>2</count>
<action>add</action><operator>Test</operator>
<action>add</action><operator>Verify</operator>
<action>add</action><operator>Format</operator>
<action>finish</action>

Example 4 - Parallel + Conditional hybrid (6 operators):
DSL: Plan -> [Programmer, Custom] -> Aggregate -> (Verify ? Revise : done) -> Format
<action>add</action><operator>Plan</operator>
<action>add</action><structure>parallel</structure><operators>Programmer,Custom</operators>
<action>add</action><operator>Aggregate</operator>
<action>add</action><structure>conditional</structure><condition>Verify</condition><true>Revise</true><false>done</false>
<action>add</action><operator>Format</operator>
<action>finish</action>

Example 5 - Self-correcting CODE workflow (RECOMMENDED for code tasks):
DSL: Programmer -> (Test ? Programmer : done) -> Format
<action>add</action><operator>Programmer</operator>
<action>add</action><structure>conditional</structure><condition>Test</condition><true>Programmer</true><false>done</false>
<action>add</action><operator>Format</operator>
<action>finish</action>

## When to Finish
- Before finishing, add Format to extract concise answer from the solution
- When Format has extracted the answer and you are satisfied → <action>finish</action>
- When the result is wrong or needs improvement → Add more operators

You will see the execution result after each operator. Compare it with the original problem and decide your next action.

## Rules (MUST follow)
- Output ONLY the XML tags, nothing else
- ONE action per turn, no more
- NO markdown code blocks (never use ```)
- NO text or explanations before or after the XML tags
- Do NOT repeat same operator more than 3 times consecutively
- ScEnsemble/Aggregate/MdEnsemble can ONLY be added AFTER a parallel block, never inside [...]

## FORBIDDEN TRANSITIONS

NEVER add Plan/Decompose after:
- Format: ONLY use FINISH or Revise
- Verify/Review/Test: ONLY use Format, Revise, or FINISH
- Programmer/Custom: ONLY use Verify, Test, Review, Format, or FINISH

Plan/Decompose ONLY at workflow START. Once Solver exists, no backward to Plan.

Correct flow: Plan -> Solver -> Checker -> Format -> FINISH"""


SYSTEM_PROMPT = SYSTEM_PROMPT_TRAINING

ACTION_EXAMPLES = ""

PROBLEM_TYPE_HINTS = {
    "default": "Think step by step. Break down the problem, solve each part carefully, then combine for the final answer.",
    "qa": "This is a QA (question answering) task. Think step by step before giving your final answer.",
    "math": "This is a MATH task. IMPORTANT: Solve step by step. Show your work for each calculation. Double-check arithmetic before giving the final answer.",
    "code": "This is a CODE task. CRITICAL: (1) You MUST use the Programmer operator to generate executable Python code. Plan/Custom operators only produce text descriptions, NOT code. (2) For checking code correctness, use TEST operator (not Verify). Test runs actual test cases. Recommended workflow: Programmer -> (Test ? Programmer : done) -> Format",
    "humaneval": "This is a CODE task (HumanEval). CRITICAL: (1) You MUST use the Programmer operator to generate executable Python code. Do NOT use Plan/Custom alone - they cannot produce code. (2) Use TEST operator (not Verify) to check code. Recommended: Programmer -> (Test ? Programmer : done) -> Format",
    "mbpp": "This is a CODE task (MBPP). CRITICAL: (1) You MUST use the Programmer operator to generate executable Python code. Do NOT use Plan/Custom alone - they cannot produce code. (2) Use TEST operator (not Verify) to check code. Recommended: Programmer -> (Test ? Programmer : done) -> Format",
    "mathqa_mc": """This is a MULTIPLE-CHOICE math question with options (a, b, c, d, e).
CRITICAL INSTRUCTIONS:
1. Solve the math problem step by step
2. Calculate the exact numerical answer
3. Compare your answer to ALL options and find the matching one
4. Your FINAL answer MUST be a SINGLE LETTER: a, b, c, d, or e

DO NOT output:
- The numerical value
- The option text
- Explanations in the final answer
- "unanswerable" (this is NOT a passage-based QA task)

ONLY output the letter (a/b/c/d/e) that matches your calculated answer.""",
}


STATE_TEMPLATE = """
## Current State
- Current DSL: {current_dsl}
- Total Operators: {total_operators}
- Unique Types: {unique_types}
- Round: {round_number}/{max_rounds}
- Node IDs: {node_ids}"""


FEEDBACK_TEMPLATE = """
## Last Action Feedback
{result_preview}

Based on the result above, output your next action:"""


HISTORY_TEMPLATE = """
## Conversation History
{history}"""


class InteractivePromptBuilder:

    def __init__(self, config: Optional[PromptConfig] = None):
        self.config = config or PromptConfig()

    def build_system_prompt(
        self,
        problem_type: str = "default",
    ) -> str:
        hint = get_problem_type_hint(problem_type)
        if hint:
            return f"{SYSTEM_PROMPT}\n\n{hint}"
        return SYSTEM_PROMPT

    def build_user_prompt(
        self,
        problem: str,
        current_dsl: str = "",
        total_operators: int = 0,
        unique_types: int = 0,
        round_number: int = 1,
        max_rounds: int = 15,
        node_ids: List[str] = None,
        last_success: Optional[bool] = None,
        last_result: str = "",
        last_message: str = "",
        history: List[Dict] = None,
    ) -> str:
        parts = []

        parts.append(f"## Problem\n{problem}")

        if self.config.include_current_state and round_number > 1:
            state = STATE_TEMPLATE.format(
                current_dsl=current_dsl or "(empty)",
                total_operators=total_operators,
                unique_types=unique_types,
                round_number=round_number,
                max_rounds=max_rounds,
                node_ids=", ".join(node_ids) if node_ids else "(none)",
            )
            parts.append(state)

        if last_success is not None:
            feedback = FEEDBACK_TEMPLATE.format(
                result_preview=last_result[:800] + "..." if len(last_result) > 800 else last_result,
            )
            parts.append(feedback)

        if history and len(history) > 0:
            recent = history[-self.config.max_history_turns:]
            history_text = ""
            for h in recent:
                history_text += f"\nTurn {h.get('turn', '?')}:\n"
                history_text += f"  Action: {h.get('action', 'N/A')}\n"
                result_val = h.get('result', 'N/A')
                history_text += f"  Result: {str(result_val)[:100]}...\n"
            parts.append(HISTORY_TEMPLATE.format(history=history_text))

        parts.append(f"\n## Your Turn\nRound {round_number}. Output your action:")

        return "\n".join(parts)

    def build_initial_prompt(
        self,
        problem: str,
        problem_type: str = "default",
    ) -> str:
        system = self.build_system_prompt(problem_type)
        user = self.build_user_prompt(
            problem=problem,
            round_number=1,
        )

        return f"{system}\n\n{user}"

    def build_continuation_prompt(
        self,
        problem: str,
        current_dsl: str,
        total_operators: int,
        unique_types: int,
        round_number: int,
        max_rounds: int,
        node_ids: List[str],
        last_success: bool,
        last_result: str,
        last_message: str,
        history: List[Dict] = None,
        problem_type: str = "default",
    ) -> str:
        system = self.build_system_prompt(problem_type)
        user = self.build_user_prompt(
            problem=problem,
            current_dsl=current_dsl,
            total_operators=total_operators,
            unique_types=unique_types,
            round_number=round_number,
            max_rounds=max_rounds,
            node_ids=node_ids,
            last_success=last_success,
            last_result=last_result,
            last_message=last_message,
            history=history,
        )
        return f"{system}\n\n{user}"

    def build_prompt_request(
        self,
        problem: str,
        operator_name: str,
        operator_description: str = "",
        context: str = "",
        problem_type: str = "default",
    ) -> str:
        pt = str(problem_type or "default").strip().lower()
        op = str(operator_name or "").strip()

        operator_rules: List[str] = []
        if op == "Programmer":
            if pt == "code":
                operator_rules = [
                    "OPERATOR-SPECIFIC RULES (Programmer / CODE):",
                    "- You are writing an instruction for a code generation model.",
                    "- Mention required signature / entry_point if provided in Context/tests.",
                    "- DO NOT write a step-by-step plan (\"Step 1/2/3\"); ask to implement code directly.",
                    "- Require: output ONLY Python code (no explanations, no markdown).",
                ]
            elif pt == "math":
                operator_rules = [
                    "OPERATOR-SPECIFIC RULES (Programmer / MATH):",
                    "- Write Python that computes the answer; define solve() and RETURN the final value.",
                    "- For EXACT answers: use sympy.Rational(a,b) for fractions, sympy.sqrt(n) for radicals.",
                    "- Example: return str(sympy.Rational(5,9)) → '5/9' (NOT 5/9 which gives 0.555...)",
                    "- Example: return str(sympy.sqrt(3)/3) → 'sqrt(3)/3' (NOT 0.577...)",
                    "- PREFER symbolic form over decimal approximations for MATH problems!",
                    "- Output ONLY Python code (no explanations, no markdown).",
                ]
        elif pt == "qa" and op in {"Custom", "Verify", "AnswerGenerate"}:
            operator_rules = [
                "OPERATOR-SPECIFIC RULES (QA):",
                "- Force SHORT final answer (usually 1-5 words).",
                "- No full sentences or extra words in the final answer field.",
            ]

        operator_rules_block = "\n".join(operator_rules).strip()
        if operator_rules_block:
            operator_rules_block += "\n"

        return PROMPT_REQUEST_TEMPLATE.format(
            operator_name=operator_name,
            operator_description=operator_description or f"A workflow operator that processes input and produces output.",
            context_block=(f"Context: {context}\n\n" if context else ""),
            problem=problem,
            operator_rules_block=operator_rules_block,
        )


# ============================================
# Second-step Prompt Request - only request prompt (no system prompt)
# ============================================

PROMPT_REQUEST_TEMPLATE = """Read the problem carefully, think step by step, then write a prompt for {operator_name} operator.

{context_block}Problem: {problem}

{operator_name}: {operator_description}

{operator_rules_block}

CRITICAL RULES - MUST FOLLOW:
1. DO NOT use <think> tags - just write the prompt directly!
2. NO XML tags of any kind - no <action>, no <think>, no <operator>
3. Write 1-3 sentences describing what the operator should do
4. Be specific to THIS problem
5. If the context mentions a parallel prompt (p0/p1/...), make this branch DIFFERENT from other branches (different method/check/perspective)
6. DO NOT restate, rephrase, or modify the problem facts/numbers/relationships
7. If you must mention values, copy them EXACTLY from the problem - never paraphrase
8. Your prompt should ONLY describe method/output format, not retell the problem

Example for Programmer (GOOD): "Implement the required function signature exactly. Output ONLY Python code, no explanation."
Example for Programmer (BAD): "Step 1: ... Step 2: ..." (this often causes code leakage)

Your prompt for {operator_name}:"""


COMPACT_SYSTEM_PROMPT = """Build workflow step by step. Actions:
- Add operator: <action>add</action><operator>NAME</operator>
- Add parallel: <action>add</action><structure>parallel</structure><operators>A,B,C</operators>
- Delete: <action>delete</action><target>node_ID</target>
- Finish: <action>finish</action>

Operators: Custom, AnswerGenerate, Programmer, ScEnsemble, Test, Review, Revise, Decompose, Verify, Plan, Aggregate, Format

One action per turn. Observe feedback. Prefer Verify/Test/Review for checking. Always add Format before finishing to extract concise answer."""


class CompactPromptBuilder:

    def build_prompt(
        self,
        problem: str,
        current_dsl: str = "",
        round_number: int = 1,
        last_result: str = "",
    ) -> str:
        parts = [COMPACT_SYSTEM_PROMPT]
        parts.append(f"\nProblem: {problem}")

        if current_dsl:
            parts.append(f"Current workflow: {current_dsl}")

        if last_result:
            parts.append(f"Last result: {last_result[:150]}...")

        parts.append(f"Round {round_number}. Your action:")

        return "\n".join(parts)


def create_prompt_builder(
    compact: bool = False,
    config: Optional[PromptConfig] = None,
) -> InteractivePromptBuilder:
    if compact:
        return CompactPromptBuilder()
    return InteractivePromptBuilder(config)


def get_problem_type_hint(problem_type: str) -> str:
    norm_type = str(problem_type).lower().strip()
    return PROBLEM_TYPE_HINTS.get(norm_type, PROBLEM_TYPE_HINTS["default"])
