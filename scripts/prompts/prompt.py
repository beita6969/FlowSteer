# -*- coding: utf-8 -*-
# @Desc: Prompts for operators
# Adapted from AFlow's prompts/prompt.py

ANSWER_GENERATION_PROMPT = """
Think step by step and solve the problem.

Your task: {input}

You MUST respond in the following XML format:
<thought>
Explain your thinking process in detail here.
</thought>
<answer>
Provide ONLY the final answer span here (as short as possible).

Rules for <answer>:
- Do NOT include explanations or full sentences.
- Do NOT prefix with words like "Answer:" / "The answer is" / "Therefore".
- If the answer is a name/entity/title: output just that name/entity/title.
- If the answer is yes/no: output only "yes" or "no".
- If the answer is a number: output only the number (and unit ONLY if the question explicitly requires the unit).

Examples:
- Good: Paris
- Bad: The capital of France is Paris.
- Good: Roe v. Wade
- Bad: She is best known for the case Roe v. Wade.
</answer>

IMPORTANT: You MUST wrap your response in <thought></thought> and <answer></answer> XML tags.
"""

FORMAT_PROMPT = """
The solution to the problem "{problem_description}" has ALREADY been computed as: {solution}

Your ONLY job is to extract the final answer from the solution.
- DO NOT re-calculate or verify the answer
- DO NOT apply any additional operations
- Just extract and return the answer that appears in the solution

**CRITICAL FORMATTING RULES:**

1. **OUTPUT ANSWER VALUE ONLY - NO EQUATIONS:**
   ❌ WRONG: "cos(45°) = 0.707" or "The answer is 42"
   ✓ CORRECT: Just output "sqrt(2)/2" or "42"
   - If solution contains "x = 5", output ONLY "5"
   - If solution contains "area = 36π", output ONLY "36π" or "36*pi"
   - NEVER include variable names, "=", or explanatory text

2. **PRESERVE SYMBOLIC FORMS** (for MATH problems):
   - Keep fractions as fractions: 5/9, 3/4, 7/12 (NOT 0.555..., 0.75, 0.583...)
   - Keep radicals: sqrt(2), sqrt(3), sqrt(5) (NOT 1.414, 1.732, 2.236)
   - Keep π expressions: pi/4, 2*pi, 3*pi/2 (NOT 0.785, 6.283, 4.712)
   - Keep combined forms: sqrt(3)/3, sqrt(2)/2, 5*sqrt(2) (NOT 0.577, 0.707, 7.071)

3. **COMPLEX NUMBER FORMAT:**
   - Use lowercase 'i' for imaginary unit: -1-5i (NOT -1 - 5*I or -1-5j)
   - No spaces around operators: 3+4i (NOT 3 + 4i)
   - Pure imaginary: 5i (NOT 5*i or 5j)

4. **VECTOR/TUPLE/COORDINATE FORMAT:**
   - Use parentheses with commas: (2, -1, 2) for 3D points
   - Use commas without brackets for lists: -3, 1 (NOT [-3, 1])
   - Remove quotes from string tuples: (2, -1, 2) (NOT ('2', '-1', '2'))

5. **SET FORMAT:**
   - Multiple values comma-separated: -3, 1 (NOT {{-3, 1}} or [-3, 1])
   - Sorted from smallest to largest when applicable

6. **INTEGER RULES:**
   - For integer answers: output ONLY the integer WITHOUT decimal point (70000, NOT 70000.0)
   - Remove .0 or .00 suffix (42.0 -> 42, 16.00 -> 16)
   - Keep non-zero decimals (42.5 -> 42.5)

7. **EXAMPLES:**
   - "x = 5/9" -> 5/9 ✓
   - "result = sqrt(3)/3" -> sqrt(3)/3 ✓
   - "cos(45°) = √2/2" -> sqrt(2)/2 ✓
   - "conjugate = -1 - 5i" -> -1-5i ✓
   - "point = (2, -1, 2)" -> (2, -1, 2) ✓
   - "k values are -3 and 1" -> -3, 1 ✓
   - "70000.0" -> 70000 ✓

8. **PRESERVE ORIGINAL NON-MATH FORMATS (IMPORTANT):**
   - If the answer is already a short phrase, keep it EXACTLY (no translation/reformatting).
   - **Dates**: Do NOT convert month names to numeric dates and do NOT add a year unless the solution explicitly contains it.
     * Example: "June 20" -> June 20 ✓ (NOT 06/20/2023, NOT 2023-06-20)
   - **Names/labels**: Do NOT convert to IDs/codes (e.g., "Angela" stays "Angela").

Return ONLY the final answer value, nothing else.
"""

SC_ENSEMBLE_PROMPT = """
Given the question described as follows: {question}
Several solutions have been generated to address the given question. They are as follows:
{solutions}

Carefully evaluate these solutions and identify the answer that appears most frequently across them. This consistency in answers is crucial for determining the most reliable solution.

You MUST respond in the following XML format (and ONLY this format):
<thought>...</thought>
<solution_letter>...</solution_letter>

In <thought>, explain your reasoning briefly. In <solution_letter>, output ONLY one letter ID (A, B, C, etc.) corresponding to the most consistent solution. Do NOT include any extra text.
"""

PYTHON_CODE_VERIFIER_PROMPT = """
You are a professional Python programmer. Your task is to write complete, self-contained code based on a given problem and output the answer. The code should include all necessary imports and be ready to run without additional setup.

Problem description: {problem}
Other analysis: {analysis}
{feedback}

## Code Requirements:

1. **CRITICAL - Entry point**: You MUST define a `def solve():` function that returns the final answer.
   - If problem defines `def xxx(params):`, implement that function AND call it from `solve()`.
   - Example: If problem wants `def find_Min_Swaps(arr, n):`, write:
     ```
     def find_Min_Swaps(arr, n):
         # your implementation
         return result

     def solve():
         # Use test case from problem or reasonable defaults
         return find_Min_Swaps([0,1,0,1,0], 5)
     ```
2. Return basic types: str, int, float, list, tuple, dict. NOT matplotlib objects.
3. All necessary imports should be at the top of the code.

## CRITICAL ERRORS TO AVOID:

### 1. NEVER use input() - Code runs in sandbox without stdin!
WRONG: x = input("Enter: ")  # Causes EOFError
RIGHT: x = "test_value"  # Hardcode test values if no function params

### 2. NEVER hardcode placeholder file paths!
WRONG: directory = "path/to/your/json/files"  # FileNotFoundError
RIGHT: Use the path from the problem, or create temp files:
       import tempfile; tmpdir = tempfile.mkdtemp()

### 3. NEVER mix SymPy symbols with NumPy functions!
WRONG: import sympy; x = sympy.Symbol('x'); np.cos(x)  # TypeError
RIGHT: Use ONLY sympy functions with sympy objects:
       sympy.cos(x), sympy.solve(), float(result.evalf())

### 3b. SymPy API pitfalls - AVOID THESE COMMON MISTAKES:
- NEVER pass a Python list to sympy.poly():
  WRONG: sympy.poly([1, -3, 2])  # AttributeError: 'list' has no attribute 'is_Poly'
  RIGHT: sympy.Poly([1, -3, 2], x)  # Pass symbol as second argument
- NEVER try to unpack sympy.Mul or sympy.Add directly:
  WRONG: a, b = expr  # TypeError if expr is Mul
  RIGHT: Use expr.args to get operands: a, b = expr.args
- For Abs(x) equations, specify real domain:
  WRONG: sympy.solve(sympy.Abs(x) - 1)  # Error with complex
  RIGHT: sympy.solve(sympy.Abs(x) - 1, x, domain=sympy.S.Reals)
- NEVER use as_int() on float:
  WRONG: sympy.as_int(1.5)  # Error
  RIGHT: int(round(value)) or sympy.Integer(value)

### 4. Pandas API - Check correct method signatures!
WRONG: df['col'].str.startswith('x', case=False)  # No 'case' param
RIGHT: df['col'].str.lower().str.startswith('x')  # Convert case first

### 5. For plotting tasks - Return data, NOT Axes objects!
WRONG: return plt.gca()  # Returns Axes, hard to validate
RIGHT: return (outlier_df, plt.gca())  # Return data + axes as tuple

## AVAILABLE LIBRARIES:
- Standard: os, sys, math, re, collections, itertools, functools, json, datetime, random, string, decimal, fractions, statistics, heapq, bisect, copy, typing, glob, tempfile, shutil
- Scientific: numpy (as np), scipy, sympy (KEEP SEPARATE from numpy!)
- ML: sklearn, tensorflow, keras
- Data: pandas (as pd), prettytable (PrettyTable)
- Plotting: matplotlib.pyplot (as plt), seaborn (as sns) - NO plt.show()!
- NLP: nltk

## PROHIBITED:
- input(), open() for reading user files, plt.show(), plt.savefig()
- External: flask, django, requests, aiohttp, faker, sqlalchemy, tkinter
- Network/DB operations

## MATH SYMBOLIC OUTPUT (for math problems requiring exact answers):
- Use sympy for symbolic results instead of float approximations:
  * sympy.Rational(5, 9) instead of 5/9 (which gives float 0.555...)
  * sympy.sqrt(3)/3 instead of 0.5773...
  * sympy.pi/4 instead of 0.785...
- Convert sympy expressions to string: str(expr) or str(expr.simplify())
- Example:
  ```
  import sympy
  def solve():
      # For answer 5/9, use:
      return str(sympy.Rational(5, 9))  # Returns "5/9"
      # For answer √3/3, use:
      return str(sympy.sqrt(3)/3)  # Returns "sqrt(3)/3"
  ```
- IMPORTANT: For MATH problems, prefer symbolic form over decimal approximations!

**Your output should match MATH benchmark ground truth format:**

1. **FRACTIONS**: Return symbolic fractions, NOT decimals
   ✓ return str(sympy.Rational(5, 9))  # "5/9"
   ✗ return 5/9  # 0.555...

2. **RADICALS**: Return symbolic sqrt, NOT float approximations
   ✓ return str(sympy.sqrt(2)/2)  # "sqrt(2)/2"
   ✗ return math.sqrt(2)/2  # 0.7071...

3. **COMPLEX NUMBERS**: Use lowercase 'i' format
   ✓ return f"{{real}}{{'+' if imag >= 0 else ''}}{{imag}}i"  # "-1-5i"
   ✗ return complex(real, imag)  # "(-1-5j)" - Python format
   ✗ return str(sympy.I * imag)  # Uses uppercase I

4. **MULTIPLE VALUES/SETS**: Comma-separated, sorted
   ✓ return ", ".join(sorted([str(x) for x in values], key=lambda x: float(x) if x.lstrip('-').isdigit() else x))
   ✗ return values  # Returns list like [-3, 1]
   ✗ return set(values)  # Returns {{-3, 1}}

5. **VECTORS/COORDINATES**: Parentheses with numeric values
   ✓ return f"({{x}}, {{y}}, {{z}})"  # "(2, -1, 2)"
   ✗ return (x, y, z)  # Python tuple
   ✗ return str((x, y, z))  # "(2, -1, 2)" but be careful with formatting

6. **PI EXPRESSIONS**: Use 'pi' string
   ✓ return f"{{coef}}*pi" or str(sympy.pi * coef)  # "2*pi"
   ✗ return math.pi * coef  # 6.283...

## CRITICAL - NEGATIVE VALUES (MUST READ):
- **Divisors/Factors**: When asked for divisors or factors, include BOTH positive AND negative:
  * Divisors of 12: ±1, ±2, ±3, ±4, ±6, ±12 (NOT just 1, 2, 3, 4, 6, 12)
  * Use: `list(sympy.divisors(n)) + [-d for d in sympy.divisors(n)]` or iterate ±range
- **Polynomial roots**: Consider all roots including negative ones
- **Integer solutions**: Check negative values when iterating possible solutions
- **Factor pairs**: (a,b) where a*b=N includes negative pairs: (-a,-b)
  * Factor pairs of 12: (1,12), (2,6), (3,4), (-1,-12), (-2,-6), (-3,-4)

## OUTPUT FORMAT:
Output ONLY Python code. No explanations, no markdown, no ```python blocks.
The function must return a single value (the answer).

CRITICAL WARNING: If the instruction asks you to "describe steps", "explain the algorithm", or "outline the approach",
IGNORE that instruction and write EXECUTABLE Python code instead. Your output MUST be valid Python code that can run.
NEVER output text like "Step 1: ..., Step 2: ..." - that is NOT code and will cause execution failure.
"""

REFLECTION_ON_PUBLIC_TEST_PROMPT = """
Given a code problem and a Python solution that failed, analyze the failure and provide a corrected solution.

### Problem
{problem}

### Failed Code
{solution}

### Execution Result
{exec_pass}

### Failed Test Case
{test_fail}

## Common Errors - Check These First:
1. EOFError → Remove any input() calls. Hardcode values instead.
2. FileNotFoundError → Don't use placeholder paths like "path/to/files". Use tempfile or problem-specified paths.
3. TypeError with SymPy → Don't pass sympy symbols to numpy functions. Use sympy.cos(), sympy.sin() etc.
4. pandas API error → Check method signatures. str.startswith() has no 'case' parameter.
5. Function signature error → If problem defines `def xxx(params):`, use EXACTLY that signature.
6. AssertionError (especially empty message) → Usually wrong logic or wrong interpretation. Re-read the spec carefully and check for common traps (off-by-one, inclusive/exclusive, sorting keys vs sorting inner lists, return type/ordering).

## Your Task:
Provide ONLY the corrected Python code. No explanations. The function must:
- ALWAYS include a `def solve():` entry point that returns the answer
- If problem defines `def xxx(params):`, implement that AND call it from `solve()`
- Return basic types (str, int, float, list, dict)
- NOT use input(), plt.show(), or network calls
"""

MD_ENSEMBLE_PROMPT = """
Given the question described as follows: {question}
Several solutions have been generated to address the given question. They are as follows:
{solutions}

Carefully evaluate these solutions and identify the solution that is more capable of solving the problem compared to other solutions, as this is crucial for problem-solving.

In the "thought" field, provide a detailed explanation of your thought process. In the "solution_letter" field, output only the single letter ID (A, B, C, etc.) corresponding to the solution. Do not include any additional text or explanation in the "solution_letter" field.
"""

REVIEW_PROMPT = """
Given a problem and a thoughtful solution, your task is to using critical thinking (questioning) to review the solution's correctness and provide a review result in boolean format.

problem: {problem}
solution: {solution}

If you are more than 95 percent confident that the final answer is incorrect, set review_result to false and give feedback explaining the error. Otherwise, set review_result to true.

Respond in XML format:
<review_result>
true or false
</review_result>
<feedback>
[If review_result is false: explain the error and what the correct answer should be]
[If review_result is true: write "nothing here"]
</feedback>

IMPORTANT: You MUST wrap your response in <review_result></review_result> and <feedback></feedback> XML tags.
"""

REVISE_PROMPT = """
Given a problem and a thoughtful solution which is just reviewed as incorrect, your task is to revise the solution to solve the question.

problem: {problem}
solution: {solution}
feedback: {feedback}

Please provide your revised solution in the following XML format:
<solution>
Your complete revised solution here. If it's code, include the full code.
</solution>

Important: You MUST wrap your answer in <solution></solution> tags.
"""

# ============================================
# ============================================

DECOMPOSE_PROMPT = """
You are an expert problem solver. Your task is to break down a complex problem into smaller, manageable sub-problems.

Problem: {problem}

Please decompose this problem into a series of simpler sub-problems that, when solved in sequence, will lead to the final answer.

You MUST respond in the following XML format:
<sub_problems>
1. [First sub-problem - describe clearly what needs to be solved]
2. [Second sub-problem - describe clearly what needs to be solved]
3. [Continue with more sub-problems as needed...]
</sub_problems>
<reasoning>
Explain why you decomposed the problem this way and how solving these sub-problems leads to the final solution.
</reasoning>

Guidelines:
- Each sub-problem should be simpler than the original
- Sub-problems should be independent where possible
- The sequence should lead logically to the final answer
- Typically 2-5 sub-problems is appropriate
"""

VERIFY_PROMPT = """
You are a meticulous verifier with mathematical and logical reasoning expertise. Your SOLE task is to verify correctness through INDEPENDENT recalculation.

Problem: {problem}
Proposed Answer: {answer}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**MANDATORY VERIFICATION PROCEDURE** (FAILURE TO FOLLOW = INVALID OUTPUT)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️ **CRITICAL RULE**: You MUST solve the problem INDEPENDENTLY **BEFORE** looking at the Proposed Answer.

**Step 1: READ AND UNDERSTAND**
- Read the problem carefully
- Identify what is being asked
- Note all given information

**Step 2: INDEPENDENT CALCULATION (🚨 REQUIRED - FAILURE = INVALID OUTPUT)**
- Do NOT look at the Proposed Answer yet
- Solve the problem completely from scratch
- Show EVERY step of your work

**For math problems**:
  - Write out EVERY arithmetic operation explicitly
  - Example format:
    * "Total = (10 apples) × ($3 per apple) = $30"
    * "Discount = $30 × 0.20 = $6"
    * "Final = $30 - $6 = $24"
  - Do NOT skip any intermediate steps

**For reasoning problems**:
  - State each logical inference explicitly
  - Explain the reasoning chain step by step

**Step 3: COMPARE**
- NOW look at the Proposed Answer
- Does your calculated answer match the Proposed Answer?
- If YES → is_correct = true
- If NO → is_correct = false (YOUR answer is likely correct)

**Step 4: IDENTIFY ERRORS** (if answers don't match)
- Pinpoint WHERE the Proposed Answer went wrong
- What calculation/logic was incorrect?
- What was the error (e.g., "used 8 instead of 10", "forgot to subtract discount")

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**REQUIRED OUTPUT FORMAT**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You MUST output in this EXACT XML format:

<verification_steps>
[Step 1] Problem Understanding:
  - What is asked: [state the question]
  - Given information: [list all given data]

[Step 2] Independent Calculation:
  🔍 SOLVING FROM SCRATCH (without looking at Proposed Answer):

  [For math: Show EVERY arithmetic step]
  Calculation 1: [expression] = [result]
  Calculation 2: [expression] = [result]
  ...

  [OR for reasoning: Show logical chain]
  Premise 1: [statement]
  Inference 1: [logical step]
  ...

  MY CALCULATED ANSWER: [your result]

[Step 3] Comparison:
  - My answer: [your calculated answer]
  - Proposed Answer: [the given answer]
  - Match? [YES/NO]

[Step 4] Error Analysis (if mismatch):
  - Error location: [where the Proposed Answer went wrong]
  - Error type: [what mistake was made]
  - Correct approach: [what should have been done]
</verification_steps>

<is_correct>
true or false
</is_correct>

<confidence>
high, medium, or low
</confidence>

<answer>
[OUTPUT THE CORRECT ANSWER HERE]

📋 Answer Format Rules:
- If Proposed Answer is WRONG: output YOUR independently calculated answer
- If Proposed Answer is CORRECT: output it (cleaned up if verbose)
- Do NOT add explanations, prefixes, or suffixes
- Do NOT write "Answer:" / "The answer is:" / etc.
- Format based on answer type:
  * Name/entity/title → output just the name (e.g., "Paris")
  * Yes/no → output only "yes" or "no"
  * Number → output only the number with unit if required (e.g., "24" or "24 dollars")
  * Code → output ONLY the code (no markdown fences, no comments)
</answer>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ **VERIFICATION INTEGRITY CHECK**: If your <verification_steps> does not contain detailed independent calculation in Step 2, your output is INVALID.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

PLAN_PROMPT = """
You are a strategic problem solver. Your task is to create a detailed plan for solving the given problem.

Problem: {problem}

Create a structured plan that outlines the approach and steps needed to solve this problem.

You MUST respond in the following XML format:
<approach>
[Describe the overall strategy or method you will use - e.g., "Use dynamic programming", "Apply mathematical induction", "Break into cases"]
</approach>
<key_insights>
[List key observations or insights about the problem that inform your solution]
</key_insights>
<plan>
Step 1: [First concrete action to take]
Step 2: [Second concrete action to take]
Step 3: [Continue with specific steps...]
Final Step: [How to derive the final answer]
</plan>

Guidelines:
- The approach should identify the main solving strategy
- Key insights should highlight important patterns or properties
- Each step in the plan should be specific and actionable
- The plan should be complete enough to follow to a solution
- **CRITICAL FOR MATH**: When finding divisors, factors, or roots:
  * ALWAYS consider BOTH positive AND negative values
  * For divisors of N: include ±1, ±2, ..., ±N (not just 1, 2, ..., N)
  * For integer roots: consider negative solutions
  * For factorization: include negative factor pairs
"""

AGGREGATE_PROMPT = """
You are selecting the best answer from multiple candidate answers.

Original Problem: {problem}

Candidate answers:
{sub_answers}

CRITICAL RULES:
1. Do NOT re-solve or re-compute the problem - trust the candidate answers
2. For numeric answers: use MAJORITY VOTING (select the most frequent value)
3. If candidates are all different: select the one with clearest reasoning or most complete solution
4. The candidates were computed by specialized solvers - do not second-guess their calculations

You MUST respond in the following XML format:
<selection_reason>
[Brief explanation: which answer you selected and why (e.g., "Selected 42 - majority vote from 3 candidates")]
</selection_reason>
<aggregated_answer>
[The selected answer - for math problems, output ONLY the number, nothing else]
</aggregated_answer>

STRICT OUTPUT RULES (for math/quantitative problems):
- Output the EXACT answer from the selected candidate - do NOT modify, simplify, or truncate
- If the answer is a single number: output just the number (e.g., 42, 3/4, 0.5)
- If the answer has MULTIPLE VALUES (e.g., eigenvalues, roots): output ALL values separated by commas (e.g., "1, -3" or "-4, 4")
- If the answer contains VARIABLES: KEEP the variables exactly (e.g., "500a^7", "x + 1", "2n")
- If the answer contains pi: keep pi (e.g., "900*pi" or "900π")
- If the answer is a coordinate/point: keep the format (e.g., "(2, -1, 2)")
- Do NOT strip variables, do NOT take only the numeric coefficient
- Do NOT reduce multiple answers to just one
"""

APPS_PLAN_PROMPT = """
You are an expert competitive programmer. Analyze the problem and create a detailed algorithm plan.

## Problem
{problem}

## Your Task
Create a step-by-step algorithm plan. Focus on:
1. **Input/Output Format**: How to parse input and format output
2. **Algorithm Choice**: What algorithm/data structure to use (e.g., DP, greedy, BFS, binary search)
3. **Edge Cases**: What special cases to handle (empty input, single element, large numbers)
4. **Time Complexity**: Ensure the solution is efficient enough

## Response Format
<algorithm>
[Name the algorithm approach: e.g., "Dynamic Programming", "Two Pointers", "BFS/DFS", "Greedy", "Binary Search", "Simulation"]
</algorithm>
<complexity>
[Expected time and space complexity: e.g., "O(n log n) time, O(n) space"]
</complexity>
<steps>
Step 1: [Parse input - describe format]
Step 2: [Initialize data structures]
Step 3: [Main algorithm logic]
Step 4: [Handle edge cases]
Step 5: [Output result]
</steps>
<key_insight>
[The critical insight that makes this problem solvable - what pattern or property to exploit]
</key_insight>
"""

APPS_CODE_PROMPT = """
You are an expert competitive programmer. Think step by step, then write Python code to solve the following problem.

## Problem
{problem}

{analysis}
{feedback}

## CRITICAL REQUIREMENTS:
1. **Input**: Read from stdin using `sys.stdin.read()` or `input()`
2. **Output**: Use `print()` - do NOT use `return` for the final answer
3. **Call solve()**: Always call the function at the end

Write complete, executable Python code.
"""

# Export all prompts
__all__ = [
    'ANSWER_GENERATION_PROMPT',
    'FORMAT_PROMPT',
    'SC_ENSEMBLE_PROMPT',
    'PYTHON_CODE_VERIFIER_PROMPT',
    'APPS_CODE_PROMPT',
    'APPS_PLAN_PROMPT',
    'REFLECTION_ON_PUBLIC_TEST_PROMPT',
    'MD_ENSEMBLE_PROMPT',
    'REVIEW_PROMPT',
    'REVISE_PROMPT',
    'DECOMPOSE_PROMPT',
    'VERIFY_PROMPT',
    'PLAN_PROMPT',
    'AGGREGATE_PROMPT'
]
