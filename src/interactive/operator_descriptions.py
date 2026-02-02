from typing import Dict, Any


OPERATOR_TEMPLATES: Dict[str, Dict[str, Any]] = {

    "Custom": {
        "name": "Custom",
        "description": "Most flexible general-purpose reasoning operator for any task requiring LLM reasoning",
        "input": "input (problem/context), instruction (your defined instruction)",
        "output": "response (generated answer)",
        "use_cases": [
            "Problem analysis and understanding",
            "Intermediate reasoning steps",
            "Any task requiring LLM reasoning"
        ],
        "default_prompt": """Analyze the given problem carefully and provide a clear, step-by-step solution.

Think step by step:
1. Read and understand the problem thoroughly
2. Identify key information and constraints
3. Break down the problem into smaller steps
4. Solve each step carefully, showing your work
5. Double-check your reasoning and calculations
6. Provide a clear final answer

IMPORTANT: Show your reasoning process step by step. Do not skip steps.
Focus on accuracy and clarity in your response.""",
        "customization_hints": [
            "Can specify analysis perspective (e.g., from mathematical angle, from logical angle)",
            "Can require specific output format (e.g., bullet points, table format)",
            "Can emphasize key focus areas (e.g., pay attention to edge cases)",
            "Can limit answer length or detail level"
        ],
        "examples": [
            {
                "problem_type": "math",
                "prompt": "Solve this math problem step by step. Show all calculations clearly. Double-check arithmetic operations. Present the final answer in a boxed format."
            },
            {
                "problem_type": "qa",
                "prompt": "Answer this question based on the given context. Extract relevant information first. Provide a concise, direct answer. If unsure, explain your reasoning."
            }
        ]
    },

    "Plan": {
        "name": "Plan",
        "description": "Create solution strategy and step planning",
        "input": "problem (problem description)",
        "output": "approach (strategy), key_insights (key insights), plan (specific steps) - XML format",
        "use_cases": [
            "Initial analysis of complex problems",
            "Planning for multi-step problems",
            "Problems requiring strategy selection"
        ],
        "default_prompt": """Create a strategic plan to solve this problem.

Your plan should include:
1. APPROACH: Identify the overall solving strategy (e.g., algebraic manipulation, pattern recognition, case analysis)
2. KEY INSIGHTS: List important observations about the problem
3. PLAN: Provide specific, actionable steps to reach the solution

Output in XML format with <approach>, <key_insights>, and <plan> tags.
Be specific and actionable - each step should be clear enough to follow.""",
        "customization_hints": [
            "Can specify preferred solving method (e.g., prefer algebraic approach)",
            "Can require considering multiple solutions",
            "Can emphasize efficiency (find the most concise method)",
            "Can adjust focus based on problem type"
        ],
        "examples": [
            {
                "problem_type": "math",
                "prompt": "Plan the solution for this math problem. Consider: Is this algebraic, geometric, or combinatorial? What formulas might apply? Plan 3-5 clear steps to the solution."
            },
            {
                "problem_type": "code",
                "prompt": "Plan the algorithm for this coding problem. Identify: input/output format, edge cases, time complexity requirements. Outline the main logic steps."
            }
        ]
    },

    "Programmer": {
        "name": "Programmer",
        "description": "Write and execute Python code. MATH tasks use solve() to return calculation results; CODE tasks match the given function signature",
        "input": "problem (problem), analysis (optional analysis)",
        "output": "code (code), output (execution result)",
        "use_cases": [
            "MATH: Math problems requiring precise calculation - use solve() function",
            "MATH: Complex numerical and symbolic computation",
            "CODE: HumanEval/MBPP code generation - match the given function signature",
            "CODE: Code tasks requiring test verification"
        ],
        "default_prompt": """Write Python code to solve this problem.

Requirements:
1. Define a function named `solve()` that returns the answer
2. Use clear variable names and add comments
3. Handle edge cases appropriately
4. Prefer EXACT answers when possible (fractions / radicals / π). Only output decimals if the problem explicitly asks for an approximation.

Available libraries: numpy, sympy, scipy, pandas, math, itertools, collections

IMPORTANT - COMPLETE ANSWER RULES (CompleteAnswer/CompleteAnswer):
- The solve() function must RETURN the answer, not print it
- If using sympy: prefer exact objects (Rational, sqrt, pi) and return `str(expr)` or `str(expr.simplify())` (avoid float unless required)
- For VECTORS: Return the COMPLETE vector/tuple, e.g., return (-2, 3, 3) not just -2
- For COMPLEX NUMBERS: Return the FULL complex number with imaginary part, e.g., return "5-10i" not just 5
- For COORDINATES: Return ALL coordinates as tuple, e.g., return (5*sqrt(2), 5*sqrt(2)) not just 5*sqrt(2)
- For MULTIPLE SOLUTIONS: Return ALL solutions, e.g., return [1, -3] or "1, -3" not just -3
- For EXPRESSIONS WITH VARIABLES: Keep the variable, e.g., return "500*a^7" not just 500
- Consider NEGATIVE numbers: Check if negative values are valid (e.g., smallest integer x where 12/(x+1) is integer -> x=-13)
- Ensure the code is complete and executable""",
        "customization_hints": [
            "CRITICAL: Your prompt MUST instruct the model to output Python CODE, NOT step descriptions!",
            "BAD prompt: 'Describe the algorithm step by step' -> causes code leakage (Step 1: ..., Step 2: ...)",
            "GOOD prompt: 'Implement using dynamic programming. Return the result.'",
            "Can specify using specific library (e.g., use sympy for symbolic computation)",
            "Can require specific precision (e.g., keep 4 decimal places)",
            "Can specify algorithm type (e.g., use dynamic programming)",
            "Can emphasize efficiency requirements"
        ],
        "examples": [
            {
                "problem_type": "math",
                "prompt": "Write Python code using sympy. Define solve() and return an EXACT value when possible: use sympy.Rational, sympy.sqrt, sympy.pi, and return str(expr.simplify()). Avoid float() unless the question asks for decimals."
            },
            {
                "problem_type": "code",
                "prompt": "Implement the function exactly as specified in the problem signature. Focus on correctness and edge cases. Do NOT use solve() - match the given function name."
            }
        ]
    },

    "Review": {
        "name": "Review",
        "description": "Review solution correctness using critical thinking",
        "input": "problem (problem), solution (solution to review)",
        "output": "review_result (true/false), feedback (feedback) - XML format",
        "use_cases": [
            "Verify calculation correctness",
            "Check reasoning logic",
            "Discover potential errors",
            "Quality control"
        ],
        "default_prompt": """Review this solution using critical thinking. Check each step carefully.

Think step by step - Check for:
1. Arithmetic errors - verify ALL calculations step by step
2. Logic errors - check each reasoning step
3. Missing cases - are edge cases handled?
4. Format errors - is the answer in correct format?

IMPORTANT: Recalculate key calculations to verify. Do not just read - actually redo the math.

Output in XML format:
<review_result>true or false</review_result>
<feedback>Detailed explanation of your review findings, including your verification calculations</feedback>

NOTE: Only return false if you are MORE THAN 95% confident the answer is wrong.
If the solution looks reasonable, return true with positive feedback.""",
        "customization_hints": [
            "Can specify focus areas (e.g., focus on unit conversion)",
            "Can adjust strictness level",
            "Can require checking specific error types",
            "Can require providing improvement suggestions"
        ],
        "examples": [
            {
                "problem_type": "math",
                "prompt": "Review this math solution. Verify: 1) All arithmetic is correct 2) Formulas are applied correctly 3) Units are consistent 4) Answer is in simplest form. Return review_result and detailed feedback in XML."
            },
            {
                "problem_type": "qa",
                "prompt": "Review this answer. Check: 1) Answer addresses the question 2) Information is accurate 3) Reasoning is sound 4) No contradictions. Return review_result and feedback in XML."
            }
        ]
    },

    "Revise": {
        "name": "Revise",
        "description": "Revise solution based on feedback",
        "input": "problem (problem), solution (original solution), feedback (feedback)",
        "output": "solution (revised solution) - XML format",
        "use_cases": [
            "Fix errors found by Review",
            "Improve solution quality",
            "Iterative optimization"
        ],
        "default_prompt": """Revise the solution based on the feedback provided.

Instructions:
1. Carefully read the feedback to understand what needs correction
2. Address each issue mentioned in the feedback
3. Maintain correct parts of the original solution
4. Provide the complete revised solution

Output in XML format:
<solution>Your complete revised solution here</solution>

The revised solution should be self-contained and complete.""",
        "customization_hints": [
            "Can specify revision focus",
            "Can require preserving specific format",
            "Can require adding more explanation",
            "Can specify output format requirements"
        ],
        "examples": [
            {
                "problem_type": "math",
                "prompt": "Revise this math solution based on the feedback. Fix any calculation errors. Keep the overall approach if it's correct. Show the corrected steps clearly. Output in <solution> tags."
            },
            {
                "problem_type": "qa",
                "prompt": "Revise this answer based on the feedback. Address the issues mentioned. Keep the answer concise (1-5 words). Output the corrected answer in <solution> tags."
            }
        ]
    },

    "Verify": {
        "name": "Verify",
        "description": "Independently verify answer correctness",
        "input": "problem (problem), answer (answer to verify)",
        "output": "is_correct (true/false), verification_steps (verification steps), confidence (confidence level), answer (final answer) - XML format",
        "use_cases": [
            "Final answer verification",
            "Independent review",
            "Cross-validation"
        ],
        "default_prompt": """Independently verify if this answer is correct by recalculating step by step.

Verification process - Think step by step:
1. Re-read the problem requirements carefully
2. Recalculate the answer from scratch, showing each step
3. Compare your calculation with the given answer
4. Check the answer format and completeness
5. Assess your confidence level

Output in XML format:
<verification_steps>Your step-by-step recalculation and verification</verification_steps>
<is_correct>true or false</is_correct>
<confidence>high, medium, or low</confidence>
<answer>The verified/corrected answer</answer>

CRITICAL: The <answer> field must contain the actual answer, not just true/false.
IMPORTANT: Show all calculation steps in verification_steps. Do not skip steps.""",
        "customization_hints": [
            "Can specify verification method (e.g., verify using different approach)",
            "Can require specific confidence threshold",
            "Can emphasize specific aspects to check"
        ],
        "examples": [
            {
                "problem_type": "math",
                "prompt": "Verify this answer by recalculating from scratch. Use a different method if possible. Check: numerical accuracy, correct formula usage, proper simplification. Output verification_steps, is_correct, confidence, and the final answer in XML."
            },
            {
                "problem_type": "qa",
                "prompt": "Verify this answer by re-reading the context. Check: Does it answer the question? Is the information accurate? Is it concise (1-5 words)? Output verification_steps, is_correct, confidence, and the corrected answer in XML."
            }
        ]
    },

    "Decompose": {
        "name": "Decompose",
        "description": "Break complex problem into sub-problems for divide-and-conquer approach",
        "input": "problem (problem)",
        "output": "sub_problems (sub-problem list), reasoning (decomposition reasoning) - XML format",
        "use_cases": [
            "Complex multi-step problems",
            "Problems requiring divide-and-conquer",
            "Breaking down large tasks"
        ],
        "default_prompt": """Break down this complex problem into smaller, manageable sub-problems.

Guidelines:
1. Each sub-problem should be simpler than the original
2. Sub-problems should be relatively independent
3. Solving all sub-problems should lead to the final answer
4. Typically 2-5 sub-problems is appropriate

Output in XML format:
<sub_problems>
1. [First sub-problem]
2. [Second sub-problem]
...
</sub_problems>
<reasoning>Explain why this decomposition makes sense</reasoning>""",
        "customization_hints": [
            "Can specify decomposition granularity",
            "Can require specific number of sub-problems",
            "Can specify decomposition strategy (e.g., by steps, by components)"
        ],
        "examples": [
            {
                "problem_type": "math",
                "prompt": "Decompose this problem into 2-4 sequential sub-problems. Each should represent a clear calculation step. Ensure solving them in order gives the final answer."
            }
        ]
    },

    "Aggregate": {
        "name": "Aggregate",
        "description": "Combine multiple sub-answers into final answer",
        "input": "problem (problem), sub_answers (sub-answer list)",
        "output": "synthesis (synthesis process), aggregated_answer (final answer) - XML format",
        "use_cases": [
            "Merge sub-answers from Decompose",
            "Synthesize multiple results",
            "Final answer generation"
        ],
        "default_prompt": """Select the best final answer from the given candidate sub-answers.

CRITICAL RULE: Do NOT re-solve or re-compute the original problem. Only select/aggregate based on the candidates.

Guidelines:
1. If the candidates are numeric answers: use MAJORITY VOTING on the final value
2. If candidates disagree: prefer the one with clearer reasoning / fewer assumptions / correct format
3. If a candidate is verbose: extract its final answer span, don't rewrite it

Output in XML format:
<synthesis>Briefly explain which candidate you selected and why</synthesis>
<aggregated_answer>The selected final answer only</aggregated_answer>""",
        "customization_hints": [
            "Can specify aggregation strategy",
            "Can require specific output format",
            "Can emphasize conflict resolution method"
        ],
        "examples": [
            {
                "problem_type": "math",
                "prompt": "Select the best answer from the candidates. Do NOT re-solve. If multiple candidates give the same number, choose that. Output only the final value in <aggregated_answer>."
            },
            {
                "problem_type": "qa",
                "prompt": "Select the best answer from the candidates. Do NOT re-solve. If candidates agree, use that answer. Output the concise answer span in <aggregated_answer> (1-5 words, no full sentence)."
            }
        ]
    },

    "AnswerGenerate": {
        "name": "AnswerGenerate",
        "description": "Generate answer with thinking process",
        "input": "input (problem)",
        "output": "thought (thinking process), answer (final answer) - XML format",
        "use_cases": [
            "Problems requiring reasoning process display",
            "Direct final answer generation",
            "Quick solution for simple problems"
        ],
        "default_prompt": """Solve this problem step by step and provide the final answer.

Think step by step:
1. Understand what the problem is asking
2. Identify all given information
3. Break down the solution into clear steps
4. Solve each step carefully, showing your work
5. Double-check your calculations
6. Provide a clear, concise final answer

Output in XML format:
<thought>Your detailed step-by-step thinking process. Show ALL calculations.</thought>
<answer>Your final answer (concise and clear)</answer>

IMPORTANT: Show all work in the thought section. The answer should be direct - just the answer, no extra explanation.""",
        "customization_hints": [
            "Can specify thinking detail level",
            "Can require specific answer format",
            "Can emphasize aspects to consider"
        ],
        "examples": [
            {
                "problem_type": "math",
                "prompt": "Think through this math problem step by step. Show all calculations in <thought>. Give only the final numerical answer in <answer> - no units or explanation, just the number."
            },
            {
                "problem_type": "qa",
                "prompt": "Answer this question. In <thought>, analyze the question and relevant information. In <answer>, provide a direct, concise answer - ideally one word or short phrase."
            }
        ]
    },

    "Format": {
        "name": "Format",
        "description": "Extract concise final answer from solution",
        "input": "problem (problem), solution (solution), instruction (optional: custom extraction guidance)",
        "output": "response (formatted answer)",
        "use_cases": [
            "Extract answer from long solution",
            "Standardize answer format",
            "Final output processing"
        ],
        "default_prompt": """Extract a short, concise final answer from this solution.

CRITICAL RULE: Do NOT re-solve the problem. Only extract what already appears in the solution.

Requirements:
1. Output ONLY the answer - no explanations, no extra text
2. Preserve the original answer form whenever possible (especially for fractions / radicals / π / trig)
3. If it's a number, output just the number (e.g., "42" not "The answer is 42")
4. If it's text (names/dates): keep it as-is (do NOT add a year or convert month names to numeric dates)

IMPORTANT - COMPLETE ANSWER (CompleteAnswer/CompleteAnswer):
- For VECTORS: Output the COMPLETE vector, e.g., "(-2, 3, 3)" not just "-2"
- For COMPLEX NUMBERS: Include the imaginary part, e.g., "5-10i" not just "5"
- For COORDINATES: Output ALL coordinates, e.g., "(5*sqrt(2), 5*sqrt(2))" not just "5*sqrt(2)"
- For MULTIPLE SOLUTIONS: List ALL solutions, e.g., "1, -3" not just "-3"
- For EXPRESSIONS: Keep variables, e.g., "500*a^7" not just "500"
- Keep π if the answer involves circles/areas (e.g., "900*pi" or "900π" not "2827.43")

The output should be directly usable as the final answer.""",
        "customization_hints": [
            "Can specify answer format (number, fraction, percentage, etc.)",
            "Can specify precision requirements",
            "Can require specific units or format"
        ],
        "examples": [
            {
                "problem_type": "math",
                "prompt": "Extract the final answer only. Preserve exact forms: keep fractions as a/b, keep sqrt/pi symbolic. Output just the final answer span with no extra words."
            },
            {
                "problem_type": "qa",
                "prompt": "Extract the answer in the most concise form possible. For names, output just the name. For yes/no, output just yes or no."
            }
        ]
    },

    "ScEnsemble": {
        "name": "ScEnsemble",
        "description": "Self-consistency ensemble - select most consistent answer from multiple solutions",
        "input": "solutions (solution list), problem (problem)",
        "output": "response (selected best solution)",
        "use_cases": [
            "Selection after parallel structure",
            "Multi-solution voting",
            "Improve answer reliability"
        ],
        "default_prompt": """Select the most consistent and reliable answer from the given solutions.

Process:
1. Analyze each solution carefully
2. Look for consistency - which answer appears most often or is most well-supported
3. Consider the quality of reasoning in each solution
4. Select the most reliable answer

Choose the solution that is most likely to be correct based on consistency and quality.""",
        "customization_hints": [
            "Can specify selection criteria (consistency, detail level, confidence)",
            "Can require explaining selection reason",
            "Can set consistency threshold"
        ],
        "examples": [
            {
                "problem_type": "math",
                "prompt": "Compare the solutions and select the one with correct calculations. Prioritize: 1) Numerical accuracy 2) Complete reasoning 3) Proper format. Output the letter of the best solution."
            },
            {
                "problem_type": "qa",
                "prompt": "Compare the answers and select the most consistent one. Prioritize: 1) Accuracy with the context 2) Conciseness (1-5 words) 3) Direct answer to the question. Output the letter of the best solution."
            }
        ]
    },

    "MdEnsemble": {
        "name": "MdEnsemble",
        "description": "Multi-dimensional ensemble - multiple voting rounds to select best solution",
        "input": "solutions (solution list), problem (problem)",
        "output": "solution (selected best solution)",
        "use_cases": [
            "High-reliability selection",
            "Important answers requiring multiple verification"
        ],
        "default_prompt": """Evaluate these solutions and select the best one.

Evaluation criteria:
1. Correctness - Is the logic and calculation correct?
2. Completeness - Does it fully answer the question?
3. Clarity - Is the explanation clear?

Select the solution that best meets all criteria.""",
        "customization_hints": [
            "Can adjust evaluation dimension weights",
            "Can add specific evaluation criteria",
            "Can require stricter evaluation"
        ],
        "examples": [
            {
                "problem_type": "math",
                "prompt": "Evaluate solutions by: 1) Calculation correctness 2) Reasoning completeness 3) Answer format. Select the best one."
            },
            {
                "problem_type": "qa",
                "prompt": "Evaluate answers by: 1) Factual accuracy 2) Relevance to question 3) Conciseness (1-5 words). Select the best one."
            }
        ]
    },

    "Test": {
        "name": "Test",
        "description": "Test code and reflect on errors",
        "input": "problem (problem), solution (code solution), entry_point (function name)",
        "output": "result (pass/fail), solution (possibly corrected code)",
        "use_cases": [
            "Code verification",
            "Automatic testing and fixing",
            "Code-type problems"
        ],
        "default_prompt": """Test this code solution and fix any errors.

Process:
1. Analyze the code logic
2. Identify potential issues
3. Test with edge cases mentally
4. Fix any bugs found

Ensure the code is correct and handles all cases properly.""",
        "customization_hints": [
            "Can specify edge cases to test",
            "Can require specific testing strategy",
            "Can specify error handling requirements"
        ],
        "examples": []
    }
}


def get_operator_template(operator_name: str) -> Dict[str, Any]:
    """Get template info for specified operator

    Args:
        operator_name: Operator name

    Returns:
        Dictionary containing template info, returns generic template if not found
    """
    if operator_name in OPERATOR_TEMPLATES:
        return OPERATOR_TEMPLATES[operator_name]

    # Return generic template
    return {
        "name": operator_name,
        "description": f"{operator_name} operator",
        "input": "varies",
        "output": "varies",
        "use_cases": ["General use"],
        "default_prompt": f"""Use the {operator_name} operator to process this input.
Follow the operator's standard behavior and output format.""",
        "customization_hints": [
            "Can customize this operator's behavior as needed"
        ],
        "examples": []
    }


def format_prompt_guidance(operator_name: str, problem_preview: str = "", problem_type: str = "") -> str:
    """Format prompt customization guidance for model

    Args:
        operator_name: Operator name
        problem_preview: Problem preview (optional)
        problem_type: Problem type (optional, math/code/qa)

    Returns:
        Formatted guidance text
    """
    template = get_operator_template(operator_name)
    pt = (problem_type or "").strip().lower()

    if pt == "mathqa_mc":
        pt = "math"

    # Make a shallow copy so per-problem overrides don't mutate global templates.
    template = dict(template)

    # Problem-type-specific guidance (important for code vs math).
    if operator_name == "Programmer":
        if pt == "code":
            template["default_prompt"] = """Write correct, minimal Python code for the given task.

Rules:
1) Follow the REQUIRED signature shown in the problem EXACTLY (function/class name and parameters).
2) Do NOT wrap everything in def solve() unless the problem explicitly asks for solve().
3) Do NOT add input()/print() I/O or any extra explanation; just implement the required API.
4) Handle edge cases and match the expected return type precisely.
5) Keep it simple and deterministic (no randomness, no external deps)."""
            template["customization_hints"] = [
                "Your prompt will be passed to code generation model, so write prompts that guide code writing",
                "WRONG example: 'Step 1: Validate input...' (this is step description, not a prompt)",
                "CORRECT example: 'Implement function xxx, handle empty list edge case'",
                "Emphasize strictly following the function/class signature given in problem (don't change to solve())",
                "Emphasize return type/sort order/case sensitivity/edge cases (empty input, duplicates, etc.)",
            ]
        elif pt == "math":
            # Keep math-oriented solve() guidance (Programmer is used as calculator).
            template["default_prompt"] = """Write Python code to compute the exact answer for this math problem.

Rules:
1) Prefer a pure function style; if no signature is given, define solve() and RETURN the final value.
2) Use sympy for symbolic math when helpful, but convert symbolic results to numbers (float/str) as needed.
3) Avoid input()/print(); return the answer."""
            template["customization_hints"] = [
                "Your prompt will be passed to code generation model, so write prompts that guide code writing",
                "WRONG example: 'Step 1: Calculate...' (this is step description, not a prompt)",
                "CORRECT example: 'Use sympy to solve equation, define solve() to return numerical result'",
                "Emphasize using solve() function and RETURN the result",
                "If symbolic computation needed, suggest using sympy and converting to numerical value",
            ]


    elif operator_name == "Plan":
        if pt == "math":
            template["customization_hints"] = [
                "Choose appropriate math method (algebra, geometry, number theory, etc.)",
                "Pay attention to precision and unit requirements",
                "Consider whether symbolic or numerical computation is needed",
            ]
        elif pt == "code":
            template["customization_hints"] = [
                "Clarify algorithm choice and time complexity requirements",
                "List edge cases to handle",
                "Determine input/output format and function signature",
            ]
        elif pt == "qa":
            template["customization_hints"] = [
                "Locate key information needed for the question",
                "Determine answer type (entity, number, yes/no, etc.)",
                "Plan information extraction and verification steps",
                "Verify the question's key premise exists in the passage before extracting answer",
            ]
    elif operator_name == "Decompose":
        if pt == "math":
            template["customization_hints"] = [
                "Decompose by calculation steps, each step with clear intermediate result",
                "Identify sub-problems that can be computed independently",
            ]
        elif pt == "code":
            template["customization_hints"] = [
                "Decompose by functional modules",
                "Clarify input/output interface for each module",
            ]
        elif pt == "qa":
            template["customization_hints"] = [
                "First check if the question can be answered from the passage",
                "If premise doesn't match passage, question is unanswerable",
            ]
    elif operator_name == "Review":
        if pt == "math":
            template["customization_hints"] = [
                "Focus on checking calculation process and numerical precision",
                "Verify unit consistency",
            ]
        elif pt == "code":
            template["customization_hints"] = [
                "Check edge case handling",
                "Verify return type and function signature",
            ]
        elif pt == "qa":
            template["customization_hints"] = [
                "Check if question premise matches passage content",
                "Ensure the answer is a verbatim span from the passage (no paraphrase)",
                "If premise is wrong, answer should be unanswerable",
            ]
    elif operator_name == "Revise":
        if pt == "math":
            template["customization_hints"] = [
                "Keep correct calculation steps, fix erroneous parts",
            ]
        elif pt == "code":
            template["customization_hints"] = [
                "Keep function signature unchanged, fix logic errors",
            ]
        elif pt == "qa":
            template["customization_hints"] = [
                "If premise was wrong, change answer to unanswerable",
                "If answer was unanswerable but premise is correct, find the real answer",
            ]
    elif operator_name == "Aggregate":
        if pt == "math":
            template["customization_hints"] = [
                "Merge calculation results, ensure unit consistency",
            ]
        elif pt == "code":
            template["customization_hints"] = [
                "Integrate code modules, ensure interface consistency",
            ]
        elif pt == "qa":
            template["customization_hints"] = [
                "Synthesize information from all parts, output concise answer",
            ]
    elif operator_name == "Format":
        if pt == "math":
            template["customization_hints"] = [
                "Extract final numerical result",
                "Preserve necessary precision",
            ]
        elif pt == "code":
            template["customization_hints"] = [
                "Output pure code, no explanatory text",
            ]
        elif pt == "qa":
            template["customization_hints"] = [
                "Copy the exact answer span from the passage, preserve original wording/case",
                "Do not paraphrase or truncate; keep all required words",
                "If no valid answer was found in the passage, output exactly 'unanswerable'",
            ]
    elif operator_name in {"Custom", "Verify", "AnswerGenerate"}:
        # QA task strict short answer constraint: reduce cases where answer is correct but marked wrong due to verbose/irregular output
        if pt == "qa":
            if operator_name == "Custom":
                template["default_prompt"] = """First, check if this question is answerable from the passage:
1. Verify the question's premise matches the passage (numbers, names, dates, events)
2. If the passage lacks required info or premise mismatches, output "unanswerable"
3. If answerable, COPY the exact answer span from the passage (no paraphrase, no truncation)

Return ONLY the exact span or "unanswerable".
"""
                template["customization_hints"] = [
                    "Check premise against passage; if mismatch, output 'unanswerable'",
                    "Copy the exact answer span (preserve wording/case/number format)",
                    "Output only the span itself, no explanation",
                ]
            elif operator_name == "Verify":
                template["default_prompt"] = """Verify whether the proposed answer is correct.

CRITICAL OUTPUT RULES (QA):
- In <answer>, output ONLY the exact answer span from the passage.
- Do NOT paraphrase or normalize numbers; preserve original wording/case.
- Keep all required words even if the span is longer than 5 words.
- If unanswerable, output ONLY "unanswerable".
"""
                template["customization_hints"] = [
                    "Ensure <answer> is a verbatim span from passage, not a paraphrase",
                    "Don't put verification process in <answer>; keep it in verification_steps",
                ]
            else:  # AnswerGenerate
                template["default_prompt"] = """Solve the question.

CRITICAL OUTPUT RULES (QA):
- Put reasoning in <thought>.
- In <answer>, output ONLY the exact answer span from the passage.
- Do NOT paraphrase or normalize numbers; preserve original wording/case.
- Keep all required words even if the span is longer than 5 words.
- If unanswerable, output ONLY "unanswerable".
"""
                template["customization_hints"] = [
                    "<answer> must be the exact span from passage, not a paraphrase",
                    "If explanation needed, put it in <thought>, keep <answer> clean",
                ]

    lines = []
    lines.append(f"## {template['name']} Operator")
    lines.append(f"**Description**: {template['description']}")
    lines.append(f"**Input**: {template['input']}")
    lines.append(f"**Output**: {template['output']}")

    lines.append("")
    lines.append("### Use Cases")
    for use_case in template.get('use_cases', []):
        lines.append(f"- {use_case}")

    lines.append("")
    lines.append("### Default Prompt Template")
    lines.append("```")
    lines.append(template['default_prompt'])
    lines.append("```")

    lines.append("")
    lines.append("### Customization Tips")
    for hint in template.get('customization_hints', []):
        lines.append(f"- {hint}")

    if problem_preview:
        lines.append("")
        lines.append("### Problem Context")
        lines.append(f"Current problem: {problem_preview[:200]}...")
        lines.append("")
        lines.append("Consider how to customize the prompt for THIS specific problem.")

    examples = template.get('examples', [])
    if examples:
        lines.append("")
        lines.append("### Example Customizations")
        for i, ex in enumerate(examples, 1):
            lines.append(f"**Example {i}** ({ex.get('problem_type', 'general')}):")
            lines.append(f"```")
            lines.append(ex.get('prompt', ''))
            lines.append(f"```")

    lines.append("")
    lines.append("---")
    lines.append("Now write YOUR custom prompt for this operator.")
    lines.append("IMPORTANT: Just write the prompt text directly (no XML/action tags).")

    return "\n".join(lines)


# Export
__all__ = [
    'OPERATOR_TEMPLATES',
    'get_operator_template',
    'format_prompt_guidance'
]
