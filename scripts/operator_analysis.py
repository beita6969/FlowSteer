# -*- coding: utf-8 -*-
# @Desc: Pydantic models for operator responses
# Pydantic models for operator responses (adapted from AFlow)

from typing import List
from pydantic import BaseModel, Field


class GenerateOp(BaseModel):
    """Generic generation response - used by Custom operator"""
    response: str = Field(default="", description="Your solution for this problem")


class CodeGenerateOp(BaseModel):
    """Code generation response - used by Programmer operator"""
    code: str = Field(default="", description="Your complete code solution for this problem")


class AnswerGenerateOp(BaseModel):
    """Structured answer with reasoning - used by AnswerGenerate operator"""
    thought: str = Field(default="", description="The step by step thinking process")
    answer: str = Field(default="", description="The final answer to the question")


class FormatOp(BaseModel):
    """Formatted answer extraction - used by Format operator"""
    solution: str = Field(default="", description="Your formatted answer for this problem")


class ScEnsembleOp(BaseModel):
    """Self-consistency ensemble response - used by ScEnsemble operator"""
    thought: str = Field(default="", description="The thought of the most consistent solution.")
    solution_letter: str = Field(default="", description="The letter of most consistent solution.")


class ReflectionTestOp(BaseModel):
    """Reflection and correction response - used by Test operator"""
    reflection_and_solution: str = Field(
        default="",
        description="Corrective solution for code execution errors or test case failures"
    )


class MdEnsembleOp(BaseModel):
    """Majority voting ensemble response - used by MdEnsemble operator"""
    thought: str = Field(default="", description="Step-by-step analysis of the solutions to determine the best one.")
    solution_letter: str = Field(default="", description="The letter of the chosen best solution (only one letter).")


class ReviewOp(BaseModel):
    """Solution review response - used by Review operator"""
    review_result: bool = Field(
        default=False,
        description="The Review Result (Bool). If you think this solution looks good for you, return 'true'; If not, return 'false'",
    )
    feedback: str = Field(
        default="",
        description="Your FeedBack for this problem based on the criteria. If the review result is true, you can put it 'nothing here'.",
    )


class ReviseOp(BaseModel):
    """Solution revision response - used by Revise operator"""
    solution: str = Field(default="", description="Based on the feedback, revised solution for this problem")


# ============================================
# ============================================

class DecomposeOp(BaseModel):
    """Problem decomposition response - used by Decompose operator"""
    sub_problems: str = Field(
        default="",
        description="A numbered list of sub-problems that break down the original problem into simpler steps"
    )
    reasoning: str = Field(
        default="",
        description="Explanation of how the sub-problems relate to solving the original problem"
    )
    sub_problems_list: List[str] = Field(
        default_factory=list,
        description="Parsed list of sub-problems for dynamic workflow execution"
    )


class VerifyOp(BaseModel):
    """Answer verification response - used by Verify operator"""
    is_correct: bool = Field(
        default=False,
        description="Whether the answer is verified to be correct (true/false)"
    )
    verification_steps: str = Field(
        default="",
        description="Step-by-step verification process showing how the answer was checked"
    )
    confidence: str = Field(
        default="medium",
        description="Confidence level: high, medium, or low"
    )
    answer: str = Field(
        default="",
        description="The verified correct answer (concise, final-answer-only)"
    )


class PlanOp(BaseModel):
    """Solution planning response - used by Plan operator"""
    plan: str = Field(
        default="",
        description="A structured plan with numbered steps for solving the problem"
    )
    approach: str = Field(
        default="",
        description="The overall approach or strategy being used"
    )
    key_insights: str = Field(
        default="",
        description="Key insights or observations that inform the solution"
    )
    steps_list: List[str] = Field(
        default_factory=list,
        description="Parsed list of steps for dynamic workflow execution"
    )


class AggregateOp(BaseModel):
    """Multi-answer aggregation response - used by Aggregate operator"""
    aggregated_answer: str = Field(
        default="",
        description="The final aggregated answer combining all sub-answers"
    )
    synthesis: str = Field(
        default="",
        description="Explanation of how the sub-answers were combined"
    )


# Export all models
__all__ = [
    'GenerateOp',
    'CodeGenerateOp',
    'AnswerGenerateOp',
    'FormatOp',
    'ScEnsembleOp',
    'ReflectionTestOp',
    'MdEnsembleOp',
    'ReviewOp',
    'ReviseOp',
    'DecomposeOp',
    'VerifyOp',
    'PlanOp',
    'AggregateOp'
]
