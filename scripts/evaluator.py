#!/usr/bin/env python3
"""
AFlowevaluator
DatasetType
"""
from enum import Enum
from typing import Optional


class DatasetType(Enum):
    """"""
    GSM8K = "gsm8k"
    MATH = "math"
    HOTPOTQA = "hotpotqa"
    HUMANEVAL = "humaneval"
    MBPP = "mbpp"
    COMMONSENSEQA = "commonsenseqa"
    MMLU = "mmlu"
    SQUAD_V2 = "squad_v2"
    DROP = "drop"
    CUSTOM = "custom"

    @classmethod
    def from_string(cls, name: str) -> 'DatasetType':
        """"""
        name_lower = name.lower().replace('-', '_').replace(' ', '_')
        for dataset in cls:
            if dataset.value == name_lower:
                return dataset
        return cls.CUSTOM

    @classmethod
    def get_problem_type(cls, dataset_type: 'DatasetType') -> str:
        """"""
        math_datasets = {cls.GSM8K, cls.MATH}
        code_datasets = {cls.HUMANEVAL, cls.MBPP}
        qa_datasets = {cls.HOTPOTQA, cls.COMMONSENSEQA, cls.MMLU, cls.SQUAD_V2, cls.DROP}

        if dataset_type in math_datasets:
            return "math"
        elif dataset_type in code_datasets:
            return "code"
        elif dataset_type in qa_datasets:
            return "qa"
        else:
            return "unknown"


def get_dataset_type_from_source(source: str) -> DatasetType:
    """sourceDatasetType"""
    return DatasetType.from_string(source)


def evaluate_answer(
    prediction: str,
    ground_truth: str,
    dataset_type: DatasetType = DatasetType.CUSTOM
) -> float:
    """

    Args:
        prediction: 
        ground_truth: 
        dataset_type: 

    Returns:
         (0.0  1.0)
    """
    pred = str(prediction).strip().lower()
    truth = str(ground_truth).strip().lower()

    if pred == truth:
        return 1.0

    if truth in pred or pred in truth:
        return 0.7

    return 0.0
