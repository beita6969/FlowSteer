#!/usr/bin/env python3
"""
GRPO

:
- Step = 48 = 48 workflow = 6GRPO
- GRPO = 8(domain+difficulty)workflow
- advantage

:
- math_easy:  8
- math_hard:  8
- code_easy:  8
- code_hard:  8
- qa_easy:    8
- qa_hard:    8
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class CrossProblemSampler:
    """
    GRPO

     domain + difficulty 8
    """

    SUBSETS = [
        ("math", "easy"),   # GSM8K
        ("math", "hard"),   # MATH
        ("code", "easy"),   # code_exercises
        ("code", "hard"),   # BigCodeBench
        ("qa", "easy"),     # SQuAD v2
        ("qa", "hard"),     # HotpotQA
    ]

    def __init__(
        self,
        train_dataset_path: str,
        test_dataset_path: str,
        samples_per_group: int = 8
    ):
        """
        Args:
            train_dataset_path: JSONL
            test_dataset_path: JSONL
            samples_per_group: GRPO8
        """
        self.train_path = Path(train_dataset_path)
        self.test_path = Path(test_dataset_path)
        self.samples_per_group = samples_per_group

        self.train_data = defaultdict(list)  # {(domain, difficulty): [samples]}
        self.test_data = defaultdict(list)

        self.train_indices = defaultdict(list)  # {(domain, difficulty): [indices]}

        self._load_data()

    def _load_data(self):
        """"""
        print(f"\n📂 CrossProblemSampler ")
        print(f"  : {self.train_path}")
        print(f"  : {self.test_path}")
        print(f"  : {self.samples_per_group}")

        if self.train_path.exists():
            self._load_jsonl(self.train_path, self.train_data, "train")
        else:
            print(f"  ⚠️ : {self.train_path}")

        if self.test_path.exists():
            self._load_jsonl(self.test_path, self.test_data, "test")
        else:
            print(f"  ⚠️ : {self.test_path}")

        self._reset_indices()

        self._print_stats()

    def _load_jsonl(self, path: Path, data_dict: dict, split: str):
        """JSONL"""
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())

                domain = sample.get('_domain', sample.get('problem_type', 'unknown'))
                difficulty = sample.get('_difficulty', sample.get('difficulty', 'easy'))

                domain = self._normalize_domain(domain)
                difficulty = self._normalize_difficulty(difficulty)

                key = (domain, difficulty)
                data_dict[key].append(sample)

    def _normalize_domain(self, domain: str) -> str:
        """domain"""
        domain = domain.lower()
        if domain in ['math', 'mathematics']:
            return 'math'
        elif domain in ['code', 'coding', 'programming']:
            return 'code'
        elif domain in ['qa', 'question_answering', 'reading']:
            return 'qa'
        return domain

    def _normalize_difficulty(self, difficulty: str) -> str:
        """difficulty"""
        difficulty = difficulty.lower()
        if difficulty in ['easy', 'simple', 'basic']:
            return 'easy'
        elif difficulty in ['hard', 'difficult', 'challenging', 'medium']:
            return 'hard'
        return difficulty

    def _reset_indices(self):
        """epoch"""
        self.train_indices.clear()
        for key, samples in self.train_data.items():
            indices = list(range(len(samples)))
            random.shuffle(indices)
            self.train_indices[key] = indices

    def _print_stats(self):
        """"""
        print(f"\n📊 :")
        print(f"  {'':<15} {'':<10} {'':<10}")
        print(f"  {'-'*35}")

        total_train = 0
        total_test = 0

        for domain, difficulty in self.SUBSETS:
            key = (domain, difficulty)
            train_count = len(self.train_data.get(key, []))
            test_count = len(self.test_data.get(key, []))
            total_train += train_count
            total_test += test_count

            subset_name = f"{domain}_{difficulty}"
            print(f"  {subset_name:<15} {train_count:<10} {test_count:<10}")

        print(f"  {'-'*35}")
        print(f"  {'':<15} {total_train:<10} {total_test:<10}")
        print(f"\n  : {len(self.SUBSETS)} × {self.samples_per_group} = {len(self.SUBSETS) * self.samples_per_group}")

    def sample_step_batch(self, split: str = "train") -> Dict[Tuple[str, str], List[dict]]:
        """
        Step

        Args:
            split: "train"  "test"

        Returns:
            {(domain, difficulty): [8]} 
        """
        data_source = self.train_data if split == "train" else self.test_data

        step_batch = {}

        for domain, difficulty in self.SUBSETS:
            key = (domain, difficulty)
            samples = data_source.get(key, [])

            if len(samples) == 0:
                print(f"  ⚠️  {key} ")
                continue

            if split == "train":
                if len(self.train_indices[key]) < self.samples_per_group:
                    indices = list(range(len(samples)))
                    random.shuffle(indices)
                    self.train_indices[key] = indices

                selected_indices = self.train_indices[key][:self.samples_per_group]
                self.train_indices[key] = self.train_indices[key][self.samples_per_group:]

                step_batch[key] = [samples[i] for i in selected_indices]
            else:
                step_batch[key] = random.sample(samples, min(self.samples_per_group, len(samples)))

        return step_batch

    def flatten_batch(self, step_batch: Dict[Tuple[str, str], List[dict]]) -> List[dict]:
        """
        batch

        Args:
            step_batch: sample_step_batch

        Returns:
            _grpo_group_id
        """
        flat_list = []

        for group_idx, ((domain, difficulty), samples) in enumerate(step_batch.items()):
            for sample in samples:
                sample_with_group = sample.copy()
                sample_with_group['_grpo_group_id'] = group_idx
                sample_with_group['_grpo_group_key'] = f"{domain}_{difficulty}"
                flat_list.append(sample_with_group)

        return flat_list

    def get_group_keys(self) -> List[Tuple[str, str]]:
        """"""
        return self.SUBSETS.copy()

    def get_step_stats(self, step_batch: Dict[Tuple[str, str], List[dict]]) -> Dict:
        """"""
        stats = {
            "total_samples": sum(len(samples) for samples in step_batch.values()),
            "num_groups": len(step_batch),
            "groups": {}
        }

        for (domain, difficulty), samples in step_batch.items():
            key_str = f"{domain}_{difficulty}"
            stats["groups"][key_str] = len(samples)

        return stats


if __name__ == "__main__":
    sampler = CrossProblemSampler(
        train_dataset_path="data/train_balanced_12k_humaneval36_fixed.jsonl",
        test_dataset_path="data/test_balanced_768_no_overlap.jsonl",
        samples_per_group=8
    )

    print("\n=== Test Sampling ===")
    step_batch = sampler.sample_step_batch("train")
    stats = sampler.get_step_stats(step_batch)
    print(f"Step stats: {stats}")

    flat = sampler.flatten_batch(step_batch)
    print(f": {len(flat)}")

    for i, sample in enumerate(flat[:3]):
        print(f"  {i}: group_id={sample['_grpo_group_id']}, group_key={sample['_grpo_group_key']}")
