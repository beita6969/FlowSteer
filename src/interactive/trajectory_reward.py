from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import math
import re

from .workflow_builder import Trajectory


@dataclass
class TrajectoryRewardResult:
    total_reward: float
    base_reward: float
    structure_reward: float
    correctness_reward: float
    checker_score: float
    format_score: float
    operator_score: float
    special_structure_score: float
    correctness_activated: bool
    total_turns: int
    total_operators: int
    has_checker: bool
    has_format: bool
    has_special_structure: bool
    token_rewards: Optional[List[float]] = None
    reward_position: Optional[int] = None
    complexity_analysis: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_reward": self.total_reward,
            "base_reward": self.base_reward,
            "structure_reward": self.structure_reward,
            "correctness_reward": self.correctness_reward,
            "checker_score": self.checker_score,
            "format_score": self.format_score,
            "operator_score": self.operator_score,
            "special_structure_score": self.special_structure_score,
            "correctness_activated": self.correctness_activated,
            "total_turns": self.total_turns,
            "total_operators": self.total_operators,
            "has_checker": self.has_checker,
            "has_format": self.has_format,
            "has_special_structure": self.has_special_structure,
            "reward_position": self.reward_position,
            "complexity_analysis": self.complexity_analysis,
        }


@dataclass
class EfficiencyConfig:
    optimal_min: int = 1
    optimal_max: int = 100
    max_turns: int = 100
    optimal_bonus: float = 0.0
    too_few_penalty: float = 0.0
    too_many_penalty: float = 0.0
    penalty_per_extra_turn: float = 0.0


class TrajectoryRewardCalculator:

    CHECKER_SCORE = 0.2
    FORMAT_SCORE = 0.2
    OPERATOR_SCORE = 0.2
    SPECIAL_STRUCTURE_SCORE = 0.4
    STRUCTURE_CAP = 1.0
    BASE_REWARD = -1.0

    CHECKER_OPERATORS = {'Verify', 'Test', 'Review'}

    ALL_OPERATORS = {
        'Plan', 'Decompose', 'Programmer', 'Custom', 'Test',
        'Review', 'Verify', 'Revise', 'ScEnsemble', 'Aggregate', 'AnswerGenerate', 'Format'
    }

    def __init__(
        self,
        base_reward: float = -1.0,
        checker_score: float = 0.2,
        format_score: float = 0.2,
        operator_score: float = 0.2,
        special_structure_score: float = 0.4,
        min_operators_for_score: int = 6,
        structure_cap: float = 1.0,
        correctness_activation_threshold: float = 0.0,
        min_operators_for_correctness: int = 5,
        require_special_structure: bool = True,
        simple_workflow_penalty: float = 0.0,
        efficiency_config: Optional[EfficiencyConfig] = None,
        debug: bool = False,
    ):
        self.base_reward = base_reward
        self.checker_score = checker_score
        self.format_score = format_score
        self.operator_score = operator_score
        self.special_structure_score = special_structure_score
        self.min_operators_for_score = min_operators_for_score
        self.structure_cap = structure_cap
        self.debug = debug
        self.correctness_activation_threshold = correctness_activation_threshold
        self.efficiency_config = efficiency_config or EfficiencyConfig()

    def _analyze_dsl(self, dsl: str) -> Dict[str, Any]:
        if not dsl:
            return {
                'total_operators': 0,
                'unique_operators': set(),
                'has_checker': False,
                'has_format': False,
                'has_parallel': False,
                'has_conditional': False,
                'has_loop': False,
                'has_special_structure': False,
            }

        operators = re.findall(
            r'\b(' + '|'.join(self.ALL_OPERATORS) + r')\b',
            dsl
        )

        unique_operators = set(operators)
        total_operators = len(operators)
        has_checker = bool(unique_operators & self.CHECKER_OPERATORS)
        has_format = 'Format' in unique_operators
        has_parallel = '[' in dsl and ']' in dsl
        has_conditional = '?' in dsl and ':' in dsl
        has_loop = bool(re.search(r'x\s*\d+', dsl))
        has_special_structure = has_parallel or has_conditional or has_loop

        return {
            'total_operators': total_operators,
            'unique_operators': unique_operators,
            'has_checker': has_checker,
            'has_format': has_format,
            'has_parallel': has_parallel,
            'has_conditional': has_conditional,
            'has_loop': has_loop,
            'has_special_structure': has_special_structure,
        }

    def compute_reward(
        self,
        trajectory: Trajectory,
        correctness: float = 0.0,
        ground_truth: Optional[str] = None,
        difficulty: str = "easy",
        domain: str = "math",
    ) -> TrajectoryRewardResult:
        final_dsl = trajectory.final_dsl
        total_turns = trajectory.total_rounds

        analysis = self._analyze_dsl(final_dsl)

        checker_score = self.checker_score if analysis['has_checker'] else 0.0
        format_score = self.format_score if analysis['has_format'] else 0.0
        operator_score = self.operator_score if analysis['total_operators'] >= self.min_operators_for_score else 0.0
        special_structure_score = self.special_structure_score if analysis['has_special_structure'] else 0.0

        structure_reward = min(
            self.structure_cap,
            checker_score + format_score + operator_score + special_structure_score
        )

        correctness_activated = (structure_reward >= self.structure_cap)

        if correctness_activated:
            correctness_reward = correctness
        else:
            correctness_reward = 0.0

        total_reward = self.base_reward + structure_reward + correctness_reward

        token_rewards = None
        reward_position = None
        if trajectory.action_mask:
            token_rewards, reward_position = self._assign_token_rewards(
                trajectory.action_mask,
                total_reward
            )

        trajectory.reward = total_reward
        trajectory.reward_breakdown = {
            "base": self.base_reward,
            "structure": structure_reward,
            "correctness": correctness_reward,
            "checker_score": checker_score,
            "format_score": format_score,
            "operator_score": operator_score,
            "special_structure_score": special_structure_score,
        }

        result = TrajectoryRewardResult(
            total_reward=total_reward,
            base_reward=self.base_reward,
            structure_reward=structure_reward,
            correctness_reward=correctness_reward,
            checker_score=checker_score,
            format_score=format_score,
            operator_score=operator_score,
            special_structure_score=special_structure_score,
            correctness_activated=correctness_activated,
            total_turns=total_turns,
            total_operators=analysis['total_operators'],
            has_checker=analysis['has_checker'],
            has_format=analysis['has_format'],
            has_special_structure=analysis['has_special_structure'],
            token_rewards=token_rewards,
            reward_position=reward_position,
            complexity_analysis={
                'unique_operators': list(analysis['unique_operators']),
                'total_operator_count': analysis['total_operators'],
                'has_parallel': analysis['has_parallel'],
                'has_conditional': analysis['has_conditional'],
                'has_loop': analysis['has_loop'],
                'has_checker': analysis['has_checker'],
                'has_format': analysis['has_format'],
                'has_special_structure': analysis['has_special_structure'],
            },
        )

        if self.debug:
            self._print_debug(result, final_dsl)

        return result

    def _assign_token_rewards(
        self,
        action_mask: List[int],
        total_reward: float
    ) -> Tuple[List[float], int]:
        token_rewards = [0.0] * len(action_mask)

        last_model_token_pos = -1
        for i in range(len(action_mask) - 1, -1, -1):
            if action_mask[i] == 1:
                last_model_token_pos = i
                break

        if last_model_token_pos >= 0:
            token_rewards[last_model_token_pos] = total_reward

        return token_rewards, last_model_token_pos

    def compute_batch_rewards(
        self,
        trajectories: List[Trajectory],
        correctness_scores: List[float],
        difficulties: Optional[List[str]] = None,
        domains: Optional[List[str]] = None,
    ) -> Tuple[List[float], List[TrajectoryRewardResult]]:
        n = len(trajectories)
        difficulties = difficulties or ["easy"] * n
        domains = domains or ["math"] * n

        rewards = []
        results = []

        for i in range(n):
            result = self.compute_reward(
                trajectory=trajectories[i],
                correctness=correctness_scores[i],
                difficulty=difficulties[i],
                domain=domains[i],
            )
            rewards.append(result.total_reward)
            results.append(result)

        return rewards, results

    def compute_group_advantages(
        self,
        rewards: List[float],
        min_std: float = 0.01
    ) -> List[float]:
        if len(rewards) == 0:
            return []

        if len(rewards) == 1:
            return [0.0]

        mean_r = sum(rewards) / len(rewards)
        variance = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
        std_r = max(math.sqrt(variance), min_std)

        advantages = [(r - mean_r) / std_r for r in rewards]
        return advantages

    def _print_debug(self, result: TrajectoryRewardResult, dsl: str):
        print("\n" + "=" * 60)
        print("TrajectoryRewardCalculator Debug")
        print("=" * 60)
        print(f"DSL: {dsl}")
        print(f"Total Reward: {result.total_reward:.4f}")
        print(f"  Base: {result.base_reward:.4f}")
        print(f"  Structure: {result.structure_reward:.4f}")
        print(f"    - Checker: {result.checker_score:.2f} (has_checker={result.has_checker})")
        print(f"    - Format: {result.format_score:.2f} (has_format={result.has_format})")
        print(f"    - Operator: {result.operator_score:.2f} (count={result.total_operators}, need>=6)")
        print(f"    - Special: {result.special_structure_score:.2f} (has_special={result.has_special_structure})")
        print(f"  Correctness: {result.correctness_reward:.4f} (activated={result.correctness_activated})")
        print(f"  Total Turns: {result.total_turns}")
        print("=" * 60)


def create_reward_calculator(
    base_reward: float = -1.0,
    correctness_threshold: float = 0.0,
    optimal_turns: Tuple[int, int] = (1, 100),
    efficiency: Optional[Dict[str, Any]] = None,
    debug: bool = False,
) -> TrajectoryRewardCalculator:
    return TrajectoryRewardCalculator(
        base_reward=base_reward,
        debug=debug,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("TrajectoryRewardCalculator Test")
    print("=" * 60)

    from .workflow_builder import Trajectory, TurnRecord

    calculator = TrajectoryRewardCalculator(debug=True)

    test_cases = [
        ("Custom", 1.0, 0.0, False),
        ("Plan -> Verify", 1.0, 0.2, False),
        ("Plan -> Verify -> Format", 1.0, 0.4, False),
        ("Plan -> [Programmer, Custom] -> Verify -> Format", 1.0, 0.8, False),
        ("Plan -> Decompose -> [Programmer, Custom] -> Aggregate -> Verify -> Format", 1.0, 1.0, True),
        ("Plan -> Decompose -> Programmer -> Custom -> Test -> Verify -> Format", 1.0, 0.6, False),
        ("Plan -> Decompose -> [Programmer, Custom] -> Test -> Verify -> Format", 1.0, 1.0, True),
    ]

    print("\n--- Test DSL Structures ---")
    for dsl, correctness, expected_structure, expected_activated in test_cases:
        trajectory = Trajectory(
            problem="Test problem",
            problem_type="math",
        )
        trajectory.final_dsl = dsl
        trajectory.total_rounds = 5

        result = calculator.compute_reward(
            trajectory=trajectory,
            correctness=correctness,
        )

        status = "✓" if (abs(result.structure_reward - expected_structure) < 0.01 and
                         result.correctness_activated == expected_activated) else "✗"

        print(f"\n{status} DSL: {dsl}")
        print(f"   Structure: {result.structure_reward:.2f} (expected: {expected_structure:.2f})")
        print(f"   Activated: {result.correctness_activated} (expected: {expected_activated})")
        print(f"   Total: {result.total_reward:.2f}")

    print("\n" + "=" * 60)
    print("Test Complete!")
