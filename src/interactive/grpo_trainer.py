import os
import gc
import torch
import torch.nn.functional as F
import asyncio
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from tqdm import tqdm
import time
import json
import logging

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from .workflow_builder import (
    InteractiveWorkflowBuilder,
    BatchWorkflowBuilder,
    Trajectory,
    TurnRecord,
    merge_trajectory_masks,
    create_aflow_executor_wrapper,
)
from .trajectory_reward import (
    TrajectoryRewardCalculator,
    TrajectoryRewardResult,
    EfficiencyConfig,
)

try:
    from ..cross_problem_sampler import CrossProblemSampler
    from ..aflow_executor import AFlowExecutor
    from ..reward_computer import RewardComputer
except ImportError:
    from src.cross_problem_sampler import CrossProblemSampler
    from src.aflow_executor import AFlowExecutor
    from src.reward_computer import RewardComputer


# ============================================
# ============================================

@dataclass
class InteractiveGRPOConfig:
    """GRPO"""
    exp_name: str = "interactive_grpo"
    output_dir: str = "checkpoints/interactive"
    log_dir: str = "logs"

    max_steps: int = 100
    save_every: int = 20
    eval_every: int = 10
    log_every: int = 1

    max_rounds: int = 15
    execute_each_step: bool = True

    samples_per_group: int = 8
    clip_range: float = 0.2        # PPO clip range
    kl_coef: float = 0.005
    entropy_coef: float = 0.005

    base_reward: float = -1.0
    correctness_activation_threshold: float = 0.6

    efficiency_optimal_min: int = 3
    efficiency_optimal_max: int = 30
    efficiency_optimal_bonus: float = 0.1
    efficiency_too_many_penalty: float = 0.0

    base_model: str = ""
    use_lora: bool = True
    lora_rank: int = 64
    lora_alpha: int = 64
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj"
    lora_dropout: float = 0.05

    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 10

    temperature: float = 0.0
    max_new_tokens: int = 512

    use_vllm_api: bool = False
    vllm_base_url: str = "http://localhost:8003/v1"

    device: str = "cuda:0"
    bf16: bool = True


# ============================================
# ============================================

class InteractiveGRPOTrainer:
    """
    GRPO

    :
    1. workflow ()
    2. action_mask/token
    3. GRPOadvantage
    4. 
    """

    def __init__(self, config: Optional[InteractiveGRPOConfig] = None, config_path: Optional[str] = None):
        """

        Args:
            config: InteractiveGRPOConfig 
            config_path: YAML (config)
        """
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = self._load_config(config_path)
        else:
            self.config = InteractiveGRPOConfig()

        print("=" * 60)
        print("🚀 Interactive GRPO Trainer ")
        print("=" * 60)

        self._setup_logging()

        self._initialize_components()

        self.global_step = 0
        self.best_accuracy = 0.0

        print("=" * 60)
        print("✅ Interactive GRPO Trainer ")
        print("=" * 60)

    def _load_config(self, config_path: str) -> InteractiveGRPOConfig:
        """YAML"""
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)

        interactive_config = yaml_config.get('interactive_grpo', {})

        merged = {**yaml_config, **interactive_config}

        config = InteractiveGRPOConfig()
        for key, value in merged.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config

    def _setup_logging(self):
        """"""
        log_dir = Path(self.config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger("InteractiveGRPO")
        self.logger.setLevel(logging.INFO)

        fh = logging.FileHandler(log_dir / "interactive_grpo.log")
        fh.setLevel(logging.DEBUG)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def _initialize_components(self):
        """"""

        print("\n🤖 ...")
        self._load_model()

        print("\n⚙️  AFlow...")
        self.executor = None

        print("\n🎯 ...")
        efficiency_config = EfficiencyConfig(
            optimal_min=self.config.efficiency_optimal_min,
            optimal_max=self.config.efficiency_optimal_max,
            optimal_bonus=self.config.efficiency_optimal_bonus,
            too_many_penalty=self.config.efficiency_too_many_penalty,
        )
        self.reward_calculator = TrajectoryRewardCalculator(
            base_reward=self.config.base_reward,
            correctness_activation_threshold=self.config.correctness_activation_threshold,
            efficiency_config=efficiency_config,
        )

        print("\n🔬 ...")
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.config.max_steps,
        )

        print(f"  : {self.config.learning_rate}")
        print(f"  Warmup: {self.config.warmup_steps}")

    def _load_model(self):
        """"""
        device = self.config.device

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Model
        dtype = torch.bfloat16 if self.config.bf16 else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            torch_dtype=dtype,
            device_map={"": device},
            trust_remote_code=True,
        )

        # LoRA
        if self.config.use_lora:
            lora_config = LoraConfig(
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                target_modules=self.config.lora_target_modules.split(','),
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_config)
            print("✅ LoRA")
            self.model.print_trainable_parameters()

    def _create_generate_fn(self) -> Callable[[str], str]:
        """InteractiveWorkflowBuilder"""

        def generate_fn(prompt: str) -> str:
            """"""
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096,
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

            return response

        return generate_fn

    async def run_interactive_loop(
        self,
        problem: str,
        problem_type: str = "math",
        ground_truth: str = "",
        use_two_step_interaction: bool = True,
        dataset: str = "",
        simple_feedback: bool = True,
        **kwargs,
    ) -> Tuple[Trajectory, float]:
        """

        Args:
            problem: 
            problem_type: 
            ground_truth: 
            use_two_step_interaction: False
            dataset: 
            simple_feedback: feedbackTrue
            **kwargs:  test_type, test_inputs, test_outputs

        Returns:
            (Trajectory, correctness_score)
        """
        executor_fn = None
        if self.config.execute_each_step and self.executor is not None:

            executor_fn = create_aflow_executor_wrapper(
                self.executor,
                problem_type,
            )

        builder = InteractiveWorkflowBuilder(
            generate_fn=self._create_generate_fn(),
            tokenizer=self.tokenizer,
            executor_fn=executor_fn,
            max_rounds=self.config.max_rounds,
            execute_each_step=self.config.execute_each_step,
            verbose=False,
        )

        trajectory = builder.run_loop(
            problem=problem,
            problem_type=problem_type,
            use_two_step_interaction=use_two_step_interaction,
            dataset=dataset,
            simple_feedback=simple_feedback,
            **kwargs,
        )

        correctness = 0.0
        if trajectory.final_answer and ground_truth:
            if str(trajectory.final_answer).strip() == str(ground_truth).strip():
                correctness = 1.0
            elif str(ground_truth).lower() in str(trajectory.final_answer).lower():
                correctness = 0.5

        return trajectory, correctness

    def compute_grpo_loss(
        self,
        trajectories: List[Trajectory],
        rewards: List[TrajectoryRewardResult],
    ) -> torch.Tensor:
        """
        GRPO

        action_masktoken

        Args:
            trajectories: 
            rewards: 

        Returns:
            GRPO
        """
        reward_values = [r.total_reward for r in rewards]

        mean_reward = np.mean(reward_values)
        std_reward = np.std(reward_values) + 1e-8
        advantages = [(r - mean_reward) / std_reward for r in reward_values]

        total_loss = 0.0
        num_samples = 0

        for traj, reward_result, adv in zip(trajectories, rewards, advantages):
            action_mask = traj.action_mask
            if not action_mask:
                continue

            all_text = ""
            for turn in traj.turns:
                all_text += turn.model_response
                if turn.feedback:
                    all_text += "\n" + turn.feedback + "\n"

            # Tokenize
            inputs = self.tokenizer(
                all_text,
                return_tensors="pt",
                truncation=True,
                max_length=4096,
            ).to(self.model.device)

            # Forward pass
            outputs = self.model(**inputs, labels=inputs['input_ids'])

            logits = outputs.logits[:, :-1, :]  # (1, seq_len-1, vocab_size)
            labels = inputs['input_ids'][:, 1:]  # (1, seq_len-1)

            log_probs = F.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # (1, seq_len-1)

            mask_tensor = torch.tensor(action_mask[:token_log_probs.shape[1]],
                                       device=token_log_probs.device, dtype=torch.float32)

            masked_log_probs = token_log_probs * mask_tensor

            sample_loss = -adv * masked_log_probs.sum()

            total_loss += sample_loss
            num_samples += mask_tensor.sum().item()

        if num_samples > 0:
            total_loss = total_loss / num_samples

        return total_loss

    async def train_step(self, step: int, problems: List[Dict]) -> Dict:
        """

        Args:
            step: 
            problems:  [{problem, problem_type, ground_truth}, ...]

        Returns:
        """
        self.model.train()

        print(f"\n{'='*60}")
        print(f"📦 Step {step}: {len(problems)} ")
        print(f"{'='*60}")

        trajectories = []
        correctness_scores = []

        for i, prob in enumerate(tqdm(problems, desc="Running interactive loops")):
            traj, correctness = await self.run_interactive_loop(
                problem=prob['problem'],
                problem_type=prob.get('problem_type', 'math'),
                ground_truth=prob.get('ground_truth', ''),
                use_two_step_interaction=prob.get('use_two_step_interaction', True),
                dataset=prob.get('source', prob.get('dataset', '')),
                simple_feedback=prob.get('simple_feedback', True),
                test_type=prob.get('test_type'),
                test_inputs=prob.get('test_inputs'),
                test_outputs=prob.get('test_outputs'),
            )
            trajectories.append(traj)
            correctness_scores.append(correctness)

            print(f"\n[{i+1}/{len(problems)}] {prob.get('problem_type', 'math')}")
            print(f"  : {traj.total_rounds}")
            print(f"  DSL: {traj.final_dsl}")
            print(f"  : {correctness:.2f}")

        reward_results = []
        for traj, correctness in zip(trajectories, correctness_scores):
            result = self.reward_calculator.compute_reward(
                trajectory=traj,
                correctness=correctness,
            )
            reward_results.append(result)

        loss = self.compute_grpo_loss(trajectories, reward_results)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm,
        )

        if (step + 1) % self.config.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        avg_reward = np.mean([r.total_reward for r in reward_results])
        avg_turns = np.mean([t.total_rounds for t in trajectories])
        avg_correctness = np.mean(correctness_scores)

        stats = {
            'step': step,
            'loss': loss.item(),
            'avg_reward': avg_reward,
            'avg_turns': avg_turns,
            'avg_correctness': avg_correctness,
            'num_samples': len(problems),
        }

        print(f"\n📊 Step {step} :")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Avg Reward: {avg_reward:.4f}")
        print(f"  Avg Turns: {avg_turns:.2f}")
        print(f"  Avg Correctness: {avg_correctness:.2%}")

        return stats

    async def train(self, train_data: List[Dict]):
        """

        Args:
            train_data: 
        """
        print("\n" + "=" * 60)
        print("🚀 Interactive GRPO")
        print("=" * 60)

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        batch_size = self.config.samples_per_group * 6

        for step in range(self.config.max_steps):
            batch_indices = np.random.choice(len(train_data), batch_size, replace=True)
            batch = [train_data[i] for i in batch_indices]

            stats = await self.train_step(step, batch)

            if (step + 1) % self.config.save_every == 0:
                self._save_checkpoint(step)

            gc.collect()
            torch.cuda.empty_cache()

        print("\n✅ !")

    def _save_checkpoint(self, step: int):
        """"""
        output_dir = Path(self.config.output_dir)
        checkpoint_path = output_dir / f"checkpoint_step_{step}"

        self.model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)

        print(f"💾 : {checkpoint_path}")


# ============================================
# ============================================

def create_interactive_trainer(
    config_path: Optional[str] = None,
    **kwargs,
) -> InteractiveGRPOTrainer:
    """
    GRPO

    Args:
        config_path: 
        **kwargs: 

    Returns:
        InteractiveGRPOTrainer
    """
    if config_path:
        trainer = InteractiveGRPOTrainer(config_path=config_path)
    else:
        config = InteractiveGRPOConfig(**kwargs)
        trainer = InteractiveGRPOTrainer(config=config)

    return trainer
