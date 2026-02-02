from .workflow_graph import (
    WorkflowGraph,
    WorkflowNode,
    NodeType,
    VALID_OPERATORS,
    create_workflow_from_dsl,
)

from .action_parser import (
    ActionParser,
    ParsedAction,
    ActionType,
    StructureType,
    parse_action,
    extract_actions,
)

from .workflow_env import (
    InteractiveWorkflowEnv,
    StepResult,
    create_env,
)

from .workflow_builder import (
    Trajectory,
    TurnRecord,
    create_action_mask,
    merge_trajectory_masks,
    InteractiveWorkflowBuilder,
    BatchWorkflowBuilder,
    create_aflow_executor_wrapper,
    create_builder,
)

from .trajectory_reward import (
    TrajectoryRewardCalculator,
    TrajectoryRewardResult,
    EfficiencyConfig,
    create_reward_calculator,
)

try:
    from .grpo_trainer import (
        InteractiveGRPOTrainer,
        InteractiveGRPOConfig,
        create_interactive_trainer,
    )
except Exception as _e:
    InteractiveGRPOTrainer = None
    InteractiveGRPOConfig = None

    def create_interactive_trainer(*args, **kwargs):
        raise ImportError("Missing training dependencies for `interactive.grpo_trainer`.") from _e

try:
    from .batch_inference import (
        BatchInferenceConfig,
        BatchGenerationManager,
        BatchInteractiveLoopManager,
        OptimizedAsyncLLMClient,
        create_batch_generator,
        create_optimized_client,
    )
except Exception as _e:
    BatchInferenceConfig = None
    BatchGenerationManager = None
    BatchInteractiveLoopManager = None
    OptimizedAsyncLLMClient = None

    def create_batch_generator(*args, **kwargs):
        raise ImportError("Missing optional dependencies for `interactive.batch_inference`.") from _e

    def create_optimized_client(*args, **kwargs):
        raise ImportError("Missing optional dependencies for `interactive.batch_inference`.") from _e

from .prompt_templates import (
    PromptConfig,
    InteractivePromptBuilder,
    CompactPromptBuilder,
    SYSTEM_PROMPT,
    ACTION_EXAMPLES,
    PROBLEM_TYPE_HINTS,
    create_prompt_builder,
    get_problem_type_hint,
)

__all__ = [
    'WorkflowGraph',
    'WorkflowNode',
    'NodeType',
    'VALID_OPERATORS',
    'create_workflow_from_dsl',
    'ActionParser',
    'ParsedAction',
    'ActionType',
    'StructureType',
    'parse_action',
    'extract_actions',
    'InteractiveWorkflowEnv',
    'StepResult',
    'create_env',
    'Trajectory',
    'TurnRecord',
    'create_action_mask',
    'merge_trajectory_masks',
    'InteractiveWorkflowBuilder',
    'BatchWorkflowBuilder',
    'create_aflow_executor_wrapper',
    'create_builder',
    'TrajectoryRewardCalculator',
    'TrajectoryRewardResult',
    'EfficiencyConfig',
    'create_reward_calculator',
    'InteractiveGRPOTrainer',
    'InteractiveGRPOConfig',
    'create_interactive_trainer',
    'BatchInferenceConfig',
    'BatchGenerationManager',
    'BatchInteractiveLoopManager',
    'OptimizedAsyncLLMClient',
    'create_batch_generator',
    'create_optimized_client',
    'PromptConfig',
    'InteractivePromptBuilder',
    'CompactPromptBuilder',
    'SYSTEM_PROMPT',
    'ACTION_EXAMPLES',
    'PROBLEM_TYPE_HINTS',
    'create_prompt_builder',
    'get_problem_type_hint',
]
