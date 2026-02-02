#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Interactive GRPO Training Script
================================
GRPO

:
    python train_interactive.py --config config/training_interactive.yaml

: Claude Code
: 2024-12-09
"""

import os
import sys
import re
import yaml
import argparse
import logging
import warnings

warnings.filterwarnings("ignore", message=".*Event loop is closed.*")
logging.getLogger("asyncio").setLevel(logging.CRITICAL)
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
except Exception:  # pragma: no cover
    AutoTokenizer = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore

try:
    from peft import LoraConfig, get_peft_model, TaskType  # type: ignore
except Exception:  # pragma: no cover
    LoraConfig = None  # type: ignore
    get_peft_model = None  # type: ignore
    TaskType = None  # type: ignore

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.interactive import (
    InteractiveWorkflowEnv,
    create_env,
    InteractiveWorkflowBuilder,
    create_builder,
    Trajectory,
    TurnRecord,
    TrajectoryRewardCalculator,
    create_reward_calculator,
    EfficiencyConfig,
    InteractiveGRPOTrainer,
    InteractiveGRPOConfig,
    create_interactive_trainer,
    InteractivePromptBuilder,
    create_prompt_builder,
)
from src.interactive.workflow_env import EnvState
from src.interactive.operator_descriptions import get_operator_template
from typing import Tuple


def create_demo_executor(problem: str, problem_type: str = "math"):
    """executor - 

    AFlow executor
    """
    raise NotImplementedError(
        "Demo executorAFlow executor"
        ""
    )


def setup_logging(log_dir: str, exp_name: str) -> logging.Logger:
    """"""
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"{exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger("InteractiveTraining")
    logger.info(f"Logging to {log_file}")
    return logger


LORA_SYNC_PATH = "/tmp/vllm_lora_sync"
LORA_ADAPTER_NAME = "training_lora"
LORA_SYNC_SUCCESS = False


def set_lora_sync_path(path: str):
    """ LoRA """
    global LORA_SYNC_PATH
    LORA_SYNC_PATH = path


def sync_lora_to_vllm(model, vllm_base_url: str = "http://localhost:8003/v1", logger=None) -> bool:
    """LoRAvLLM

    verloptimizer.step()LoRA
    vLLM

    Args:
        model: PEFTLoRA
        vllm_base_url: vLLM API
        logger: 

    Returns:
        bool: 
    """
    import os
    import shutil
    import requests

    if logger is None:
        logger = logging.getLogger("InteractiveTraining")

    try:
        if not hasattr(model, 'save_pretrained') or not hasattr(model, 'peft_config'):
            logger.warning("[P-verl] Model is not a PEFT model, skipping LoRA sync")
            return False

        if os.path.exists(LORA_SYNC_PATH):
            shutil.rmtree(LORA_SYNC_PATH)
        os.makedirs(LORA_SYNC_PATH, exist_ok=True)

        model.save_pretrained(LORA_SYNC_PATH)
        logger.info(f"[P-verl] LoRA weights saved to {LORA_SYNC_PATH}")

        success = _load_lora_to_vllm(vllm_base_url, LORA_ADAPTER_NAME, LORA_SYNC_PATH, logger)

        global LORA_SYNC_SUCCESS
        if success:
            logger.info(f"[P-verl] LoRA adapter '{LORA_ADAPTER_NAME}' synced to vLLM successfully")
            LORA_SYNC_SUCCESS = True
        else:
            logger.warning("[P-verl] Failed to sync LoRA to vLLM, rollout will use base model")
            LORA_SYNC_SUCCESS = False

        return success

    except Exception as e:
        logger.error(f"[LoRA] sync failed: {e}")
        return False


def _load_lora_to_vllm(base_url: str, lora_name: str, lora_path: str, logger) -> bool:
    """vLLM APILoRA

    unload+loadload400
    """
    return _try_unload_reload_lora(base_url, lora_name, lora_path, logger)


def _try_unload_reload_lora(base_url: str, lora_name: str, lora_path: str, logger) -> bool:
    """unload + loadLoRA

    vLLMunloadload
    """
    import requests

    unload_url = base_url.rstrip('/').replace('/v1', '') + '/v1/unload_lora_adapter'
    load_url = base_url.rstrip('/').replace('/v1', '') + '/v1/load_lora_adapter'

    try:
        requests.post(unload_url, json={"lora_name": lora_name}, timeout=30)

        response = requests.post(
            load_url,
            json={"lora_name": lora_name, "lora_path": lora_path},
            timeout=60,
        )
        return response.status_code == 200

    except Exception as e:
        logger.debug(f"[LoRA] unload/reload attempt failed: {e}")
        return False


VLLM_PROCESS = None
VLLM_RESTART_INTERVAL = 5


def restart_vllm_with_lora(model, config: dict, logger) -> bool:
    """vLLMLoRA

    vLLM 0.12.0LoRAAPI
    vLLMon-policy

    Args:
        model: PEFTLoRA
        config: 
        logger: 

    Returns:
        bool: 
    """
    import subprocess
    import time
    import requests
    import os
    import shutil
    global VLLM_PROCESS

    base_model = config.get('base_model', '/path/to/Qwen3-8B')
    port = 8003

    try:
        if os.path.exists(LORA_SYNC_PATH):
            shutil.rmtree(LORA_SYNC_PATH)
        os.makedirs(LORA_SYNC_PATH, exist_ok=True)
        model.save_pretrained(LORA_SYNC_PATH)
        logger.info(f"[LoRA] weights saved to {LORA_SYNC_PATH}")

        logger.info("[vLLM] Stopping existing vLLM server...")
        subprocess.run(
            "pkill -f 'vllm.entrypoints.openai.api_server.*--port 8003'",
            shell=True, capture_output=True, timeout=10
        )
        time.sleep(3)

        logger.info("[vLLM] Starting vLLM with updated LoRA...")
        cmd = [
            "/data/conda_envs/colab-grpo/bin/python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", base_model,
            "--served-model-name", "Qwen3-8B",
            "--port", str(port),
            "--gpu-memory-utilization", "0.85",
            "--max-model-len", "16384",
            "--enable-lora",
            "--max-loras", "2",
            "--lora-modules", f"{LORA_ADAPTER_NAME}={LORA_SYNC_PATH}",
            "--trust-remote-code",
            "--dtype", "bfloat16",
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"

        VLLM_PROCESS = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )

        logger.info("[vLLM] Waiting for vLLM to be ready...")
        for i in range(60):
            time.sleep(2)
            try:
                resp = requests.get(f"http://localhost:{port}/v1/models", timeout=5)
                if resp.status_code == 200:
                    models = resp.json().get("data", [])
                    model_ids = [m.get("id") for m in models]
                    if LORA_ADAPTER_NAME in model_ids:
                        logger.info(f"[vLLM] vLLM ready with LoRA: {model_ids}")
                        global LORA_SYNC_SUCCESS
                        LORA_SYNC_SUCCESS = True
                        return True
                    elif "Qwen3-8B" in model_ids:
                        logger.info(f"[vLLM] vLLM ready, waiting for LoRA... ({model_ids})")
            except:
                pass

        logger.warning("[vLLM] vLLM restart timeout, using base model")
        return False

    except Exception as e:
        logger.error(f"[vLLM] vLLM restart failed: {e}")
        return False


def get_vllm_model_name(base_model: str, use_lora_sync: bool = False) -> str:
    """vLLM API

    LoRALoRAvLLM
    vLLMLoRAidbase_model
    LoRA
    """
    if use_lora_sync and LORA_SYNC_SUCCESS:
        return LORA_ADAPTER_NAME
    return base_model


# ======================= End LoRA Sync =======================


def load_config(config_path: str) -> dict:
    """"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model_and_tokenizer(config: dict, device: str, resume_path: str = None):
    """tokenizer

    Args:
        config: 
        device: 
        resume_path: checkpoint
    """
    logger = logging.getLogger("InteractiveTraining")

    if torch is None or AutoTokenizer is None or AutoModelForCausalLM is None:
        raise ImportError(
            "Missing training dependencies. Install `torch`, `transformers`, and `peft` "
            "(see requirements.txt) to run training."
        )

    model_path = config['base_model']
    logger.info(f"Loading model from {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side='left',
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if config.get('bf16', True) else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=device,
    )

    if config.get('use_lora', True):
        if resume_path:
            from peft import PeftModel
            logger.info(f"Loading LoRA weights from checkpoint: {resume_path}")
            model = PeftModel.from_pretrained(model, resume_path, is_trainable=True)
            logger.info("✅ LoRA weights loaded from checkpoint")
        else:
            logger.info("Applying LoRA...")
            target_modules = config.get('lora_target_modules', 'q_proj,k_proj,v_proj,o_proj').split(',')

            lora_config = LoraConfig(
                r=config.get('lora_rank', 64),
                lora_alpha=config.get('lora_alpha', 64),
                target_modules=target_modules,
                lora_dropout=config.get('lora_dropout', 0.05),
                task_type=TaskType.CAUSAL_LM,
            )

            model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    if config.get('gradient_checkpointing', False):
        model.gradient_checkpointing_enable()
        if hasattr(model, 'enable_input_require_grads'):
            model.enable_input_require_grads()
        logger.info("✅ Gradient checkpointing enabled - ~7GB")

    model.train()

    return model, tokenizer


def load_dataset(data_path: str, max_samples: int = None, filter_unanswerable: bool = True) -> list:
    """

    Args:
        filter_unanswerable: ground_truthunanswerable (True)
    """
    logger = logging.getLogger("InteractiveTraining")

    data = []
    skipped_unanswerable = 0
    with open(data_path, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            if filter_unanswerable:
                gt = str(item.get('ground_truth', '')).strip()
                if _is_unanswerable(gt):
                    skipped_unanswerable += 1
                    continue
            data.append(item)
            if max_samples and len(data) >= max_samples:
                break

    if skipped_unanswerable > 0:
        logger.info(f"[Filter] Filtered {skipped_unanswerable} unanswerable samples")
    logger.info(f"Loaded {len(data)} samples from {data_path}")
    return data


def _load_code_public_tests_map(config: dict) -> Dict[Tuple[str, str], Dict[str, str]]:
    """Load AFlow-style public tests for code datasets.

    Mapping key: (source_lower, task_id_str) -> {"test": str, "entry_point": str, "prompt": str}
    """
    mapping: Dict[Tuple[str, str], Dict[str, str]] = {}
    cfg = (config.get("code_public_tests", {}) or {}) if isinstance(config, dict) else {}
    if not cfg:
        return mapping

    base_dir = Path(__file__).resolve().parent
    for src, p in cfg.items():
        src_l = str(src).strip().lower()
        path = Path(str(p))
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        if not path.exists():
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    task_id = row.get("task_id")
                    if task_id is None:
                        continue
                    key = (src_l, str(task_id))
                    mapping[key] = {
                        "test": str(row.get("test") or ""),
                        "entry_point": str(row.get("entry_point") or ""),
                        "prompt": str(row.get("prompt") or ""),
                    }
        except Exception:
            continue

    return mapping


def _apply_code_public_test(meta: Dict[str, Any], source: str, public_map: Dict[Tuple[str, str], Dict[str, str]]) -> Dict[str, Any]:
    """Override meta.test/entry_point using the public test map (if available)."""
    if not public_map:
        return dict(meta or {})

    src_l = str(source or "").strip().lower()
    m = dict(meta or {})
    task_id = m.get("task_id")
    if task_id is None:
        return m
    key = (src_l, str(task_id))
    row = public_map.get(key)
    if not row:
        return m

    if row.get("test"):
        m["test"] = row["test"]
    if row.get("entry_point"):
        m["entry_point"] = row["entry_point"]
    # Optional: keep prompt copy for convenience (HumanEval uses prompt+completion+test).
    if src_l == "humaneval" and row.get("prompt"):
        m["prompt"] = row["prompt"]
    return m


def create_generate_fn(model, tokenizer, config: dict):
    """Qwen3 thinking mode  vLLM API"""
    gen_config = config.get('generation_config', {})
    debug = bool(config.get("debug", False))

    use_vllm_api = config.get('use_vllm_api', False)
    vllm_base_url = config.get('vllm_base_url', 'http://localhost:8003/v1')

    if use_vllm_api:
        print(f"[INFO] vLLM API: {vllm_base_url}")
        from openai import OpenAI
        import threading

        _thread_local = threading.local()

        def _get_client() -> OpenAI:
            client = getattr(_thread_local, "client", None)
            if client is None:
                client = OpenAI(
                    base_url=vllm_base_url,
                    api_key="EMPTY",
                    timeout=300.0,
                )
                _thread_local.client = client
            return client

        base_model_name = config.get('vllm_served_model_name') or os.path.basename(config.get('base_model', 'Qwen3-8B'))
        enable_lora_sync = config.get('enable_lora_sync', False)

        def _get_model_name() -> str:
            return get_vllm_model_name(base_model_name, enable_lora_sync)

        if enable_lora_sync:
            print(f"[INFO] P-verl: LoRA: {_get_model_name()}")

        def generate_vllm(prompt: str, disable_thinking: bool = False) -> str:
            """

            Args:
                prompt: 
                disable_thinking:  thinking mode ( AWAITING_PROMPT )
            """
            try:
                import re

                context_limit = int(gen_config.get("vllm_context_limit", 16384))
                estimated_input_tokens = max(1, len(prompt) // 3)
                reserve = 128

                if disable_thinking:
                    desired_max_tokens = int(gen_config.get("vllm_prompt_max_tokens", 256))
                    enable_thinking = False
                    if debug:
                        print("[DEBUG] Thinking mode DISABLED for this call (AWAITING_PROMPT state)")
                else:
                    desired_max_tokens = int(gen_config.get("vllm_action_max_tokens", 512))
                    enable_thinking = bool(gen_config.get("enable_thinking", True))

                available = max(16, context_limit - estimated_input_tokens - reserve)
                max_tokens = max(16, min(desired_max_tokens, available))

                client = _get_client()
                current_model = _get_model_name()
                response = client.chat.completions.create(
                    model=current_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=gen_config.get('temperature', 0.6),
                    max_tokens=max_tokens,
                    top_p=gen_config.get('top_p', 0.95),
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": enable_thinking}
                    },
                )
                result = response.choices[0].message.content

                if '<think>' in result.lower():
                    think_match = re.search(r'<think>(.*?)</think>', result, re.DOTALL | re.IGNORECASE)
                    if think_match:
                        thinking = think_match.group(1).strip()
                        if debug:
                            preview = thinking[:300] + "..." if len(thinking) > 300 else thinking
                            print(f"[THINKING] {preview}")
                        result_after_think = re.sub(r'<think>.*?</think>\s*', '', result, flags=re.DOTALL | re.IGNORECASE)
                        if result_after_think.strip():
                            result = result_after_think
                    else:
                        incomplete = re.search(r'<think>(.*)', result, re.DOTALL | re.IGNORECASE)
                        if incomplete:
                            truncated = incomplete.group(1).strip()[:200]
                            if debug:
                                print(f"[WARNING] Thinking truncated (len={len(truncated)}), treating as Think-Only.")
                            result = f"<think_only>{truncated}</think_only>"

                result = _truncate_after_first_action(result)

                if not disable_thinking:
                    has_action = re.search(r'[<\[]action[>\]]', str(result), re.IGNORECASE) is not None
                    if not has_action:
                        repair_max = int(gen_config.get("vllm_repair_max_tokens", 128))
                        repair_max = max(16, min(repair_max, available))
                        repair_prompt = (
                            prompt
                            + "\n\nIMPORTANT: Output EXACTLY ONE XML <action>...</action> now. "
                            + "No <think> tags. No extra text."
                        )
                        repair_resp = client.chat.completions.create(
                            model=current_model,
                            messages=[{"role": "user", "content": repair_prompt}],
                            temperature=gen_config.get('temperature', 0.6),
                            max_tokens=repair_max,
                            top_p=gen_config.get('top_p', 0.95),
                            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                        )
                        repaired = repair_resp.choices[0].message.content
                        repaired = _truncate_after_first_action(repaired)
                        if re.search(r'[<\[]action[>\]]', str(repaired), re.IGNORECASE):
                            result = repaired

                return result
            except Exception as e:
                print(f"[ERROR] vLLM API: {e}")
                return "<action>add</action><operator>Custom</operator><prompt>API error fallback</prompt>"

        return generate_vllm

    is_qwen3 = hasattr(model.config, 'model_type') and 'qwen3' in model.config.model_type.lower()
    if is_qwen3:
        print("[DEBUG] Detected Qwen3 model, DISABLED thinking mode to prevent truncation issues")
        default_temp = 0.6
        default_top_p = 0.95
        default_top_k = 20
    else:
        default_temp = 0.3
        default_top_p = 0.9
        default_top_k = 50

    def generate(prompt: str, disable_thinking: bool = False) -> str:
        if is_qwen3 and hasattr(tokenizer, 'apply_chat_template'):
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=(not disable_thinking)
            )
            inputs = tokenizer(formatted_prompt, return_tensors='pt', truncation=True, max_length=4096)
        else:
            inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048)

        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=gen_config.get('max_new_tokens', 2048 if is_qwen3 else 256),
                temperature=gen_config.get('temperature', default_temp),
                top_p=gen_config.get('top_p', default_top_p),
                top_k=gen_config.get('top_k', default_top_k),
                do_sample=gen_config.get('do_sample', True),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        if is_qwen3 and '<think>' in response:
            import re
            think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
            if think_match:
                thinking = think_match.group(1).strip()
                print(f"[THINKING] {thinking[:200]}..." if len(thinking) > 200 else f"[THINKING] {thinking}")
                response_after_think = re.sub(r'<think>.*?</think>\s*', '', response, flags=re.DOTALL)

                if not response_after_think.strip() or '<action>' not in response_after_think.lower():
                    print(f"[WARNING] Think-Only detected! Adding marker for better feedback.")
                    response = f"<think_only>{thinking[:300]}</think_only>"
                else:
                    response = response_after_think
            else:
                incomplete_think = re.search(r'<think>(.*)', response, re.DOTALL)
                if incomplete_think:
                    truncated_thinking = incomplete_think.group(1).strip()[:200]
                    print(f"[THINKING TRUNCATED] {truncated_thinking}...")
                    print(f"[WARNING] Thinking truncated, treating as Think-Only.")
                    response = f"<think_only>{truncated_thinking}</think_only>"

        response = _truncate_after_first_action(response)
        return response

    return generate


def _truncate_after_first_action(response: str) -> str:
    """ (action + )

     action action
     action  parallel  <operators>

    -  </action> <action>add</action> 
    - “ action ” <action> 
    -  <tag>  [tag] 
    """
    import re

    think_match = re.search(r'<think>.*?</think>\s*', response, re.DOTALL | re.IGNORECASE)
    if think_match:
        response = response[think_match.end():]

    action_pattern = re.compile(
        r'[<\[]action[>\]]\s*(add|delete|modify|set_prompt|finish)\s*[<\[]/action[>\]]',
        re.IGNORECASE
    )
    open_action_pattern = re.compile(r'[<\[]action[>\]]', re.IGNORECASE)
    structure_pattern = re.compile(
        r'[<\[]structure[>\]]\s*(parallel|conditional|loop)\s*[<\[]/structure[>\]]',
        re.IGNORECASE
    )

    first_action = action_pattern.search(response)
    if not first_action:
        return response

    next_action_open = open_action_pattern.search(response, first_action.end())
    boundary = next_action_open.start() if next_action_open else len(response)

    segment = response[first_action.start():boundary]
    action_type = first_action.group(1).lower()

    structure_match = structure_pattern.search(segment)
    structure_type = structure_match.group(1).lower() if structure_match else None

    closing_tags = {
        "operator": re.compile(r'(</operator>|\[/operator\])', re.IGNORECASE),
        "operators": re.compile(r'(</operators>|\[/operators\])', re.IGNORECASE),
        "target": re.compile(r'(</target>|\[/target\])', re.IGNORECASE),
        "condition": re.compile(r'(</condition>|\[/condition\])', re.IGNORECASE),
        "true": re.compile(r'(</true>|\[/true\])', re.IGNORECASE),
        "false": re.compile(r'(</false>|\[/false\])', re.IGNORECASE),
        "count": re.compile(r'(</count>|\[/count\])', re.IGNORECASE),
        "position": re.compile(r'(</position>|\[/position\])', re.IGNORECASE),
        "prompt": re.compile(r'(</prompt>|\[/prompt\])', re.IGNORECASE),
        "answer": re.compile(r'(</answer>|\[/answer\])', re.IGNORECASE),
    }

    def _last_end(tag_name: str) -> int:
        """ segment  closing tag  end  -1"""
        pat = closing_tags[tag_name]
        m = None
        for m in pat.finditer(segment):
            pass
        return m.end() if m else -1

    end_in_segment = first_action.end() - first_action.start()

    if action_type == "finish":
        end_in_segment = max(end_in_segment, _last_end("answer"))
    elif action_type == "set_prompt":
        end_in_segment = max(end_in_segment, _last_end("prompt"))
    elif action_type == "delete":
        end_in_segment = max(end_in_segment, _last_end("target"))
    elif action_type == "modify":
        end_in_segment = max(end_in_segment, _last_end("target"), _last_end("operator"))
    elif action_type == "add":
        if structure_type == "parallel":
            end_in_segment = max(end_in_segment, _last_end("operators"))
        elif structure_type == "conditional":
            end_in_segment = max(end_in_segment, _last_end("condition"), _last_end("true"), _last_end("false"))
        elif structure_type == "loop":
            end_in_segment = max(end_in_segment, _last_end("operators"), _last_end("count"))
        else:
            end_in_segment = max(end_in_segment, _last_end("operator"))

        end_in_segment = max(end_in_segment, _last_end("position"), _last_end("prompt"))

    if end_in_segment <= 0:
        end_index = boundary
    else:
        end_index = first_action.start() + min(end_in_segment, len(segment))

    return response[:end_index]


def _sanitize_feedback_for_prompt(feedback: str) -> str:
    """ feedback“”

    /DSL/ <action> 
    """
    if not feedback:
        return ""

    filtered_lines = []
    for line in str(feedback).splitlines():
        lower = line.lower()
        if "<action>" in lower or "</action>" in lower:
            continue
        if lower.startswith("[action hint]:"):
            continue
        filtered_lines.append(line)

    return "\n".join(filtered_lines)


def run_interactive_loop(
    problem: str,
    problem_type: str,
    generate_fn,
    prompt_builder: InteractivePromptBuilder,
    max_rounds: int = 15,
    verbose: bool = True,
) -> Trajectory:
    """"""
    logger = logging.getLogger("InteractiveTraining")
    print(f"[DEBUG] run_interactive_loop started, problem={problem[:30]}...", flush=True)

    executor = create_demo_executor(problem, problem_type)
    print(f"[DEBUG] demo executor created", flush=True)

    env = create_env(problem=problem, problem_type=problem_type, executor=executor, max_rounds=max_rounds)
    print(f"[DEBUG] env created with executor", flush=True)

    trajectory = Trajectory(problem=problem, problem_type=problem_type)

    prompt = prompt_builder.build_initial_prompt(problem=problem, problem_type=problem_type)
    print(f"[DEBUG] Initial prompt built, len={len(prompt)}", flush=True)

    for round_idx in range(max_rounds):
        print(f"[DEBUG] Round {round_idx+1}/{max_rounds} - generating...", flush=True)
        is_awaiting_prompt = hasattr(env, 'env_state') and env.env_state == EnvState.AWAITING_PROMPT
        response = generate_fn(prompt, disable_thinking=is_awaiting_prompt)
        print(f"[DEBUG] Round {round_idx+1} - got response len={len(response)}", flush=True)
        print(f"[RESPONSE] {response[:500]}...", flush=True)

        if verbose:
            logger.info(f"  Round {round_idx + 1}: {response[:100]}...")

        feedback, success_list, active = env.step(response)
        done = not active
        print(f"[FEEDBACK] success={success_list}, active={active}, feedback={feedback[:200]}...", flush=True)

        turn_record = TurnRecord(
            round_idx=round_idx,
            model_response=response,
            feedback=feedback,
            success=all(success_list) if success_list else True,
            dsl_snapshot=env.get_dsl(),
        )
        trajectory.add_turn(turn_record)

        if done:
            break

        if hasattr(env, 'env_state') and env.env_state == EnvState.AWAITING_PROMPT:
            operator_name = env.pending_operator or "Custom"
            op_template = get_operator_template(operator_name)
            op_description = op_template.get("description", "") if op_template else ""
            prompt = prompt_builder.build_prompt_request(
                problem=problem,
                operator_name=operator_name,
                operator_description=op_description,
                context=getattr(env, "pending_prompt_context", "") or "",
                problem_type=problem_type,
            )
            print(f"[DEBUG] Using PROMPT_REQUEST prompt for {operator_name}", flush=True)
        else:
            stats = env.graph.get_statistics() if hasattr(env, 'graph') else {}
            prompt = prompt_builder.build_continuation_prompt(
                problem=problem,
                current_dsl=env.get_dsl(),
                total_operators=stats.get('total_operators', 0),
                unique_types=stats.get('unique_types', 0),
                round_number=round_idx + 2,
                max_rounds=max_rounds,
                node_ids=[n.id for n in env.graph.nodes] if hasattr(env, 'graph') else [],
                last_success=all(success_list) if success_list else True,
                last_result=_sanitize_feedback_for_prompt(feedback),
                last_message="",
                problem_type=problem_type,
            )

    trajectory.finalize(
        final_dsl=env.get_dsl(),
    )

    return trajectory


def demo_interactive_building(config: dict, model, tokenizer):
    """"""
    logger = logging.getLogger("InteractiveTraining")

    print("[DEBUG] Entering demo_interactive_building", flush=True)
    logger.info("=" * 60)
    logger.info("Demo: Interactive Workflow Building")
    logger.info("=" * 60)
    print("[DEBUG] Creating generate_fn...", flush=True)

    generate_fn = create_generate_fn(model, tokenizer, config)
    print("[DEBUG] generate_fn created", flush=True)

    prompt_builder = create_prompt_builder(compact=False)
    print("[DEBUG] prompt_builder created", flush=True)

    test_problems = [
        {
            "problem": "What is 15 * 23?",
            "problem_type": "math",
        },
        {
            "problem": "Calculate 2^10 (2 to the power of 10)",
            "problem_type": "math",
        },
        {
            "problem": "What is the sum of the first 100 natural numbers?",
            "problem_type": "math",
        },
        {
            "problem": "If a train travels at 60 km/h, how far will it travel in 2.5 hours?",
            "problem_type": "reasoning",
        },
    ]
    print(f"[DEBUG] Testing {len(test_problems)} problems", flush=True)

    for i, prob in enumerate(test_problems):
        print(f"\n[DEBUG] === Problem {i+1} ===", flush=True)
        logger.info(f"\n--- Problem {i+1}: {prob['problem'][:50]}... ---")

        trajectory = run_interactive_loop(
            problem=prob['problem'],
            problem_type=prob['problem_type'],
            generate_fn=generate_fn,
            prompt_builder=prompt_builder,
            max_rounds=config.get('interactive_grpo', {}).get('max_rounds', 15),
            verbose=True,
        )

        logger.info(f"Final DSL: {trajectory.final_dsl}")
        logger.info(f"Total rounds: {len(trajectory.turns)}")
        logger.info(f"Turns: {len(trajectory.turns)}")

        reward_calculator = create_reward_calculator(
            base_reward=config.get('progressive_reward', {}).get('base_reward', -1.0),
        )

        reward_result = reward_calculator.compute_reward(
            trajectory=trajectory,
            correctness=0.5,
        )

        logger.info(f"Reward: {reward_result.total_reward:.4f}")
        logger.info(f"  - Structure: {reward_result.structure_reward:.4f}")
        logger.info(f"  - Correctness: {reward_result.correctness_reward:.4f}")


def demo_dataset_building(
    config: dict,
    model,
    tokenizer,
    num_problems: int = 5,
    use_aflow: bool = False
):
    """"""
    import random
    logger = logging.getLogger("InteractiveTraining")

    print("[DEBUG] Entering demo_dataset_building", flush=True)
    logger.info("=" * 60)
    logger.info("Demo: Interactive Workflow Building (Dataset Mode)")
    logger.info("=" * 60)

    train_path = config.get('train_dataset', 'data/train_balanced_12k.jsonl')
    print(f"[DEBUG] Loading dataset from {train_path}...", flush=True)

    problems_by_source = {}
    with open(train_path, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            source = item.get('source', item.get('dataset', 'unknown'))
            if source not in problems_by_source:
                problems_by_source[source] = []
            problems_by_source[source].append(item)

    print(f"[DEBUG] Loaded problems from {len(problems_by_source)} sources", flush=True)
    for src, probs in problems_by_source.items():
        print(f"  - {src}: {len(probs)} problems", flush=True)

    random.seed(42)
    test_problems = []

    priority_sources = ['gsm8k', 'math']
    other_sources = [s for s in problems_by_source.keys() if s not in priority_sources]

    for src in priority_sources:
        if src in problems_by_source:
            selected = random.sample(
                problems_by_source[src],
                min(2, len(problems_by_source[src]))
            )
            for p in selected:
                test_problems.append({
                    "problem": p.get('problem', p.get('question', '')),
                    "problem_type": "math",
                    "source": src,
                    "ground_truth": p.get('ground_truth', p.get('answer', '')),
                })

    remaining = num_problems - len(test_problems)
    if remaining > 0:
        for src in random.sample(other_sources, min(remaining, len(other_sources))):
            if src in problems_by_source:
                p = random.choice(problems_by_source[src])
                ptype = "code" if src in ['humaneval', 'bigcodebench'] else "qa"
                test_problems.append({
                    "problem": p.get('problem', p.get('question', '')),
                    "problem_type": ptype,
                    "source": src,
                    "ground_truth": p.get('ground_truth', p.get('answer', '')),
                })

    print(f"\n[DEBUG] Selected {len(test_problems)} problems for testing:", flush=True)
    for i, p in enumerate(test_problems):
        print(f"  {i+1}. [{p['source']}] {p['problem'][:60]}...", flush=True)

    print("\n[DEBUG] Creating generate_fn...", flush=True)
    generate_fn = create_generate_fn(model, tokenizer, config)
    prompt_builder = create_prompt_builder(compact=False)

    aflow_executor = None
    if use_aflow:
        print("[DEBUG] Creating AFlow executor...", flush=True)
        try:
            from src.aflow_executor import AFlowExecutor
            aflow_executor = AFlowExecutor(
                llm_config_path=config.get('aflow_config_path', 'config/aflow_llm.yaml'),
                llm_model_name=config.get('aflow_executor_model', 'gpt-4o-mini'),
                timeout=int(config.get('execution_timeout', 300) or 300),
            )
            print("[DEBUG] AFlow executor created!", flush=True)
        except Exception as e:
            print(f"[WARNING] Failed to create AFlow executor: {e}", flush=True)
            print("[WARNING] Falling back to demo executor", flush=True)
            aflow_executor = None

    results = []
    for i, prob in enumerate(test_problems):
        print(f"\n{'='*60}", flush=True)
        print(f"[DEBUG] === Problem {i+1}/{len(test_problems)} [{prob['source']}] ===", flush=True)
        print(f"[PROBLEM] {prob['problem'][:200]}...", flush=True)
        print(f"[EXPECTED] {str(prob.get('ground_truth', 'N/A'))[:100]}...", flush=True)
        logger.info(f"\n--- Problem {i+1}: {prob['problem'][:50]}... [{prob['source']}] ---")

        if aflow_executor and use_aflow:
            from src.interactive import create_aflow_executor_wrapper
            executor = create_aflow_executor_wrapper(aflow_executor, prob['problem_type'])
        else:
            executor = create_demo_executor(prob['problem'], prob['problem_type'])

        finish_cfg = config.get('interactive_grpo', {}).get('finish_constraints', {}) or {}
        env = create_env(
            problem=prob['problem'],
            problem_type=prob['problem_type'],
            executor=executor,
            max_rounds=config.get('interactive_grpo', {}).get('max_rounds', 15),
            finish_min_total_operators=finish_cfg.get('min_total_operators', 1),
            finish_require_checker=finish_cfg.get('require_checker', False),
            finish_require_structure=finish_cfg.get('require_structure', False),
        )

        trajectory = Trajectory(problem=prob['problem'], problem_type=prob['problem_type'])

        prompt = prompt_builder.build_initial_prompt(
            problem=prob['problem'],
            problem_type=prob['problem_type']
        )

        max_rounds = config.get('interactive_grpo', {}).get('max_rounds', 15)
        for round_idx in range(max_rounds):
            print(f"[Round {round_idx+1}/{max_rounds}] generating...", flush=True)

            is_awaiting_prompt = hasattr(env, 'env_state') and env.env_state == EnvState.AWAITING_PROMPT
            response = generate_fn(prompt, disable_thinking=is_awaiting_prompt)
            print(f"[RESPONSE] {response[:200]}", flush=True)

            feedback, success_list, active = env.step(response)
            done = not active
            print(f"[FEEDBACK] success={success_list}, done={done}", flush=True)
            print(f"[FEEDBACK_PREVIEW] {feedback[:300]}...", flush=True)

            turn_record = TurnRecord(
                round_idx=round_idx,
                model_response=response,
                feedback=feedback,
                success=all(success_list) if success_list else True,
                dsl_snapshot=env.get_dsl(),
            )
            trajectory.add_turn(turn_record)

            if done:
                break

            if hasattr(env, 'env_state') and env.env_state == EnvState.AWAITING_PROMPT:
                operator_name = env.pending_operator or "Custom"
                op_template = get_operator_template(operator_name)
                op_description = op_template.get("description", "") if op_template else ""
                prompt = prompt_builder.build_prompt_request(
                    problem=prob['problem'],
                    operator_name=operator_name,
                    operator_description=op_description,
                    context=getattr(env, "pending_prompt_context", "") or "",
                    problem_type=prob.get('problem_type', 'default'),
                )
                print(f"[DEBUG] Using PROMPT_REQUEST prompt for {operator_name}", flush=True)
            else:
                stats = env.graph.get_statistics() if hasattr(env, 'graph') else {}
                prompt = prompt_builder.build_continuation_prompt(
                    problem=prob['problem'],
                    current_dsl=env.get_dsl(),
                    total_operators=stats.get('total_operators', 0),
                    unique_types=stats.get('unique_types', 0),
                    round_number=round_idx + 2,
                    max_rounds=max_rounds,
                    node_ids=[n.id for n in env.graph.nodes] if hasattr(env, 'graph') else [],
                    last_success=all(success_list) if success_list else True,
                    last_result=_sanitize_feedback_for_prompt(feedback),
                    last_message="",
                    problem_type=prob['problem_type'],
                )

        trajectory.finalize(final_dsl=env.get_dsl())

        result = {
            "problem": prob['problem'][:60],
            "source": prob['source'],
            "rounds": len(trajectory.turns),
            "final_dsl": trajectory.final_dsl,
        }
        results.append(result)

        print(f"\n[RESULT] Completed in {result['rounds']} rounds", flush=True)
        print(f"[RESULT] Final DSL: {result['final_dsl']}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 60, flush=True)
    for i, r in enumerate(results):
        print(f"{i+1}. [{r['source']}] {r['problem']}...", flush=True)
        print(f"   Rounds: {r['rounds']}, DSL: {r['final_dsl']}", flush=True)


def run_full_training(config: dict, model, tokenizer, args, resume_path: str = None):
    """GRPO

    Qwen3(thinking disabled, 512 tokens)

    Args:
        resume_path: checkpointstep
    """
    import gc
    import hashlib
    import numpy as np
    import random
    import torch.nn.functional as F
    from tqdm import tqdm
    from transformers import get_cosine_schedule_with_warmup

    logger = logging.getLogger("InteractiveTraining")

    print("=" * 60)
    print("🚀 Interactive GRPO Training - Full Mode")
    print("=" * 60)

    code_eval_cfg = config.get("code_eval", {}) or {}
    if code_eval_cfg.get("backend"):
        os.environ["COLAB_GRPO_CODE_EVAL_BACKEND"] = str(code_eval_cfg["backend"])
    if code_eval_cfg.get("docker_image"):
        os.environ["COLAB_GRPO_CODE_EVAL_DOCKER_IMAGE"] = str(code_eval_cfg["docker_image"])
    if code_eval_cfg:
        print(
            f"[INFO] CodeEval backend={os.environ.get('COLAB_GRPO_CODE_EVAL_BACKEND')} "
            f"image={os.environ.get('COLAB_GRPO_CODE_EVAL_DOCKER_IMAGE')}"
        )

    output_dir = Path(config.get('output_dir', 'checkpoints/interactive'))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prompt logging (optional): record small-model custom prompts for later analysis.
    prompt_log_cfg = config.get("prompt_logging", {}) or {}
    prompt_log_enabled = bool(prompt_log_cfg.get("enabled", False))
    prompt_log_fh = None
    prompt_log_sample_rate = float(prompt_log_cfg.get("sample_rate", 1.0) or 1.0)
    prompt_log_sample_rate = max(0.0, min(1.0, prompt_log_sample_rate))
    prompt_log_max_prompt_chars = int(prompt_log_cfg.get("max_prompt_chars", 2000) or 2000)
    prompt_log_max_problem_chars = int(prompt_log_cfg.get("max_problem_chars", 2000) or 2000)

    if prompt_log_enabled:
        try:
            log_dir = Path(config.get("log_dir", "logs"))
            log_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            prompt_log_path = str(log_dir / f"prompt_events_{config.get('exp_name', 'interactive_grpo')}_{ts}.jsonl")
            prompt_log_fh = open(prompt_log_path, "a", encoding="utf-8")
            logger.info(f"[PromptLog] enabled: {prompt_log_path} (sample_rate={prompt_log_sample_rate})")
        except Exception as e:
            logger.warning(f"[PromptLog] failed to open log file: {e}")
            prompt_log_fh = None

    def _maybe_log_prompt_event(step_idx: int, prob: dict, env, dsl_snapshot: str) -> None:
        try:
            event = getattr(env, "last_prompt_set", None)
            if not event:
                return

            # Clear immediately to avoid duplicated logging.
            env.last_prompt_set = None

            if not prompt_log_fh:
                return
            if prompt_log_sample_rate < 1.0 and random.random() > prompt_log_sample_rate:
                return

            problem_text = str(prob.get("problem", "") or "")
            problem_hash = hashlib.sha1(problem_text.encode("utf-8", errors="ignore")).hexdigest()[:12]

            record = {
                "ts": time.time(),
                "train_step": int(step_idx),
                "source": prob.get("source", ""),
                "problem_type": prob.get("problem_type", ""),
                "problem_hash": problem_hash,
                "problem_preview": problem_text[:prompt_log_max_problem_chars],
                "round": event.get("round"),
                "node_id": event.get("node_id"),
                "operator": event.get("operator"),
                "context": event.get("context"),
                "prompt": str(event.get("prompt", "") or "")[:prompt_log_max_prompt_chars],
                "dsl_snapshot": dsl_snapshot,
            }
            prompt_log_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"[PromptLog] write failed: {e}")

    train_path = config.get('train_dataset', 'data/train_balanced_12k.jsonl')
    logger.info(f"Loading training data from {train_path}")

    raw_data = load_dataset(train_path, filter_unanswerable=True)

    # AFlow-style public tests (HumanEval/MBPP) for training-time feedback/eval
    public_tests_map = _load_code_public_tests_map(config)
    if public_tests_map:
        logger.info(f"[INFO] Loaded code public tests: {len(public_tests_map)} entries")

    train_data = []
    for item in raw_data:
        src = item.get('source', item.get('dataset', 'unknown'))
        inferred_ptype = _infer_problem_type(src)
        meta = item.get('meta', {}) or {}
        if inferred_ptype == "code" and public_tests_map:
            meta = _apply_code_public_test(meta, src, public_tests_map)
        train_data.append({
            'problem': item.get('problem', item.get('question', '')),
            'problem_type': inferred_ptype,
            'ground_truth': item.get('ground_truth', item.get('answer', '')),
            'source': src,
            'meta': meta,
        })

    logger.info(f"Loaded {len(train_data)} training samples (after filtering)")

    max_steps = config.get('max_steps', 100)
    batch_size = config.get('samples_per_group', 8)
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 4)
    save_every = config.get('save_every', 20)

    generate_fn = create_generate_fn(model, tokenizer, config)
    prompt_builder = create_prompt_builder(compact=False)

    if config.get('use_vllm_api', False) and config.get('enable_lora_sync', False):
        logger.info("[P-verl] Performing initial LoRA sync to vLLM...")
        vllm_base_url = config.get('vllm_base_url', 'http://localhost:8003/v1')
        init_sync_success = sync_lora_to_vllm(model, vllm_base_url, logger)
        if init_sync_success:
            logger.info("[P-verl] Initial LoRA sync successful - vLLM ready for on-policy rollout")
        else:
            logger.warning("[P-verl] Initial LoRA sync failed - first rollout will use base model")
            logger.warning("[P-verl] This may cause on-policy/off-policy mismatch until next sync")

    from src.aflow_executor import AFlowExecutor
    aflow_executor = AFlowExecutor(
        llm_config_path=config.get('aflow_config_path', 'config/aflow_llm.yaml'),
        llm_model_name=config.get('aflow_executor_model', 'gpt-4o-mini'),
        timeout=int(config.get('execution_timeout', 300) or 300),
    )
    logger.info("✅ AFlow executor initialized (REAL execution mode)")

    progressive_cfg = config.get("progressive_reward", {}) or {}
    base_reward = float(progressive_cfg.get("base_reward", -0.5))
    base_threshold = float(progressive_cfg.get("correctness_activation_threshold", 0.6))
    threshold_schedule = progressive_cfg.get("threshold_schedule", {}) or {}
    schedule_enabled = bool(threshold_schedule.get("enabled", False))
    schedule_start = float(threshold_schedule.get("start", 0.0))
    schedule_end = float(threshold_schedule.get("end", base_threshold))
    schedule_warmup = int(threshold_schedule.get("warmup_steps", 0) or 0)

    eff_cfg = (config.get("interactive_grpo", {}) or {}).get("efficiency", {}) or {}
    optimal_min = int(eff_cfg.get("optimal_min", 3) or 3)
    optimal_max = int(eff_cfg.get("optimal_max", 30) or 30)

    reward_calculator = create_reward_calculator(
        base_reward=base_reward,
        correctness_threshold=(schedule_start if schedule_enabled else base_threshold),
        optimal_turns=(optimal_min, optimal_max),
        efficiency=eff_cfg,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get('learning_rate', 1e-5),
        weight_decay=config.get('weight_decay', 0.01),
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.get('warmup_steps', 10),
        num_training_steps=max_steps,
    )

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    logger.info("Random seeds set to 42 for reproducibility")

    model.train()
    global_step = 0
    accumulated_loss = 0.0

    logger.info(f"Starting training for {max_steps} steps")
    logger.info(f"Batch size: {batch_size}, Gradient accumulation: {gradient_accumulation_steps}")

    wandb_run = None
    if config.get('wandb', {}).get('enabled', False):
        try:
            import wandb
            wandb_resume_id = config.get('wandb', {}).get('resume_id', None)
            wandb_resume_mode = config.get('wandb', {}).get('resume', None)  # "must", "allow", "never", None

            wandb_settings = wandb.Settings(
                x_file_stream_transmit_interval=5,
            )

            init_kwargs = {
                "project": config['wandb'].get('project', 'aflow-interactive-grpo'),
                "name": config['wandb'].get('run_name', f'interactive-grpo-{datetime.now().strftime("%Y%m%d_%H%M%S")}'),
                "config": config,
                "settings": wandb_settings,
            }

            if wandb_resume_id:
                init_kwargs["id"] = wandb_resume_id
                init_kwargs["resume"] = wandb_resume_mode or "must"
                logger.info(f"Wandb resuming run: {wandb_resume_id}")
            elif wandb_resume_mode:
                init_kwargs["resume"] = wandb_resume_mode

            wandb_run = wandb.init(**init_kwargs)
            logger.info(f"Wandb initialized (run_id={wandb_run.id})")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")

    # ===== Rollout parallelism (vectorized rollouts across problems) =====
    interactive_cfg = config.get("interactive_grpo", {}) or {}
    vectorized_rollout = bool(interactive_cfg.get("vectorized_rollout", False))
    gen_cfg = config.get("generation_config", {}) or {}

    rollout_pool = None
    if vectorized_rollout:
        if not config.get("use_vllm_api", False):
            logger.warning("interactive_grpo.vectorized_rollout=true but use_vllm_api=false; falling back to serial rollout.")
            vectorized_rollout = False
        else:
            from concurrent.futures import ThreadPoolExecutor

            override_workers = int(interactive_cfg.get("vectorized_rollout_workers", 0) or 0)
            if override_workers > 0:
                workers = override_workers
            else:
                workers = int(gen_cfg.get("vllm_max_concurrency", 32))
                workers = max(1, min(workers, batch_size))
            rollout_pool = ThreadPoolExecutor(max_workers=workers)
            logger.info(f"✅ Vectorized rollout enabled (workers={workers})")

    start_step = 0
    if resume_path:
        import re
        match = re.search(r'checkpoint_step_(\d+)', resume_path)
        if match:
            start_step = int(match.group(1))
            logger.info(f"📍 Resuming from step {start_step}")

            training_state_path = Path(resume_path) / 'training_state.pt'
            if training_state_path.exists():
                training_state = torch.load(training_state_path, map_location='cpu')
                optimizer.load_state_dict(training_state['optimizer_state_dict'])
                scheduler.load_state_dict(training_state['scheduler_state_dict'])
                logger.info(f"✅ Optimizer and scheduler state restored from checkpoint")
                logger.info(f"   Current LR: {scheduler.get_last_lr()[0]:.2e}")

                if 'random_state' in training_state:
                    random.setstate(training_state['random_state'])
                    np.random.set_state(training_state['np_random_state'])
                    torch.set_rng_state(training_state['torch_random_state'])
                    if torch.cuda.is_available() and 'torch_cuda_random_state' in training_state:
                        torch.cuda.set_rng_state_all(training_state['torch_cuda_random_state'])
                    logger.info(f"✅ Random state restored from checkpoint")
                else:
                    logger.warning(f"⚠️ No random state in checkpoint, advancing random state for {start_step} steps")
                    for _ in range(start_step):
                        for _ in range(6):
                            np.random.choice(100, config['grpo'].get('samples_per_source', 6), replace=False)
                    logger.info(f"   Random state advanced to step {start_step}")
            else:
                logger.warning(f"⚠️ No training_state.pt found, manually stepping scheduler to step {start_step}")
                for _ in range(start_step):
                    scheduler.step()
                logger.info(f"   Adjusted LR: {scheduler.get_last_lr()[0]:.2e}")
        else:
            logger.warning(f"⚠️ Could not parse step from resume_path: {resume_path}, starting from 0")

    for step in range(start_step, max_steps):
        step_start = time.time()

        if schedule_enabled and schedule_warmup > 0:
            frac = min(max(step / schedule_warmup, 0.0), 1.0)
            current_threshold = schedule_start + (schedule_end - schedule_start) * frac
        else:
            current_threshold = schedule_end if schedule_enabled else base_threshold

        try:
            reward_calculator.correctness_activation_threshold = float(current_threshold)
            if hasattr(reward_calculator, "complexity_calculator"):
                reward_calculator.complexity_calculator.correctness_activation_threshold = float(current_threshold)
        except Exception:
            pass

        samples_per_source = config.get('samples_per_source', 6)

        from collections import defaultdict
        source_to_indices = defaultdict(list)
        for idx, item in enumerate(train_data):
            source_to_indices[item.get('source', 'unknown')].append(idx)

        batch_indices = []
        for source, indices in sorted(source_to_indices.items()):
            if len(indices) >= samples_per_source:
                sampled = np.random.choice(indices, samples_per_source, replace=False)
            else:
                sampled = np.random.choice(indices, samples_per_source, replace=True)
            batch_indices.extend(sampled)

        batch = [train_data[i] for i in batch_indices]
        batch_size = len(batch)

        print(f"\n{'='*60}")
        print(f"📦 Step {step+1}/{max_steps}: Processing {batch_size} problems ({samples_per_source} per source)")
        print(f"{'='*60}")

        trajectories = []
        correctness_scores = []
        correctness_details = []

        max_rounds = int(interactive_cfg.get('max_rounds', 20) or 20)
        execute_each_step = bool(interactive_cfg.get('execute_each_step', True))
        use_custom_prompts_in_execution = bool(interactive_cfg.get('use_custom_prompts_in_execution', True))
        finish_cfg = interactive_cfg.get('finish_constraints', {}) or {}
        debug = bool(config.get("debug", False))
        verbose = bool(config.get("verbose", False) or debug)
        rollout_t0 = time.time()
        vllm_time = 0.0
        env_step_time = 0.0

        if vectorized_rollout and rollout_pool is not None:
            from concurrent.futures import as_completed

            from src.interactive import create_aflow_executor_wrapper

            envs: List[Any] = []
            prompts: List[str] = []
            trajs: List[Trajectory] = []

            for i, prob in enumerate(batch):
                print(f"  [{i+1}/{batch_size}] {prob['source']}: {prob['problem'][:50]}...")

                executor = create_aflow_executor_wrapper(aflow_executor, prob['problem_type'])

                executor_kwargs = {}
                if prob.get('problem_type') == 'code':
                    meta = prob.get('meta', {}) or {}
                    test = meta.get('test')
                    entry_point = meta.get('entry_point')
                    if not entry_point and test:
                        entry_point = _extract_entry_point_from_test(test)
                    if test:
                        executor_kwargs['test'] = test
                    if entry_point:
                        executor_kwargs['entry_point'] = entry_point

                env = create_env(
                    problem=prob['problem'],
                    problem_type=prob['problem_type'],
                    executor=executor,
                    executor_kwargs=executor_kwargs if executor_kwargs else None,
                    max_rounds=max_rounds,
                    execute_each_step=execute_each_step,
                    finish_min_total_operators=finish_cfg.get('min_total_operators', 1),
                    finish_require_checker=finish_cfg.get('require_checker', False),
                    finish_require_structure=finish_cfg.get('require_structure', False),
                    use_custom_prompts_in_execution=use_custom_prompts_in_execution,
                )
                envs.append(env)

                trajs.append(Trajectory(problem=prob['problem'], problem_type=prob['problem_type']))
                prompts.append(
                    prompt_builder.build_initial_prompt(
                        problem=prob['problem'],
                        problem_type=prob['problem_type'],
                    )
                )

            active_mask = [True] * len(batch)

            for _ in range(max_rounds):
                active_indices = [i for i, a in enumerate(active_mask) if a]
                if not active_indices:
                    break

                if verbose:
                    print(f"    [Vectorized] Active problems: {len(active_indices)}/{len(batch)}")

                gen_round_t0 = time.time()
                futures = {}
                for idx in active_indices:
                    env = envs[idx]
                    is_awaiting_prompt = hasattr(env, 'env_state') and env.env_state == EnvState.AWAITING_PROMPT
                    fut = rollout_pool.submit(generate_fn, prompts[idx], disable_thinking=is_awaiting_prompt)
                    futures[fut] = idx

                responses_by_idx: Dict[int, str] = {}
                for fut in as_completed(futures):
                    idx = futures[fut]
                    try:
                        responses_by_idx[idx] = fut.result()
                    except Exception as e:
                        if debug:
                            print(f"[WARNING] vLLM generation failed for idx={idx}: {e}")
                        responses_by_idx[idx] = "<action>add</action><operator>Custom</operator><prompt>generation error</prompt>"
                vllm_time += time.time() - gen_round_t0

                env_round_t0 = time.time()

                def _execute_env_step(idx):
                    """step"""
                    env = envs[idx]
                    response = responses_by_idx.get(idx, "")
                    feedback, success_list, still_active = env.step(response)
                    current_dsl = env.get_dsl()
                    return {
                        'idx': idx,
                        'feedback': feedback,
                        'success_list': success_list,
                        'still_active': still_active,
                        'current_dsl': current_dsl,
                        'response': response,
                    }

                env_futures = {}
                for idx in active_indices:
                    fut = rollout_pool.submit(_execute_env_step, idx)
                    env_futures[fut] = idx

                step_results = {}
                for fut in as_completed(env_futures):
                    try:
                        result = fut.result()
                        step_results[result['idx']] = result
                    except Exception as e:
                        idx = env_futures[fut]
                        if debug:
                            print(f"[WARNING] env.step() failed for idx={idx}: {e}")
                        step_results[idx] = {
                            'idx': idx,
                            'feedback': f"Execution error: {str(e)}",
                            'success_list': [False],
                            'still_active': False,
                            'current_dsl': envs[idx].get_dsl() if idx < len(envs) else "",
                            'response': responses_by_idx.get(idx, ""),
                        }

                for idx in active_indices:
                    result = step_results.get(idx)
                    if not result:
                        continue

                    env = envs[idx]
                    trajectory = trajs[idx]
                    prob = batch[idx]

                    feedback = result['feedback']
                    success_list = result['success_list']
                    still_active = result['still_active']
                    current_dsl = result['current_dsl']
                    response = result['response']

                    turn_record = TurnRecord(
                        round_idx=len(trajectory.turns),
                        model_response=response,
                        feedback=feedback,
                        success=all(success_list) if success_list else True,
                        dsl_snapshot=current_dsl,
                    )
                    trajectory.add_turn(turn_record)
                    _maybe_log_prompt_event(step + 1, prob, env, current_dsl)

                    active_mask[idx] = still_active
                    if not still_active:
                        continue

                    if hasattr(env, 'env_state') and env.env_state == EnvState.AWAITING_PROMPT:
                        operator_name = env.pending_operator or "Custom"
                        op_template = get_operator_template(operator_name)
                        op_description = op_template.get("description", "") if op_template else ""
                        prompts[idx] = prompt_builder.build_prompt_request(
                            problem=prob['problem'],
                            operator_name=operator_name,
                            operator_description=op_description,
                            context=getattr(env, "pending_prompt_context", "") or "",
                            problem_type=prob.get('problem_type', 'default'),
                        )
                    else:
                        stats = env.graph.get_statistics() if hasattr(env, 'graph') else {}
                        prompts[idx] = prompt_builder.build_continuation_prompt(
                            problem=prob['problem'],
                            current_dsl=current_dsl,
                            total_operators=stats.get('total_operators', 0),
                            unique_types=stats.get('unique_types', 0),
                            round_number=(getattr(env, "round_count", 0) + 1),
                            max_rounds=max_rounds,
                            node_ids=[n.id for n in env.graph.nodes] if hasattr(env, 'graph') else [],
                            last_success=all(success_list) if success_list else True,
                            last_result=_sanitize_feedback_for_prompt(feedback),
                            last_message="",
                            problem_type=prob['problem_type'],
                        )
                env_step_time += time.time() - env_round_t0

            # finalize + correctness
            for i, prob in enumerate(batch):
                env = envs[i]
                trajectory = trajs[i]

                raw_result = getattr(env, 'last_execution_result', None)
                final_answer = _extract_operator_answer(raw_result, problem_type=prob['problem_type'])
                trajectory.finalize(final_dsl=env.get_dsl(), final_answer=final_answer)
                trajectories.append(trajectory)

                meta = prob.get('meta', {})
                code_for_test = None
                if prob['problem_type'] == 'code' and isinstance(raw_result, dict):
                    code_for_test = raw_result.get('code')

                gt = prob.get('ground_truth', '')
                meta_with_problem = dict(meta)
                meta_with_problem['problem'] = prob.get('problem', '')
                meta_with_problem['prompt'] = prob.get('prompt', '')

                correctness = _compute_correctness(
                    model_answer=final_answer,
                    ground_truth=gt,
                    problem_type=prob['problem_type'],
                    source=prob.get('source', ''),
                    test=meta_with_problem.get('test'),
                    entry_point=meta_with_problem.get('entry_point'),
                    code_for_test=code_for_test,
                    meta=meta_with_problem,
                    use_llm_judge=True,
                    return_details=True,
                )
                if isinstance(correctness, tuple) and len(correctness) == 2:
                    correctness_score, detail = correctness
                else:
                    correctness_score, detail = correctness, {}

                correctness_scores.append(correctness_score)
                correctness_details.append({
                    "source": prob.get("source", ""),
                    "problem_type": prob.get("problem_type", ""),
                    "correctness": float(correctness_score),
                    "details": detail or {},
                })

                print(f"    -> Rounds: {len(trajectory.turns)}, DSL: {trajectory.final_dsl}")
                print(f"    -> Answer: {str(final_answer)[:100]}..., Correctness: {correctness_score:.2f}")

        else:
            # ===== Serial rollouts (legacy) =====
            for i, prob in enumerate(batch):
                print(f"  [{i+1}/{batch_size}] {prob['source']}: {prob['problem'][:50]}...")

                from src.interactive import create_aflow_executor_wrapper
                executor = create_aflow_executor_wrapper(aflow_executor, prob['problem_type'])

                executor_kwargs = {}
                if prob.get('problem_type') == 'code':
                    meta = prob.get('meta', {}) or {}
                    test = meta.get('test')
                    entry_point = meta.get('entry_point')
                    if not entry_point and test:
                        entry_point = _extract_entry_point_from_test(test)
                    if test:
                        executor_kwargs['test'] = test
                    if entry_point:
                        executor_kwargs['entry_point'] = entry_point

                env = create_env(
                    problem=prob['problem'],
                    problem_type=prob['problem_type'],
                    executor=executor,
                    executor_kwargs=executor_kwargs if executor_kwargs else None,
                    max_rounds=max_rounds,
                    execute_each_step=execute_each_step,
                    finish_min_total_operators=finish_cfg.get('min_total_operators', 1),
                    finish_require_checker=finish_cfg.get('require_checker', False),
                    finish_require_structure=finish_cfg.get('require_structure', False),
                    use_custom_prompts_in_execution=use_custom_prompts_in_execution,
                )

                trajectory = Trajectory(problem=prob['problem'], problem_type=prob['problem_type'])
                prompt = prompt_builder.build_initial_prompt(problem=prob['problem'], problem_type=prob['problem_type'])

                if verbose:
                    print(f"\n    {'─'*50}")
                    print(f"    🎯 Problem: {prob['problem'][:80]}...")
                    print(f"    📋 Type: {prob['problem_type']}, Source: {prob['source']}")
                    print(f"    {'─'*50}")

                for round_idx in range(max_rounds):
                    is_awaiting_prompt = hasattr(env, 'env_state') and env.env_state == EnvState.AWAITING_PROMPT
                    gen_t0 = time.time()
                    response = generate_fn(prompt, disable_thinking=is_awaiting_prompt)
                    vllm_time += time.time() - gen_t0

                    step_t0 = time.time()
                    feedback, success_list, active = env.step(response)
                    env_step_time += time.time() - step_t0
                    done = not active
                    current_dsl = env.get_dsl()

                    if verbose:
                        print(f"\n    ┌─ Round {round_idx + 1}/{max_rounds} ─────────────────────────────────┐")
                        print(f"    │ 🤖 Model Response:")
                        resp_lines = response[:500].split('\n')
                        for line in resp_lines:
                            print(f"    │   {line}")
                        if len(response) > 500:
                            print(f"    │   ... (truncated, total {len(response)} chars)")
                        env_state_str = env.env_state.name if hasattr(env, 'env_state') else 'N/A'
                        pending_op = getattr(env, 'pending_operator', None)
                        print(f"    │")
                        print(f"    │ 📊 Parse Result:")
                        print(f"    │   - Env State: {env_state_str}")
                        print(f"    │   - Pending Operator: {pending_op}")
                        print(f"    │   - Success: {success_list}")
                        print(f"    │   - Active: {active}")
                        print(f"    │")
                        print(f"    │ 🔧 Current DSL: {current_dsl or '(empty)'}")
                        print(f"    │")
                        print(f"    │ 📝 Feedback:")
                        fb_lines = feedback[:400].split('\n')
                        for line in fb_lines[:8]:
                            print(f"    │   {line}")
                        if len(fb_lines) > 8 or len(feedback) > 400:
                            print(f"    │   ... (truncated)")
                        print(f"    └────────────────────────────────────────────────┘")

                    turn_record = TurnRecord(
                        round_idx=round_idx,
                        model_response=response,
                        feedback=feedback,
                        success=all(success_list) if success_list else True,
                        dsl_snapshot=current_dsl,
                    )
                    trajectory.add_turn(turn_record)
                    _maybe_log_prompt_event(step + 1, prob, env, current_dsl)

                    if done:
                        if verbose:
                            print(f"    ✅ Workflow completed at round {round_idx + 1}")
                        break

                    if hasattr(env, 'env_state') and env.env_state == EnvState.AWAITING_PROMPT:
                        operator_name = env.pending_operator or "Custom"
                        op_template = get_operator_template(operator_name)
                        op_description = op_template.get("description", "") if op_template else ""
                        prompt = prompt_builder.build_prompt_request(
                            problem=prob['problem'],
                            operator_name=operator_name,
                            operator_description=op_description,
                            context=getattr(env, "pending_prompt_context", "") or "",
                            problem_type=prob.get('problem_type', 'default'),
                        )
                    else:
                        stats = env.graph.get_statistics() if hasattr(env, 'graph') else {}
                        prompt = prompt_builder.build_continuation_prompt(
                            problem=prob['problem'],
                            current_dsl=current_dsl,
                            total_operators=stats.get('total_operators', 0),
                            unique_types=stats.get('unique_types', 0),
                            round_number=round_idx + 2,
                            max_rounds=max_rounds,
                            node_ids=[n.id for n in env.graph.nodes] if hasattr(env, 'graph') else [],
                            last_success=all(success_list) if success_list else True,
                            last_result=_sanitize_feedback_for_prompt(feedback),
                            last_message="",
                            problem_type=prob['problem_type'],
                        )

                raw_result = getattr(env, 'last_execution_result', None)
                final_answer = _extract_operator_answer(raw_result, problem_type=prob['problem_type'])
                trajectory.finalize(final_dsl=env.get_dsl(), final_answer=final_answer)
                trajectories.append(trajectory)

                meta = prob.get('meta', {})
                code_for_test = None
                if prob['problem_type'] == 'code' and isinstance(raw_result, dict):
                    code_for_test = raw_result.get('code')

                gt = prob.get('ground_truth', '')
                meta_with_problem = dict(meta)
                meta_with_problem['problem'] = prob.get('problem', '')
                meta_with_problem['prompt'] = prob.get('prompt', '')
                correctness = _compute_correctness(
                    model_answer=final_answer,
                    ground_truth=gt,
                    problem_type=prob['problem_type'],
                    source=prob.get('source', ''),
                    test=meta_with_problem.get('test'),
                    entry_point=meta_with_problem.get('entry_point'),
                    code_for_test=code_for_test,
                    meta=meta_with_problem,
                    use_llm_judge=True,
                    return_details=True,
                )
                if isinstance(correctness, tuple) and len(correctness) == 2:
                    correctness_score, detail = correctness
                else:
                    correctness_score, detail = correctness, {}

                correctness_scores.append(correctness_score)
                correctness_details.append({
                    "source": prob.get("source", ""),
                    "problem_type": prob.get("problem_type", ""),
                    "correctness": float(correctness_score),
                    "details": detail or {},
                })

                print(f"    -> Rounds: {len(trajectory.turns)}, DSL: {trajectory.final_dsl}")
                print(f"    -> Answer: {str(final_answer)[:100]}..., Correctness: {correctness_score:.2f}")

        rollout_wall = time.time() - rollout_t0
        print(f"    ⏱️ Rollout: {rollout_wall:.1f}s (vLLM={vllm_time:.1f}s, env_step={env_step_time:.1f}s)")

        print(f"\n    {'═'*50}")
        print(f"    📊 Reward Calculation Summary")
        print(f"    {'═'*50}")
        reward_results = []
        for idx, (traj, correctness) in enumerate(zip(trajectories, correctness_scores)):
            result = reward_calculator.compute_reward(
                trajectory=traj,
                correctness=correctness,
            )
            reward_results.append(result)
            print(f"    [{idx+1}] DSL: {traj.final_dsl}")
            print(f"        Total: {result.total_reward:.3f} = Base({result.base_reward:.2f}) + Structure({result.structure_reward:.3f}) + Correctness({result.correctness_reward:.3f})")

        sources = [cd.get("source", "") for cd in correctness_details]

        loss = compute_grpo_loss(model, tokenizer, trajectories, reward_results, config, sources=sources)
        accumulated_loss += loss.item()

        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.get('max_grad_norm', 1.0)
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if config.get('use_vllm_api', False) and config.get('enable_lora_sync', False):
                vllm_base_url = config.get('vllm_base_url', 'http://localhost:8003/v1')
                sync_success = sync_lora_to_vllm(model, vllm_base_url, logger)
                if sync_success:
                    print(f"    ✅ [vLLM] LoRA weights synced to vLLM (on-policy)")
                else:
                    print(f"    ⚠️ [vLLM] LoRA sync failed, next rollout uses base model")

            avg_reward = np.mean([r.total_reward for r in reward_results])
            avg_turns = np.mean([len(t.turns) for t in trajectories])

            try:
                from collections import defaultdict

                grouped = defaultdict(list)
                for row in correctness_details:
                    grouped[row.get("source", "")].append(row)

                print(f"\n📈 Step {step+1} Metrics (Intl):")
                for src in sorted(k for k in grouped.keys() if k):
                    rows = grouped[src]
                    ptype = rows[0].get("problem_type", "")
                    n = len(rows)

                    if ptype == "qa":
                        em = sum(float(r.get("details", {}).get("qa_em", 0.0)) for r in rows) / max(n, 1)
                        f1 = sum(float(r.get("details", {}).get("qa_f1", r.get("correctness", 0.0))) for r in rows) / max(n, 1)
                        print(f"  {src}: EM={em:.3f} | F1={f1:.3f} (n={n})")
                    elif ptype == "code":
                        pass_1 = sum(1 for r in rows if float(r.get("correctness", 0.0)) >= 1.0) / max(n, 1)
                        print(f"  {src}: pass@1={pass_1:.3f} (n={n})")
                    else:
                        acc = sum(1 for r in rows if float(r.get("correctness", 0.0)) >= 1.0) / max(n, 1)
                        print(f"  {src}: acc={acc:.3f} (n={n})")
            except Exception as e:
                print(f"  ⚠️ Metrics summary failed: {e}")

            print(f"\n📊 Step {step+1} Stats:")
            print(f"  Loss: {accumulated_loss:.4f}")
            print(f"  Avg Reward: {avg_reward:.4f}")
            print(f"  Avg Turns: {avg_turns:.1f}")
            print(f"  Time: {time.time() - step_start:.1f}s")

            if wandb_run:
                log_dict = {
                    'step': step + 1,
                    'loss': accumulated_loss,
                    'avg_reward': avg_reward,
                    'avg_turns': avg_turns,
                    'learning_rate': scheduler.get_last_lr()[0],
                    'correctness_threshold': current_threshold,
                }

                log_dict['reward/total_mean'] = np.mean([r.total_reward for r in reward_results])
                log_dict['reward/total_std'] = np.std([r.total_reward for r in reward_results])
                log_dict['reward/base_mean'] = np.mean([r.base_reward for r in reward_results])
                log_dict['reward/structure_mean'] = np.mean([r.structure_reward for r in reward_results])
                log_dict['reward/correctness_mean'] = np.mean([r.correctness_reward for r in reward_results])
                log_dict['reward/correctness_activated_ratio'] = np.mean([1.0 if r.correctness_activated else 0.0 for r in reward_results])

                log_dict['trajectory/avg_turns'] = avg_turns
                log_dict['trajectory/max_turns'] = max(len(t.turns) for t in trajectories)
                log_dict['trajectory/min_turns'] = min(len(t.turns) for t in trajectories)

                try:
                    import re as re_module
                    from collections import Counter
                    dsl_list = [t.final_dsl for t in trajectories if t.final_dsl]

                    unique_dsls = set(dsl_list)
                    log_dict['workflow/diversity'] = len(unique_dsls) / max(len(dsl_list), 1)
                    log_dict['workflow/unique_count'] = len(unique_dsls)
                    log_dict['workflow/total_count'] = len(dsl_list)

                    all_operators = []
                    unique_ops_per_workflow = []
                    for dsl in dsl_list:
                        ops = re_module.findall(r'\b(Plan|Decompose|Programmer|Custom|Test|Review|Verify|Revise|ScEnsemble|Aggregate|AnswerGenerate|Format)\b', dsl)
                        all_operators.extend(ops)
                        unique_ops_per_workflow.append(len(set(ops)))

                    op_counter = Counter(all_operators)
                    total_ops = sum(op_counter.values())

                    log_dict['operator/unique_types_mean'] = np.mean(unique_ops_per_workflow) if unique_ops_per_workflow else 0
                    log_dict['operator/unique_types_max'] = max(unique_ops_per_workflow) if unique_ops_per_workflow else 0
                    log_dict['operator/unique_types_min'] = min(unique_ops_per_workflow) if unique_ops_per_workflow else 0
                    log_dict['operator/total_types_used'] = len(op_counter)

                    for op_name in ['Plan', 'Decompose', 'Programmer', 'Custom', 'Test', 'Review', 'Verify', 'Revise', 'ScEnsemble', 'Aggregate', 'AnswerGenerate']:
                        count = op_counter.get(op_name, 0)
                        log_dict[f'operator/{op_name}_count'] = count
                        log_dict[f'operator/{op_name}_freq'] = count / max(total_ops, 1)

                    parallel_count = sum(1 for dsl in dsl_list if '[' in dsl)
                    conditional_count = sum(1 for dsl in dsl_list if '?' in dsl)
                    loop_count = sum(1 for dsl in dsl_list if 'x' in dsl and any(c.isdigit() for c in dsl))

                    log_dict['structure/parallel_ratio'] = parallel_count / max(len(dsl_list), 1)
                    log_dict['structure/conditional_ratio'] = conditional_count / max(len(dsl_list), 1)
                    log_dict['structure/loop_ratio'] = loop_count / max(len(dsl_list), 1)

                    avg_ops = total_ops / max(len(dsl_list), 1)
                    log_dict['workflow/avg_operators'] = avg_ops

                except Exception as e:
                    print(f"  ⚠️ Workflow diversity metrics failed: {e}")

                try:
                    for src in sorted(k for k in grouped.keys() if k):
                        rows = grouped[src]
                        ptype = rows[0].get("problem_type", "")
                        n = len(rows)
                        src_key = src.replace("-", "_").replace("/", "_")

                        correctness_vals = [float(r.get("correctness", 0.0)) for r in rows]
                        log_dict[f'dataset/{src_key}/correctness_mean'] = np.mean(correctness_vals)
                        log_dict[f'dataset/{src_key}/correctness_std'] = np.std(correctness_vals)
                        log_dict[f'dataset/{src_key}/sample_count'] = n

                        if ptype == "qa":
                            em = sum(float(r.get("details", {}).get("qa_em", 0.0)) for r in rows) / max(n, 1)
                            f1 = sum(float(r.get("details", {}).get("qa_f1", r.get("correctness", 0.0))) for r in rows) / max(n, 1)
                            log_dict[f'dataset/{src_key}/em'] = em
                            log_dict[f'dataset/{src_key}/f1'] = f1
                        elif ptype == "code":
                            pass_1 = sum(1 for r in rows if float(r.get("correctness", 0.0)) >= 1.0) / max(n, 1)
                            log_dict[f'dataset/{src_key}/pass_at_1'] = pass_1
                        else:
                            acc = sum(1 for r in rows if float(r.get("correctness", 0.0)) >= 1.0) / max(n, 1)
                            log_dict[f'dataset/{src_key}/accuracy'] = acc

                        src_rewards = [r.total_reward for r, cd in zip(reward_results, correctness_details) if cd.get("source") == src]
                        if src_rewards:
                            log_dict[f'dataset/{src_key}/reward_mean'] = np.mean(src_rewards)
                            log_dict[f'dataset/{src_key}/reward_std'] = np.std(src_rewards)
                except Exception as e:
                    print(f"  ⚠️ wandb dataset metrics failed: {e}")

                wandb.log(log_dict, step=step+1, commit=True)

            accumulated_loss = 0.0

        if (step + 1) % save_every == 0:
            checkpoint_path = output_dir / f"checkpoint_step_{step+1}"
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
            training_state = {
                'step': step + 1,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'random_state': random.getstate(),
                'np_random_state': np.random.get_state(),
                'torch_random_state': torch.get_rng_state(),
            }
            if torch.cuda.is_available():
                training_state['torch_cuda_random_state'] = torch.cuda.get_rng_state_all()
            torch.save(training_state, checkpoint_path / 'training_state.pt')
            logger.info(f"💾 Checkpoint saved: {checkpoint_path} (with optimizer/scheduler/random state)")

        gc.collect()
        torch.cuda.empty_cache()

        global_step = step + 1

    # Close prompt log (best-effort)
    try:
        if prompt_log_fh:
            prompt_log_fh.close()
    except Exception:
        pass

    final_path = output_dir / "final"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"✅ Training complete! Final model saved to {final_path}")

    if wandb_run:
        wandb.finish()
    if rollout_pool is not None:
        rollout_pool.shutdown(wait=True)


def compute_grpo_loss(model, tokenizer, trajectories, reward_results, config, sources=None):
    """Compute GRPO policy gradient loss with action_mask."""
    import torch.nn.functional as F
    import numpy as np
    import gc
    from collections import defaultdict

    if sources is not None and len(sources) == len(reward_results):
        source_groups = defaultdict(list)
        for idx, (r, src) in enumerate(zip(reward_results, sources)):
            source_groups[src].append((idx, r.total_reward))

        advantages = [0.0] * len(reward_results)
        print(f"    [Loss] GRPO (with action_mask):")
        for src, group in source_groups.items():
            indices, rewards = zip(*group)
            group_mean = np.mean(rewards)
            group_std = max(np.std(rewards), 0.01)
            for idx, reward in zip(indices, rewards):
                advantages[idx] = (reward - group_mean) / group_std
            print(f"      {src}: n={len(group)}, mean={group_mean:.3f}, std={group_std:.3f}")
    else:
        reward_values = [r.total_reward for r in reward_results]
        mean_reward = np.mean(reward_values)
        std_reward = max(np.std(reward_values), 0.01)
        advantages = [(r - mean_reward) / std_reward for r in reward_values]
        print(f"    [GRPO] : n={len(reward_values)}, mean={mean_reward:.3f}, std={std_reward:.3f}")

    accumulated_loss = 0.0
    valid_samples = 0
    total_model_tokens = 0
    total_masked_tokens = 0
    CLEANUP_INTERVAL = 2

    for idx, (traj, reward_result, adv) in enumerate(zip(trajectories, reward_results, advantages)):
        all_token_ids = []
        action_mask = []

        for turn in traj.turns:
            if turn.model_response:
                model_ids = tokenizer.encode(turn.model_response, add_special_tokens=False)
                all_token_ids.extend(model_ids)
                action_mask.extend([1] * len(model_ids))

            if turn.feedback:
                feedback_ids = tokenizer.encode(turn.feedback, add_special_tokens=False)
                all_token_ids.extend(feedback_ids)
                action_mask.extend([0] * len(feedback_ids))

        if not all_token_ids or sum(action_mask) == 0:
            continue

        max_length = 1024
        if len(all_token_ids) > max_length:
            head_len = max_length // 2
            tail_len = max_length - head_len
            all_token_ids = all_token_ids[:head_len] + all_token_ids[-tail_len:]
            action_mask = action_mask[:head_len] + action_mask[-tail_len:]

        input_ids = torch.tensor([all_token_ids], dtype=torch.long, device=model.device)
        mask_tensor = torch.tensor([action_mask], dtype=torch.float, device=model.device)

        try:
            with torch.amp.autocast('cuda', enabled=True):
                outputs = model(input_ids=input_ids)
                logits = outputs.logits[:, :-1, :]
                labels = input_ids[:, 1:]
                shifted_mask = mask_tensor[:, 1:]

                log_probs = F.log_softmax(logits, dim=-1)
                token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

                masked_log_probs = token_log_probs * shifted_mask
                n_model_tokens = shifted_mask.sum()

                if n_model_tokens > 0:
                    sample_loss = -adv * masked_log_probs.sum() / n_model_tokens
                else:
                    continue

            sample_loss.backward()
            accumulated_loss += sample_loss.item()
            valid_samples += 1
            total_model_tokens += int(n_model_tokens.item())
            total_masked_tokens += int((1 - shifted_mask).sum().item())

            del outputs, logits, labels, log_probs, token_log_probs, sample_loss
            del input_ids, mask_tensor, shifted_mask, masked_log_probs

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"[WARNING] OOM at sample {idx}, skipping...")
                torch.cuda.empty_cache()
                gc.collect()
                continue
            raise

        if (idx + 1) % CLEANUP_INTERVAL == 0:
            torch.cuda.empty_cache()
            gc.collect()

    if valid_samples > 0:
        avg_loss = accumulated_loss / valid_samples
        mask_ratio = total_masked_tokens / (total_model_tokens + total_masked_tokens) if (total_model_tokens + total_masked_tokens) > 0 else 0
        print(f"    [Loss] Loss stats: samples={valid_samples}, model_tokens={total_model_tokens}, masked_tokens={total_masked_tokens}, mask_ratio={mask_ratio:.2%}")
    else:
        avg_loss = 0.0

    torch.cuda.empty_cache()
    gc.collect()

    return torch.tensor(avg_loss, device=model.device)


def _llm_judge_answer(model_answer: str, ground_truth: str, problem_type: str) -> Optional[float]:
    """Use LLM for semantic judgment with retry mechanism."""
    import time
    MAX_RETRIES = 3

    for attempt in range(MAX_RETRIES):
        try:
            from openai import OpenAI

            base_url = "http://localhost:8006/v1"

            client = OpenAI(
                api_key="EMPTY",
                base_url=base_url,
                timeout=60
            )

            prompt = f"""

: {problem_type}
 (Ground Truth): {ground_truth[:500]}
: {model_answer[:500]}


1. ****: CORRECT
   - : "Logan""The film is Logan""...film 'Logan'..." → CORRECT
2. : 1.5707963 ≈ π/230 = 30.06.633 ≈ 2√11 → CORRECT
3. : (2/5, 3/5) = (2/5,3/5)i√2 = i*sqrt(2) → CORRECT
4. Unanswerable: "unanswerable"
5. : markdown



: CORRECT  INCORRECT"""

            response = client.chat.completions.create(
                model="gpt-oss-120b",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0
            )

            msg = response.choices[0].message
            result = msg.content or ""
            if not result and hasattr(msg, 'reasoning_content') and msg.reasoning_content:
                result = msg.reasoning_content

            result = result.strip().upper() if result else ""

            if "INCORRECT" in result:
                print(f"    [LLMJudge] LLM Judge (gpt-oss-120b): INCORRECT")
                return 0.0
            elif "CORRECT" in result:
                print(f"    [LLMJudge] LLM Judge (gpt-oss-120b): CORRECT")
                return 1.0
            print(f"    [LLMJudge] LLM Judge:  (result={result[:50]}...)")
            return None

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"    [Retry] LLM Judge retry {attempt+1}/{MAX_RETRIES}: {e}")
                time.sleep(1 * (attempt + 1))
                continue
            print(f"    [LLMJudge] LLM judge error: {e}")
            return None

    return None


def _is_unanswerable(text: str) -> bool:
    """Detect if answer indicates 'unanswerable'."""
    if not text:
        return False

    text_lower = text.lower().strip()

    unanswerable_markers = [
        'unanswerable',
        'no answer',
        'cannot be answered',
        'not answerable',
        'cannot answer',
        "can't be answered",
        'no information',
        'not mentioned',
        'not provided',
        'insufficient information',
        'unknown',
    ]

    for marker in unanswerable_markers:
        if marker in text_lower:
            return True

    import string
    cleaned = ''.join(c for c in text_lower if c not in string.punctuation and not c.isspace())
    if not cleaned or cleaned in ['none', 'na', 'n/a', '']:
        return True

    return False


def _compute_correctness(
    model_answer: str,
    ground_truth: str,
    problem_type: str,
    source: str = None,
    test: Optional[str] = None,
    entry_point: Optional[str] = None,
    code_for_test: Optional[str] = None,
    use_llm_judge: bool = False,
    meta: Optional[Dict] = None,
    return_details: bool = False,
) -> Any:
    """Compute correctness score between model answer and ground truth.

    Scoring logic:
    1. QA: F1 Score (token-level after normalize_answer)
    2. Math: math_equal (string -> numeric 1e-3 -> symbolic SymPy)
    3. Code: binary (PASS=1.0, FAIL=0.0)

    Args:
        model_answer: Model generated answer
        ground_truth: Expected correct answer
        problem_type: Problem type (math/code/qa)
        source: Data source
        test: Code test cases
        entry_point: Code entry function name
        meta: Full meta info (including problem for BigCodeBench)
        code_for_test: Source code for code tasks
        use_llm_judge: Whether to use LLM Judge as fallback

    Returns:
        correctness score (default), or (score, details) if return_details=True
    """
    try:
        from src.answer_extractor import AnswerExtractor

        extractor = AnswerExtractor()

        details: Dict[str, Any] = {}

        if not model_answer:
            if return_details:
                details["skip_reason"] = "empty_model_answer"
                return 0.0, details
            return 0.0

        model_str = str(model_answer).strip()
        truth_str = str(ground_truth).strip() if ground_truth else ""

        if not truth_str:
            if return_details:
                details["skip_reason"] = "empty_ground_truth"
                return 0.0, details
            return 0.0

        if problem_type == 'qa':
            extracted_model = extractor.extract_answer(model_str, problem_type='qa', is_ground_truth=False)
            extracted_truth = extractor.extract_answer(truth_str, problem_type='qa', is_ground_truth=True)

            normalized_model = _normalize_qa_answer(extracted_model)
            normalized_truth = _normalize_qa_answer(extracted_truth)
            if normalized_model != extracted_model or normalized_truth != extracted_truth:
                print(f"    [QANorm] QA: '{extracted_model[:30]}' -> '{normalized_model[:30]}'")

            score = _aflow_qa_f1_score(normalized_model, normalized_truth)
            em = _aflow_qa_em_score(normalized_model, normalized_truth)
            details.update({"qa_f1": float(score), "qa_em": float(em)})
            print(f"    [Eval] QA: EM={em:.3f}, F1={score:.3f}")

            if score < 0.5:
                gt_lower = extracted_truth.lower().strip()
                model_lower = extracted_model.lower().strip()
                if gt_lower and len(gt_lower) >= 3 and gt_lower in model_lower:
                    print(f"    [QA] QA: GT '{gt_lower[:30]}' found in model answer -> 0.9")
                    score = 0.9
                    details["qa_f1"] = float(score)
                    if return_details:
                        return score, details
                    return score

            if score < 0.3 and use_llm_judge:
                llm_score = _llm_judge_answer(model_str, truth_str, problem_type)
                if llm_score is not None:
                    print(f"    [Eval] LLM Judge override (QA): {score:.2f} -> {llm_score:.2f}")
                    score = llm_score
            if return_details:
                return score, details
            return score

        elif problem_type == 'math':
            extracted_model = extractor.extract_answer(model_str, problem_type='math', is_ground_truth=False)
            extracted_truth = extractor.extract_answer(truth_str, problem_type='math', is_ground_truth=True)

            if _aflow_math_equal(extracted_model, extracted_truth):
                score = 1.0
            else:
                score = 0.0
            print(f"    [Eval] Math score: {score:.1f} (model='{extracted_model[:30]}', truth='{extracted_truth[:30]}')")
            details.update({"math_equal": bool(score == 1.0)})

            if score < 0.5 and use_llm_judge:
                llm_score = _llm_judge_answer(model_str, truth_str, problem_type)
                if llm_score is not None:
                    print(f"    [Eval] LLM Judge override (math): {score:.2f} -> {llm_score:.2f}")
                    score = llm_score
            if return_details:
                return score, details
            return score

        elif problem_type == 'code':
            code_to_test = code_for_test if code_for_test else model_str

            meta = meta or {}
            source_lower = (source or '').lower()
            code_prompt = ""

            if 'humaneval' in source_lower:
                prompt_for_humaneval = meta.get('problem', '')
                if not prompt_for_humaneval:
                    prompt_for_humaneval = meta.get('prompt', '')
                meta['prompt'] = prompt_for_humaneval
                print(f"    [CodeEval] HumanEval prompt extracted (len={len(prompt_for_humaneval)})")

            elif 'bigcodebench' in source_lower:
                code_prompt = meta.get('code_prompt', '')
                if not code_prompt:
                    problem_text = meta.get('problem', '')
                    if problem_text:
                        code_prompt = _extract_code_prompt_from_problem(problem_text)
                if code_prompt:
                    print(f"    [CodeEval] BigCodeBench code_prompt extracted (len={len(code_prompt)})")

            code_source = "code_for_test" if code_for_test else "model_str"
            print(f"    [Code] Code source: {code_source}")
            print(f"    [Code] Code preview: {code_to_test[:150] if code_to_test else 'None'}...")

            invalid_code_placeholders = {'CODE_GENERATED', 'NO_CODE', 'EMPTY', ''}
            if code_to_test and code_to_test.strip() in invalid_code_placeholders:
                print(f"    [CodeCheck] Invalid code placeholder detected: '{code_to_test[:50]}' -> 0.0")
                details.update({"code_invalid_placeholder": True, "placeholder": code_to_test.strip()})
                if return_details:
                    return 0.0, details
                return 0.0

            if not entry_point and test:
                entry_point = _extract_entry_point_from_test(test)
                if entry_point:
                    print(f"    [Eval] Auto-extracted entry_point: {entry_point}")

            if entry_point and test and code_to_test:
                code_to_test = _validate_and_fix_signature(code_to_test, entry_point, test)

            print(f"    [Eval] Code evaluation: test={bool(test)}, entry_point={entry_point}")

            if test and code_to_test:
                prompt_for_eval = meta.get('prompt', '')
                passed = _aflow_code_check(
                    code_to_test, test, entry_point,
                    prompt=prompt_for_eval,
                    source=source or '',
                    problem=meta.get('problem', ''),
                    code_prompt=code_prompt,
                )
                score = 1.0 if passed else 0.0
                print(f"    [CodeEval] Code test result: {'PASS' if passed else 'FAIL'} -> {score}")
                details.update({"code_pass": bool(passed)})
                if return_details:
                    return score, details
                return score
            else:
                try:
                    compile(code_to_test, '<string>', 'exec')
                    print(f"    [Eval] No test cases, syntax OK -> 0.5")
                    score = 0.5
                    details.update({"code_syntax_ok": True})
                    if return_details:
                        return score, details
                    return score
                except:
                    print(f"    [Eval] No test cases, syntax error -> 0.0")
                    score = 0.0
                    details.update({"code_syntax_ok": False})
                    if return_details:
                        return score, details
                    return score

        else:
            extracted_model = extractor.extract_answer(model_str, problem_type='math', is_ground_truth=False)
            extracted_truth = extractor.extract_answer(truth_str, problem_type='math', is_ground_truth=True)

            if _aflow_math_equal(extracted_model, extracted_truth):
                score = 1.0
            else:
                score = 0.0

            if score < 0.5 and use_llm_judge:
                llm_score = _llm_judge_answer(model_str, truth_str, problem_type)
                if llm_score is not None:
                    score = llm_score
            if return_details:
                return score, details
            return score

    except Exception as e:
        print(f"[WARNING] Evaluation error: {e}")
        score = _simple_compare(model_answer, ground_truth, problem_type)
        if return_details:
            return score, {"error": str(e)}
        return score



def _aflow_normalize_answer(s: str) -> str:
    """AFlow (F1)

    1. 
    2. 
    3.  (a, an, the)
    4. 
    """
    import string

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _normalize_qa_answer(answer: str) -> str:
    """Normalize QA answer format to improve EM matching."""
    if not answer:
        return ""

    answer = answer.strip()

    prefixes_to_remove = [
        'the ', 'a ', 'an ',
        'in ', 'on ', 'at ', 'by ', 'to ', 'for ', 'from ', 'with ',
        'it is ', 'it was ', 'they are ', 'they were ',
        'this is ', 'that is ', 'there is ', 'there are ',
    ]

    answer_lower = answer.lower()
    for prefix in prefixes_to_remove:
        if answer_lower.startswith(prefix):
            answer = answer[len(prefix):]
            answer_lower = answer.lower()

    answer = answer.rstrip('.,;:!?')

    return answer.strip()


def _aflow_qa_f1_score(prediction: str, ground_truth: str) -> float:
    """AFlowQA F1"""
    from collections import Counter

    answers = ground_truth.split("|")
    f1_scores = []

    for answer in answers:
        answer = answer.strip()
        if not answer:
            continue

        prediction_tokens = _aflow_normalize_answer(prediction).split()
        ground_truth_tokens = _aflow_normalize_answer(answer).split()

        if not prediction_tokens or not ground_truth_tokens:
            f1_scores.append(0.0)
            continue

        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            f1_scores.append(0.0)
            continue

        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        f1_scores.append(f1)

    return max(f1_scores) if f1_scores else 0.0


def _aflow_qa_em_score(prediction: str, ground_truth: str) -> float:
    """AFlow/ QA EM |"""
    answers = [a.strip() for a in str(ground_truth).split("|") if a.strip()]
    pred_norm = _aflow_normalize_answer(str(prediction))
    for ans in answers:
        if pred_norm == _aflow_normalize_answer(ans):
            return 1.0
    return 0.0


def _normalize_math_answer(s: str) -> str:
    """Normalize math answer format."""
    import re
    if not s:
        return s

    result = str(s).strip()

    frac_pattern = r'(-?)\\d?frac\{([^}]+)\}\{([^}]+)\}'
    match = re.search(frac_pattern, result)
    if match:
        sign = match.group(1)
        result = f"{sign}{match.group(2)}/{match.group(3)}"

    result = re.sub(r'\^\\circ', '', result)
    result = re.sub(r'\\circ', '', result)
    result = re.sub(r'°', '', result)
    result = re.sub(r'\s*degrees?\s*$', '', result, flags=re.IGNORECASE)

    result = re.sub(r'\\%', '%', result)

    result = re.sub(r'\*([a-zA-Z])', r'\1', result)

    bracket_match = re.match(r'^([+-]?\d+\.?\d*)\s*\(', result)
    if bracket_match:
        result = bracket_match.group(1)

    # 6. LaTeX sqrt → sqrt
    result = re.sub(r'\\sqrt\{([^}]+)\}', r'sqrt(\1)', result)
    result = re.sub(r'\\sqrt', 'sqrt', result)

    # 7. LaTeX pi → pi
    result = re.sub(r'\\pi', 'pi', result)

    result = result.strip()

    return result


def _normalize_latex_fraction(s: str) -> str:
    """Backward compatible wrapper for _normalize_math_answer."""
    return _normalize_math_answer(s)


def _aflow_math_equal(prediction: str, reference: str) -> bool:
    """AFlow →  → 

    AFlow
    """
    from math import isclose

    if not prediction or not reference:
        return False

    prediction = str(prediction).strip()
    reference = str(reference).strip()

    prediction = _normalize_latex_fraction(prediction)
    reference = _normalize_latex_fraction(reference)

    if prediction == reference:
        return True

    if prediction.lower() == reference.lower():
        return True

    try:
        pred_num = _aflow_parse_digits(prediction)
        ref_num = _aflow_parse_digits(reference)
        if pred_num is not None and ref_num is not None:
            return isclose(pred_num, ref_num, abs_tol=1e-3)
    except:
        pass

    try:
        return _aflow_symbolic_equal(prediction, reference)
    except:
        pass

    return False


def _aflow_parse_digits(num: str) -> Optional[float]:
    """Parse digits with support for comma, percentage, fraction and LaTeX symbols."""
    import math

    num = re.sub(",", "", str(num)).strip()

    base_match = re.fullmatch(r'([0-9A-Za-z_]+)_\{?(\d{1,2})\}?', num)
    if base_match:
        digits_part, base_part = base_match.group(1), base_match.group(2)
        try:
            base = int(base_part)
            if 2 <= base <= 36:
                digits_clean = digits_part.replace('_', '')
                value = int(digits_clean, base)
                return float(value)
        except Exception:
            pass

    try:
        return float(num)
    except:
        pass

    if '/' in num and '\\' not in num:
        try:
            parts = num.split('/')
            if len(parts) == 2:
                return float(parts[0].strip()) / float(parts[1].strip())
        except:
            pass

    if num.endswith("%"):
        num_clean = num[:-1]
        if num_clean.endswith("\\"):
            num_clean = num_clean[:-1]
        try:
            return float(num_clean) / 100
        except:
            pass

    pi_match = re.match(r'^(-?\d+\.?\d*)\s*\\?\\?pi$', num, re.IGNORECASE)
    if pi_match:
        try:
            return float(pi_match.group(1)) * math.pi
        except:
            pass

    if num.lower() in ['\\pi', 'pi', '\\\\pi']:
        return math.pi

    sqrt_match = re.match(r'^(-?\d+\.?\d*)\s*\\?\\?sqrt\{(\d+\.?\d*)\}$', num, re.IGNORECASE)
    if sqrt_match:
        try:
            coef = float(sqrt_match.group(1))
            radicand = float(sqrt_match.group(2))
            return coef * math.sqrt(radicand)
        except:
            pass

    sqrt_alone = re.match(r'^\\?\\?sqrt\{(\d+\.?\d*)\}$', num, re.IGNORECASE)
    if sqrt_alone:
        try:
            return math.sqrt(float(sqrt_alone.group(1)))
        except:
            pass

    frac_match = re.match(r'^(-?)\\?\\?d?frac\{([^}]+)\}\{([^}]+)\}$', num)
    if frac_match:
        try:
            sign = -1 if frac_match.group(1) == '-' else 1
            numer = float(frac_match.group(2))
            denom = float(frac_match.group(3))
            return sign * numer / denom
        except:
            pass

    pi_frac_match = re.match(r'^(-?\d+\.?\d*)\s*\\?\\?pi\s*/\s*(\d+\.?\d*)$', num, re.IGNORECASE)
    if pi_frac_match:
        try:
            numer = float(pi_frac_match.group(1))
            denom = float(pi_frac_match.group(2))
            return numer * math.pi / denom
        except:
            pass

    return None


def _aflow_symbolic_equal(a: str, b: str) -> bool:
    """AFlow"""
    from math import isclose
    try:
        from sympy import N, simplify
        from sympy.parsing.latex import parse_latex
        from sympy.parsing.sympy_parser import parse_expr

        def _parse(s):
            for f in [parse_latex, parse_expr]:
                try:
                    return f(s)
                except:
                    pass
            return s

        a_parsed = _parse(a)
        b_parsed = _parse(b)

        try:
            if simplify(a_parsed - b_parsed) == 0:
                return True
        except:
            pass

        try:
            if isclose(float(N(a_parsed)), float(N(b_parsed)), abs_tol=1e-3):
                return True
        except:
            pass

        return False
    except ImportError:
        return False


def _extract_code_prompt_from_problem(problem: str) -> str:
    """Extract code_prompt from BigCodeBench problem field."""
    if not problem:
        return ''

    import re
    match = re.search(r'```(?:python)?\s*\n(.*?)```', problem, re.DOTALL)
    if match:
        code_prompt = match.group(1).strip()
        if 'def ' in code_prompt:
            return code_prompt

    if 'starting with:' in problem.lower():
        parts = re.split(r'starting with:\s*', problem, flags=re.IGNORECASE)
        if len(parts) > 1:
            code_part = parts[1]
            match = re.search(r'```(?:python)?\s*\n(.*?)```', code_part, re.DOTALL)
            if match:
                return match.group(1).strip()

    return ''


def _validate_and_fix_signature(code: str, entry_point: str, test: str) -> str:
    """Validate and fix function signature - disabled due to bugs."""
    return code


def _extract_entry_point_from_test(test: str) -> Optional[str]:
    """Extract entry_point from test code."""
    if 'task_func(' in test:
        return 'task_func'

    match = re.search(r'candidate\s*=\s*([a-zA-Z_][a-zA-Z0-9_]*)', test)
    if match:
        return match.group(1)

    match = re.search(r'result\s*=\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', test)
    if match:
        return match.group(1)

    BUILTIN_FUNCS = ('True', 'False', 'None', 'len', 'str', 'int', 'float', 'list', 'dict', 'set', 'tuple', 'abs', 'sum', 'max', 'min', 'sorted', 'type', 'isinstance', 'print')
    match = re.search(r'assert\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', test)
    if match and match.group(1) not in BUILTIN_FUNCS:
        return match.group(1)

    match = re.search(r'self\.assert\w+\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', test)
    if match and match.group(1) not in BUILTIN_FUNCS:
        return match.group(1)

    return None


def _aflow_code_check(solution: str, test: str, entry_point: Optional[str],
                      timeout: int = 15,
                      prompt: str = '',
                      source: str = '',
                      problem: str = '',
                      code_prompt: str = '') -> bool:
    """Evaluate code according to official standards."""
    if not solution or not test:
        return False

    try:
        from src.code_execution import run_code_check
    except Exception as e:
        print(f"    [CodeEval] Code check backend import failed: {e}")
        return False

    result = run_code_check(
        solution=solution,
        test=test,
        entry_point=entry_point,
        prompt=prompt or "",
        problem=problem or "",
        source=source or "",
        timeout=timeout,
        code_prompt=code_prompt or "",
    )

    if not result.passed and result.error_type:
        err = (result.error or "")[:200]
        print(f"    [CodeEval] Code check ({result.backend}) failed: {result.error_type}: {err}")

    return bool(result.passed)


def _simple_compare(model_answer: str, ground_truth: str, problem_type: str) -> float:
    """"""
    import re

    if not model_answer or not ground_truth:
        return 0.0

    model_str = str(model_answer).strip().lower()
    truth_str = str(ground_truth).strip().lower()

    boxed_match = re.search(r'\\boxed\{([^}]+)\}', str(ground_truth))
    if boxed_match:
        truth_str = boxed_match.group(1).strip().lower()

    if model_str == truth_str:
        return 1.0

    def extract_number(s):
        s = s.replace('$', '').replace(',', '')
        numbers = re.findall(r'-?\d+\.?\d*', s)
        if numbers:
            return float(numbers[-1])
        return None

    model_num = extract_number(model_str)
    truth_num = extract_number(truth_str)

    if model_num is not None and truth_num is not None:
        if abs(model_num - truth_num) < 1e-6:
            return 1.0
        elif abs(model_num - truth_num) / max(abs(truth_num), 1e-8) < 0.01:
            return 0.8

    if truth_str in model_str:
        return 0.8
    if model_str in truth_str:
        return 0.5

    return 0.0


def _clean_code_markdown(code: str) -> str:
    """Clean markdown pollution from code."""
    if not code or not isinstance(code, str):
        return code if code else ""

    original_len = len(code)

    code_block_match = re.search(r'```(?:python)?\s*\n(.*?)```', code, re.DOTALL)
    if code_block_match:
        code = code_block_match.group(1)
        print(f"    [Markdown] Extracted code from markdown block")

    lines = code.split('\n')
    clean_lines = []
    skip_until_code = True

    for line in lines:
        stripped = line.strip()

        if stripped.startswith('**') and stripped.endswith('**'):
            continue
        if stripped.startswith('#'):
            continue

        if stripped.startswith(('def ', 'class ', 'import ', 'from ', '@', 'async def ')):
            skip_until_code = False

        if skip_until_code:
            if stripped and stripped[0].isalpha() and not any(
                stripped.startswith(kw) for kw in ['def ', 'class ', 'import ', 'from ', 'return ', 'if ', 'for ', 'while ', 'try:', 'except', 'with ', 'async ']
            ):
                if len(stripped) > 50 and ' ' in stripped:
                    continue

        clean_lines.append(line)

    code = '\n'.join(clean_lines).strip()

    if len(code) < original_len:
        print(f"    [Markdown] Cleaned markdown: {original_len} -> {len(code)} chars")

    return code


def _extract_operator_answer(result: any, problem_type: str = None) -> str:
    """Extract answer string from operator output.

    Priority:
    - Code tasks: code > output > others
    - Other tasks: output > response > answer > solution > others

    Args:
        result: Raw result from operator (may be dict, str or None)
        problem_type: Problem type (code/math/qa/reasoning)

    Returns:
    """
    if result is None:
        return ""

    if isinstance(result, dict):
        if problem_type == 'code':
            if 'code' in result and result['code']:
                code_str = str(result['code'])
                code_str = _clean_code_markdown(code_str)
                print(f"    [Code] Extracted and cleaned code field (len={len(code_str)})")
                return code_str
            if 'output' in result and result['output']:
                output_str = str(result['output'])
                if 'def ' in output_str or 'class ' in output_str or 'import ' in output_str:
                    output_str = _clean_code_markdown(output_str)
                    return output_str

        if 'output' in result and result['output']:
            output_val = result['output']
            if isinstance(output_val, list) and len(output_val) > 0:
                for item in output_val:
                    if isinstance(item, str) and item.strip():
                        return item.strip()
                return str(output_val[0])
            # Programmer: {'code': '...', 'output': '42'}
            return str(output_val)
        elif 'response' in result and result['response']:
            # Custom: {'response': '...'}
            return str(result['response'])
        elif 'answer' in result and result['answer']:
            # AnswerGenerate/Verify: {'answer': '...', ...}
            return str(result['answer'])
        elif 'solution' in result and result['solution']:
            # Revise/MdEnsemble: {'solution': '...'}
            return str(result['solution'])
        elif 'aggregated_answer' in result and result['aggregated_answer']:
            # Aggregate: {'aggregated_answer': '...'}
            return str(result['aggregated_answer'])
        elif 'code' in result and result['code']:
            return str(result['code'])
        elif 'thought' in result:
            thought = str(result.get('thought', ''))
            numbers = re.findall(r'-?\d+\.?\d*', thought)
            if numbers:
                return numbers[-1]
            return thought
        else:
            print(f"[WARNING] Unknown operator output format: {list(result.keys())}")
            return str(result)
    else:
        return str(result) if result else ""


def _infer_problem_type(source: str) -> str:
    """"""
    source_lower = source.lower()
    if source_lower in ['gsm8k', 'math']:
        return 'math'
    elif source_lower in ['humaneval', 'bigcodebench', 'mbpp', 'mbppplus', 'code_exercises']:
        return 'code'
    elif source_lower in ['hotpotqa', 'squad_v2']:
        return 'qa'
    else:
        return 'reasoning'


def main():
    parser = argparse.ArgumentParser(description="Interactive GRPO Training")
    parser.add_argument('--config', type=str, default='config/training_interactive.yaml',
                        help='Path to config file')
    parser.add_argument('--demo', action='store_true',
                        help='Run demo mode (no training)')
    parser.add_argument('--demo-dataset', action='store_true',
                        help='Run demo with problems from dataset')
    parser.add_argument('--use-aflow', action='store_true',
                        help='Use real AFlow executor (requires API key)')
    parser.add_argument('--max_samples', type=int, default=10,
                        help='Max samples for demo')
    parser.add_argument('--num-problems', type=int, default=5,
                        help='Number of problems to test from dataset')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path (e.g., checkpoints/interactive/checkpoint_step_30)')
    args = parser.parse_args()

    print("[DEBUG] Arguments parsed", flush=True)

    config = load_config(args.config)
    print(f"[DEBUG] Config loaded: {args.config}", flush=True)

    lora_sync_path = config.get('lora_sync_path')
    if lora_sync_path:
        set_lora_sync_path(lora_sync_path)
        print(f"[INFO] LoRA sync path set to: {lora_sync_path}", flush=True)

    logger = setup_logging(
        config.get('log_dir', 'logs'),
        config.get('exp_name', 'interactive_grpo'),
    )

    print(f"[DEBUG] Logger setup complete", flush=True)
    logger.info(f"Config: {args.config}")
    logger.info(f"Demo mode: {args.demo}")

    use_vllm_api = config.get('use_vllm_api', False)
    if use_vllm_api:
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if cuda_visible:
            device = 'cuda:0'
            print(f"[INFO] CUDA_VISIBLE_DEVICES={cuda_visible} cuda:0")
        else:
            device = 'cuda:1'
            print(f"[INFO] vLLM API:  {device}vLLM GPU 0")
    else:
        device = config.get('device', 'cuda:0')
    logger.info(f"Using device: {device}")
    print(f"[DEBUG] Using device: {device}", flush=True)

    print("[DEBUG] Loading model...", flush=True)
    model, tokenizer = load_model_and_tokenizer(config, device, resume_path=args.resume)
    print("[DEBUG] Model loaded!", flush=True)

    if args.demo_dataset:
        print("[DEBUG] Running demo-dataset mode...", flush=True)
        demo_dataset_building(
            config, model, tokenizer,
            num_problems=args.num_problems,
            use_aflow=args.use_aflow
        )
    elif args.demo:
        print("[DEBUG] Running demo mode...", flush=True)
        demo_interactive_building(config, model, tokenizer)
    else:
        logger.info("Starting full GRPO training mode...")
        run_full_training(config, model, tokenizer, args, resume_path=args.resume)


if __name__ == "__main__":
    main()
