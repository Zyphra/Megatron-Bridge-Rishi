import sys
import os
import argparse

# --- START PATH HACK ---
# Get the absolute path to the root of your project
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Construct the paths to your src and submodule directories
src_path = os.path.join(project_root, 'src')
submodule_path = os.path.join(project_root, '3rdparty', 'Megatron-LM')

# Forcefully insert these paths at the beginning of sys.path
if submodule_path not in sys.path:
    sys.path.insert(0, submodule_path)
    print(f"[DEBUG] Inserted submodule path: {submodule_path}")
if src_path not in sys.path:
    sys.path.insert(0, src_path)
    print(f"[DEBUG] Inserted src path: {src_path}")

# --- END PATH HACK ---

import megatron
print("--- Python Search Path (sys.path) ---")
for p in sys.path:
    print(p)
print("--- Location of 'megatron' module ---")
try:
    print(megatron.__path__)
except Exception as e:
    print(f"Could not find megatron module path: {e}")
print("-------------------------------------")

import torch
from omegaconf import OmegaConf

from megatron.bridge.recipes.gemma import gemma3_1b_pretrain_config
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.utils.omegaconf_utils import (
    apply_overrides,
    create_omegaconf_dict_config,
    parse_hydra_overrides,
)
from megatron.bridge.utils.common_utils import get_rank_safe

# --- NEW: Import the default forward step function ---
from megatron.bridge.training.gpt_step import forward_step


# --- All custom loss and forward_step functions have been REMOVED ---


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Add an argument specifically for data_path needed early
    parser.add_argument(
        '--initial_data_path',
        type=str,
        default=None, # Default to None if not provided
        help="Dataset path prefix needed during initial config creation."
    )
    # Parse known args, let the rest be handled by OmegaConf later
    known_args, cli_dotlist_overrides = parser.parse_known_args()

    # 1. Get base config, PASSING data_path if provided
    initial_data_paths_list = [known_args.initial_data_path] if known_args.initial_data_path else None

    config: ConfigContainer = gemma3_1b_pretrain_config(
        data_paths=initial_data_paths_list # Pass it here
    )

    # 2. Convert Python dataclass TO OmegaConf DictConfig
    merged_omega_conf, excluded_fields = create_omegaconf_dict_config(config)

    # 3. Apply command-line overrides (OmegaConf + OmegaConf merge)
    if cli_dotlist_overrides:
        print(f"Applying CLI overrides: {cli_dotlist_overrides}")
        merged_omega_conf = parse_hydra_overrides(merged_omega_conf, cli_dotlist_overrides)

    # 4. Apply the final merged OmegaConf values BACK to the Python dataclass
    final_overrides_as_dict = OmegaConf.to_container(merged_omega_conf, resolve=True)
    apply_overrides(config, final_overrides_as_dict, excluded_fields)

    # Optional: Print final config on rank 0
    if get_rank_safe() == 0:
        print("--- Final Merged Configuration ---")
        config.print_yaml()
        print("----------------------------------")
        
    # 5. Run finetuning using the default gpt_step
    finetune(
        config=config,
        forward_step_func=forward_step, # <-- USE THE DEFAULT STEP
    )

    # Cleanup (Optional but good practice)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()