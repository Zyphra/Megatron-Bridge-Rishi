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
# This ensures they are checked first during imports
if submodule_path not in sys.path:
    sys.path.insert(0, submodule_path)
    print(f"[DEBUG] Inserted submodule path: {submodule_path}")
if src_path not in sys.path:
    sys.path.insert(0, src_path)
    print(f"[DEBUG] Inserted src path: {src_path}")

# --- END PATH HACK ---
#my_gemma3_diffusion_test.py
# --- Keep the path printing for verification ---
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
# --- End path printing ---
from pathlib import Path
from typing import Dict, Union

import torch
from megatron.core import parallel_state as mpu # <-- NEW: Import modern mpu
from omegaconf import OmegaConf

from megatron.bridge.recipes.gemma import gemma3_1b_pretrain_config
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.state import GlobalState # <-- NEW: Replaces get_args()
from megatron.bridge.training.utils.omegaconf_utils import (
    apply_overrides,
    create_omegaconf_dict_config,
    parse_hydra_overrides,
)
from megatron.bridge.utils.common_utils import get_rank_safe

# --- START: COPY-PASTE YOUR LOSS FUNCTION HERE ---
# I've adapted it to work with the new framework

def my_custom_sbd_loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    """
    Your custom SBD loss function, adapted for Megatron-Bridge.
    """
    # 1. Get config from GlobalState instead of get_args()
    state = GlobalState()
    args = state.cfg # args is now the ConfigContainer

    # The output_tensor is now shaped [batch, 2 * seq_len]
    seq_len = output_tensor.shape[1] // 2
    #ntp_losses = output_tensor[:, :seq_len].float()
    #sbd_losses = output_tensor[:, seq_len:].float()
    ntp_losses = output_tensor.float()
    sbd_losses = output_tensor.float()

    loss_mask = loss_mask.view(-1).float()
    total_tokens = loss_mask.sum()

    # Calculate scalar loss for each component
    ntp_loss_sum = torch.sum(ntp_losses.contiguous().view(-1) * loss_mask)
    sbd_loss_sum = torch.sum(sbd_losses.contiguous().view(-1) * loss_mask)
    
    # The combined loss for the backward pass
    total_loss_for_backward = ntp_loss_sum + sbd_loss_sum
    
    loss_vector = torch.stack([total_loss_for_backward, total_tokens, ntp_loss_sum, sbd_loss_sum])

    # 2. Use new parallel_state (mpu) for communication
    #CP IS NOT SUPPORTED RIGHT NOW 
    #if args.dist.context_parallel_size > 1:
    #    torch.distributed.all_reduce(loss_vector, group=mpu.get_context_parallel_group())

    # 3. Check for NaN (using the config flag)
    #if args.check_for_nan_in_loss_and_grad:
    #    global_rank = torch.distributed.get_rank()
    #    assert not loss_vector[0].isnan(), (
    #        f'Rank {global_rank}: found NaN in local forward loss calculation.'
    #    )

    # 4. Reduce for logging (using new mpu)
    reporting_loss_vector = loss_vector.clone().detach()
    torch.distributed.all_reduce(reporting_loss_vector, group=mpu.get_data_parallel_group())

    # 5. Prepare metrics dictionary
    reporting_total_tokens = reporting_loss_vector[1] + 1e-6 # Avoid division by zero
    metrics = {
        'lm loss': (reporting_loss_vector[0], reporting_loss_vector[1]),
        'ntp loss': (reporting_loss_vector[2], reporting_loss_vector[1]),
        'sbd loss': (reporting_loss_vector[3], reporting_loss_vector[1]),
        'ntp loss avg': (reporting_loss_vector[2] / reporting_total_tokens, reporting_loss_vector[1]),
        'sbd loss avg': (reporting_loss_vector[3] / reporting_total_tokens, reporting_loss_vector[1]),
    }
    
    # 6. Get the final loss for backward
    loss_for_backward = loss_vector[0] #* args.dist.context_parallel_size

    # The train function needs the loss and the metrics dict
    return loss_for_backward, metrics
# --- END: YOUR LOSS FUNCTION ---


# This is the function you pass to pretrain/finetune
def my_custom_diffusion_forward_step(
    data_iterator: Union[Dict, torch.Tensor], model: torch.nn.Module
) -> (torch.Tensor, Dict):
    """
    This is the main forward_step_func.
    It gets data, runs the model, and then calls your custom loss function.
    """
    try:
        # --- NEW: Get the actual batch dictionary from the iterator ---
        batch = next(data_iterator)
        # --- END NEW ---
        #print("[DEBUG] Batch Keys:", batch.keys())
        # 1. Get data from the batch dictionary
        #    Adapt keys if your dummy dataset uses different names
        tokens = batch["tokens"].cuda()
        loss_mask = batch["loss_mask"].cuda()
        attention_mask = batch["attention_mask"].cuda()
        position_ids = batch["position_ids"].cuda()
        labels=batch["labels"].cuda()

        # 2. Run your modified model (from your fork)
        #    This will return the concatenated (ntp_loss, sbd_loss) tensor
        # --- Ensure your model expects these args ---
        output_tensor = model(
            input_ids=tokens,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels # Your model computes loss internally? Or pass batch["labels"]?
        )
        # --- Check model output format ---


        # 3. Call your custom loss function
        loss, metrics = my_custom_sbd_loss_func(loss_mask, output_tensor)

        # Optional logging
        state = GlobalState()
        #if state.iter % state.cfg.logger.log_interval == 0:
        
        print(f"Iter: nah, Total Loss: {loss.item()}")

        # 4. Return what the `train` loop expects
        return loss, metrics

    except Exception as e:
        print(f"Error in forward step: {e}")
        raise e


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
    # This is the most important step!
    known_args, cli_dotlist_overrides = parser.parse_known_args()

    # 1. Get base config, PASSING data_path if provided
    #    The recipe expects a list for data_paths, even if it's just one.
    initial_data_paths_list = [known_args.initial_data_path] if known_args.initial_data_path else None

    config: ConfigContainer = gemma3_1b_pretrain_config(
        data_paths=initial_data_paths_list # Pass it here
    )

    # 2. Get command line overrides (Hydra-style dotlist)
    #    cli_dotlist_overrides is already correctly populated by parser.parse_known_args()
    #    It now *only* contains things like 'train.train_iters=10'

    # --- START: ADOPT EXAMPLE SCRIPT'S PATTERN ---

    # 3. Convert Python dataclass TO OmegaConf DictConfig
    merged_omega_conf, excluded_fields = create_omegaconf_dict_config(config)

    # 4. Apply command-line overrides (OmegaConf + OmegaConf merge)
    if cli_dotlist_overrides:
        print(f"Applying CLI overrides: {cli_dotlist_overrides}")
        merged_omega_conf = parse_hydra_overrides(merged_omega_conf, cli_dotlist_overrides)

    # 5. Apply the final merged OmegaConf values BACK to the Python dataclass
    final_overrides_as_dict = OmegaConf.to_container(merged_omega_conf, resolve=True)
    apply_overrides(config, final_overrides_as_dict, excluded_fields)

    # --- END: ADOPT EXAMPLE SCRIPT'S PATTERN ---

    # Optional: Print final config on rank 0
    if get_rank_safe() == 0: # <-- Use the safe version
        print("--- Final Merged Configuration ---")
        config.print_yaml()
        print("----------------------------------")
        
    # 6. Run finetuning using the correctly updated Python dataclass
    finetune(
        config=config,
        forward_step_func=my_custom_diffusion_forward_step,
    )

    # Cleanup (Optional but good practice)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()