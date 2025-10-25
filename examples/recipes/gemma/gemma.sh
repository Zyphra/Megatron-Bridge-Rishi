#!/bin/bash

# --- Environment ---
# Ensure your Python environment (e.g., venv or conda) is activated if needed
# source /path/to/your/venv/bin/activate
export TOKENIZERS_PARALLELISM=false
# Set PYTHONPATH if the sys.path hack isn't used or preferred
export PYTHONPATH=/workspace/Megatron-Bridge-Rishi/src:/workspace/Megatron-Bridge-Rishi/3rdparty/Megatron-LM:$PYTHONPATH

# --- Cluster Configuration ---
# Modify this section for your specific cluster
declare -A name2ip
name2ip[node-host-1]=192.168.1.1 # Replace with your node hostnames and IPs
name2ip[node-host-2]=192.168.1.2
name2ip[node-host-3]=192.168.1.3
name2ip[node-host-4]=192.168.1.4
# Add all nodes participating in the job

declare -A rank
rank[node-host-1]=0 # Rank 0 is the master
rank[node-host-2]=1
rank[node-host-3]=2
rank[node-host-4]=3
# Assign ranks from 0 to NNODES-1

NNODES=4 # Total number of nodes
GPUS_PER_NODE=8 # Number of GPUs per node
MASTER_PORT=6121 # An open port for communication

# --- Distributed Setup Logic (Copied from your example) ---
current_node=$(hostname)
current_rank=${rank[$current_node]}

master_node=NOT_FOUND
for k in "${!rank[@]}"; do
    if [ ${rank[$k]} = 0 ]; then
        master_node="$k"
    fi
done
if [ $master_node = NOT_FOUND ]; then
    echo "Must specify rank 0 node!"
    exit 1
fi
echo "Current node: $current_node. Current rank: $current_rank. Master node: $master_node. Master address: ${name2ip[$master_node]}"

NODE_RANK=$current_rank
export MASTER_ADDR=${name2ip[$master_node]}
export MASTER_PORT=$MASTER_PORT
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
# --- End Distributed Setup ---


# --- Script and Paths ---
MEGATRON_BRIDGE_SCRIPT="/workspace/Megatron-Bridge-Rishi/examples/recipes/gemma/my_gemma3_diffusion_test.py"
CHECKPOINT_PATH="/checkpoints/my_gemma3_diffusion_run" # Example path
DATA_PATH="/workspace/Megatron-Bridge-Rishi/my-dummy-dataset" # Path PREFX for your .bin/.idx files
LOG_DIR="/logs/my_gemma3_diffusion_run" # Example log directory
RUN_NAME="gemma3_1b_diffusion_test"

mkdir -p $LOG_DIR

# --- Megatron-Bridge Configuration Overrides (Hydra/OmegaConf Style) ---
# Start with your single-node test overrides and adjust/add as needed for multi-node
# Use '+' prefix only if you encounter the specific Hydra error again.
MBRIDGE_ARGS=(
    # Training Loop
    train.train_iters=100000 # Set a realistic number of iterations
    train.eval_interval=1000
    train.eval_iters=10

    # Checkpointing
    checkpoint.save=$CHECKPOINT_PATH
    checkpoint.load=$CHECKPOINT_PATH # Optional: Load from checkpoint if resuming
    checkpoint.save_interval=500 # Example save frequency

    # Distributed Settings (Passed as overrides to the script)
    # These will be applied to the 'dist' section of the ConfigContainer
    dist.tensor_model_parallel_size=2 # Example TP=2
    dist.pipeline_model_parallel_size=2 # Example PP=2
    # Add other dist args as needed (e.g., dist.expert_model_parallel_size)

    # Scheduler (Adjust to match train_iters)
    scheduler.lr_warmup_iters=100 # Example warmup
    # Let decay_iters be calculated from fraction or set explicitly:
    # scheduler.lr_decay_iters=100000

    # Dataset
    dataset.data_path=$DATA_PATH
    # Add other dataset args like dataset.split if needed

    # Model Parameters (Use the *actual* Gemma 3 1B defaults from the recipe)
    # You usually DON'T need to override these unless making changes
    # model.num_layers=26
    # model.hidden_size=1152
    # model.num_attention_heads=4
    # model.ffn_hidden_size=6912
    # model.seq_length=32768
    # model.num_query_groups=1 # Specific to Gemma 3 1B

    # Batch Size
    train.micro_batch_size=1 # Adjust based on memory
    train.global_batch_size=32 # Example: (MBS=1) * (PP*TP*DP=4*8) = 32

    # Optimizer
    optimizer.lr=1e-4 # Example LR

    # Logging
    logger.log_interval=10
    # wandb.enabled=true # Example WandB enablement
    # wandb.project="my_project"
    # wandb.name=$RUN_NAME
)


# --- Execute Training ---
echo "Starting torchrun..."
torchrun ${DISTRIBUTED_ARGS[@]} $MEGATRON_BRIDGE_SCRIPT \
    "${MBRIDGE_ARGS[@]}" \
    2>&1 | tee "${LOG_DIR}/${RUN_NAME}_rank${NODE_RANK}.log"

echo "Training finished."