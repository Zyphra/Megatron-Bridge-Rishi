
# Keep this unchanged, if running on the same cluster

declare -A name2ip



# VP second config

name2ip[g293.cluster.local]=10.10.33.7

name2ip[g294.cluster.local]=10.10.33.8

name2ip[g295.cluster.local]=10.10.33.9

name2ip[g296.cluster.local]=10.10.33.10

name2ip[g297.cluster.local]=10.10.33.11

name2ip[g298.cluster.local]=10.10.33.12

name2ip[g299.cluster.local]=10.10.33.13

name2ip[g300.cluster.local]=10.10.33.14

name2ip[g301.cluster.local]=10.10.33.15

name2ip[g302.cluster.local]=10.10.33.16

name2ip[g303.cluster.local]=10.10.34.1

name2ip[g304.cluster.local]=10.10.34.2

name2ip[g305.cluster.local]=10.10.34.3

name2ip[g306.cluster.local]=10.10.34.4

name2ip[g307.cluster.local]=10.10.34.5

name2ip[g308.cluster.local]=10.10.34.6

name2ip[g309.cluster.local]=10.10.34.7

name2ip[g310.cluster.local]=10.10.34.8

name2ip[g311.cluster.local]=10.10.34.9

name2ip[g312.cluster.local]=10.10.34.10

name2ip[g313.cluster.local]=10.10.34.11

name2ip[g314.cluster.local]=10.10.34.12

name2ip[g315.cluster.local]=10.10.34.13

name2ip[g316.cluster.local]=10.10.34.14

name2ip[g317.cluster.local]=10.10.34.15

name2ip[g318.cluster.local]=10.10.34.16



############################################################################################################################################

# This is the beginning of the block that configures distributed arguments, like node ranks, master address/port, number of nodes, etc.

# Modify to suit your multi-node needs: set ranks appropriately, and update NNODES.



declare -A rank



#rank[g293.cluster.local]=0

#rank[g294.cluster.local]=1

#rank[g295.cluster.local]=2

#rank[g296.cluster.local]=3

#rank[g297.cluster.local]=4

#rank[g298.cluster.local]=5

rank[g299.cluster.local]=0

#rank[g300.cluster.local]=7

#rank[g301.cluster.local]=8

#rank[g302.cluster.local]=9

#rank[g303.cluster.local]=10

#rank[g304.cluster.local]=11

#rank[g305.cluster.local]=12

#rank[g306.cluster.local]=13

#rank[g307.cluster.local]=14

#rank[g308.cluster.local]=15

#rank[g309.cluster.local]=0

#rank[g310.cluster.local]=1

#rank[g311.cluster.local]=0

#rank[g312.cluster.local]=1

#rank[g313.cluster.local]=0

#rank[g314.cluster.local]=1

#rank[g315.cluster.local]=2

#rank[g316.cluster.local]=3

#rank[g317.cluster.local]=0

#rank[g318.cluster.local]=1



NNODES=1
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

export WANDB_API_KEY=6f0443d34d41df289b878635a247d89381c06271

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
HF_MODEL_PATH="/datasets/processed_gemma3/models"
CHECKPOINT_PATH="/checkpoints/gemmasbd" # Example path
DATA_PATH="/datasets/processed_gemma3/consolidated/zyda/train_text_document" # Path PREFX for your .bin/.idx files
LOG_DIR="/logs/my_gemma3_diffusion_run" # Example log directory
RUN_NAME="gemma3_1b_diffusion_test"
WANDB_PROJECT="sbdgemma_discrete" # From old script
WANDB_SAVE_DIR="/logs/$RUN_NAME" # From old script
mkdir -p $LOG_DIR

# --- Megatron-Bridge Configuration Overrides (Hydra/OmegaConf Style) ---
# Start with your single-node test overrides and adjust/add as needed for multi-node
# Use '+' prefix only if you encounter the specific Hydra error again.
MBRIDGE_ARGS=(
    # Training Loop
    checkpoint.pretrained_checkpoint="/checkpoints/gemmasbd"
    train.train_iters=1000 # Short test run
    train.eval_interval=1000000000000
    train.eval_iters=0
    model.seq_length=2048
    dataset.sequence_length=2048
    # Checkpointing
    checkpoint.save=null # Disable saving for test
    # checkpoint.load=$CHECKPOINT_PATH

    # --- REMOVE THESE LINES TEMPORARILY ---
    # dist.tensor_model_parallel_size=1
    # dist.pipeline_model_parallel_size=1
    # --- END REMOVAL ---

    # Scheduler (Adjust to match train_iters)
    scheduler.lr_warmup_iters=1 # Adjust for short run
    scheduler.lr_decay_iters=10 # Adjust for short run

    # Dataset
    #dataset.data_path=$DATA_PATH
    #+dataset.data_path=$DATA_PATH
    # Size (Keep small for single node test)
    train.micro_batch_size=1
    train.global_batch_size=8 # GBS = MBS * DP = 1 * 8 (since TP=1, PP=1)

    # Optimizer
    optimizer.lr=1e-4

    # Logging
    logger.log_interval=1 # Log every step for debugging

    # --- WANDB ARGS (New Hydra format) ---
    logger.wandb_project=$WANDB_PROJECT
    logger.wandb_exp_name=$RUN_NAME
    logger.wandb_save_dir=$WANDB_SAVE_DIR
)


# --- Execute Training ---
echo "Starting torchrun..."
torchrun ${DISTRIBUTED_ARGS[@]} $MEGATRON_BRIDGE_SCRIPT \
    --initial_data_path $DATA_PATH \
    "${MBRIDGE_ARGS[@]}" \
    2>&1 | tee "${LOG_DIR}/${RUN_NAME}_rank${NODE_RANK}.log"

echo "Training finished."