# 1. Set your paths
export HF_MODEL_PATH="/datasets/processed_gemma3/models"
export MEGATRON_CHECKPOINT_PATH="/checkpoints/gemmasbd" # Example new path

# 2. Run the import script
python /workspace/Megatron-Bridge-Rishi/scripts/import_hf_ckpt.py \
    --model-id $HF_MODEL_PATH \
    --output-path $MEGATRON_CHECKPOINT_PATH \
    --trust-remote-code