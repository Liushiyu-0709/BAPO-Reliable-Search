export TOKENIZERS_PARALLELISM=true


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB__SERVICE_WAIT=300
# export PYTHONPATH=/src/verl:$PYTHONPATH
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU

actor_model_path={your_path}/Qwen2.5-3B-Instruct

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

experiment_name=rl_bapo_3b
wandb_key=

bash scripts/train/train.sh \
    --train_batch_size 64 \
    --ppo_mini_batch_size 16 \
    --rollout_n 8 \
    --apply_chat True \
    --prompt_template_name re_search_faith_template \
    --actor_model_path ${actor_model_path} \
    --project_name RAPO \
    --experiment_name ${experiment_name} \
    --nnodes 1 \
    --n_gpus_per_node 8 \
    --save_freq 10 \
    --test_freq 1 \
    --total_epochs 2 \
    --oversample_faith True \
    --idk_ratio 0.05 \
    --stage_level True \
    --sample_level True \
    --tensor_model_parallel_size 1 \
    --reward_manager re_search_faith \
    --optimizer_offload False \
    --wandb_api_key ${wandb_key} \
    --save_path {your_output_path}/out/${experiment_name} \
    --train_files ./dataset/train_qa.parquet \
    --test_files ./dataset/test_qa.parquet
# fi