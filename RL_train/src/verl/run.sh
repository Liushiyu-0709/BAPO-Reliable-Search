export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export VLLM_ATTENTION_BACKEND=XFORMERS



module load cuda/12.1.1
module load gcc/9.5.0 

set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_API_KEY=0a7b6feda1659e1159902ec41129bc9f0fe81019



PROMPT_KEY=question
TRAIN_BATCH_SIZE=256
PPO_MINI_BATCH_SIZE=256
MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=8192
APPLY_CHAT=False
PROMPT_TEMPLATE_NAME=qa_template
ACTOR_MODEL_PATH=/fs/archive/share/u2024001021/huggingface_models/Qwen2.5-3B-instruct  # your model path here
REWARD_MANAGER=qa
ROLLOUT_N=5
SEARCH_URL=http://0.0.0.0:1243  # your host search url here
PROJECT_NAME=hotpot_v2
EXPERIMENT_NAME=Qwen2.5-3B-Instruct_310_v1
NNODES=1
SAVE_FREQ=1
TEST_FREQ=1
TOTAL_EPOCHS=3
WANDB_API_KEY=0a7b6feda1659e1159902ec41129bc9f0fe81019
SAVE_PATH=/home/u2024001021/ReSearch/hotpotqa/dgt_model  # your save path here

TRAIN_DATA_PATH=/home/u2024001021/ReSearch/hotpotqa/train.parquet  # your train data path here
DEV_DATA_PATH=/home/u2024001021/ReSearch/hotpotqa/dev.parquet  # your dev data path here

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prompt_key) PROMPT_KEY="$2"; shift 2;;
        --train_batch_size) TRAIN_BATCH_SIZE="$2"; shift 2;;
        --ppo_mini_batch_size) PPO_MINI_BATCH_SIZE="$2"; shift 2;;
        --max_prompt_length) MAX_PROMPT_LENGTH="$2"; shift 2;;
        --max_response_length) MAX_RESPONSE_LENGTH="$2"; shift 2;;
        --apply_chat) APPLY_CHAT="$2"; shift 2;;
        --prompt_template_name) PROMPT_TEMPLATE_NAME="$2"; shift 2;;
        --actor_model_path) ACTOR_MODEL_PATH="$2"; shift 2;;
        --reward_manager) REWARD_MANAGER="$2"; shift 2;;
        --rollout_n) ROLLOUT_N="$2"; shift 2;;
        --search_url) SEARCH_URL="$2"; shift 2;;
        --project_name) PROJECT_NAME="$2"; shift 2;;
        --experiment_name) EXPERIMENT_NAME="$2"; shift 2;;
        --nnodes) NNODES="$2"; shift 2;;
        --save_freq) SAVE_FREQ="$2"; shift 2;;
        --test_freq) TEST_FREQ="$2"; shift 2;;
        --total_epochs) TOTAL_EPOCHS="$2"; shift 2;;
        --wandb_api_key) WANDB_API_KEY="$2"; shift 2;;
        --save_path) SAVE_PATH="$2"; shift 2;;
        --train_data_path) TRAIN_DATA_PATH="$2"; shift 2;;
        --dev_data_path) DEV_DATA_PATH="$2"; shift 2;;
        *)
            echo "Error: unknown argument '$1'" >&2
            exit 1;;
    esac
done

# if [ "$WANDB_API_KEY" != "None" ]; then
#     wandb login --relogin $WANDB_API_KEY
# fi

# make output directory
if [ ! -d "$SAVE_PATH" ]; then
    mkdir -p $SAVE_PATH
fi

train_files=$TRAIN_DATA_PATH
test_files=$DEV_DATA_PATH

wandb login --relogin $WANDB_API_KEY

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.001 \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.prompt_key=${PROMPT_KEY} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.apply_chat=${APPLY_CHAT} \
    data.prompt_template_name=${PROMPT_TEMPLATE_NAME} \
    actor_rollout_ref.model.path=${ACTOR_MODEL_PATH} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((2*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((4*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm_with_search \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((4*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager=${REWARD_MANAGER} \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=${SAVE_PATH} \
    hydra.run.dir=${SAVE_PATH}/outputs \
    +trainer.val_before_train=True \
    +actor_rollout_ref.rollout.search_url=${SEARCH_URL} \
    +data.save_path=${SAVE_PATH}/rollout_result.jsonl | tee $SAVE_PATH/run.log