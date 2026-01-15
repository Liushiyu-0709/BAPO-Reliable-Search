LOCAL_DIR="/home/u2024001021/research_checkpoint/global_step_40"
Target_dir=${LOCAL_DIR}+"/hf"
# 执行模型合并
python model_merge.py \
    --local_dir ${LOCAL_DIR} \
    --hf_upload_path ${Target_dir}