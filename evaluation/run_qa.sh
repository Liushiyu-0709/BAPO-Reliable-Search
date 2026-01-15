export TOKENIZERS_PARALLELISM=true

batch_size=100
search_template=search_faith
model_path=
# hotpotqa musique 2wiki bamboogle mix_grpo
for dataset_name in hotpotqa musique 2wiki bamboogle
do
python run.py \
    --model_path ${model_path} \
    --dataset_name ${dataset_name} \
    --task qa \
    --gpu_use 0.70 \
    --max_tokens 8192 \
    --max_input_len 16384 \
    --output_path ${output_path_prefix}/${search_template}_${dataset_name}.json \
    --counts 200 \
    --batch_size ${batch_size} \
    --prompt_type ${search_template}
done

