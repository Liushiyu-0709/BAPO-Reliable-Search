output_path=

for dataset_name in hotpotqa musique 2wiki bamboogle
do
    python evaluate/scripts/evaluate.py \
        --output_path ${output_path} \
        --task qa \
        --dataset_name ${dataset_name} \
        --extract_answer \
        --use_llm
done

