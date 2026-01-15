
## BAPO: Boundary-Aware Policy Optimization for Reliable Agentic Search
<div align="center"> 

[![Paper](https://img.shields.io/badge/Paper-arXiv-b5212f.svg?logo=arxiv)](https://arxiv.org/abs/2505.16410)
[![License](https://img.shields.io/badge/LICENSE-MIT-green.svg)](https://opensource.org/licenses/MIT) 
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/) 

</div> 

## 💡 Overview
**Boundary-Aware Policy Optimization（BAPO）** is a novel **reinforcement learning-based framework** for training reliable agentic search models. Beyond correctness rewards, BAPO incorporates boundary-aware rewards to encourage appropriate "I Don't Know" (IDK) responses. To tackle the tradeoff between exploration and exploitation during RL training, we introduce an adaptive reward modulator to prevent the model from being over-encouraged to admit ignorance. 


# 🧑‍🏫 Guide for Implementation


### 1. Environment Setup

You can install the required packages by checking requirements.txt and following the steps below:

```bash
#create conda env
conda create -n bapo python==3.10
conda activate bapo

# install torch
pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# install faiss gpu
# Due to the incompatibility when installing faiss using pip, it is necessary to use the following conda command for installation.
conda install -c pytorch -c nvidia faiss-gpu=1.8.0


# install RL basic env
cd RL_train
pip3 install -e .

# install other requirements
cd ../
pip install -r requirements.txt
```

### 2. Model Download
Our default training is based on the Qwen series models, including the Qwen2.5-instruct series from 3B, 7B to 14B. You can download the corresponding checkpoints from Hugging Face:

- [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct)



### 3. Retriever Serving Deployment

In this section, we will deploy the retriever for performing search tasks on Wikipedia-based datasets. We provide a Wikipedia retriever service implemented using FlashRAG.

More details can be found in the [FlashRAG documentation](https://github.com/RUC-NLPIR/FlashRAG/tree/main?tab=readme-ov-file#rocket-quick-start).

#### Configuration

The config file is located at `evaluation/search/serving_config.yaml`:

```yaml
retrieval_method: "{your_path}/e5-base-v2"  # name or path of the retrieval model.
index_path: "{your_path}/e5_flat_inner.index" # path to the indexed file
faiss_gpu: True # whether use gpu to hold index
corpus_path: "{your_path}/wiki18_100w.jsonl"  # path to corpus in '.jsonl' format that store the documents
```

#### Required Downloads

You need to download the following components:

**Corpus:** [wiki18_100w.jsonl](https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset/tree/master/retrieval_corpus/wiki18_100w.jsonl)

**Retrieval Model:** [e5-base-v2](https://huggingface.co/intfloat/e5-base-v2)

**Index:** [wiki18_100w_e5_index.zip](https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset/tree/master/retrieval_corpus/wiki18_100w_e5_index.zip)

#### Starting the Service

To start the retriever serving, first fill in `evaluation/search/serving_config.yaml` with the correct paths to the retrieval model, index, and corpus, as well as available GPU IDs. Then, run the following command:

```bash
cd evaluation/search
python host_wiki.py \
    --config serving_config.yaml \
    --num_retriever {num_retriever} \
    --port {port}
```

### 4. BAPO Training

Our training framework is based on [verl](https://github.com/volcengine/verl). The training scripts can be found under `scripts/train`.

#### Dataset

The training dataset is located at `dataset/train_qa.parquet` and the test set is at `dataset/test_qa.parquet`. Our data consists of the QA part of the dataset released by [Tool-Star](https://github.com/RUC-NLPIR/Tool-Star), totaling 5k samples.

**Original dataset links:**
- [Training parquet](https://huggingface.co/datasets/dongguanting/Multi-Tool-RL-10K)
- [Test parquet](https://github.com/dongguanting/Tool-Star/blob/main/Tool_Star_RL/mix_grpo/grpo_mix_test.parquet)

#### Training Command

```bash
export PYTHONPATH=/src/verl:$PYTHONPATH
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU

bash scripts/train/train.sh \
    --train_batch_size 64 \
    --ppo_mini_batch_size 16 \
    --rollout_n 8 \
    --apply_chat True \
    --prompt_template_name re_search_faith_template \
    --actor_model_path ${actor_model_path} \
    --project_name Tool-Star \
    --experiment_name ${experiment_name} \
    --nnodes 1 \
    --n_gpus_per_node 8 \
    --save_freq 10 \
    --test_freq 1 \
    --total_epochs 2 \
    --oversample_faith True \
    --idk_ratio 0.1 \
    --stage_level True \
    --sample_level True \
    --tensor_model_parallel_size 1 \
    --reward_manager re_search_faith \
    --wandb_api_key {wandb_key} \
    --save_path {your_output_path}/out/${experiment_name} \
    --train_files ./dataset/train_qa.parquet \
    --test_files ./dataset/test_qa.parquet \
    --search_url http://0.0.0.0:1243
```

**Note:** `train.sh` is a training script template where you can pass several key hyperparameters to configure the training, while some fixed hyperparameters can be adjusted within the `train.sh` file itself.

#### Quick Start

We provide several example training scripts. You can run the following script to start training:

```bash
cd ./RL_train
bash scripts/train/run_bapo.sh
```

**Important:** If you encounter OOM (Out of Memory) issues (typically when training 14B models), we recommend increasing the `tensor_model_parallel_size` parameter (e.g., set it to 2).

#### Checkpoint Conversion

For the trained RL checkpoint, you can follow the code below to convert the weights to Hugging Face format:
```bash
# Merge RL weights and save in the same path.
python ./RL_train/model_merger.py \
    --local_dir {save_path}/global_step_156/actor \
```

### 5. Evaluation

For evaluation of our trained checkpoints, we include four QA datasets: **HotpotQA**, **2WikiMultiHopQA**, **Bamboogle**, and **MuSiQue**.

#### Configuration Setup

First, replace the search URL in `evaluation/utils.py` with the configuration from [3. Retriever Serving Deployment](#3-retriever-serving-deployment):

```python
def search(query: str):
    if query == '':
        return 'invalid query'

    url = f'your_search_api_url'  # Replace with your actual search API URL
    ...

def batch_search(query: Union[str, List[str]], top_n=5) -> List[str]:
    if len(query) == 0:
        return 'invalid query'

    url = f'your_search_api_url'  # Replace with your actual search API URL
    ...
```

#### Running Inference

Start the inference using the following command with recommended default parameters:

```bash
cd evaluation
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=true
export PYTHONPATH=/path/to/your_path:$PYTHONPATH

python run.py \
    --model_path /path/to/your_model_path \
    --dataset_name HotpotQA \
    --task qa \
    --gpu_use 0.95 \
    --max_tokens 8192 \
    --max_input_len 8192 \
    --output_path /path/to/your_results/your_exp_result.json \
    --counts 200 \
    --batch_size 100
```


#### Parameter Explanations

| Parameter | Description |
|-----------|-------------|
| `--model_path` | Path to your trained model |
| `--dataset_name` | Dataset name (supports: HotpotQA, 2WikiMultiHopQA, Bamboogle, MuSiQue) |
| `--task` | Set to `qa` for QA reasoning datasets |
| `--gpu_use` | GPU memory utilization ratio |
| `--max_tokens` | Maximum number of tokens the model can generate |
| `--max_input_len` | Maximum input tokens the model can accept |
| `--output_path` | Path to save the evaluation results |
| `--counts` | Number of samples to evaluate from the test set |
| `--batch_size` | Batch size for parallel inference |

### 6. Calculate Metrics

This section describes how to calculate evaluation metrics using the LLM-as-judge mechanism.

#### API Configuration

First, replace the API URL and API key with your own in `evaluation/evaluate/scripts/evaluate.py`:

```python
async def llm_evaluate_equivalence_batch(
    questions: List[str],
    labeled_answers: List[str],
    pred_answers: List[str],
    api_base_url: str = "", #input your api base url
    model_name: str = "", # input the model name
    api_key: str = "", #input your api key
    concurrent_limit: int = 50,
    extract_answer: bool = False
) -> List[bool]:
    """
    Evaluate multiple answer pairs concurrently using LLM
    """
    ...
```
#### Running Evaluation

Execute the following command to calculate metrics:

```bash
cd evaluation
python evaluate/scripts/evaluate.py \
    --output_path /path/to/your_results/your_exp_result.json \
    --task qa \
    --dataset_name HotpotQA \
    --use_llm \
    --extract_answer
```

#### Parameter Explanations

| Parameter | Description |
|-----------|-------------|
| `--output_path` | Path to the evaluation results file |
| `--task` | Task type: `qa` for QA reasoning |
| `--dataset_name` | Dataset name for evaluation |
| `--use_llm` | Enable LLM-as-judge evaluation mechanism |
| `--extract_answer` | Enable exact matching |

---
## 📚 Documentation
We have provided a comprehensive code implementation guide for BAPO to help users understand the codebase and facilitate replication and modification.

- [Code Implementation Guide](./RL_train/BAPO_IMPLEMENTATION_README.md)

---

## 🙏 Acknowledgements

This work is implemented based on [verl](https://github.com/volcengine/verl), [ReCall](https://github.com/Agent-RL/ReCall) and [Tool-Star](https://github.com/RUC-NLPIR/Tool-Star). We sincerely thank the authors of these projects for their valuable contributions to the open-source community.


