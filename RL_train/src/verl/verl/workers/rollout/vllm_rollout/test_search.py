from typing import List
from contextlib import contextmanager
from omegaconf import DictConfig
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
import time
import requests
from functools import wraps
from typing import Union


from verl.workers.rollout.vllm_rollout.python_executor import PythonExecutor
from verl import DataProto
from verl.utils.torch_functional import get_eos_mask, pad_sequence_to_length
from verl.workers.rollout.base import BaseRollout
from verl.third_party.vllm import LLM, vllm_version
from verl.third_party.vllm import parallel_state as vllm_ps
from vllm import SamplingParams

def batch_search(query: Union[str, List[str]], top_n=5):
    if len(query) == 0:
        return 'invalid query'

    url = f'http://183.174.229.164:1243/batch_search'
    if isinstance(query, str):
        query = [query]
    data = {'query': query, 'top_n': top_n}
    response = requests.post(url, json=data)
    
    result_list = []
    for item in response.json():
        curr_result = ''
        for line in item:
            curr_result += f"{line['contents']}\n\n"
    result_list.append(curr_result.strip())
    
    return result_list

if __name__ == "__main__":
    query = "What is the capital of France?"
    top_n = 5
    result = batch_search(query, top_n)
    print(result)
