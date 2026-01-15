# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""
import sys
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
import re
from openai import OpenAI


from verl.workers.rollout.vllm_rollout.python_executor import PythonExecutor
from verl import DataProto
from verl.utils.torch_functional import get_eos_mask, pad_sequence_to_length
from verl.workers.rollout.base import BaseRollout
from verl.third_party.vllm import LLM, vllm_version
from verl.third_party.vllm import parallel_state as vllm_ps
from vllm import SamplingParams

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


class vLLMRollout(BaseRollout):

    def __init__(self, actor_module: nn.Module, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get('max_num_batched_tokens', 8192)

        if kwargs.get('train_tp', None) is not None:
            # deployed with megatron
            import os
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            train_tp = kwargs.get('train_tp', None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
                vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                                  num_tp_per_train_tp=num_tp_per_train_tp)

        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, \
            "model context length should be greater than total sequence length"
        self.inference_engine = LLM(
            actor_module,
            tokenizer=tokenizer,
            model_hf_config=model_hf_config,
            tensor_parallel_size=tensor_parallel_size,
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            skip_tokenizer_init=False,
            max_model_len=config.prompt_length + config.response_length,
            load_format=config.load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.offload_model_weights()

        kwargs = dict(
            n=1,
            logprobs=1,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # we may detokenize the result all together later
        if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
            kwargs['detokenize'] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']

        batch_size = idx.size(0)

        idx_list = []
        # parse idx from torch.Tensor to List[List[str]]
        for i in range(batch_size):
            idx_list.append(_pre_process_inputs(self.pad_token_id, idx[i]))

        do_sample = prompts.meta_info.get('do_sample', True)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            output = self.inference_engine.generate(
                prompts=None,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                prompt_token_ids=idx_list,
                use_tqdm=False)
        # TODO(sgm): disable logprob when recompute_log_prob is enable
        # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)
        response = output[0].to(idx.device)
        log_probs = output[1].to(idx.device)

        if response.shape[1] < self.config.response_length: 
            response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
            log_probs = pad_sequence_to_length(log_probs, self.config.response_length, self.pad_token_id)

        if self.config.n > 1 and do_sample:
            idx = idx.repeat_interleave(self.config.n, dim=0)
            attention_mask = attention_mask.repeat_interleave(self.config.n, dim=0)
            position_ids = position_ids.repeat_interleave(self.config.n, dim=0)
            batch_size = batch_size * self.config.n
        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                'prompts': idx,
                'responses': response,
                'input_ids': seq,  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                'attention_mask': attention_mask,
                'position_ids': position_ids
            },
            batch_size=batch_size)

        # free vllm cache engine
        if self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch)

def retry(max: int=10, sleep: int=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(max):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if i == max - 1:
                        print(f"Retry {func.__name__} failed after {max} times")
                    elif sleep:
                        time.sleep(sleep)
        return wrapper
    return decorator

def extract_question_from_prompt(prompt: str) -> str:
    """
    Extract the question part from the complete prompt

    Args:
        prompt: Complete conversation prompt

    Returns:
        Extracted question text
    """
    user_start = prompt.find("<|im_start|>user\n")
    user_end = prompt.find("<|im_end|>\n<|im_start|>assistant")

    if user_start != -1 and user_end != -1:
        question = prompt[user_start + len("<|im_start|>user\n"):user_end].strip()
        return question

    raise ValueError("Failed to extract question from prompt")



class vLLMRolloutWithSearch(vLLMRollout):
    def __init__(self, actor_module: nn.Module, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        super().__init__(actor_module, config, tokenizer, model_hf_config, **kwargs)
        self.tokenizer = tokenizer

    @retry(max=5, sleep=1)
    def batch_search(self, query: Union[str, List[str]], top_n=5) -> List[str]:
        if len(query) == 0:
            return 'invalid query'

        url = f'http://localhost:1243/batch_search' #your local search path
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


    def batch_python(self, query: List[str]) -> List[str]:
        batch_code = query
        results = []
        executor = PythonExecutor(get_answer_from_stdout=True)
        for code in batch_code:
            result = executor.apply(code)
            if result[0] != "":
                results.append(result[0])
            else:
                results.append(result[1])
        return results


    @retry(max=5, sleep=1)
    def search(self, query: str):
        if query == '':
            return 'invalid query'

        url = f'{self.config.search_url}/search'
        data = {'query': query, 'top_n': 5} ### top1 for math
        response = requests.post(url, json=data)
        retrieval_text = ''
        for line in response.json():
            retrieval_text += f"{line['contents']}\n\n"
        retrieval_text = retrieval_text.strip()
        return retrieval_text

    def extract_search_content(self, text: str) -> str:
        try:
            start_tag = '<search>'
            end_tag = '</search>'
            end_pos = text.rindex(end_tag)
            start_pos = text.rindex(start_tag, 0, end_pos)
            return text[start_pos + len(start_tag):end_pos].strip()
        except ValueError:
            return ""
    def extract_python_content(self, text: str) -> str:
        try:
            start_tag = '<python>'
            end_tag = '</python>'
            end_pos = text.rindex(end_tag)
            start_pos = text.rindex(start_tag, 0, end_pos)
            return text[start_pos + len(start_tag):end_pos].strip()
        except ValueError:
            return ""

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        ori_input_ids = prompts.batch['input_ids']  # (bs, prompt_length)

        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']
        
        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']
        prompts_ls = self.tokenizer.batch_decode(ori_input_ids)
        prompts_ls = [prompt.replace('<|endoftext|>', '') for prompt in prompts_ls]

        #prompts_ls移除掉所有<|endoftext|>
        questions_ls = [extract_question_from_prompt(p) for p in prompts_ls]
        print("len promprs ls:", len(prompts_ls))
        assert len(questions_ls) == len(prompts_ls)
        batch_size = ori_input_ids.size(0)

        idx_list = []
        # parse idx from torch.Tensor to List[List[str]]
        for i in range(batch_size):
            idx_list.append(_pre_process_inputs(self.pad_token_id, ori_input_ids[i]))

        do_sample = prompts.meta_info.get('do_sample', True)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }

            
        with self.update_sampling_params(**kwargs):
            print(f"--------------------------------rollout开始--------------------------------")
            # prepare n copies for each input
            curr_inputs = []
            for input_ids in idx_list:
                for _ in range(self.sampling_params.n):
                    curr_inputs.append(input_ids.copy())
            init_inputs = [ids.copy() for ids in curr_inputs]
            print("len curr inputs:", len(curr_inputs))

            # track the status of each input
            curr_max_tokens = [self.sampling_params.max_tokens] * len(curr_inputs)
            active_indices = list(range(len(curr_inputs)))
            
            # Add counter to track search/python calls for each sample
            call_counters = [0] * len(curr_inputs)

            # collect the result mask of each rollout
            result_mask_list = [[] for _ in range(len(curr_inputs))]

            # generate until all inputs are finished
            while active_indices:
                # only process the active inputs
                active_inputs = [curr_inputs[i] for i in active_indices]
                active_max_tokens = [curr_max_tokens[i] for i in active_indices]

                # generate in batch, according to active max tokens
                with self.update_sampling_params(n=1, stop=['</search>','</python>'], max_tokens=max(active_max_tokens), detokenize=True):
                    outputs = self.inference_engine.generate(
                        prompts=None,
                        sampling_params=self.sampling_params,
                        prompt_token_ids=active_inputs,
                        use_tqdm=False
                    )
                
                # collect the queries to search
                search_queries = []
                search_indices = []
                python_queries = []
                python_indices = []
                # process each output
                new_active_indices = []
                for i, idx in enumerate(active_indices):
                    # print(f"--------------------------------batch样本第：{i}样本--------------------------------")
                    output_ids = outputs[0][i].tolist()
                    if self.tokenizer.eos_token_id in output_ids:
                        first_eos_idx = output_ids.index(self.tokenizer.eos_token_id)
                    else:
                        first_eos_idx = len(output_ids)
                    
                    if self.tokenizer.pad_token_id in output_ids:
                        first_pad_idx = output_ids.index(self.tokenizer.pad_token_id)
                    else:
                        first_pad_idx = len(output_ids)
                    
                    finish_reason = outputs[2][i]
                    stop_reason = outputs[3][i]

                    if finish_reason == 'stop' and isinstance(stop_reason, str) and '</search>' in stop_reason:
                        # need to search
                        ## check if we've exceeded the call limit
                        if call_counters[idx] >= 3:
                            # exceed limit, directly truncate and add EOS token
                            print(f"--------------------------------tool call limit reached, truncating with EOS--------------------------------")
                            # Add the current output truncated and append EOS
                            output_ids = output_ids[:first_pad_idx]
                            # Add EOS token to end the generation
                            output_ids.append(self.tokenizer.eos_token_id)
                            curr_inputs[idx] += output_ids
                            result_mask_list[idx] += [1] * len(output_ids)
                            # Don't add to new_active_indices to skip further generation
                            continue
                            
                        call_counters[idx] += 1
                        ## truncate from the first pad token
                        output_ids = output_ids[:first_pad_idx]
                        output_str = self.tokenizer.decode(output_ids)
                        ## process the search
                        search_content = self.extract_search_content(output_str)
                        search_queries.append(search_content)
                        search_indices.append(idx)
                        new_active_indices.append(idx)
                        ## update the current input
                        curr_inputs[idx] += output_ids
                        result_mask_list[idx] += [1] * len(output_ids)

                    if finish_reason == 'stop' and isinstance(stop_reason, str) and '</python>' in stop_reason:
                        # need to execute the python code
                        ## check if we've exceeded the call limit
                        if call_counters[idx] >= 3:
                            # exceed limit, directly truncate and add EOS token
                            print(f"--------------------------------tool call limit reached, truncating with EOS--------------------------------")
                            # Add the current output truncated and append EOS
                            output_ids = output_ids[:first_pad_idx]
                            # Add EOS token to end the generation
                            output_ids.append(self.tokenizer.eos_token_id)
                            curr_inputs[idx] += output_ids
                            result_mask_list[idx] += [1] * len(output_ids)
                            # Don't add to new_active_indices to skip further generation
                            continue
                            
                        call_counters[idx] += 1
                        output_ids = output_ids[:first_pad_idx]
                        output_str = self.tokenizer.decode(output_ids)

                        ## process the python code
                        python_content = self.extract_python_content(output_str)
                        python_queries.append(python_content)
                        python_indices.append(idx)
                        new_active_indices.append(idx)
                        ## update the current input
                        curr_inputs[idx] += output_ids
                        result_mask_list[idx] += [1] * len(output_ids)
                        
                    elif finish_reason == 'stop' and stop_reason == None:
                        # output eos, indicating finished; truncate from the first eos token
                        output_ids = output_ids[:first_eos_idx+1]
                        curr_inputs[idx] += output_ids
                        result_mask_list[idx] += [1] * len(output_ids)
                    elif finish_reason == 'stop' and stop_reason == self.tokenizer.pad_token_id:
                        # for instruction model, there is a chance that the end is endoftext, not im_end, this case needs special handling
                        output_ids = output_ids[:first_pad_idx+1]
                        curr_inputs[idx] += output_ids
                        result_mask_list[idx] += [1] * len(output_ids)
                    elif finish_reason == 'length':
                        # output is too long
                        curr_inputs[idx] += output_ids
                        result_mask_list[idx] += [1] * len(output_ids)

                # batch process the search requests
                if search_queries:
                    '''local search mode'''
                    print("---------------------wikisearch------------------")

                    search_results = self.batch_search(search_queries)
                    #-------------------------local search mode------------

                    '''Here, we use web search snippets to accelerate the training process (we recommend this mode). It is worth noting that in web_search/web_search_main.py, we support multiple web search modes, including "Web Search + Browser" and "Web Search + Browser + Summarize". '''
                    print(f"search queries: {search_queries}")
                    
                    print("search results 1 as example: ", search_results[0])
                    

                    for idx, result in zip(search_indices, search_results):
                        # update the output, add the search result
                        output_ids = self.tokenizer.encode(f" <result>\n{result}\n</result>")
                        curr_inputs[idx] += output_ids
                        result_mask_list[idx] += [0] * len(output_ids)
                if python_queries:
                    python_results = self.batch_python(python_queries)
                    print(f"python results: {python_results}")
                    for idx, result in zip(python_indices, python_results):
                        output_ids = self.tokenizer.encode(f" <result>\n{result}\n</result>")
                        curr_inputs[idx] += output_ids
                        result_mask_list[idx] += [0] * len(output_ids)

                # check if need to truncate for active indices
                length_checked_active_indices = []
                for idx in active_indices:
                    assert len(curr_inputs[idx]) - len(init_inputs[idx]) == len(result_mask_list[idx]), f"curr_inputs: {len(curr_inputs[idx])}, init_inputs: {len(init_inputs[idx])}, result_mask_list: {len(result_mask_list[idx])}"
                    if len(curr_inputs[idx]) - len(init_inputs[idx]) >= self.config.response_length:
                        curr_inputs[idx] = init_inputs[idx] \
                            + curr_inputs[idx][len(init_inputs[idx]):len(init_inputs[idx])+self.config.response_length]
                        result_mask_list[idx] = result_mask_list[idx][:self.config.response_length]
                    else:
                        curr_max_tokens[idx] = self.config.response_length - len(curr_inputs[idx]) + len(init_inputs[idx])
                        if idx in new_active_indices:
                            length_checked_active_indices.append(idx)
                active_indices = length_checked_active_indices

            output_ids_list = []
            # collect the results
            for i, input_ids in enumerate(idx_list):
                for j in range(self.sampling_params.n):
                    idx = i * self.sampling_params.n + j
                    input_len = len(input_ids)
                    output_ids_list.append(curr_inputs[idx][input_len:])

        response_list = []
        result_mask_list_padded = []
        for output_ids, result_mask in zip(output_ids_list, result_mask_list):
            assert len(output_ids) == len(result_mask), f"output_ids: {len(output_ids)}, result_mask: {len(result_mask)}"
            response = torch.tensor(output_ids, device=ori_input_ids.device)
            response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
            result_mask = torch.tensor(result_mask, device=ori_input_ids.device)
            result_mask = pad_sequence_to_length(result_mask, self.config.response_length, 0)
            response_list.append(response)
            result_mask_list_padded.append(result_mask)
        response = torch.stack(response_list, dim=0)
        result_mask = torch.stack(result_mask_list_padded, dim=0)

        if self.config.n > 1 and do_sample:
            ori_input_ids = ori_input_ids.repeat_interleave(self.config.n, dim=0)
            attention_mask = attention_mask.repeat_interleave(self.config.n, dim=0)
            position_ids = position_ids.repeat_interleave(self.config.n, dim=0)
            batch_size = batch_size * self.config.n
        seq = torch.cat([ori_input_ids, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
                
        response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # result mask: result part is 0, other part is 1
        loss_mask = result_mask * response_attention_mask
        
        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict({
            'prompts': ori_input_ids,
            'responses': response,
            'input_ids': seq,  # here input_ids become the whole sentences
            'attention_mask': attention_mask,
            'loss_mask': loss_mask,
            'position_ids': position_ids
        }, batch_size=batch_size)

        # free vllm cache engine
        if self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch)
    @torch.no_grad()
    def generate_sequences_oversample(self, prompts: DataProto, ground_truth_ls, max_resample_attempts= 2, **kwargs) -> DataProto:
        """
        Generation function with repeated sampling based on specified conditions
        For groups where all scores are 0 but no "I don't know" responses, continue sampling up to `max_resample_attempts` times
        """
        from verl.utils.reward_score.re_search_faith import compute_score_single, is_rejection

        # Get ground truth information
        labels = prompts.non_tensor_batch.get('reward_model', None)
        if labels is None:
            # If no ground truth, raise error
            raise ValueError("not ground truth key")
            # return self.generate_sequences(prompts, **kwargs)
        # ground_truth_ls = [data.non_tensor_batch['reward_model']['ground_truth'] for data in prompts]
        print("ground_truth_ls:", ground_truth_ls)
        
        rollout_size = self.config.n   # 8 samples per group in GRPO
        batch_size = prompts.batch['input_ids'].size(0) * rollout_size # This is real_batch_size
        print("real batch_size:", batch_size)
        num_groups = batch_size // rollout_size

        group_resample_counts = [0] * num_groups  # Record resample count for each group

        # First generation
        current_prompts = prompts
        current_results = self.generate_sequences(current_prompts, **kwargs)

        if prompts.meta_info.get('validate', False):
            print("===This is validation set===")
            return current_results
        print("===This is train set===")
        
        for attempt in range(max_resample_attempts):
            print(f"--------------------------------Sampling round: {attempt + 1}--------------------------------")
            # =======Extract decoded sequences for scoring========
            sequences_str_ls = []
            valid_response_length_ls = []
            
            
            for i in range(len(current_results)):
                data_item = current_results[i]

                prompt_ids = data_item.batch['prompts']
                prompt_length = prompt_ids.shape[-1]
                valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]

                response_ids = data_item.batch['responses']
                valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]
                valid_response_length_ls.append(valid_response_length)
                # decode
                sequences = torch.cat((valid_prompt_ids, valid_response_ids))
                sequences_str = self.tokenizer.decode(sequences)

                sequences_str_ls.append(sequences_str)


        # =======end============
            # Decode generated sequences for scoring
            # generated_sequences = current_results.batch['input_ids']
            # solution_str_ls = self.tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)
            #solution_str_ls = current_results.batch['responses']
            print("solution str example:", sequences_str_ls[0])
            print("ground truth example:", ground_truth_ls[0])

            # Calculate score and rejection status for each sample
            scores = []
            is_rejection_ls = []

            for i in range(batch_size):
                solution_str = sequences_str_ls[i]
                ground_truth = ground_truth_ls[i // rollout_size]  # 处理ground_truth可能比batch_size小的情况

                try:
                    score, reason = compute_score_single(self.tokenizer, solution_str, ground_truth)
                    scores.append(score)
                    is_rejection_ls.append(is_rejection(solution_str))
                except Exception as e:
                    print(f"Error calculating score: {e}")
                    scores.append(-1)
                    is_rejection_ls.append(False)

            # Analyze each group's situation and decide whether to resample
            groups_to_resample = []

            for group_idx in range(num_groups):
                group_start = group_idx * rollout_size
                group_end = group_start + rollout_size

                group_scores = scores[group_start:group_end]
                group_rejections = is_rejection_ls[group_start:group_end]

                # Check resampling conditions
                should_resample = False

                # Condition: all scores are 0, but no "I don't know" answers
                all_zero_scores = all(score <= 0 for score in group_scores)
                no_rejection = not any(group_rejections)

                if all_zero_scores and no_rejection:
                    # Check if maximum resample attempts reached
                    if group_resample_counts[group_idx] < max_resample_attempts - 1:
                        should_resample = True
                        group_resample_counts[group_idx] += 1
                        groups_to_resample.append(group_idx)
                        print(f"Group {group_idx} needs resampling due to lack of encouragement (attempt {group_resample_counts[group_idx]})")
                    else:
                        print(f"Group {group_idx} has reached maximum resample attempts, using current results")
                has_good_answer = any(score > 0.0 for score in group_scores)
                if has_good_answer:
                    print(f"Group {group_idx} already has high-score answers, no resampling needed")

                # Check if there are rejection answers
                has_rejection = any(group_rejections)
                if has_rejection and all_zero_scores:
                    print(f"Group {group_idx} has rejection answers and no correct answers, no resampling needed")

            # If no groups need resampling, end the loop
            if not groups_to_resample:
                print("All groups meet stopping conditions, ending sampling")
                break

            # If this is the last attempt, also end the loop
            if attempt == max_resample_attempts - 1:
                print("Reached maximum sampling attempts, ending sampling")
                break

            # Prepare prompts for resampling
            resample_indices = []
            for group_idx in groups_to_resample:
                group_start = group_idx * rollout_size
                group_end = group_start + rollout_size
                resample_indices.append(group_idx)

            if resample_indices:
                print(f"Resampling indices: {resample_indices}")

                # Create new prompts, only including parts that need resampling
                resample_input_ids = prompts.batch['input_ids'][resample_indices]
                resample_attention_mask = prompts.batch['attention_mask'][resample_indices]
                resample_position_ids = prompts.batch['position_ids'][resample_indices]

                resample_prompts = DataProto(
                    batch=TensorDict({
                        'input_ids': resample_input_ids,
                        'attention_mask': resample_attention_mask,
                        'position_ids': resample_position_ids
                    }, batch_size=len(resample_indices)),
                    meta_info=prompts.meta_info.copy()
                )

                # Regenerate these samples
                resample_results = self.generate_sequences(resample_prompts, **kwargs)

                # Replace the regenerated results into the original results
                for i, original_idx in enumerate(resample_indices):
                    # Replace each field
                    original_group_start = original_idx * rollout_size
                    original_group_end = original_group_start + rollout_size
                    resample_group_start = i * rollout_size
                    resample_group_end = resample_group_start + rollout_size
                    current_results.batch['prompts'][original_group_start:original_group_end] = resample_results.batch['prompts'][resample_group_start:resample_group_end]
                    current_results.batch['responses'][original_group_start:original_group_end] = resample_results.batch['responses'][resample_group_start:resample_group_end]
                    current_results.batch['input_ids'][original_group_start:original_group_end] = resample_results.batch['input_ids'][resample_group_start:resample_group_end]
                    current_results.batch['attention_mask'][original_group_start:original_group_end] = resample_results.batch['attention_mask'][resample_group_start:resample_group_end]
                    current_results.batch['position_ids'][original_group_start:original_group_end] = resample_results.batch['position_ids'][resample_group_start:resample_group_end]
                    if 'loss_mask' in current_results.batch:
                        current_results.batch['loss_mask'][original_group_start:original_group_end] = resample_results.batch['loss_mask'][resample_group_start:resample_group_end]
        return current_results
