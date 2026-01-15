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

from verl import DataProto
from verl.utils.reward_score import barrel_compute_score
import torch
import json
import math
import os

class ReSearchRewardManagerWithSaveBarrel():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, save_path=None, all_steps=156) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = barrel_compute_score
        self.save_path = save_path
        self.all_steps = all_steps

    
    def __call__(self, data: DataProto, curr_save_path=None):
        """We will expand this function gradually based on the available datasets"""

        if curr_save_path is not None:
            save_path = curr_save_path
        else:
            save_path = self.save_path
        # train_{self.global_steps}.jsonl
        if 'val' in save_path:
            step = 0
        else:
            filename = os.path.basename(curr_save_path)
            step = int(filename.replace("train_","").replace(".jsonl", ""))

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32) #最后要把这个reward_tensor填好

        already_print_data_sources = {}

        

        if save_path is not None:
            save_file = open(save_path, 'a')

        sequences_str_ls, ground_truth_ls, data_source_ls = [], [], []
        valid_response_length_ls = []
        for i in range(len(data)):

            data_item = data[i]

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

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']
            sequences_str_ls.append(sequences_str)
            ground_truth_ls.append(ground_truth)
            data_source_ls.append(data_source)

        scores, reasons, advs = self.compute_score(
                tokenizer=self.tokenizer,
                solution_str_ls=sequences_str_ls,
                ground_truth_ls=ground_truth_ls,
                step=step
            )
        for i in range(len(scores)):
            score, reason = scores[i], reasons[i]
            reward_tensor[i, valid_response_length_ls[i] - 1] = score
            if save_path is not None:
                save_json_line = {
                    'data_source': data_source_ls[i],
                    'sequences_str': sequences_str_ls[i],
                    'ground_truth': ground_truth_ls[i],
                    'score': scores[i],
                    'reason': reasons[i],
                    'advs': advs[i]
                }
                save_file.write(json.dumps(save_json_line, ensure_ascii=False) + '\n')
            data_source = data_source_ls[i]
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print('-' * 20)
                print(f"data_source: \n{data_source}")
                print(f"sequences_str: \n{sequences_str_ls[i]}")
                print(f"ground_truth: \n{ground_truth_ls[i]}")
                print(f"score: \n{score}")  
                print(f"reason: \n{reason}")
                print('-' * 20)

        if save_path is not None:
            save_file.close()

        return reward_tensor