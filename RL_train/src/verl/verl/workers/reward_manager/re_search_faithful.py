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
from verl.utils.reward_score import faithful_compute_score
import torch
import json
import math
import os




def cal_validation_score(filename):
    total_score = 0.0
    total_num = 0
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # skip empty lines
                data = json.loads(line)
                total_num += 1
                if 'score' in data:
                    total_score += float(data['score'])
    
    return total_score / total_num


class ReSearchRewardManagerWithSaveFaithful():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, save_path=None, all_steps=156, stage_level=None, sample_level=None, idk_ratio=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = faithful_compute_score
        self.save_path = save_path
        self.all_steps = all_steps
        self.stage_level = stage_level
        self.sample_level = sample_level
        self.idk_ratio = idk_ratio
        # == following variables for stage transition ==
        self.early_stage = True # initialization is the early stage
        self.tolerance_step = 0
        self.max_val_score = 0

    
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

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32) # this reward_tensor needs to be filled properly

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

        # To detect the stage split (when consecutive 5 validations show no accuracy improvement)
        if step > 1 and self.early_stage: # check if the stage transition happens in train stage
            # read the previous 5 validation scores
            # validation_files = [os.path.join(os.path.dirname(curr_save_path), f"val_{s}") for s in range(step-5, step)]
            validation_file = os.path.join(os.path.dirname(curr_save_path), f"val_{step-1}.jsonl")
            curr_score = cal_validation_score(validation_file)
            self.max_val_score = max(curr_score, self.max_val_score)
            if curr_score < self.max_val_score:
                self.tolerance_step += 1
                if self.tolerance_step > 5:
                    self.early_stage = False

        print("early_stage:", self.early_stage)
        scores, reasons, advs = self.compute_score(
                tokenizer=self.tokenizer,
                solution_str_ls=sequences_str_ls,
                ground_truth_ls=ground_truth_ls,
                step=step,
                stage_level=self.stage_level,
                sample_level=self.sample_level,
                idk_ratio=self.idk_ratio,
                early_stage=self.early_stage
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