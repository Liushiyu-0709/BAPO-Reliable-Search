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
# from . import gsm8k, math, prime_math, prime_code


def _default_compute_score(data_source, tokenizer,solution_str, ground_truth):
    print(f"--------------------------------compute_score开始--------------------------------")
    # import pdb
    # pdb.set_trace()
    if data_source == 'openai/gsm8k':
        from . import gsm8k
        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ['lighteval/MATH', 'DigitalLearningGmbH/MATH-lighteval']:
        from . import math_ori
        res = math_ori.compute_score(solution_str, ground_truth)
    elif data_source in [
            'numina_aops_forum', 'numina_synthetic_math', 'numina_amc_aime', 'numina_synthetic_amc', 'numina_cn_k12',
            'numina_olympiads'
    ]:
        from . import prime_math
        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ['codecontests', 'apps', 'codeforces', 'taco']:
        from . import prime_code
        res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source in ['musique', 'hotpotqa']:
        from . import re_search
        res = re_search.compute_score(tokenizer, solution_str, ground_truth)
    
    elif data_source in ['grpo_math']:
        from . import re_search_math
        res = re_search_math.compute_score(tokenizer, solution_str, ground_truth, reward_type='default')

    else:
        print(f"--------------------------------data_source not found--------------------------------")
        print(f"data_source: {data_source}")
        from . import re_search
        res = re_search.compute_score(tokenizer, solution_str, ground_truth)

    return res

def faithful_compute_score(tokenizer,solution_str_ls, ground_truth_ls, step, stage_level, sample_level, idk_ratio, early_stage):
    print(f"--------------------------------compute_score开始--------------------------------")

    print(f"--------------------------------faithfully compute score--------------------------------")
    # print(f"data_source: {data_source}")
    from . import re_search_faith
    res = re_search_faith.compute_score(tokenizer, solution_str_ls, ground_truth_ls, step=step, stage_level=stage_level, sample_level=sample_level, idk_ratio=idk_ratio, early_stage=early_stage)

    return res

def barrel_compute_score(tokenizer,solution_str_ls, ground_truth_ls, step):
    print(f"--------------------------------compute_score开始--------------------------------")

    print(f"--------------------------------faithfully compute score--------------------------------")
    # print(f"data_source: {data_source}")
    from . import re_search_barrel
    res = re_search_barrel.compute_score(tokenizer, solution_str_ls, ground_truth_ls, step=step)

    return res